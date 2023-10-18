"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import random
import re, string, os
import json 
import time
import tiktoken
from langchain.llms.base import BaseLLM
from langchain import OpenAI, Wikipedia
from langchain.docstore.base import Docstore
from langchain.agents.react.base import DocstoreExplorer
from langchain.prompts import PromptTemplate
from collections import Counter

from src.pre_prompt import (react_agent_prompt, zeroshot_agent_prompt, 
                         plan_prompt, planner_agent_prompt, plannerreact_agent_prompt)
from src.fewshots import REACT_EXAMPLE, PLANNER_EXAMPLE, PLAN_EXAMPLE, PLANNERREACT_EXAMPLE

from src.llms import token_enc

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        action_type, argument = fuzzy_parse_action(string)
        return action_type, argument
        
def fuzzy_parse_action(text):
    text = text.strip(' ').strip('.')
    pattern = r'^(\w+)\[(.+)\]'
    match = re.match(pattern, text)
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        return text, ''

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer = token_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(token_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
  
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)


class BaseAgent:
    def __init__(self,
                 question: str,
                 key: str,
                 llm: BaseLLM,
                 context_len: int = 2000,
                 max_steps: int= 10,
                 docstore: Docstore = Wikipedia()
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = ""
        self.examples = ""
        self.context_len = context_len
        self.run_error = False
        self.name = "Base_HotPotQA_run_Agent"

        self.docstore = DocstoreExplorer(docstore) # Search, Lookup
        self.llm = llm
        
        self.enc = token_enc
        self.__reset_agent()
    
    def run(self, reset = True) -> None:
        if reset:
            self.__reset_agent()
        
        while not self.is_halted() and not self.is_finished() and not self.run_error:
            self.step()
    
    def prompt_agent(self) -> str:
        generation = self.llm(self._build_agent_prompt())
        self.check_run_error(generation)
        return format_step(generation)
 
    def check_run_error(self, text):
        if text in ["No response"]:
            self.run_error = True
            
    def is_finished(self) -> bool:
        return self.finished
    
    def reward(self) -> float:
        return f1_score(self.answer, self.key)   
    
    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps)
                or (len(self.enc.encode(self._build_agent_prompt())) > self.context_len)
                ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

    def _think(self):
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])
    
    def _action(self):
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])
        return action_type, argument
        
    def step(self) -> None:
        
        # agent forward
        ret = self.forward()
        if ret:
            action_type, argument = ret[0], ret[1]
        else:
            action_type = ret
        
        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            self.step_n += 1
            return

        if action_type == 'Search':
            try:
                self.scratchpad += format_step(self.docstore.search(argument))
            except Exception as e:
                print(e)
                self.scratchpad += f'Could not find that page, please try again.'
            
        elif action_type == 'Lookup':
            try:
                self.scratchpad += format_step(self.docstore.lookup(argument))
            except ValueError:
                self.scratchpad += f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'

        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

        print(self.scratchpad.split('\n')[-1])

        self.step_n += 1
    
    def _build_agent_prompt(self) -> str:
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
    
class ReactAgent(BaseAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 llm,
                 context_len: int = 2000
                 ) -> None:
        super().__init__(question, key, llm, context_len)

        self.examples = REACT_EXAMPLE
        self.agent_prompt = react_agent_prompt
        self.name = "React_HotPotQA_run_Agent"
    
    def forward(self):
        self._think()
        action_type, argument = self._action()
        return action_type, argument

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.examples,
                            question = self.question,
                            scratchpad = self.scratchpad)
        
class ZeroshotAgent(BaseAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 llm,
                 context_len: int = 2000
                 ) -> None:
        super().__init__(question, key, llm, context_len)

        self.examples = ""
        self.agent_prompt = zeroshot_agent_prompt
        self.name = "Zeroshot_HotPotQA_run_Agent"
    
    def forward(self):
        action_type, argument = self._action()
        return action_type, argument

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            question = self.question,
                            scratchpad = self.scratchpad)
        
class ZeroshotThinkAgent(BaseAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 llm,
                 context_len: int = 2000
                 ) -> None:
        super().__init__(question, key, llm, context_len)

        self.examples = ""
        self.agent_prompt = zeroshot_agent_prompt
        self.name = "ZeroshotThink_HotPotQA_run_Agent"
    
    def forward(self):
        self._think()
        action_type, argument = self._action()
        return action_type, argument

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            question = self.question,
                            scratchpad = self.scratchpad)
        
class PlannerAgent(BaseAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 llm,
                 context_len: int = 2000
                 ) -> None:
        super().__init__(question, key, llm, context_len)

        self.examples = PLANNER_EXAMPLE
        self.plan_example = PLAN_EXAMPLE
        self.agent_prompt = planner_agent_prompt
        self.plan_prompt  = plan_prompt
        self.name = "Planner_HotPotQA_run_Agent"
        self._plan()
        
    def _plan(self):
        self.plan = format_step(self.llm(self._build_plan_prompt()))

        
    def _build_plan_prompt(self):
        return self.plan_prompt.format(
            examples = self.plan_example,
            question = self.question,
        )
    
    def forward(self):
        action_type, argument = self._action()
        return action_type, argument

    def _build_agent_prompt(self) -> str:
        prompt = self.agent_prompt.format(
            examples = self.examples,
            question = self.question,
            plan = self.plan,
            scratchpad = self.scratchpad)
        return prompt

class PlannerReactAgent(PlannerAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 llm,
                 context_len: int = 2000
                 ) -> None:
        super().__init__(question, key, llm, context_len)

        self.examples = PLANNERREACT_EXAMPLE
        self.plan_example = PLAN_EXAMPLE
        self.agent_prompt = plannerreact_agent_prompt
        self.plan_prompt  = plan_prompt
        self.name = "PlannerReact_HotPotQA_run_Agent"
        self._plan()
    
    def forward(self):
        self._think()
        action_type, argument = self._action()
        return action_type, argument


def get_agent(agent_name):
    if agent_name in ["Zeroshot_HotPotQA_run_Agent"]:
        return ZeroshotAgent
    if agent_name in ["ZeroshotThink_HotPotQA_run_Agent"]:
        return ZeroshotThinkAgent
    if agent_name in ["React_HotPotQA_run_Agent"]:
        return ReactAgent
    if agent_name in ["Planner_HotPotQA_run_Agent"]:
        return PlannerAgent
    if agent_name in ["PlannerReact_HotPotQA_run_Agent"]:
        return PlannerReactAgent