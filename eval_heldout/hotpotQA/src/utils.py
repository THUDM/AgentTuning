"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import os
import joblib
import json

def summarize_trial(agents):
    correct = [a for a in agents if a.is_correct()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    not_finish = [a for a in agents if not a.is_finished()]
    return correct, incorrect, not_finish

def remove_fewshot(prompt: str) -> str:
    prefix = prompt.split('Here are some examples:')[0]
    suffix = prompt.split('(END OF EXAMPLES)')[1]
    return prefix.strip('\n').strip() + '\n' +  suffix.strip('\n').strip()

def log_trial(agents, trial_n):
    correct, incorrect, not_finish = summarize_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)} , Not Finished: {len(not_finish)}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        # log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
        log += agent._build_agent_prompt() + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        # log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
        log += agent._build_agent_prompt() + f'\nCorrect answer: {agent.key}\n\n'
    
    log += '------------- BEGIN NOT_FINISH AGENTS -----------\n\n'
    for agent in not_finish:
        log += agent._build_agent_prompt() + f'\nCorrect answer: {agent.key}\n\n'

    return log

def summarize_trial_detailed(agents):
    correct = [a.is_correct() for a in agents]
    reward = [a.reward()[0] for a in agents]
    halted = [a for a in agents if a.is_halted()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    error = [a.run_error for a in agents]
    return correct, reward, error, halted, incorrect

def log_agent(agent, file_path):
    question = agent.question
    g_truth = agent.key
    correct = agent.is_correct()
    reward = agent.reward()[0]
    halted = agent.is_halted()
    error = agent.run_error
    prompt = agent._build_agent_prompt()
    save_dict = {"question":question, "answer":g_truth, "correct":correct, "reward":reward, 
                 "halted":halted, "error":error,"prompt":prompt}
    with open(file_path, 'w') as f:
        json.dump(save_dict, f)
        f.write("\n")


def get_all_agent_sessions(file_name):
    sessions = []
    with open(file_name) as f:
        for line in f:
            session = json.loads(line)
            sessions.append(session)
    return sessions

def get_error_tasks(sessions):
    error_tasks = []
    for sess in sessions:
        if sess["error"]:
            task = (sess["question"], sess["answer"])
            error_tasks.append(task)
    error_tasks = list(set(error_tasks))
    return error_tasks

def get_non_error_tasks(sessions):
    tasks = []
    for sess in sessions:   
        if not sess["error"]:    
            task = (sess["question"], sess["answer"])
            tasks.append(task)
    tasks = list(set(tasks))
    return tasks

def delete_error(file_name):
    sessions = get_all_agent_sessions(file_name)
    non_error_sessions = [sess for sess in sessions if not sess["error"]]
    with open(file_name+'.back', 'a') as b_f:
        for sess in sessions:
            json.dump(sess, b_f)
            b_f.write('\n')
    with open(file_name, 'w') as f:
        for sess in non_error_sessions:
            json.dump(sess, f)
            f.write('\n')
            
def summarize_react_trial(agents):
    correct = [a for a in agents if a.is_correct()]
    halted = [a for a in agents if a.is_halted()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect, halted

def summarize_react_trial_detailed(agents):
    correct = [a.is_correct() for a in agents]
    reward = [a.reward()[0] for a in agents]
    return correct, reward

def log_react_trial(agents, trial_n):
    correct, incorrect, halted = summarize_react_trial(agents)

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}
#######################################
"""

    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    log += '------------- BEGIN HALTED AGENTS -----------\n\n'
    for agent in halted:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    return log

def save_agents(agents, dir: str):
    os.makedirs(dir, exist_ok=True)
    for i, agent in enumerate(agents):
        agent.enc = None
        joblib.dump(agent, os.path.join(dir, f'{i}.joblib'))

def load_agents(dir:str):
    import tiktoken
    agents = []
    for f in os.listdir(dir):
        agent = joblib.load(os.path.join(dir, f))
        agent.enc = tiktoken.encoding_for_model("text-davinci-003")
        agents.append(agent)
    return agents