import json
from prompt import Prompt
import time
import openai
from pathlib import Path
from selenium.webdriver.common.keys import Keys
import os
import logging

import random
import traceback

from computergym.miniwob.miniwob_interface.action import (
    MiniWoBType,
    MiniWoBElementClickId,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
)
import re

from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
import requests
from requests.exceptions import Timeout
import urllib3

import tiktoken

FEW_SHOT = 'prompt/few-shot.json'

urllib3.disable_warnings()

CONTROLLER_ADDR = os.environ['CONTROLLER_ADDR'].split(',')

BIG_PROMPT = '''
You are an agent embarking on a computer task. Each turn, you will be provided a task and an accessibility tree describing what is on the screen now, and you should either devise a overall plan to solve this task or to provide an instruction to execute. The plan could be multi-step, and each step should strictly corresponds to one instruction to execute. When devising a plan to execute, list the steps in order and precede each step with a numerical index starting from 1, e.g. "1." or "2.", and when executing, follow the plan strictly. When asked to provide an action to execute, refer strictly to the regular expression to ensure that your action is valid to execute.
'''.strip()

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def llm_gpt(prompt: list[dict[str, str]], model='gpt-3.5-turbo') -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    for _ in range(7):
        try:
            response = requests.post(
                f"{api_base}/v1/chat/completions",
                headers={
                    'Authorization': f'Bearer {api_key}'
                },
                json={
                    'model': model,
                    'messages': prompt,
                    'temperature': 0.0,
                    'max_tokens': 256,
                },
                timeout=120,
            )
            text = response.json()['choices'][0]['message']['content']
            return text.strip()
        # if timeout or connection error, retry
        except Timeout: 
            print("Timeout, retrying...")
        except ConnectionError:
            print("Connection error, retrying...")
        except Exception:
            traceback.print_exc()
            try:
                print(response)
                print(response.text)
                print('===REQUEST===')
                print(response.request)
                print(response.request.body)
            except:
                pass
        time.sleep(5 * random.random())
    else:
        raise Exception("Timeout after 7 retries.")

trajs = []

def get_prompt(conv: Conversation) -> str:
    if conv.name == 'openchat':
        ret = ''
        for role, message in conv.messages:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
        return ret
    else:
        return conv.get_prompt()

class LLMAgent:
    def __init__(
        self,
        env: str,
        rci_plan_loop: int = 1,
        rci_limit: int = 1,
        llm="chatgpt",
        with_task=True,
        state_grounding=True,
    ) -> None:
        self.rci_limit = rci_limit
        self.rci_plan_loop = rci_plan_loop
        self.llm = llm
        self.prompt = Prompt(env=env)
        self.state_grounding = state_grounding

        self.load_model()

        self.html_state = ""
        self.task = ""
        self.with_task = with_task
        self.current_plan = ""
        self.past_plan = []
        self.past_instruction = []
        self.custom_gaol = False

        self.history_name = time.strftime("%Y%m%d-%H%M%S")
        config_string = (
            f"erci{rci_plan_loop}_state{self.state_grounding}_irci{rci_limit}"
        )
        if self.prompt.example_prompt:
            self.file_path = Path(
                f"history/{self.llm}/{env}/{config_string}/few-shot/{self.history_name}.txt"
            )
        else:
            self.file_path = Path(
                f"history/{self.llm}/{env}/{config_string}/zero-shot/{self.history_name}.txt"
            )
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self.env = env

    def load_model(self):
        if self.llm == "chatgpt":
            self.model = "gpt-3.5-turbo"
        elif self.llm == "gpt4":
            self.model = "gpt-4"
        elif self.llm == "davinci":
            self.model = "text-davinci-003"
        elif self.llm == "ada":
            self.model = "ada"
        elif self.llm == "babbage":
            self.model = "babbage"
        elif self.llm == "curie":
            self.model = "curie"
        elif self.llm == "davinci1":
            self.model = "davinci"
        elif self.llm == "davinci2":
            self.model = "text-davinci-002"
        elif 'llama' in self.llm or 'openchat' in self.llm or 'vicuna' in self.llm:
            self.model = 'tgi'
        else:
            raise NotImplemented

    def save_result(self, result):
        with open(self.file_path, "a") as f:
            if result:
                f.write("\n\nSUCCESS\n\n")
                new_file_path = self.file_path.with_name(
                    f"{self.history_name}_success.txt"
                )
            else:
                f.write("\n\nFAIL\n\n")
                new_file_path = self.file_path.with_name(
                    f"{self.history_name}_fail.txt"
                )

        with (self.file_path.parent / f'{self.file_path.name}.json').open('w') as f:
            json.dump(trajs, f)

        os.rename(self.file_path, new_file_path)

        return

    def save(self, pt):
        with open(self.file_path, "a") as f:
            f.write("\n")
            ho_line = "-" * 30
            f.write(ho_line)
            f.write("\n\n")
            f.write(pt)

        return

    def set_goal(self, goal: str):
        self.custom_gaol = True
        self.task = goal

        return

    def instruction_history_prompt(self):
        pt = "\n\n"
        pt += "We have a history of instructions that have been already executed by the autonomous agent so far.\n"
        if not self.past_instruction:
            pt += "No instruction has been executed yet."
        else:
            for idx, inst in enumerate(self.past_instruction):
                pt += f"{idx+1}: "
                pt += inst
                pt += "\n"
        pt += "\n\n"

        return pt

    def webpage_state_prompt(self, init_plan: bool = False, with_task=False):
        pt = "\n\n"
        pt += "Below is the HTML code of the webpage where the agent should solve a task.\n"
        pt += self.html_state
        pt += "\n\n"
        if self.prompt.example_prompt and (init_plan or self.rci_plan_loop == -1):
            pt += self.prompt.example_prompt
            pt += "\n\n"
        if with_task:
            pt += "Current task: "
            pt += self.task
            pt += "\n"

        return pt

    def update_html_state(self, state: str):
        self.html_state = state

        return

    def rci_plan(self, pt=None):
        pt += "\n\nFind problems with this plan for the given task compared to the example plans.\n\n"
        criticizm = self.get_response(pt)
        pt += criticizm

        pt += "\n\nBased on this, what is the plan for the agent to complete the task?\n\n"
        # pt += self.webpage_state_prompt()
        plan = self.get_response(pt)

        return pt, plan

    def rci_action(self, instruciton: str, pt=None):
        instruciton = self.process_instruction(instruciton)

        loop_num = 0
        while self.check_regex(instruciton):
            if loop_num >= self.rci_limit:
                print(instruciton)
                self.save(pt)
                raise ValueError("Action RCI failed")

            pt += self.prompt.rci_action_prompt
            instruciton = self.get_response(pt)

            pt += instruciton
            instruciton = self.process_instruction(instruciton)

            loop_num += 1

        return pt, instruciton

    def check_regex(self, instruciton):
        return (
            (not re.search(self.prompt.clickxpath_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.chatgpt_type_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.davinci_type_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.press_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.clickoption_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.movemouse_regex, instruciton, flags=re.I))
        )

    def process_instruction(self, instruciton: str):
        end_idx = instruciton.find("`")
        if end_idx != -1:
            instruciton = instruciton[:end_idx]

        instruciton = instruciton.replace("`", "")
        instruciton = instruciton.replace("\n", "")
        instruciton = instruciton.replace("\\n", "\n")
        instruciton = instruciton.strip()
        instruciton = instruciton.strip("'")

        return instruciton

    def get_plan_step(self):
        idx = 1
        while True:
            if (str(idx) + ".") not in self.current_plan:
                return (idx - 1) + 1
            idx += 1

    def initialize_plan(self):
        if not self.custom_gaol:
            if self.with_task:
                self.initialize_task()

        if not self.prompt.init_plan_prompt or self.rci_plan_loop == -1:
            return

        pt = self.prompt.base_prompt
        pt += self.webpage_state_prompt(True, with_task=self.with_task)
        pt += self.prompt.init_plan_prompt

        # print('getting response...')

        message = "\n" + self.get_response(pt)

        # print('done getting response')

        pt += message

        for _ in range(self.rci_plan_loop):
            pt, message = self.rci_plan(pt)
            pt += message

        self.current_plan = message
        self.save(pt)

        return

    def get_response(self, pt):
        import inspect

        logging.info(
            f"Send a request to the language model from {inspect.stack()[1].function}"
        )

        pt_orig = pt
        while True:
            try:
                if self.llm == "chatgpt" or self.llm == "gpt4":
                    conv = get_conversation_template('gpt-3.5-turbo')
                    conv.set_system_message("You are an autoregressive language model that completes user's sentences. You should not conversate with user.")
                    with open(FEW_SHOT) as f:
                        examples = json.load(f)
                    conv.append_message(conv.roles[0], BIG_PROMPT)
                    conv.append_message(conv.roles[1], 'Ok.')
                    conv.append_message(conv.roles[0], examples[0]['conversations'][0]['value'])
                    conv.append_message(conv.roles[1], examples[0]['conversations'][1]['value'])
                    conv.append_message(conv.roles[0], "The previous task has ended, and I'll start a new task now.\n\n" + examples[5]['conversations'][0]['value'])
                    conv.append_message(conv.roles[1], examples[5]['conversations'][1]['value'])
                    conv.append_message(conv.roles[0], "The previous task has ended, and I'll start a new task now.\n\n" + pt)
                    conv.append_message(conv.roles[1], None)
                    time.sleep(random.random())

                    model = 'gpt-4' if self.llm == 'gpt4' else 'gpt-3.5-turbo'

                    prompt = conv.to_openai_api_messages()
                    if num_tokens_from_messages(prompt, model) > 4096:
                        cut = num_tokens_from_messages(prompt, model) - (4096 - 256)
                        cut *= 4
                        conv.messages[-2][1] = conv.messages[-2][1][cut:]
                    prompt = conv.to_openai_api_messages()

                    message = llm_gpt(conv.to_openai_api_messages())
                elif 'llama' in self.llm or 'openchat' in self.llm or 'vicuna' in self.llm:
                    if 'vicuna' in self.llm:
                        conv = get_conversation_template('vicuna')
                    elif 'llama' in self.llm:
                        conv = get_conversation_template('llama-2')
                        conv.set_system_message("You are a helpful, respectful and honest assistant.")
                    elif 'openchat' in self.llm:
                        conv = Conversation(
                            name="openchat",
                            roles=("GPT4 User", "GPT4 Assistant"),
                            messages=[],
                            offset=0,
                            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
                            sep="<|end_of_turn|>",
                        )
                    with open(FEW_SHOT) as f:
                        examples = json.load(f)
                    conv.append_message(conv.roles[0], BIG_PROMPT)
                    conv.append_message(conv.roles[1], 'Ok.')
                    conv.append_message(conv.roles[0], examples[0]['conversations'][0]['value'])
                    conv.append_message(conv.roles[1], examples[0]['conversations'][1]['value'])
                    conv.append_message(conv.roles[0], "The previous task has ended, and I'll start a new task now.\n\n" + examples[5]['conversations'][0]['value'])
                    conv.append_message(conv.roles[1], examples[5]['conversations'][1]['value'])

                    conv.append_message(conv.roles[0], "The previous task has ended, and I'll start a new task now.\n\n" + pt)
                    conv.append_message(conv.roles[1], None)
                    prompt = get_prompt(conv)
                    if self.model == 'tgi':
                        data = {
                            "inputs": prompt,
                            "parameters": {
                                "max_new_tokens": 512,
                                "do_sample": False,
                                'truncate': 4000,
                            }
                        }
                        for _ in range(3):
                            try:
                                response = requests.post(
                                    random.choice(CONTROLLER_ADDR) + "/generate",
                                    headers = {'Content-Type': 'application/json'},
                                    json=data,
                                    timeout=120,
                                )
                                text = response.json()["generated_text"]
                                # print(text)
                                message = text.split('[INST]')[0].strip()
                                break
                            # if timeout or connection error, retry
                            except Timeout: 
                                print("Timeout, retrying...")
                            except ConnectionError:
                                print("Connection error, retrying...")
                            time.sleep(5)
                        else:
                            raise Exception("Timeout after 3 retries.")
                    else:
                        gen_params = {
                            'model': self.model,
                            'prompt': prompt,
                            'temperature': 0,
                            'max_new_tokens': 512,
                            'stop': conv.stop_str,
                            'sotp_token_ids': conv.stop_token_ids,
                            'echo': False,
                        }
                        for _ in range(3):
                            try:
                                response = requests.post(
                                    CONTROLLER_ADDR + "/worker_generate_stream",
                                    headers = {"User-Agent": "FastChat Client"},
                                    json=gen_params,
                                    stream=True,
                                    timeout=120,
                                )
                                text = ""
                                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                                    if line:
                                        text = json.loads(line)["text"]
                                message = text
                                break
                            # if timeout or connection error, retry
                            except Timeout: 
                                print("Timeout, retrying...")
                            except ConnectionError:
                                print("Connection error, retrying...")
                            time.sleep(5)
                        else:
                            raise Exception("Timeout after 3 retries.")
                else:
                    time.sleep(1)
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=pt,
                        temperature=0,
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )
                    message = response["choices"][0]["text"]
            except Exception as e:
                print(e)
                if "maximum context" in str(e):
                    raise ValueError
                time.sleep(10)
            else:
                if message:
                    break

        trajs.append({
            'id': f'{self.env}_{len(trajs)}',
            'conversations': [
                {
                    'from': 'human',
                    'value': pt_orig,
                    # 'value_processed': prompt,
                },
                {
                    'from': 'gpt',
                    'value': message
                },
            ]
        })

        print(message)
        return message

    def generate_action(self) -> str:
        pt = self.prompt.base_prompt
        pt += self.webpage_state_prompt(with_task=self.with_task)
        if self.prompt.init_plan_prompt and self.rci_plan_loop != -1:
            pt += self.current_plan_prompt()
        pt += self.instruction_history_prompt()
        if self.past_instruction:
            update_action_prompt = self.prompt.action_prompt.replace(
                "{prev_inst}", self.past_instruction[-1]
            )
            if len(self.past_instruction) == 1:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", "2nd"
                )
            elif len(self.past_instruction) == 2:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", "3rd"
                )
            else:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", f"{len(self.past_instruction)+1}th"
                )

            action_prompt = update_action_prompt
        else:
            action_prompt = self.prompt.first_action_prompt

        if self.rci_plan_loop == -1:
            action_prompt = "Based on the task, " + action_prompt
        else:
            action_prompt = (
                "Based on the plan and the history of instructions executed so far, "
                + action_prompt
            )

        pt += action_prompt

        message = self.get_response(pt)

        pt += self.process_instruction(message) + "`."

        pt, message = self.update_action(pt, message)

        pt, instruction = self.rci_action(pt=pt, instruciton=message)

        self.past_instruction.append(instruction)

        self.save(pt)

        return instruction

    def update_action(self, pt=None, message=None):
        if self.prompt.update_action and self.state_grounding:
            pt += self.prompt.update_action
            message = self.get_response(pt)
            pt += message

        return pt, message

    def current_plan_prompt(self, pos=None):
        pt = "\n\n"
        pt += "Here is a plan you are following now.\n"

        pt += f'{self.current_plan}'

        pt += "\n\n"

        return pt

    def convert_to_miniwob_action(self, instruction: str):
        instruction = instruction.split(" ")
        inst_type = instruction[0]
        inst_type = inst_type.lower()

        if inst_type == "type":
            characters = " ".join(instruction[1:])
            characters = characters.replace('"', "")
            return MiniWoBType(characters)
        elif inst_type == "clickid":
            element_id = " ".join(instruction[1:])
            return MiniWoBElementClickId(element_id)
        elif inst_type == "press":
            key_type = instruction[1].lower()
            if key_type == "enter":
                return MiniWoBType("\n")
            elif key_type == "space":
                return MiniWoBType(" ")
            elif key_type == "arrowleft":
                return MiniWoBType(Keys.LEFT)
            elif key_type == "arrowright":
                return MiniWoBType(Keys.RIGHT)
            elif key_type == "backspace":
                return MiniWoBType(Keys.BACKSPACE)
            elif key_type == "arrowup":
                return MiniWoBType(Keys.UP)
            elif key_type == "arrowdown":
                return MiniWoBType(Keys.DOWN)
            else:
                raise NotImplemented
        elif inst_type == "movemouse":
            xpath = " ".join(instruction[1:])
            return MiniWoBMoveXpath(xpath)
        elif inst_type == "clickxpath":
            xpath = " ".join(instruction[1:])
            return MiniWoBElementClickXpath(xpath)
        elif inst_type == "clickoption":
            xpath = " ".join(instruction[1:])
            return MiniWoBElementClickOption(xpath)
        else:
            raise ValueError("Invalid instruction")
