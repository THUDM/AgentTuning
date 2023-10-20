import argparse
import json

import requests
import os
import json
import sys
import time
import re
import math
import random
import datetime
import argparse
import requests
from typing import List, Dict, Any, Union

from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import Conversation, SeparatorStyle, Conversation
from src.agent import Agent

from requests.exceptions import Timeout, ConnectionError


class Prompter:
    @staticmethod
    def get_prompter(prompter_name: Union[str, None]):
        # check if prompter_name is a method and its variable
        if not prompter_name:
            return None
        if hasattr(Prompter, prompter_name) and callable(getattr(Prompter, prompter_name)):
            return getattr(Prompter, prompter_name)

    @staticmethod
    def claude(messages: List[Dict[str, str]]):
        prompt = ""
        role_dict = {
            "user": "Human",
            "agent": "Assistant",
        }
        for item in messages:
            prompt += f"{role_dict[item['role']]}: {item['content']}\n\n"
        prompt += "Assistant:"
        return {"prompt": prompt}

    @staticmethod
    def openchat_v3_1(messages: List[Dict[str, str]]):
        prompt = "Assistant is GPT4<|end_of_turn|>"
        role_dict = {
            "user": "User: {content}<|end_of_turn|>",
            "agent": "Assistant: {content}<|end_of_turn|>",
        }
        for item in messages:
            prompt += role_dict[item['role']].format(content=item['content'])
        prompt += "Assistant:"
        return {"prompt": prompt}

    @staticmethod
    def openchat_v3_2(messages: List[Dict[str, str]]):
        prompt = ""
        role_dict = {
            "user": "GPT4 User: {content}<|end_of_turn|>",
            "agent": "GPT4 Assistant: {content}<|end_of_turn|>",
        }
        for item in messages:
            prompt += role_dict[item['role']].format(content=item['content'])
        prompt += "GPT4 Assistant:"
        return {"prompt": prompt}


class TGIAgent(Agent):
    """This agent is a test agent, which does nothing. (return empty string for each action)"""

    def __init__(self, model_name, max_tokens, ip="0.0.0.0", address_from=23333, address_to=23334, max_new_tokens=32, temperature=1, top_p=0, prompter=None, args=None, **kwargs) -> None:
        self.controller_address = [
            ip + ":" + str(i) for i in range(address_from, address_to)]
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.prompter = Prompter.get_prompter(prompter)
        self.args = args or {}
        self.temperature = temperature
        print(self.max_new_tokens)
        super().__init__(**kwargs)

    def inference(self, history: List[dict]) -> str:
        gen_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.top_p,
            **self.args
        }
        if "AgentLM" in self.model_name:
            conv = get_conversation_template("llama-2")
            for history_item in history:
                role = history_item["role"]
                content = history_item["content"].strip()
                if role == "user":
                    conv.append_message(conv.roles[0], content)
                elif role == "agent":
                    conv.append_message(conv.roles[1], content)
                else:
                    raise ValueError(f"Unknown role: {role}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        else:
            conv = get_conversation_template("vicuna")
            for history_item in history:
                role = history_item["role"]
                content = history_item["content"].strip()
                if role == "user":
                    conv.append_message(conv.roles[0], content)
                elif role == "agent":
                    conv.append_message(conv.roles[1], content)
                else:
                    raise ValueError(f"Unknown role: {role}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

        gen_params.update({
            "prompt": prompt,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
        })
        # print("===prompt====")
        # print(prompt)
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "temperature": 0.7,
                "truncate": self.max_tokens - self.max_new_tokens
            }
        }
        headers = {"Content-Type": "application/json"}
        for _ in range(3):
            try:
                import random
                response = requests.post(
                    random.sample(self.controller_address, 1)[0] + "/generate",
                    headers=headers,
                    data=json.dumps(data),
                    timeout=120,
                )
                text = ""
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    try:
                        if line:
                            text = json.loads(line)["generated_text"]
                    except:
                        print("========")
                        print(line)
                        text = ""
                return text
            # if timeout or connection error, retry
            except Timeout:
                print("Timeout, retrying...")
            except ConnectionError:
                print("Connection error, retrying...")
            time.sleep(5)
        else:
            raise Exception("Timeout after 3 retries.")
