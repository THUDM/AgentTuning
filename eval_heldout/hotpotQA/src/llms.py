"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

from langchain.llms import HuggingFaceTextGenInference
from src.config import OPENAI_API_KEY
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate, OpenAI, LLMChain
import openai
import os
import sys
import json
import tiktoken
token_enc = tiktoken.get_encoding("cl100k_base")

OPENAI_CHAT_MODELS = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613",
                      "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k-0613"]
OPENAI_LLM_MODELS = ["text-davinci-003", "text-ada-001"]


class langchain_openai_chatllm:
    def __init__(self, llm_name):
        self.llm_name = llm_name
        human_template = "{prompt}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [human_message_prompt])

    def run(self, prompt, temperature=1, stop=['\n'], max_tokens=128):
        chat = ChatOpenAI(model_name=self.llm_name,
                          temperature=temperature, stop=stop, max_tokens=max_tokens)
        self.chain = LLMChain(llm=chat, prompt=self.chat_prompt)
        return self.chain.run(prompt)


class langchain_openai_llm:
    def __init__(self, llm_name):
        openai.api_key = OPENAI_API_KEY
        self.prompt_temp = PromptTemplate(
            input_variables=["prompt"], template="{prompt}"
        )
        self.llm_name = llm_name

    def run(self, prompt, temperature=0.9, stop=['\n'], max_tokens=128):
        llm = OpenAI(model=self.llm_name, temperature=temperature,
                     stop=['\n'], max_tokens=max_tokens)
        chain = LLMChain(llm=llm, prompt=self.prompt_temp)
        return chain.run(prompt)


class langchain_tgi_llm:
    def __init__(self, llm_name, ip, port_min, port_max):
        self.prompt_temp = PromptTemplate(
            input_variables=["prompt"], template="{prompt}"
        )
        self.ip = ip
        self.port_min = port_min
        self.port_max = port_max

    def run(self, prompt, temperature=0.9, stop=['\n'], max_tokens=128):
        ports = [i for i in range(self.port_min, self.port_max + 1)]
        ip = self.ip
        import random
        url = f"http://{ip}:{random.sample(ports, 1)[0]}/"
        # embed()
        # assert False
        llm = HuggingFaceTextGenInference(
            inference_server_url=url,
            max_new_tokens=256,
            stop_sequences=["\n"],
            # do_sample=False
            temperature=1e-5
        )

        print(f"request {url}")
        chain = LLMChain(llm=llm, prompt=self.prompt_temp)
        ans = chain.run(prompt)
        print(ans)
        return ans


def get_llm_backend(llm_name, ip, port_min, port_max):
    if llm_name in OPENAI_CHAT_MODELS:
        return langchain_openai_chatllm(llm_name)
    elif llm_name in OPENAI_LLM_MODELS:
        return langchain_openai_llm(llm_name)
    else:
        return langchain_tgi_llm(llm_name, ip, port_min, port_max)
