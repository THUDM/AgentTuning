from nodes.LLMNode import *
import time
from utils.util import *


class IO:
    def __init__(self, fewshot="\n", model_name="text-davinci-003"):
        self.fewshot = fewshot
        self.model_name = model_name
        self.llm = LLMNode("CoT", model_name, input_type=str, output_type=str)
        self.context_prompt = "Answer following questions. Respond directly with no extra words.\n"
        self.token_unit_price = get_token_unit_price(model_name)

    def run(self, input):
        result = {}
        st = time.time()
        prompt = self.context_prompt + self.fewshot + input + '\n'
        response = self.llm.run(prompt, log=True)
        result["wall_time"] = time.time() - st
        result["input"] = response["input"]
        result["output"] = response["output"]
        result["prompt_tokens"] = response["prompt_tokens"]
        result["completion_tokens"] = response["completion_tokens"]
        result["total_tokens"] = response["prompt_tokens"] + response["completion_tokens"]
        result["token_cost"] = result["total_tokens"] * self.token_unit_price
        result["tool_cost"] = 0
        result["total_cost"] = result["token_cost"] + result["tool_cost"]
        result["steps"] = 1
        return result


class CoT:
    def __init__(self, fewshot="\n", model_name="text-davinci-003"):
        self.fewshot = fewshot
        self.model_name = model_name
        self.llm = LLMNode("CoT", model_name, input_type=str, output_type=str)
        self.context_prompt = "Answer following questions. Let's think step by step. Give your reasoning process, and then answer the " \
                              "question in a new line directly with no extra words.\n"
        self.token_unit_price = get_token_unit_price(model_name)

    def run(self, input):
        result = {}
        st = time.time()
        prompt = self.context_prompt + self.fewshot + input + '\n'
        response = self.llm.run(prompt, log=True)
        result["wall_time"] = time.time() - st
        result["input"] = response["input"]
        result["output"] = response["output"]
        result["prompt_tokens"] = response["prompt_tokens"]
        result["completion_tokens"] = response["completion_tokens"]
        result["total_tokens"] = response["prompt_tokens"] + response["completion_tokens"]
        result["token_cost"] = result["total_tokens"] * self.token_unit_price
        result["tool_cost"] = 0
        result["total_cost"] = result["token_cost"] + result["tool_cost"]
        result["steps"] = response["output"].count("Step")
        return result

