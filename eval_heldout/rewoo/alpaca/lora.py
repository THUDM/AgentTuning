import os
import sys

import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from alpaca.utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


class AlpacaLora:
    def __init__(self, load_8bit: bool = True,
                 base_model: str = "decapoda-research/llama-7b-hf",
                 lora_weights: str = "tloen/alpaca-lora-7b",
                 prompt_template: str = ""):
        base_model = base_model or os.environ.get("BASE_MODEL", "")
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

        self.prompter = Prompter(prompt_template)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        if device == "cuda":
            self.model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        elif device == "mps":
            self.model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                device_map={"": device},
            )

        # unwind broken decapoda-research config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        if not load_8bit:
            self.model.half()  # seems to fix bugs for some users.

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(self.model)

    def lora_generate(self, instruction, input):
        # evaluate
        temperature = 0
        top_p = 0.75
        top_k = 40
        num_beams = 4
        max_new_tokens = 128
        stream_output = False
        prompt = self.prompter.generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        return self.prompter.get_response(output), prompt


# PARAMS
load_8bit: bool = True
base_model: str = "decapoda-research/llama-7b-hf"
lora_weights: str = "./lora-alpaca"  # "tloen/alpaca-lora-7b"
prompt_template: str = ""
server_name: str = "0.0.0.0"
share_gradio: bool = False
