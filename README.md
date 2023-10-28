# AgentTuning: Enabling Generalized Agent Abilities For LLMs

<p align="center">
🤗 <a href="https://huggingface.co/THUDM/agentlm-70b" target="_blank">Model (AgentLM-70B)</a> • 🤗 <a href="https://huggingface.co/datasets/THUDM/AgentInstruct" target="_blank">Dataset (AgentInstruct)</a> • 📃 <a href="https://arxiv.org/abs/2310.12823" target="_blank">Paper</a> • 🌐 <a href="https://thudm.github.io/AgentTuning/" target="_blank">Project Page</a> <br>
</p>
<center><img src="assets/main-figure.svg" alt="main-figure" style="zoom:50%;" /></center>

[中文版(Chinese)](./README-zh.md)

**AgentTuning** represents the very first attempt to instruction-tune LLMs using interaction trajectories across multiple agent tasks. Evaluation results indicate that AgentTuning enables the agent capabilities of LLMs with robust generalization on unseen agent tasks while remaining strong in general language abilities. We have open-sourced the AgentInstruct dataset and AgentLM.

## Main Result

<center><img src="assets/head-figure.svg" alt="head-figure" width="1500" /></center>

<center><b>Figure 1</b>&nbsp;&nbsp;Overall score in our held-in and held-out tasks</center>

## AgentInstruct

**AgentInstruct** is a meticulously curated dataset featuring 1,866 high-quality interactions designed to enhance AI agents across 6 diverse real-world tasks.

- 🔍 **CoT** - Harness the power of  [ReAct](http://arxiv.org/abs/2210.03629), offering detailed thought explanations for each action, ensuring an intricate understanding of the model's decision-making journey.
- 🌍 **Diversity** - Spanning 6 real-world scenarios, from Daily Household Routines to Database Operations, with an average turn range from 5 to 35.
- 🎯 **Precision** - Not all trajectories of GPT-4 are effective! Ours are rigorously filtered using strict rewards to ensure top-notch quality.
- ✅ **Assurance** - Rigorous checks to avoid data leakage, ensuring pristine dataset quality.

AgentInstruct dataset is available on [🤗Huggingface Repo](https://huggingface.co/datasets/THUDM/AgentInstruct).

## AgentLM

**AgentLM** models are produced by mixed training on AgentInstruct dataset and ShareGPT dataset from Llama2-chat series.

The models follow the conversation format of [Llama-2-chat](https://huggingface.co/blog/llama2#how-to-prompt-llama-2), with the system prompt fixed as `You are a helpful, respectful and honest assistant.`

The 7B, 13B, and 70B models are available on Huggingface model hub.

|    Model    |                        Huggingface Repo                        |
| :---------: | :------------------------------------------------------------: |
| AgentLM-7B  | [🤗Huggingface Repo](https://huggingface.co/THUDM/agentlm-7b)  |
| AgentLM-13B | [🤗Huggingface Repo](https://huggingface.co/THUDM/agentlm-13b) |
| AgentLM-70B | [🤗Huggingface Repo](https://huggingface.co/THUDM/agentlm-70b) |

## Run AgentLM

We use [Text-Generation-Inference](https://github.com/huggingface/text-generation-inference) to accelerate the evaluation process.

You can start a AgentLM-70b instance with:

```bash
cd docker
docker compose -f agentlm-70b.yml up
```

Upon successful execution, a client will be available on port `30070`. Here is an example of launching a request:

```bash
curl 127.0.0.1:30070/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"inputs": "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\nHello! [/INST]", "parameters":{"temperature": 1.0}}'

# {"generated_text":"Hello! How can I help you today? "}
```

You can replicate the services in the Docker Compose file to create multiple inference instances if more GPUs are available.

## Evaluation

Here are details of our evaluation task, including 6 held-in tasks and 6 held-out tasks.

### Held-in Tasks

The 6 held-in tasks are selected from [**AgentBench**](https://github.com/THUDM/AgentBench). However, since AgentBench is still under active development, the results from the latest branch might not fully reproduce the results reported in the paper. The evaluation code of this project is located in ./AgentBench.old.

### Held-out Tasks

Held-out tasks are recompiled from the following frameworks:

| Task              | AgentTuning Setup                                           | Original Repo                                                |
| ----------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| SciWorld          | [📂 eval_heldout/science-world](eval_heldout/science-world/) | [💻 allenai/ScienceWorld](https://github.com/allenai/ScienceWorld) |
| MiniWoB++         | [📂 eval_heldout/miniwob++](eval_heldout/miniwob++)          | [💻 Farama-Foundation/miniwob-plusplus](https://github.com/Farama-Foundation/miniwob-plusplus) |
| HotpotQA          | [📂 eval_heldout/hotpotQA](eval/held_out/hotpotQA)           | [💻 salesforce/BOLAA](https://github.com/salesforce/BOLAA)    |
| ReWOO             | [📂 eval_heldout/rewoo](eval_heldout/rewwo/)                 | [💻 billxbf/ReWOO](https://github.com/billxbf/ReWOO)          |
| WebArena          | [📂 eval_heldout/webarena](eval_heldout/webarena/)           | [💻 web-arena-x/webarena](https://github.com/web-arena-x/webarena) |
| Digital Card Game | [💻 AgentBench.old](./AgentBench.old) ( _Extend_ Split )     | [💻 THUDM/AgentBench](https://github.com/THUDM/AgentBench)    |

### General Tasks

**MMLU Setup**:

- Download the 14k multi-choice questions into `./data`:

  ```bash
  cd data
  wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
  tar xf data.tar
  cd ..
  ```

- Evaluate Hf model(organization/name or ckpt path)by executing the evaluation script:

  ```bash
  python eval_general/evaluate_mmlu_hf.py -c THUDM/AgentLM-70b
  ```

**GSM8k Setup**:

- Start TGI worker
- Run the evaluation

  ```bash
  python eval_general/evaluate_gsm8k_tgi.py --port 30070
  ```

  Use `--sample-input-file` to load a local dataset, or [GSM8K](https://huggingface.co/datasets/gsm8k) will be loaded for evaluation.

**MT-Bench Setup**:

- Install [FastChat](https://github.com/lm-sys/FastChat) locally

  ```bash
  git clone https://github.com/lm-sys/FastChat.git
  pip install -e FastChat
  ```

- Start TGI worker

- Run the evaluation script:

  ```bash
  python eval_general/eval_mt_bench_tgi.py --host 127.0.0.1 --port 30070 --model-id agentlm-70b
  ```

- Evaluate the answers with GPT-4
  ```bash
  cd FastChat/fastchat/llm_judge
  OPENAI_API_KEY=<your-api-key> python gen_judgment.py --model-list agentlm-70b --parallel <number-of-cuncurrent-requests>
  ```

## Citation

If you find our work useful, please consider citing AgentTuning:

```
@misc{zeng2023agenttuning,
      title={AgentTuning: Enabling Generalized Agent Abilities for LLMs},
      author={Aohan Zeng and Mingdao Liu and Rui Lu and Bowen Wang and Xiao Liu and Yuxiao Dong and Jie Tang},
      year={2023},
      eprint={2310.12823},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
