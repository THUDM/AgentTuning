# AgentTuning: Enabling Generalized Agent Abilities For LLMs

<p align="center">
🤗 <a href="https://huggingface.co/THUDM/agentlm-70b" target="_blank">模型 (AgentLM-70B)</a> • 🤗 <a href="https://huggingface.co/datasets/THUDM/AgentInstruct" target="_blank">数据集 (AgentInstruct)</a> • 📃 <a href="https://arxiv.org/abs/2310.12823" target="_blank">论文</a> • 🌐 <a href="https://thudm.github.io/AgentTuning/" target="_blank">项目主页</a> <br>
</p>
<center><img src="assets/main-figure.svg" alt="main-figure" style="zoom:50%;" /></center>

**AgentTuning** 是首次利用多个 Agent 任务交互轨迹对 LLM 进行指令调整的方法。评估结果表明，**AgentTuning** 让 LLM 在**未见过**的 Agent 任务中也展现出强大的泛化能力，同时**通用语言能力**也基本保持不变。

**AgentInstruct** 数据集和 **AgentLM** 模型均已开源。

## 主要结果

<center><img src="assets/head-figure.svg" alt="head-figure" width="1500" /></center>

<center><b>Figure 1</b>&nbsp;&nbsp; 在 held-in 和 held-out 任务上的总得分</center>

## AgentInstruct

**AgentInstruct** 是一个经过挑选的智能体数据集，包含 **1866** 个高质量交互、**6** 个多样化的真实场景任务，用于增强语言模型的 Agent 能力，有如下特性

- 🔍 **思维链** - 采用 [ReAct](http://arxiv.org/abs/2210.03629) 提示词策略，为每步操作提供详细的思维链，深入理解模型决策过程

- 🌍 **多样性** - 涵盖 6 个现实世界场景，包括日常家务到操作数据库，平均回合数 5 ~ 35 不等。

- 🎯 **精确性** - GPT-4 也不能完全做对智能体任务，使用轨迹奖励机制对数据严格筛选，确保每条数据的质量。

- ✅ **泛化性** - 严格检查，避免数据泄露，保证数据的泛化性

**AgentInstruct** 数据集开源在 [🤗Huggingface Repo](https://huggingface.co/datasets/THUDM/AgentInstruct)

## AgentLM

**AgentLM** 由 Llama2-chat 开源模型系列在 **AgentInstruct**，**ShareGPT** 混合数据集上微调得到。模型遵循 [Llama-2-chat](https://huggingface.co/blog/llama2#how-to-prompt-llama-2) 的对话格式，系统提示词固定为 `You are a helpful, respectful and honest assistant.`。

7B、13B 和 70B 模型开源网址如下

|    Model    |                        Huggingface Repo                        |
| :---------: | :------------------------------------------------------------: |
| AgentLM-7B  | [🤗Huggingface Repo](https://huggingface.co/THUDM/agentlm-7b)  |
| AgentLM-13B | [🤗Huggingface Repo](https://huggingface.co/THUDM/agentlm-13b) |
| AgentLM-70B | [🤗Huggingface Repo](https://huggingface.co/THUDM/agentlm-70b) |

## 运行 AgentLM

使用 [Text-Generation-Inference](https://github.com/huggingface/text-generation-inference) 加速评测流程，启动一个 AgentLM-70b 实例：

```bash
cd docker
docker compose -f agentlm-70b.yml up
```

成功部署后的端口位于 `30070`，可以向其发送请求：

```bash
curl 127.0.0.1:30070/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"inputs": "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\nHello! [/INST]", "parameters":{"temperature": 1.0}}'

# {"generated_text":"Hello! How can I help you today? "}
```

可在 docker compose 文件后面增加更多端口，产生多个推理实例。

## 评测

模型评测包含 6 个 held-in 任务、6 个 held-out 任务、通用任务

### Held-in 任务

6 个保留任务来源于 [**AgentBench**](https://github.com/THUDM/AgentBench)。 但是，由于 AgentBench 仍在开发中，最新版本可能无法完全重现论文中报告的结果。

本项目有关评测代码位于`./AgentBench.old` 文件夹中。

### Held-out 任务

Held-out 任务来源于以下开源框架

| 任务              | AgentTuning 评测脚本                                        | 原始仓库                                                     |
| ----------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| SciWorld          | [📂 eval_heldout/science-world](eval_heldout/science-world/) | [💻 allenai/ScienceWorld](https://github.com/allenai/ScienceWorld) |
| MiniWoB++         | [📂 eval_heldout/miniwob++](eval_heldout/miniwob++)          | [💻 Farama-Foundation/miniwob-plusplus](https://github.com/Farama-Foundation/miniwob-plusplus) |
| HotpotQA          | [📂 eval_heldout/hotpotQA](eval/held_out/hotpotQA)           | [💻 salesforce/BOLAA](https://github.com/salesforce/BOLAA)    |
| ReWOO             | [📂 eval_heldout/rewoo](eval_heldout/rewwo/)                 | [💻 billxbf/ReWOO](https://github.com/billxbf/ReWOO)          |
| WebArena          | [📂 eval_heldout/webarena](eval_heldout/webarena/)           | [💻 web-arena-x/webarena](https://github.com/web-arena-x/webarena) |
| Digital Card Game | [💻 AgentBench.old](./AgentBench.old) ( _Extend_ Split )     | [💻 THUDM/AgentBench](https://github.com/THUDM/AgentBench)    |

### 通用任务

**MMLU 配置**

- 下载 14k 多项选择题到 `./data` 文件夹：
  ```bash
  cd data
  wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
  tar xf data.tar
  cd ..
  ```
- 执行以下代码评测 Hf 模型 MMLU 得分：
  ```bash
  python eval_general/evaluate_mmlu_hf.py -c THUDM/AgentLM-70b
  ```

**GSM8k 配置**

- 部署 TGI
- 运行以下代码评测 GSM8k

  ```bash
  python eval_general/evaluate_gsm8k_tgi.py --port 30070
  ```

  使用 `--sample-input-file` 可以加载本地数据，否则脚本会下载 [GSM8K](https://huggingface.co/datasets/gsm8k)  到本地

**MT-Bench 配置**

- 本地安装 [FastChat](https://github.com/lm-sys/FastChat)

  ```bash
  git clone https://github.com/lm-sys/FastChat.git
  pip install -e FastChat
  ```

- 部署 TGI

- 运行评测脚本

  ```bash
  python eval_general/eval_mt_bench_tgi.py --host 127.0.0.1 --port 30070 --model-id agentlm-70b
  ```

- 使用 GPT-4 评测回答
  ```bash
  cd FastChat/fastchat/llm_judge
  OPENAI_API_KEY=<your-api-key> python gen_judgment.py --model-list agentlm-70b --parallel <number-of-cuncurrent-requests>
  ```

## 引用

如果你觉得我们的工作有帮助的话，请考虑引用下列论文

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
