# Readme-zh

## 配置

确认安装 docker，并能在无需 sudo 的情况下运行 docker 命令

```bash
docker --version
docker image list
docker ps
```

### 基于 Docker 任务

确保安装 docker，直接拉取 Docker 镜像

```bash
# 用于 Held-in 任务
docker pull learningrate/agentbench-alfworld
docker pull learningrate/agentbench-webshop
docker pull learningrate/agentbench-mind2web
# 用于 Held-out 任务
docker pull learningrate/agentbench-card_game
```

### 其余任务

安装全局依赖库

```bash
pip install -r requirements.txt
```

**OS 任务**

安装所需的库并创建本地镜像（大约需要 5 ~ 10 分钟）

```bash
pip install -r src/tasks/os_interaction/requirements.txt
python src/tasks/os_interaction/images.py build -c configs/tasks/os_interaction/std.yaml -r .
```

运行以下命令来测试 OS 任务：

```bash
python evaluate.py \
    --task configs/tasks/os_interaction/std.yaml \
    --agent configs/agents/do_nothing.yaml \
    --workers 30
```

**DB 任务**

安装 docker 并准备 `mysql` 镜像，确保你已经安装了全局的要求库

```bash
pip install -r src/tasks/dbbench/requirements.txt
```

运行以下命令来测试 DB 任务（为了避免 docker 出现问题，不建议使用太多的 workers 来运行）

```bash
python evaluate.py \
    --task configs/tasks/dbbench/std.yaml \
    --agent configs/agents/do_nothing.yaml \
    --workers 5
```

**KG 任务**

按照 [Freebase设置](https://github.com/dki-lab/Freebase-Setup) 启动本地 Virtuoso 服务器。然后在 [配置文件](https://github.com/Longin-Yu/AgentBench/tree/main/configs/tasks/knowledgegraph) 中将`sparql_url` 替换为指向本地服务器的链接。（**注意：** 你可以尝试使用仓库中默认设置的 `sparql_url`，但我们不能保证我们的 Virtuoso 服务器始终是可用的）

安装必要的 Python 包：

```bash
pip install -r src/tasks/knowledgegraph/requirements.txt
```

运行以下命令来测试 KG 任务：

```bash
python evaluate.py \
    --task configs/tasks/knowledgegraph/std.yaml \
    --agent configs/agents/do_nothing.yaml \
    --workers 30
```

## TGI配置

部署 TGI 时，为了评测更快，你可以在 docker compose 文件中增加端口，并在`/configs/agents/tgi_clients/AgentLM-{7b,13b,70b}.yaml` 中添加端口。

## 评估

运行以下 bash 文件，对 AgentLM-{7b,13b,70b} 进行评估

```bash
bash eval/AgentLM-7b-eval-all.sh
bash eval/AgentLM-13b-eval-all.sh
bash eval/AgentLM-70b-eval-all.sh
```

评估后，每个任务的结果将存储在`outputs/AgentLM-{7b,13b,70b}/{timestamp}/{task}/results.json`中。