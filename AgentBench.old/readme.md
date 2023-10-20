# Readme

[中文版(Chinese)](./readme-zh.md)

## set-up

Verify that you have installed docker and can run docker commands without sudo.

```bash
docker --version
docker image list
docker ps
```

### Docker-based tasks

Simply pull the docker image

```bash
# For Held-in tasks
docker pull learningrate/agentbench-alfworld
docker pull learningrate/agentbench-webshop
docker pull learningrate/agentbench-mind2web
# For Held-out task
docker pull learningrate/agentbench-card_game
```

### Other tasks

First install the global requirements

```bash
pip install -r requirements.txt
```

**For OSInteraction task**

Install requirements and create local images (5 ~ 10 minutes)

```bash
pip install -r src/tasks/os_interaction/requirements.txt
python src/tasks/os_interaction/images.py build -c configs/tasks/os_interaction/std.yaml -r .
```

Run the following command to test OS task:

```bash
python evaluate.py \
    --task configs/tasks/os_interaction/std.yaml \
    --agent configs/agents/do_nothing.yaml \
    --workers 30
```

**For DB task**

Install docker and prepare `mysql` image, and make sure you have already installed global requirements.

```bash
pip install -r src/tasks/dbbench/requirements.txt
```

Run the following command to test DB task (To avoid docker problem, we do not recommend run with too many workers)

```bash
python evaluate.py \
    --task configs/tasks/dbbench/std.yaml \
    --agent configs/agents/do_nothing.yaml \
    --workers 5
```

**For KG task**

Follow [Freebase Setup](https://github.com/dki-lab/Freebase-Setup) to start your own Virtuoso server. Then replace `sparql_url` with the link to your own server in the [config files](https://github.com/Longin-Yu/AgentBench/tree/main/configs/tasks/knowledgegraph). (**Caveat:** You may try the default `sparql_url` without touching this, but it is not always guaranteed that our Virtuoso server is active.)

Install necessary Python packages:

```bash
pip install -r src/tasks/knowledgegraph/requirements.txt
```

Run the following command to test KG task

```bash
python evaluate.py \
    --task configs/tasks/knowledgegraph/std.yaml \
    --agent configs/agents/do_nothing.yaml \
    --workers 30
```

## TGI config

When setting up TGI, you can add more port(s) in `/configs/agents/tgi_clients/AgentLM-{7b,13b,70b}.yaml` for faster evaluation.

## Evaluation

Running the bash file for evaluating AgentLM-{7b,13b,70b}

```bash
bash eval/AgentLM-7b-eval-all.sh
bash eval/AgentLM-13b-eval-all.sh
bash eval/AgentLM-70b-eval-all.sh
```

After evaluation, result of each task will be stored in `outputs/AgentLM-{7b,13b,70b}/{timestamp}/{task}/results.json`.
