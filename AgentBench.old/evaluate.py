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
import yaml
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Type, TypeVar

import time
import importlib
import argparse

from os.path import join, isdir, isfile, relpath
from glob import glob

from src import YAMLConfig, print_rank_0, Task, Agent, serialize


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("evaluation", "Evaluation configurations")
    group.add_argument("--task", nargs="+", required=True, help="All task config(s) to load")
    group.add_argument("--agent", type=str, required=True, help="Agent config to load")
    group.add_argument("--output_dir", type=str, default="outputs", help="Output root directory")
    group.add_argument("--workers", type=int, default=1, help="Number of workers for evaluation")
    group.add_argument("--max_new_tokens", type=int, default=None, help="Maximum number of new tokens to generate")
    group.add_argument("--no_timestamp", action="store_true", help="Do not use timestamp in output directory")
    args = parser.parse_args()
    return args


def find_all_task_files(all_task_config_path) -> List[str]:
    # print(type(all_task_config_path), all_task_config_path)
    tasks = []
    for task in all_task_config_path:
        if isdir(task):
            tasks += [relpath(path, ".") for path in glob(join(task, "**/*.yaml"), recursive=True)]
        elif isfile(task):
            tasks.append(task)
        else:
            print(f"'{task}' is not a valid file or directory, ignored.")
    return tasks


def evaluate_all_tasks(tasks: List[Task], agent: Agent):
    for task in tasks:
        task.evaluate(agent)
        task.release()
        del task


def main():
    args = parse_args()
    create_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if args.no_timestamp:
        output_root_dir = args.output_dir
    else:
        output_root_dir = os.path.join(args.output_dir, create_time)
        
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    task_files = find_all_task_files(args.task)
    tasks = []
    task_configs = []

    updt = {}
    if args.max_new_tokens is not None:
        updt["max_new_tokens"] = args.max_new_tokens
    agent_config = YAMLConfig.init_from_yaml(args.agent, updt)
    agent = agent_config.create()
    
    print("> Loading task configs")
    for task_config_path in task_files:
        updt = {"output_root_dir": output_root_dir, "workers": args.workers}
        print(updt)
        task_config = YAMLConfig.init_from_yaml(task_config_path, updt)
        task = task_config.create()
        if not task.output_root_dir:
            task.output_root_dir = output_root_dir
        os.makedirs(task.get_output_dir()) # TODO: exist_ok=True for resume
        config_path = os.path.join(task.get_output_dir(), "config.json")
        with open(config_path, "w", encoding='utf-8') as f:
            f.write(json.dumps({
                "agent": args.agent,
                "task": task_config_path,
            }, indent=4, ensure_ascii=False))
        # task.workers = args.workers or task.workers
        print(f"    Task '{task.name}' loaded from config {task_config_path}, output to {task.output_root_dir}")
        tasks.append(task)
        task_configs.append(task_config)
    print(f"> Successfully load {len(tasks)} task{'s' if len(tasks) > 1 else ''}")

    # model, tokenizer = initialize_model_and_tokenizer(args)
    # model = ModelForEvaluation(model, args.position_encoding_2d)
    

    with open(os.path.join(output_root_dir, "configs.json"), "w") as f:
        json.dump({
            "args": args.__dict__,
            "command_line": sys.argv,
            "create_time": create_time,
            "output_root_dir": output_root_dir,
            "tasks": [{
                "class": str(type(task)),
                "config": serialize(task_config),
            } for task, task_config in zip(tasks, task_configs)],
            "agent": {
                "class": str(type(agent)),
                "config": serialize(agent_config),
            },
        }, f, indent=4)

    start = time.time()
    evaluate_all_tasks(tasks, agent)
    print_rank_0(f"> Finish {len(tasks)} task{'s' if len(tasks) > 1 else ''} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
