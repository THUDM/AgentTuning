'''
This script is adopted from https://github.com/salesforce/BOLAA/tree/main/hotpotqa_run
'''
import os
import json
import jsonlines
import datetime
import argparse
import numpy as np
import pandas as pd
import concurrent
import joblib
from src.utils import summarize_trial_detailed, log_trial
import src.utils as utils
from src.agent_arch import get_agent
from src.llms import get_llm_backend
from src.config import available_agent_names
from IPython import embed


parser = argparse.ArgumentParser(
    description='Test tgi checkpoint on hotpotQA.')
parser.add_argument("--agent_name", type=str,
                    help="Name of the agent.", default="React")
parser.add_argument("--llm_name", type=str,
                    help="Name of the llm", default="gpt-3.5-turbo")
parser.add_argument("--max_context_len", type=int,
                    help="Maximum context length", default=1700)
parser.add_argument("-i", "--ip", type=str, default="127.0.0.1")
parser.add_argument("--min", type=int, default=23330)
parser.add_argument("--max", type=int, default=23337)
parser.add_argument("-p", "--port", type=int, default=23333)
args = parser.parse_args()
if args.min is None:
    args.min = args.port
    args.max = args.port

agent_name = args.agent_name
llm_name = args.llm_name
max_context_len = args.max_context_len
assert agent_name in available_agent_names

def process_agent_run_step(agent):
    agent.run()

def run_one_complex_level(args, level="easy"):
    print(llm_name)
    hotpot = joblib.load(
        f'src/data/{level}.joblib').reset_index(drop=True)
    agent_save_file = f"execution_data/{args.min}-{args.max}/{level}_{agent_name}_{llm_name}.jsonl"

    task_instructions = [(row['question'], row['answer'])
                         for _, row in hotpot.iterrows()]
    if os.path.exists(agent_save_file):
        sessions = utils.get_all_agent_sessions(agent_save_file)
        completed_tasks = utils.get_non_error_tasks(sessions)
        print(f"{level}:{len(completed_tasks)}")
        task_instructions = [
            task for task in task_instructions if task not in completed_tasks]
        utils.delete_error(agent_save_file)
    llm = get_llm_backend(llm_name, args.ip, args.min, args.max).run
    agent_cls = get_agent(agent_name)
    agents = [agent_cls(ques, ans, llm, max_context_len)
              for ques, ans in task_instructions]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(process_agent_run_step, agents)
    # for agent in agents:
    #     process_agent_run_step(agent)
    for agent in agents:
        utils.log_agent(agent, agent_save_file)
    print(f'Finished Trial. Total: {len(agents)}')


def main():
    if not os.path.exists('execution_data'):
        os.mkdir('execution_data')

    folder_path = f'execution_data/{args.min}-{args.max}'
    # embed()
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    levels = ['easy', 'medium', 'hard']
    for level in levels:
        run_one_complex_level(args, level)

    def average_reward(name):
        with open(name) as f:
            data = [i for i in jsonlines.Reader(f)]
        rewards = [i['reward'] for i in data]
        return sum(rewards) / len(rewards)

    result_dict = {}

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(root, file_name)
                avg_reward = average_reward(file_path)
                print(f"{file_name}: {avg_reward}")
                result_dict[file_name] = avg_reward

    average_score = sum(result_dict.values()) / len(result_dict)
    result_dict["average"] = average_score
    print(f"Average: {average_score}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"result_{timestamp}.json", "w") as f:
        f.write(json.dumps(result_dict))


if __name__ == '__main__':
    main()
