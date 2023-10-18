"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import json
import os
import re 
from collections import defaultdict
import pandas as pd
import hotpotqa_run.utils as utils

def eval_success(result_file) -> list:
    df = pd.read_csv(result_file)
    return df['success'].tolist()

def eval_reward(result_file) -> list:
    df = pd.read_csv(result_file)
    return df['reward'].tolist()

def eval_llm_agent(llm_name, agent_name):
    levels = ['easy','medium','hard']
    all_reward = []
    all_success = []
    for l in levels:
        file_name = f"execution_data/hotpotqa/{l}_{agent_name}_{llm_name}.csv"
        all_reward += eval_reward(file_name)
        all_success += eval_success(file_name)
    avg_reward = sum(all_reward)/len(all_reward)
    avg_success = sum(all_success)/len(all_success)
    return avg_reward, avg_success

def eval_llm_agent_level(llm_name, agent_name, level):
    file_name = f"execution_data/hotpotqa/{level}_{agent_name}_{llm_name}.csv"
    all_reward = eval_reward(file_name)
    all_success = eval_success(file_name)
    avg_reward = sum(all_reward)/len(all_reward)
    avg_success = sum(all_success)/len(all_success)
    return avg_reward, avg_success

def eval_sessions(llm_name, agent_name):
    levels = ['easy','medium','hard']
    all_reward = []
    all_success = []
    for l in levels:
        reward, success = eval_sessions_level((llm_name, agent_name,l))
        all_reward += reward
        all_success += success
    avg_reward = sum(all_reward)/len(all_reward)
    avg_success = sum(all_success)/len(all_success)
    return avg_reward, avg_success

def eval_sessions_level(llm_name, agent_name,level):
    file_name = f"execution_data/hotpotqa/{level}_{agent_name}_{llm_name}.jsonl"
    sessions = utils.get_all_agent_sessions(file_name)
    all_reward = [sess["reward"] for sess in sessions]
    all_success = [sess["correct"] for sess in sessions]
    avg_reward = sum(all_reward)/len(all_reward)
    avg_success = sum(all_success)/len(all_success)
    return avg_reward, avg_success

def get_reward_w_level(llm_name, agent_name):
    levels = ['easy','medium','hard']
    ret = []
    for l in levels:
        reward, _ = eval_sessions_level(llm_name, agent_name, l)
        ret.append(reward)
    return ret
        
