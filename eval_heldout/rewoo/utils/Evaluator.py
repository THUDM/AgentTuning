import re
import string
import traceback
from collections import Counter

import numpy as np
import pandas as pd
import tqdm
from langchain.evaluation.qa import QAEvalChain
from langchain.llms import OpenAI

from algos.PWS import PWS_Base, PWS_Extra
from algos.notool import CoT, IO
from algos.react import ReactBase

from nodes import LLMNode

import openai


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def llm_accuracy_score(query, prediction, ground_truth):
    data = [{
        'query': query,
        'answer': ground_truth,
    }]
    pred = [{
        'query': query,
        'answer': ground_truth,
        'result': prediction,
    }]
    eval_chain = QAEvalChain.from_llm(OpenAI(
        temperature=0,
    ))
    graded_outputs = eval_chain.evaluate(data, pred)
    return 1 if graded_outputs[0]['text'].strip() == 'CORRECT' else 0


class Evaluator:
    def __init__(self, task, dataset, algo, maxtry=3):
        assert task in ["hotpot_qa", "trivia_qa", "gsm8k", "physics_question", "disfl_qa",
                        "sports_understanding", "strategy_qa", "sotu_qa"]
        assert isinstance(dataset, pd.DataFrame)
        assert isinstance(algo, (PWS_Base, PWS_Extra, ReactBase, IO, CoT))

        self.task = task
        self.dataset = dataset
        self.algo = algo
        self.maxtry = maxtry
        self.failed_response = self._failed_response()
        self.eval_data = self._initialize_eval_dict()

    def run(self):
        print("\n******************* Start Evaluation *******************\n")
        if self.task in ["hotpot_qa", "sotu_qa"]:
            for i in tqdm.tqdm(range(len(self.dataset))):
                question = self.dataset["question"][i]
                label = self.dataset["answer"][i]
                for _ in range(self.maxtry):
                    try:
                        response = self.algo.run(question)
                        break
                    except Exception:
                        traceback.print_exc()
                        response = self.failed_response
                self._update_eval_dict(question, label, response)

        elif self.task == "fever":
            for i in tqdm.tqdm(range(len(self.dataset))):
                question = self.dataset["claim"][i]
                label = self.dataset["label"][i]
                for _ in range(self.maxtry):
                    try:
                        response = self.algo.run(question)
                        break
                    except:
                        response = self.failed_response
                self._update_eval_dict(question, label, response)
        elif self.task == "trivia_qa":
            for i in tqdm.tqdm(range(len(self.dataset))):
                question = self.dataset["question"][i]
                label = self.dataset["answer"][i]["value"]
                for _ in range(self.maxtry):
                    try:
                        response = self.algo.run(question)
                        break
                    except:
                        response = self.failed_response
                self._update_eval_dict(question, label, response)
        elif self.task == "gsm8k":
            for i in tqdm.tqdm(range(len(self.dataset))):
                question = self.dataset["question"][i]
                label = self.dataset["answer"][i].split("#### ")[1]
                for _ in range(self.maxtry):
                    try:
                        response = self.algo.run(question)
                        break
                    except:
                        response = self.failed_response
                self._update_eval_dict(question, label, response)
        elif self.task in ["physics_question", "sports_understanding", "strategy_qa"]:
            for i in tqdm.tqdm(range(len(self.dataset))):
                question = self.dataset["input"][i]
                label = self.dataset["target"][i]
                for _ in range(self.maxtry):
                    try:
                        response = self.algo.run(question)
                        break
                    except:
                        response = self.failed_response
                self._update_eval_dict(question, label, response)
        else:
            raise NotImplementedError

        return self._get_avg_results(), self.eval_data

    def _initialize_eval_dict(self):
        data = {}
        for d in ["label", "preds", "em", "f1", "acc", "wall_time", "total_tokens", "total_cost", "steps", "token_cost",
                  "tool_cost", "planner_log", "solver_log"]:
            data[d] = []
        return data

    def _update_eval_dict(self, question, label, response):
        print("=== Planner ===" + '\n\n' + response.get("planner_log", '') + '\n' + "=== Solver ===" + '\n\n' + response.get("solver_log", ''))

        pred = self._parse_prediction(response["output"])
        self.eval_data["label"] += [label]
        self.eval_data["preds"] += [pred]
        self.eval_data["em"] += [self.get_metrics(question, label, pred)["em"]]
        self.eval_data["f1"] += [self.get_metrics(question, label, pred)["f1"]]
        self.eval_data["acc"] += [self.get_metrics(question, label, pred)["acc"]]
        self.eval_data["wall_time"] += [response["wall_time"]]
        self.eval_data["total_tokens"] += [response["total_tokens"]]
        self.eval_data["total_cost"] += [response["total_cost"]]
        self.eval_data["steps"] += [response["steps"]]
        self.eval_data["token_cost"] += [response["token_cost"]]
        self.eval_data["tool_cost"] += [response["tool_cost"]]

        LLMNode.refresh('succ' if self.get_metrics(question, label, pred)["acc"] else 'fail')

        if "planner_log" in response:
            self.eval_data["planner_log"] += [response["planner_log"]]
        if "solver_log" in response:
            self.eval_data["solver_log"] += [response["solver_log"]]

    def _get_avg_results(self):
        result = {}
        result["avg_em"] = np.nanmean(self.eval_data["em"])
        result["avg_f1"] = np.nanmean(self.eval_data["f1"])
        result["avg_acc"] = np.nanmean(self.eval_data["acc"])
        result["avg_wall_time"] = np.nanmean(self.eval_data["wall_time"])
        result["avg_total_tokens"] = np.nanmean(self.eval_data["total_tokens"])
        result["avg_total_cost"] = np.nanmean(self.eval_data["total_cost"])
        result["avg_steps"] = np.nanmean(self.eval_data["steps"])
        result["avg_token_cost"] = np.nanmean(self.eval_data["token_cost"])
        result["avg_tool_cost"] = np.nanmean(self.eval_data["tool_cost"])
        return result

    def get_metrics(self, query, label, pred):
        if pred is None:
            return {'em': 0, 'f1': 0}
        norm_label = normalize_answer(label)
        norm_pred = normalize_answer(pred)
        em = (norm_pred == norm_label)
        f1 = f1_score(norm_pred, norm_label)
        acc = llm_accuracy_score(query, pred, label)
        return {'em': em, 'f1': f1, 'acc': acc}

    def _parse_prediction(self, output):
        if isinstance(self.algo, IO):
            return str(output).strip("\n")
        elif isinstance(self.algo, CoT):
            return str(output).split("\n")[-1].replace("Answer:", "")
        elif isinstance(self.algo, ReactBase):
            return str(output).strip("\n")
        elif isinstance(self.algo, PWS_Base):
            return str(output).strip("\n")
        elif isinstance(self.algo, PWS_Extra):
            return str(output).strip("\n")

    def _failed_response(self):
        resposne = {}
        for key in ["input", "output", "wall_time", "total_tokens", "total_cost", "steps", "token_cost", "tool_cost"]:
            resposne[key] = np.nan
        return resposne
