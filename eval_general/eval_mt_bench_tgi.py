"""
This script is adopted from https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py
"""
import argparse
import json
import os
import random
import time

import shortuuid
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import get_conversation_template
import fastchat.llm_judge as llm_judge

import requests
import multiprocessing as mp

def run_eval(
    model_id,
    question_file,
    answer_file,
    host, 
    port
):
    questions = load_questions(question_file, None, None)
    random.shuffle(questions)

    chunk_size = len(questions)
    for i in range(0, len(questions), chunk_size):
        get_model_answers(
            model_id,
            questions[i : i + chunk_size],
            answer_file,
            host, 
            port,
        )

def query_model(x):
    question, host, port = x
    choices = []
    for i in range(1):
        conv = get_conversation_template("llama-2")
        conv.set_system_message('You are a helpful, respectful and honest assistant.')
        turns = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            # print("Prompt", prompt)
            temp = temperature_config[question["category"]]
            try:
                resp = requests.post(
                    url=f"http://{host}:{port}/generate",
                    json={
                            "inputs": prompt,
                            "parameters": {
                                "decoder_input_details": True,
                                "do_sample": temp > 1e-4,
                                "max_new_tokens": 4096,
                                **({"temperature": temp} if temp > 1e-4 else {})
                            }
                        },
                )
                try:
                    output = resp.json()['generated_text']
                except:
                    import traceback
                    print(">>> ERROR getting 'generated_text' question ID: ", question["question_id"])
                    print(resp.json())
                    traceback.print_exc()
                    output = ""
                output = output.strip()
            except:
                print(">>> ERROR question ID: ", question["question_id"])
                print(output)
                output = "ERROR"
                import traceback
                traceback.print_exc()

            turns.append(output)
            conv.messages[-1][-1] = output
        choices.append({"index": i, "turns": turns})

    return choices, question

def get_model_answers(
    model_id,
    questions,
    answer_file,
    host, 
    port,
):  
    print("Evauating", len(questions), "questions")

    with mp.Pool(80) as p:
        to_be_testes = [(x, host, port) for x in questions]
        question_choices = p.imap_unordered(query_model, to_be_testes)

        for choices, question in tqdm(question_choices):
            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    args = parser.parse_args()

    question_file = f"{llm_judge.__path__[0]}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{llm_judge.__path__[0]}/data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.model_id,
        question_file,
        answer_file,
        args.host,
        args.port,
    )

    reorg_answer_file(answer_file)
