'''
This script is adopted from https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_mmlu.py
'''
import os
import pandas as pd
import numpy as np
import argparse
import torch
import datetime

from typing import List
from tqdm import tqdm
from transformers.trainer_utils import set_seed

import transformers
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig

from fastchat.model.model_adapter import get_conversation_template

'''
This script is used to evaluate the MMLU dataset.
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../
python eval/evaluate_mmlu.py -c /path/to/checkpoint
'''

def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, use_safetensors=True, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path)
    return model, tokenizer

def format_example(line, include_answer=True):
    example = 'Question: ' + line['question']
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += '\nAnswer: ' + line["answer"] + '\n\n'
    else:
        example += '\nAnswer:'
    return example


def generate_few_shot_prompt(k, subject, dev_df):

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s.strip()

    # Use llama2 template to generate answer
    prompt = f"The following is a multiple-choice question about {format_subject(subject)}. Please choose the most suitable one among A, B, C and D as the answer to this question."
    conv = get_conversation_template("llama-2")
    conv.set_system_message(f"{prompt}")
    for i in range(k):
        line = dev_df.iloc[i, :]
        conv.append_message(conv.roles[0], f'Question: {line["question"]}\n' + '\n'.join(
            [f"{choice}. {line[f'{choice}']}" for choice in ["A", "B", "C", "D"]]))
        conv.append_message(conv.roles[1], 'Answer: ' + line["answer"])

    return conv


def get_logits(tokenizer, model, inputs: List[str]):
    input_ids = tokenizer(inputs, padding=False)['input_ids']
    input_ids = torch.tensor(input_ids, device=model.device)

    if input_ids.shape[1] > args.max_seq_len:
        input_ids = input_ids[:, input_ids.shape[1]-args.max_seq_len+1:]
    tokens = {'input_ids': input_ids}

    outputs = model(input_ids)['logits']
    logits = outputs[:, -1, :]
    log_probs = torch.nn.functional.softmax(logits, dim=-1)
    return log_probs, {'tokens': tokens}


@torch.no_grad()
def eval_subject(
        model,
        tokenizer,
        subject_name,
        test_df,
        k=5,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        **kwargs
):
    result = []
    score = []

    cov = generate_few_shot_prompt(
        k, subject_name, dev_df) if few_shot else []
    all_probs = {'prob_A': [], 'prob_B': [], 'prob_C': [], 'prob_D': []}

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row, include_answer=False)
        cov.append_message(cov.roles[0], question.split("Answer:")[0].rstrip())
        cov.append_message(cov.roles[1], None)
        full_prompt = cov.get_prompt() + " Answer: "
        cov = generate_few_shot_prompt(
            k, subject_name, dev_df) if few_shot else []

        output, input_info = get_logits(tokenizer, model, [full_prompt])
        assert output.shape[0] == 1
        logits = output.flatten()

        softval = torch.nn.functional.softmax(
            torch.tensor(
                [
                    logits[tokenizer(" A")['input_ids'][-1]],
                    logits[tokenizer(" B")['input_ids'][-1]],
                    logits[tokenizer(" C")['input_ids'][-1]],
                    logits[tokenizer(" D")['input_ids'][-1]],
                ]
            ),
            dim=0,
        )
        if softval.dtype in {torch.bfloat16, torch.float16}:
            softval = softval.to(dtype=torch.float32)
        probs = softval.detach().cpu().numpy()

        for i, choice in enumerate(choices):
            all_probs[f'prob_{choice}'].append(probs[i])
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        if 'answer' in row:
            correct = 1 if pred == row['answer'] else 0
            score.append(correct)
            if args.debug:
                print(f'{question} pred: {pred} ref: {row["answer"]}')
        result.append(pred)

    if save_result_dir:
        test_df['model_output'] = result
        for i, choice in enumerate(choices):
            test_df[f'prob_{choice}'] = (all_probs[f'prob_{choice}'])
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(os.path.join(
            save_result_dir, f'{subject_name}_result.csv'), encoding="utf-8", index=False)
    return score


def cal_mmlu(res):
    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.
    cnt = 0
    hard_cnt = 0
    hard_acc_sum = 0.

    for class_ in TASK_NAME_MAPPING.keys():
        acc_sum_dict[class_] = 0.
        acc_norm_sum_dict[class_] = 0.
        cnt_dict[class_] = 0.

        for tt in TASK_NAME_MAPPING[class_]:
            acc_sum += sum(res[tt])
            cnt += len(res[tt])

            acc_sum_dict[class_] += sum(res[tt])
            cnt_dict[class_] += len(res[tt])

    for k in TASK_NAME_MAPPING.keys():
        if k in cnt_dict:
            print('%s ACC: %.2f ' % (
                k, acc_sum_dict[k] / cnt_dict[k] * 100))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"mmlu_eval_result_{timestamp}.json", "w") as f:
        result = {}
        result["acc"] = acc_sum / cnt * 100
        result["cnt"] = cnt
        result["acc_sum_dict"] = acc_sum_dict
        result["cnt_dict"] = cnt_dict
        f.write(json.dumps(result))


def main(args):
    model, tokenizer = load_models_tokenizer(args)

    dev_result = {}
    for subject_name in tqdm(SUBJECTS):
        dev_file_path = os.path.join(
            args.eval_data_path, 'dev', f'{subject_name}_dev.csv')
        test_file_path = os.path.join(
            args.eval_data_path, 'test', f'{subject_name}_test.csv')

        dev_df = pd.read_csv(dev_file_path, names=[
                             'question', 'A', 'B', 'C', 'D', 'answer'])
        test_df = pd.read_csv(test_file_path, names=[
                              'question', 'A', 'B', 'C', 'D', 'answer'])

        score = eval_subject(model, tokenizer, subject_name, test_df, dev_df=dev_df, k=5, few_shot=True,
                             save_result_dir=f"outs/mmlu_eval_result")
        dev_result[subject_name] = score
    cal_mmlu(dev_result)


TASK_NAME_MAPPING = {'stem': ['abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning'],
                     'Humanities': ['formal_logic', 'high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'international_law', 'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'prehistory', 'professional_law', 'world_religions'],
                     'other': ['business_ethics', 'college_medicine', 'human_aging', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine', 'virology', 'global_facts', 'clinical_knowledge'],
                     'social': ['econometrics', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology', 'human_sexuality', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy']}
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
choices = ["A", "B", "C", "D"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test HF checkpoint.')
    parser.add_argument('-c', '--checkpoint-path',
                        type=str, help='Checkpoint path')
    parser.add_argument('-s', '--seed', type=int,
                        default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')

    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Evaluation options')
    group.add_argument('-d', '--eval_data_path', type=str,
                       help='Path to eval data', default='../data/mmlu/data')
    group.add_argument("--max-seq-len", type=int, default=2048,
                       help='Size of the output generated text.')
    group.add_argument("--debug", action='store_true', default=False,
                       help='Print infos.')

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)
