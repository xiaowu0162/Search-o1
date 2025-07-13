import re
import os
import json
import string
from tqdm import tqdm
import torch
import numpy as np
from openai import OpenAI
from collections import Counter
import backoff
from transformers import AutoTokenizer


OPENAI_REQUEST_TIMEOUT = 60*60*24 


@backoff.on_exception(backoff.constant, Exception, interval=5)
def run_generate_with_backoff(client, **kwargs):
    return client.completions.create(**kwargs)


# load direct pred logs
def load_direct_pred_logs(task):
    if task == 'gpqa':
        log_name = '/fsx-comem/diwu0162/Search-o1/outputs/gpqa.qwq.direct/diamond.7.1,20:8.json'
    elif task == 'aime':
        log_name = '/fsx-comem/diwu0162/Search-o1/outputs/aime.qwq.direct/test.7.1,19:42.json'
    elif task == 'amc':
        log_name = '/fsx-comem/diwu0162/Search-o1/outputs/amc.qwq.direct/test.7.1,19:24.json'
    elif task == 'math500':
        log_name = '/fsx-comem/diwu0162/Search-o1/outputs/math500.qwq.direct/test.7.1,20:17.json'
    elif task == 'livecode':
        log_name = '/fsx-comem/diwu0162/Search-o1/outputs/livecode.qwq.direct/test_1to4.7.1,22:30.json'
    elif task == 'bamboogle':
        log_name = '/fsx-comem/diwu0162/Search-o1/outputs/runs.qa/bamboogle.qwq.direct/test.7.1,16:22.json'
    else:
        raise NotImplementedError
    logs = json.load(open(log_name))
    return log_name, logs


def generate_specific_hint(client, teacher_model_name, question, teacher_answer, teacher_thought_str):
    # assert teacher_answer != ""
    prompt = (f"You are an expert tutor. Given a question, a final answer written by the teacher, and a long thinking process written by a teacher, "
              f"write a brief hint that can help yourself approach similar questions without revealing the answer or any intermediate results. "
              f"The hint should outline the steps to solve these general questions and the general strategy. "
              f"The hint should also highlight key points in thinking to expedite problem solving and avoid common traps. "
              f"Utilize and pay special attention to the places where the teacher also gets confused or spends too much time. "
              f"Start by outlining the general problem under a section ### Applicable Problems . Then, start your hint on a new line by ### Hint ."
              f"Inside the hint, you must first re-state the general problem setting that the hint can apply to. "
              f"Then, use a first person perspective just like you are the student, e.g., say something like 'For problems like X, I should...'."
              f"\n\n\n### Question:\n{question}\n\n\n### Teacher's Answer:\n{teacher_answer}\n\n\n### Teacher's Thinking:\n{teacher_thought_str}"
              f"\n\n\nNow, analyze the question, answer, and teacher's thought and write your hint. Make sure your hint helps solving"
              f" approach similar questions without revealing the answer or any intermediate results. ")
    prompt_formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    response = client.completions.create(model=teacher_model_name, prompt=prompt_formatted, n=1, temperature=0.7, top_p=0.8, 
                                         max_tokens=10000, timeout=OPENAI_REQUEST_TIMEOUT,
                                         extra_body={'top_k': 20, 'include_stop_str_in_output': True, 'repetition_penalty': 1.05,})
    hint_text_raw = response.choices[0].text
    if '</think>' in hint_text_raw:
        hint_derivation_thinking = hint_text_raw.split('</think>')[0].strip()
        hint_text = hint_text_raw.split('</think>')[1].strip()
    else:
        hint_derivation_thinking = ''
        hint_text = hint_text_raw.strip()

    # further process hint_text
    hint_result_dict = {'hint_derivation_thinking': hint_derivation_thinking}
    if '### Hint' in hint_text:
        hint_result_dict['hint'] = hint_text.split('### Hint')[-1].strip()
        hint_result_dict['applicable_problems'] = hint_text.split('### Hint')[0].strip() 
        hint_result_dict['applicable_problems'] = hint_result_dict['applicable_problems'].replace('### Applicable Problems', '').strip()
    else:
        hint_result_dict['hint'] = hint_text
        hint_result_dict['applicable_problems'] = ''
    
    return hint_text_raw, hint_result_dict
 

if __name__ == '__main__':
    
    port = 8001

    teacher_model_name = 'Qwen/QwQ-32B'

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY", timeout=OPENAI_REQUEST_TIMEOUT)
    print('Started openai client')
    print(client.models.list())

    # for task in ['gpqa', 'aime', 'amc', 'math500', 'livecode', 'bamboogle']:
    for task in ['aime', 'amc', 'math500', 'livecode', 'bamboogle']:
        log_name, logs = load_direct_pred_logs(task)
        out_file_name = f'logs_hint_distillation_eval_task_{task}.jsonl'
        out_f = open(out_file_name, 'w')


        for entry in tqdm(logs):
            # question = entry['conversations'][0]['value']
            # if '</think>' in entry['conversations'][1]['value']:
            #     thoughts = entry['conversations'][1]['value'].split('</think>')[0].replace('<think>', '').strip()
            #     answer = entry['conversations'][1]['value'].split('</think>')[-1].strip()
            # else:
            #     thoughts = entry['conversations'][1]['value'].replace('<think>', '').strip()
            #     answer = 'answer unknown due to thinking unfinished'
            question = entry['Question'].split('\n\nQuestion:\n')[-1].replace('\n\n<|im_end|>\n<|im_start|>assistant\n<think>\n', "").strip()
            if '</think>' in entry['Output']:
                thoughts = entry['Output'].split('</think>')[0].replace('<think>', '').strip()
                answer = entry['Output'].split('</think>')[-1].strip()
            else:
                thoughts = entry['Output'].replace('<think>', '').strip()
                answer = 'answer unknown due to thinking unfinished'
            # pred_steps = segment_thoughts_v2('\n\n'.join(entry['Output'].split('</think>')[:-1]))

            hint_text_raw, problem_specific_hint = generate_specific_hint(client, teacher_model_name, question, answer, thoughts)

            print('Question:', question)
            print('\n\n==================================')
            print('QwQ solution after thinking:\n\n', answer, sep='')
            print('\n\n==================================')
            print('QwQ\'s hint after observing question + thinking + solution:\n\n')
            print(hint_text_raw)
            print('\n\n==================================')
            print(json.dumps(problem_specific_hint, indent=4))


            out_entry = {
                'question': question,
                'teacher_answer': answer,
                'teacher_thoughts': thoughts,
                'hint': {
                    'hint_model': teacher_model_name, 
                    'hint_text_raw': hint_text_raw,
                    'content': problem_specific_hint
                }
            }
            print(json.dumps(out_entry), file=out_f, flush=True)