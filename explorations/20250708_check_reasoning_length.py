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
from datetime import datetime
from transformers import AutoTokenizer


# @backoff.on_exception(backoff.constant, Exception, interval=5)
def run_chat_completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


@backoff.on_exception(backoff.constant, Exception, interval=5)
def run_generate_with_backoff(client, **kwargs):
    return client.completions.create(**kwargs)

# load direct pred logs
def load_direct_pred_logs(task):
    if task == 'gpqa':
        logs = json.load(open('/fsx-comem/diwu0162/Search-o1/outputs/gpqa.qwq.direct/diamond.7.1,20:8.json'))
    elif task == 'aime':
        logs = json.load(open('/fsx-comem/diwu0162/Search-o1/outputs/aime.qwq.direct/test.7.1,19:42.json'))
    elif task == 'amc':
        logs = json.load(open('/fsx-comem/diwu0162/Search-o1/outputs/amc.qwq.direct/test.7.1,19:24.json'))
    elif task == 'math500':
        logs = json.load(open('/fsx-comem/diwu0162/Search-o1/outputs/math500.qwq.direct/test.7.1,20:17.json'))
    elif task == 'livecode':
        logs = json.load(open('/fsx-comem/diwu0162/Search-o1/outputs/livecode.qwq.direct/test_1to4.7.1,22:30.json'))
    elif task == 'bamboogle':
        logs = json.load(open('/fsx-comem/diwu0162/Search-o1/outputs/runs.qa/bamboogle.qwq.direct/test.7.1,16:22.json'))
    else:
        raise NotImplementedError
    return logs


def segment_thoughts_v1(x):
    return x.strip().split('\n\n')

def segment_thoughts_v2(x):
    # note: excluding things like "so" "therefore", "but", "let me" 
    reasoning_word_list = [
        'okay', 'hmm', 'wait', 'but wait', 'oh wait', 'no wait', 'no, wait', 'but let me', 'but actually', 'alternatively', 
        'now', 'the question', 'ah', 'oh', 'next', 'another angle', 'another approach', 'also', 'hold on', 'looking it up', 
        'another point', 'I don\'t think', 'perhaps I', 'putting this together', 'Putting it all together', 'i\'m', 'but i\'m',   
        'let me think again', 'I don\'t see', 'maybe I', 'alternative', "I wonder if", "another way", 'an alternative', 
    ]
    prefix_len = max([len(x) for x in reasoning_word_list])
    newline_segmented_thoughts = segment_thoughts_v1(x)
    final_thoughts = []
    for t in newline_segmented_thoughts:
        t_lower = t.lower()
        is_segment_start = False
        for r_w in reasoning_word_list:
            if t_lower.startswith(r_w.lower()):
                is_segment_start = True
                break
        if is_segment_start or not final_thoughts:
            final_thoughts.append(t)
        else:
            final_thoughts[-1] += '\n\n' + t
    return final_thoughts


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('Qwen/QwQ-32B')
    # for task in ['gpqa', 'aime', 'amc', 'math500', 'bamboogle']: # , 'livecode']:
    for task in ['livecode']: 
        print('Task:', task)
        logs = load_direct_pred_logs(task)
        question_len, thought_len_steps, thought_len_tokens, step_len, pred_len = [], [], [], [], []
        print(logs[0].keys())
        for entry in tqdm(logs):
            question = entry['Question']
            full_output = entry['Output']
            prediction = full_output.split('</think>')[-1]
            thoughts = '</think>'.join(full_output.split('</think>')[:-1])
            thought_steps = segment_thoughts_v2(thoughts)

            question_len.append(len(tokenizer.encode(question)))
            thought_len_steps.append(len(thought_steps))
            thought_len_tokens.append(len(tokenizer.encode(thoughts)))
            for s in thought_steps:
                step_len.append(len(tokenizer.encode(s)))
            pred_len.append(len(tokenizer.encode(prediction)))
            
        
        print(f'\tquestion_len: {round(np.mean(question_len), 2)}')
        print(f'\tthought_len_steps: {round(np.mean(thought_len_steps), 2)}')
        print(f'\tthought_len_tokens: {round(np.mean(thought_len_tokens), 2)}')
        print(f'\tstep_len: {round(np.mean(step_len), 2)}')
        print(f'\tpred_len: {round(np.mean(pred_len), 2)}')