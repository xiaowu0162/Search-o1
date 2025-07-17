import re
import os
import sys
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


def get_metric_names(task):
    if task in ['gpqa']:
        metric_name = 'acc'
        answer_field = "Correct Choice"
    elif task in ['aime', 'amc', 'math500']:
        metric_name = 'math_equal'
        answer_field = 'answer'
    elif task in ['livecode']:
        metric_name = 'pass@1'
        answer_field = 'public_testcases'
    elif task in ['gaia', 'bamboogle']:
        metric_name = 'f1'
        answer_field = 'answer'
    else:
        raise NotImplementedError
    
    return metric_name, answer_field
    

if __name__ == '__main__':
    log_file = sys.argv[1]

    task = None
    if 'gpqa' in log_file.lower():
        task = 'gpqa'
    elif 'bamboogle' in log_file.lower():
        task = 'bamboogle'
    elif 'aime' in log_file.lower():
        task = 'aime'
    elif 'amc' in log_file.lower():
        task = 'amc'
    elif 'math500' in log_file.lower():
        task = 'math500'
    elif 'livecode' in log_file.lower():
        task = 'livecode'
    else:
        raise NotImplementedError
    
    try:
        logs = json.load(open(log_file))
    except:
        logs = [json.loads(line) for line in open(log_file).readlines()]

    metric_name, answer_field = get_metric_names(task)
    ids_correct = set([x['id'] for x in logs if x['Metrics'][metric_name]])
    ids_wrong = set([x['id'] for x in logs if not x['Metrics'][metric_name]])
    print(f'\tLoaded {len(logs)} examples: {len(ids_correct)} correct and {len(ids_wrong)} wrong.')
    
    n_steps_v2, n_steps_v2_correct, n_steps_v2_wrong = [], [], []
    for i, entry in enumerate(logs):
        
        thoughts = entry['Output'].split('</think>')[0]
    
        thoughts_segmented_v2 = segment_thoughts_v2(thoughts)
        n_steps_v2.append(len(thoughts_segmented_v2))

        if entry['id'] in ids_wrong:
            n_steps_v2_wrong.append(len(thoughts_segmented_v2))
        else:
            n_steps_v2_correct.append(len(thoughts_segmented_v2))

    print('\tAll ex avg steps {}, correct ex avg steps {}, wrong ex avg steps {}'.format(round(np.mean(n_steps_v2), 0), round(np.mean(n_steps_v2_correct), 0), round(np.mean(n_steps_v2_wrong), 0)))
