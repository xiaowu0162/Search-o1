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

# evaluation helper (right now only supporting choice, qa, and math)

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2
        
def extract_answer_fn(output, mode='qa', extract_answer=False):
    if extract_answer == False and mode not in ['infogen', 'summary', 'research']:
        if mode == 'qa':
            return output.strip()
        pred_answer_lines = output.replace("\n\n", "\n").strip().split('\n')
        pred_answer = '\n'.join(pred_answer_lines[-3:])
        return pred_answer
    extracted_text = ''
    if mode == 'codegen':
        pattern = r'```python\s*(.*?)\s*```'  # Extract the code between ```python and ```
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode in ['infogen', 'summary', 'research']:
        pattern_info = "**Final Information"
        if "</think>\n" in output:
            extracted_text = output.split("</think>\n")[-1].split("<|begin_click_link|>")[0].replace(pattern_info, "").strip(':**').strip('\n').strip("```").strip()  # 提取</think>后面的内容
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
        elif pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].split("<|begin_click_link|>")[0].strip('\n').strip(':**').strip("```").strip()  # 提取**Final Information**后面的内容
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
        else:
            # extracted_text = "No helpful information found."
            extracted_text = '\n'.join(output.strip().replace("</think>\n", "").replace("\n\n", "\n").split('\n')[-5:])  # 若没提取到，只保留最后5行
        if mode == 'research':
            extracted_text = extracted_text[:6000]
        else:
            extracted_text = extracted_text[:2500]
    elif mode in ['math', 'choose', 'qa']:
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
        else:
            pattern = 'ANSWER:'
            if pattern in output:
                extracted_text = output.split(pattern)[-1].strip('**').strip()
        if mode in ['choose']:
            inner_pattern = r'\\text\{(.*)\}'
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1]  # Take the last match
            extracted_text = extracted_text.strip("()")
    return extracted_text
    
def evaluate_predictions(output, labeled_answer, mode='math', use_llm=False, question=None, extract_answer=False):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0, 'llm_equal': 0}
    pred_answer = extract_answer_fn(output, mode=mode, extract_answer=extract_answer)
    pred_answer_new = pred_answer
    if pred_answer != '':
        final_metric["is_valid_answer"] = True
    else:
        # If no answer was extracted, keep only the last 3 lines
        pred_answer_new = '\n'.join(output.replace("\n\n", "\n").strip().split('\n')[-5:])

    if mode in ['qa']:
        def normalize_answer_qa(s):
            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)
            def white_space_fix(text):
                return " ".join(text.strip().split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        normalized_pred_answer = normalize_answer_qa(pred_answer_new)

        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    elif mode in ['math', 'choose']:
        def normalize_answer(text):
            text = text.lower()
            text = " ".join(text.strip().split())
            return text
        normalized_pred_answer = normalize_answer(pred_answer_new)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)
    
        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)
        
        # Add LLM-based evaluation if requested
        if use_llm and question is not None:
            final_metric["llm_equal"] = 0  # Will be updated in batch later

    return final_metric, pred_answer


def analyze_node_by_node_v0(prefix, steps, answer, eval_task_type, metric_name, n_sample_per_node, node_join_char='\n\n'):  #  
    print('Prefix:', [prefix])
    print(f'{len(steps)} nodes to run, {n_sample_per_node} samples per node...')

    prefixes_to_run = []
    for i_node in range(len(steps)):
        cur_prefix_nodes = steps[:i_node]
        cur_prefix_text = prefix + node_join_char.join(cur_prefix_nodes)
        prefixes_to_run.append(cur_prefix_text)
    
    # print(json.dumps(prefixes_to_run, indent=4))

    responses = client.completions.create(model=model_name, prompt=prefixes_to_run, n=n_sample_per_node, temperature=0.7, top_p=0.8, max_tokens=20000, timeout=3600,
                                          extra_body={'top_k': 20, 'include_stop_str_in_output': True, 'repetition_penalty': 1.05,})

    perf_dict = {}
    for i_node in range(len(steps)):
        sample_preds = [x.text for x in responses.choices[i_node*n_sample_per_node:(i_node+1)*n_sample_per_node]]
        sample_preds_processed = []
        metrics = []
        for cur_pred in sample_preds:
            metrics_dict, processed_pred = evaluate_predictions(cur_pred, answer, mode=eval_task_type, use_llm=False, question=None, extract_answer=True)
            sample_preds_processed.append(processed_pred)
            metrics.append(metrics_dict[metric_name])
        perf_dict[i_node] = {
            'preds': sample_preds,
            'preds_processed': sample_preds_processed,
            'metrics': metrics
        }
    return perf_dict


def analyze_node_by_node_v1(analysis_mode, prefix, steps, answer, eval_task_type, metric_name, n_sample_per_node, node_join_char='\n\n', max_steps=10):
    assert analysis_mode in ['wrong', 'correct']
    print('Prefix:', [prefix])
    print(f'At most {min(max_steps,len(steps))} nodes to run, {n_sample_per_node} samples per node...')

    perf_dict = {}
    
    for i_node in range(min(max_steps,len(steps))):
        cur_prefix_nodes = steps[:i_node]
        cur_prefix_text = prefix + node_join_char.join(cur_prefix_nodes)

        responses = client.completions.create(model=model_name, prompt=cur_prefix_text, n=n_sample_per_node, temperature=0.7, top_p=0.8, max_tokens=20000, timeout=3600,
                                              extra_body={'top_k': 20, 'include_stop_str_in_output': True, 'repetition_penalty': 1.05,})

        sample_preds = [x.text for x in responses.choices]
        sample_preds_processed = []
        metrics = []
        for cur_pred in sample_preds:
            metrics_dict, processed_pred = evaluate_predictions(cur_pred, answer, mode=eval_task_type, use_llm=False, question=None, extract_answer=True)
            sample_preds_processed.append(processed_pred)
            metrics.append(metrics_dict[metric_name])
        perf_dict[i_node] = {
            'preds': sample_preds,
            'preds_processed': sample_preds_processed,
            'metrics': metrics
        }
        cur_node_mean_metrics = np.mean(metrics)
        if analysis_mode == 'wrong' and cur_node_mean_metrics == 0:
            print(f'Stopping at step {i_node} due to all samples failing.')
            break
        elif analysis_mode == 'correct' and cur_node_mean_metrics == 1:
            print(f'Stopping at step {i_node} due to all samples succeeding.')
            break
        else:
            print(f'Step {i_node}, mean metrics {round(cur_node_mean_metrics, 4)}, continuing')
    return perf_dict

    
if __name__ == '__main__':
    task = 'gpqa' # 'math500' 'gpqa' 'bamboogle'
    subset = 'correct'
    assert subset in ['wrong', 'correct']
    port = 8004
    model_name = 'Qwen/QwQ-32B'
    
    OPENAI_REQUEST_TIMEOUT = 60*60*24 
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY", timeout=OPENAI_REQUEST_TIMEOUT)
    print(client.models.list())

    
    if task in ['gpqa']:
        eval_task_type = 'choose'
        metric_name = 'acc'
    elif task in ['aime', 'amc', 'math500']:
        eval_task_type = 'math'
        metric_name = 'math_equal'
    # elif task in ['livecode']:
    #     eval_task_type = 'code'
    elif task in ['gaia', 'bamboogle']:
        eval_task_type = 'qa'
        metric_name = 'f1'
    else:
        raise NotImplementedError
    assert eval_task_type in ['math', 'choose', 'qa']   # not supporting code here
    
    
    logs = load_direct_pred_logs(task)
    print(f'Task: {task}')
    
    if task in ['bamboogle']:
        ids_correct = set([x['id'] for x in logs if x['Metrics']['em']])
        ids_wrong = set([x['id'] for x in logs if not x['Metrics']['em']])
        assert len(ids_correct) + len(ids_wrong) == len(logs)
    elif task in ['livecode']:
        ids_correct = set([x['id'] for x in logs if x['Metrics']['pass@1']])
        ids_wrong = set([x['id'] for x in logs if not x['Metrics']['pass@1']])
        assert len(ids_correct) + len(ids_wrong) == len(logs)
    else:
        ids_correct = set([x['id'] for x in logs if x['Metrics']['math_equal']])
        ids_wrong = set([x['id'] for x in logs if not x['Metrics']['math_equal']])
        assert len(ids_correct) + len(ids_wrong) == len(logs)
    print(f'Loaded {len(logs)} examples: {len(ids_correct)} correct and {len(ids_wrong)} wrong.')


    logs_wrong = [x for x in logs if x['id'] in ids_wrong]
    logs_correct = [x for x in logs if x['id'] in ids_correct]

    interested_logs = logs_wrong if subset == 'wrong' else logs_correct
    
    n_sample_per_node = 20

    out_file = f'perf_dicts_{subset}_only_{task}_{datetime.now().strftime("%Y%m%d-%H%M")}.jsonl'
    out_f = open(out_file, 'w')

    # out_logs = []
    for entry in tqdm(interested_logs):
        thoughts = entry['Output'].split('</think>')[0]
        thoughts_segmented_v2 = segment_thoughts_v2(thoughts)
        answer = entry['answer'] if 'answer' in entry else entry['Correct Answer']
        # out_logs.append({
        #     'item': entry,
        #     'perf_dict': analyze_node_by_node_v1(entry['Question'], thoughts_segmented_v2, answer, eval_task_type, metric_name, n_sample_per_node=n_sample_per_node)
        # })
        print(json.dumps({
            'item': entry,
            'perf_dict': analyze_node_by_node_v1(subset, entry['Question'], thoughts_segmented_v2, answer, 
                                                 eval_task_type, metric_name, n_sample_per_node=n_sample_per_node)
        }), file=out_f, flush=True)
        
    # for i_node in range(len(thoughts_segmented_v2)):
    #     print(i_node, perf_dict[i_node]['metrics'], np.mean(perf_dict[i_node]['metrics']))

    # json.dump(out_logs, open(out_file, 'w'))
    out_f.close()

    
    

    