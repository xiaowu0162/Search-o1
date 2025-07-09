import re
import os
import json
import string
from tqdm import tqdm
import torch
import numpy as np
from openai import OpenAI, AzureOpenAI
from collections import Counter
import backoff
from datetime import datetime
from typing import List, Dict


##############################################################################
# Client backends
##############################################################################
OPENAI_REQUEST_TIMEOUT = 60*60*24 

student_model_port = 8001
student_model_name = 'Qwen/QwQ-32B'
student_model_client = OpenAI(base_url=f"http://localhost:{student_model_port}/v1", api_key="EMPTY", timeout=OPENAI_REQUEST_TIMEOUT)

teacher_model_name = "gpt-4o"

if teacher_model_name == "gpt-4o":
    teacher_model_client = AzureOpenAI(
        api_version='2024-10-21',
        api_key='f9f1776c5da04fc8ba1c3ccd6a8faeeb',
        azure_endpoint='https://azure-services-fair-openai1-westus.azure-api.net',    
    )
elif teacher_model_name == "o3":
    teacher_model_client = AzureOpenAI(
        api_version='2024-10-21',
        api_key='f9f1776c5da04fc8ba1c3ccd6a8faeeb',
        azure_endpoint='https://azure-services-fair-openai1-westus.azure-api.net',    
    )
else:
    raise NotImplementedError

def _call_teacher_llm(messages: List[Dict], *, temperature: float = 0.3, max_tokens: int = 1500):
    """Call the teacher LLM with retries."""

    teacher_model_client = globals().get("teacher_model_client")
    teacher_model_name = globals().get("teacher_model_name")
    OPENAI_REQUEST_TIMEOUT = globals().get("OPENAI_REQUEST_TIMEOUT")

    @backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=None)
    def _chat_once():
        resp = teacher_model_client.chat.completions.create(
            model=teacher_model_name,
            messages=messages,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            timeout=OPENAI_REQUEST_TIMEOUT,
        )
        return resp.choices[0].message.content

    return _chat_once()

##############################################################################
# JSON extraction helpers
##############################################################################

OPENAI_JSON_RE = re.compile(r"\{[\s\S]*?\}")  # greedy braces-based match of JSON blocks


def _extract_json_block(text: str) -> Dict:
    """Return the **last** JSON object that appears in *text*.

    Each teacher-model reply now contains visible chain-of-thought, then the
    required JSON on the *final* line.  We therefore scan for *all* JSON blocks
    and parse the last one.  If parsing fails we make a best-effort to clean it.
    """
    matches = list(OPENAI_JSON_RE.finditer(text))
    if not matches:
        raise ValueError("No JSON object found in assistant reply.")
    json_str = matches[-1].group(0)
    json_str = json_str.replace("\u201c", '"').replace("\u201d", '"')
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        cleaned = re.sub(r"[^\x20-\x7E]+", "", json_str)
        return json.loads(cleaned)


##############################################################################
# Data helper functions
##############################################################################
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


##############################################################################
# Thought segmentation
##############################################################################
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


##############################################################################
# Eval helper functions (right now only supporting choice, qa, and math)
##############################################################################
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


##############################################################################
# Rollout helper functions 
##############################################################################
def roll_out_single_node_with_hint(orig_prefix, hint, steps, step_id, answer, task, eval_task_type, metric_name, n_sample_per_node, node_join_char='\n\n', max_tokens=24000):
    def process_question_instruct_add_hint(instruction):
        if task in ['gpqa']:
            instruction_new = instruction.replace('Please answer the following multiple-choice question.',
                                                  'Please answer the following multiple-choice question. Hints might be provided during your question answering wrapped within [hint] and [end of hint]. If you see hints, try to leverage them to guide your thinking process.')
        elif task in ['aime', 'amc', 'math500']:
            instruction_new = instruction.replace('Please answer the following math question.',
                                                  'Please answer the following math question. Hints might be provided during your question answering wrapped within [hint] and [end of hint]. If you see hints, try to leverage them to guide your thinking process.')
        elif task in ['livecode']:
            raise NotImplementedError
        elif task in ['bamboogle']:
            instruction_new = instruction.replace('Please answer the following question.',
                                                  'Please answer the following question. Hints might be provided during your question answering wrapped within [hint] and [end of hint]. If you see hints, try to leverage them to guide your thinking process.')
        assert instruction_new != instruction
        return instruction_new

    perf_dict = {}
    
    cur_prefix_nodes = steps[:step_id]
    # cur_prefix_text = prefix + node_join_char.join(cur_prefix_nodes)
    if hint != "":
        formatted_question = process_question_instruct_add_hint(orig_prefix)
        hint_str = f'[hint] {hint} [end of hint]\n\nOkay,'
    else:
        formatted_question = orig_prefix
        hint_str = ''
    final_prompt = formatted_question + node_join_char.join(cur_prefix_nodes) + hint_str
    
    # print(f'Roll out with hint injected at (before) node {step_id}, {n_sample_per_node} samples per node...')
    # print('Prompt:', [final_prompt])

    responses = student_model_client.completions.create(model=student_model_name, prompt=final_prompt, n=n_sample_per_node, temperature=0.7, top_p=0.8, 
                                                        max_tokens=max_tokens, timeout=OPENAI_REQUEST_TIMEOUT,
                                                        extra_body={'top_k': 20, 'include_stop_str_in_output': True, 'repetition_penalty': 1.05,})

    sample_preds = [x.text for x in responses.choices]
    sample_preds_processed = []
    metrics = []
    for cur_pred in sample_preds:
        metrics_dict, processed_pred = evaluate_predictions(cur_pred, answer, mode=eval_task_type, use_llm=False, question=None, extract_answer=True)
        sample_preds_processed.append(processed_pred)
        metrics.append(metrics_dict[metric_name])
    perf_dict[step_id] = {
        'preds': sample_preds,
        'preds_processed': sample_preds_processed,
        'metrics': metrics
    }
    # cur_node_mean_metrics = np.mean(metrics)
    # print(metrics)
    # print(f'Mean metrics {round(cur_node_mean_metrics, 4)}')
    return perf_dict


##############################################################################
# Main optimization function
##############################################################################
def propose_initial_hint(question):
    for _ in range(5):
        system_msg = (
            "You are an expert tutor. First think step-by-step (this will be visible), "
            "then *on the very last line* output exactly one JSON object with the "
            "schema {\"hint\": \"<hint text>\"}. Do NOT wrap the JSON in code "
            "fences or add anything after it."
        )
        user_msg = (
            "Can you write a hint to this question? The hint should be several sentences and its style should follow first-person self-reflection perspective (e.g., 'I should ...'). Make sure the hints are high-level but contextualised so that similar problems can also benefit from looking at it. Also, do not give out the final answer but focus on the key high-level insight or general workflow. First think step-by-step and then output the hint in JSON in the last line.\n\nQuestion:\n" + question.strip()
        )
        try:
            assistant_reply = _call_teacher_llm([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ], temperature=0.8)
            hint_json = _extract_json_block(assistant_reply)
            return hint_json.get("hint", "")
        except:
            # print('JSON extraction error in propose_initial_hint:', assistant_reply)
            print('caught a JSON extraction error in propose_initial_hint')
            hint_json = {}
    return hint_json.get("hint", "")
        

def refine_hint(question, prev_hint, student_preds):
    for _ in range(5):
        student_structured = {f"Attempt {i}": p for i, p in enumerate(student_preds)}
        attempts_json = json.dumps(student_structured, indent=2)
        system_msg = (
            "You are an expert tutor. First think step-by-step (this will be visible), "
            "then *on the very last line* output exactly one JSON object with the "
            "schema {\"hint\": \"<hint text>\"}. Do NOT wrap the JSON in code "
            "fences or add anything after it."
        )
        user_msg = (
            "Here are ten attempts from a model trying to solve the question based on an earlier hint. Revise the hint so that the model can better understand and follow it. The style should follow first-person self-reflection perspective Make sure the hints are high-level but contextualised so that similar problems can also benefit from looking at it. Also, do not give out the final answer but focus on the key high-level insight or general workflow. First think step-by-step and then output the hint in JSON in the last line."
            "Use first-person perspective.\n\nQuestion: " + question.strip() + "\n\nPrevious Hint: " + prev_hint + "\n\nModel Attempts: " + attempts_json
        )
        try:
            assistant_reply = _call_teacher_llm([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ], max_tokens=1500, temperature=0.8)
            # hint_json = _extract_json_block(assistant_reply)
            hint_json = _extract_json_block(assistant_reply)
            return hint_json.get("hint", "")
        except:
            # print('JSON extraction error in refine_hint:', assistant_reply)
            print('caught a JSON extraction error in refine_hint')
            hint_json = {}
    return hint_json.get("hint", "")

def critique_thought_adherence(question, hint, student_preds):
    scores: List[int] = []
    for attempt in student_preds:
        system_msg = (
            "You are a meticulous grader. First think step-by-step, then on the "
            "last line output one JSON object: {\"score\": <int>} where <int> is "
            "1-10 (10 = perfect adherence). No markdown fences."
        )
        user_msg = "Evaluate how well the following attempt follows the hint.\n\nQuestion: " + question.strip() + "\n\nHint: " + hint.strip() + "\n\nAttempt: " + attempt.strip()
        try:
            assistant_reply = _call_teacher_llm([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ], max_tokens=512)
            score_json = _extract_json_block(assistant_reply)
            raw_score = int(score_json.get("score", 1))
        except Exception:
            raw_score = 1
        scores.append(max(1, min(10, raw_score)))
    return scores


def iterative_hint_optimization(question, answer, steps, hint_injection_loc, task, eval_task_type, metric_name, n_iters):
    assert hint_injection_loc == 0   # hard code to 0 for now
    hint_history = []
    cur_hint = propose_initial_hint(question)
    student_attempts = roll_out_single_node_with_hint(question, cur_hint, steps, hint_injection_loc, answer, task, 
                                                      eval_task_type, metric_name, n_sample_per_node=10, node_join_char='\n\n', max_tokens=500)[hint_injection_loc]['preds']
    student_attempt_critique = critique_thought_adherence(question, cur_hint, student_attempts)
    best_prev_adherence_score = np.mean(student_attempt_critique)
    hint_history.append({'hint': cur_hint, 'student_attempts': student_attempts, 'student_attempt_adherence': student_attempt_critique})
    print({'hint': cur_hint, 'student_attempt_critique': student_attempt_critique})
    for i in tqdm(range(n_iters), 'Optimizing single example'):
        prev_hint_for_logging = cur_hint
        update_hint = False
        new_hint = refine_hint(question, cur_hint, student_attempts)
        if new_hint == "":
            continue
        student_attempts = roll_out_single_node_with_hint(question, cur_hint, steps, hint_injection_loc, answer, task, 
                                                          eval_task_type, metric_name, n_sample_per_node=10, node_join_char='\n\n', max_tokens=500)[hint_injection_loc]['preds']
        student_attempt_critique = critique_thought_adherence(question, cur_hint, student_attempts)
        # new: update only if student thought adherence improves
        if np.mean(student_attempt_critique) >= best_prev_adherence_score:
            best_prev_adherence_score = np.mean(student_attempt_critique)
            cur_hint = new_hint
            update_hint = True
        hint_history.append({'iter': i, 'prev_hint': prev_hint_for_logging, 'proposed_hint': new_hint, 'hint_updated': update_hint,
                              'student_attempts': student_attempts, 'student_attempt_adherence': student_attempt_critique})
        print({'iter': i, 'hint': cur_hint, 'average_adherence': np.mean(student_attempt_critique)})

    return hint_history


if __name__ == '__main__':
    exp_id = '20250709_v2'

    task = 'bamboogle'   # math500 gpqa bamboogle
    n_optimize_iters = 20
    out_dir = f'202507_hint_optimization_logs/{exp_id}/{task}/'
    os.makedirs(out_dir, exist_ok=True)

    ################################################
    # load data
    ################################################
    if task in ['gpqa']:
        eval_task_type = 'choose'
        metric_name = 'acc'
        answer_field = "Correct Choice"
    elif task in ['aime', 'amc', 'math500']:
        eval_task_type = 'math'
        metric_name = 'math_equal'
        answer_field = 'answer'
    # elif task in ['livecode']:
    #     eval_task_type = 'code'
    elif task in ['gaia', 'bamboogle']:
        eval_task_type = 'qa'
        metric_name = 'f1'
        answer_field = 'answer'
    else:
        raise NotImplementedError
    assert eval_task_type in ['math', 'choose', 'qa']   # not supporting code here
    orig_pred_logs = load_direct_pred_logs(task)
    if metric_name in ['acc', 'math_equal']:
        orig_pred_logs_wrong = [x for x in orig_pred_logs if not x['Metrics'][metric_name]]
    else:
        orig_pred_logs_wrong = [x for x in orig_pred_logs if x['Metrics'][metric_name] < 0.5]

    # for each question
    #     rollout without hint
    #     get hint by calling iteative hint optimization
    #     rollout once with hint 
    #     save metrics

    for entry in tqdm(orig_pred_logs_wrong, desc='Processing examples'):
        question = entry['Question']
        answer = entry[answer_field]# ['answer'] if 'answer' in entry else entry['Answer']
        all_pred_steps = segment_thoughts_v2('\n\n'.join(entry['Output'].split('</think>')[:-1]))
        hint_injection_loc = 0   # for now we hard-code this
        hint_history = iterative_hint_optimization(question, answer, all_pred_steps, hint_injection_loc, task, eval_task_type, metric_name, n_optimize_iters)

        # find best iter idx
        # best_iter_idx = 0
        # best_critique_score = -1
        # for i_ent, hint_history_entry in enumerate(hint_history):
        #     if np.mean(hint_history_entry['student_attempt_adherence']) >= best_critique_score:
        #         best_iter_idx = i_ent
        #         best_critique_score = np.mean(hint_history_entry['student_attempt_adherence']).item()

        # rollout and eval hint quality
        out_dict = {
            'hint_optimization_history': hint_history,
            'hint_rollout_results': {
                'no_hint': roll_out_single_node_with_hint(question, "", all_pred_steps, hint_injection_loc, answer, task, 
                                                          eval_task_type, metric_name, n_sample_per_node=10, node_join_char='\n\n', max_tokens=20000),
                'initial_hint': roll_out_single_node_with_hint(question, hint_history[0], all_pred_steps, hint_injection_loc, answer, task, 
                                                          eval_task_type, metric_name, n_sample_per_node=10, node_join_char='\n\n', max_tokens=20000),
                # f'best_iter_{best_iter_idx}': roll_out_single_node_with_hint(question, hint_history[best_iter_idx], all_pred_steps, hint_injection_loc,
                #                                                              answer, task, eval_task_type, metric_name, n_sample_per_node=10,
                #                                                              node_join_char='\n\n', max_tokens=20000),
                f'last_iter_hint': roll_out_single_node_with_hint(question, hint_history[-1], all_pred_steps, hint_injection_loc, answer, task, 
                                                                  eval_task_type, metric_name, n_sample_per_node=10, 
                                                                  node_join_char='\n\n', max_tokens=20000),
            }
        }

        out_file = f'{out_dir}/trace_questionid{entry["id"]}_{n_optimize_iters}iters_{datetime.now().strftime("%Y%m%d-%H%M")}.json'
        json.dump(out_dict, open(out_file, 'w'), indent=4)

        print({'hint_rollout_metrics': {
            'no_hint': np.mean(out_dict['hint_rollout_results']['no_hint'][hint_injection_loc]['metrics']),
            'initial_hint': np.mean(out_dict['hint_rollout_results']['initial_hint'][hint_injection_loc]['metrics']),
            # f'best_iter_{best_iter_idx}': np.mean(out_dict['hint_rollout_results'][f'best_iter_{best_iter_idx}'][hint_injection_loc]['metrics']),
            f'last_iter_hint': np.mean(out_dict['hint_rollout_results'][f'last_iter_hint'][hint_injection_loc]['metrics'])
        }})
        


