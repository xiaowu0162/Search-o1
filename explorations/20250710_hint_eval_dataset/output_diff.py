import os
import numpy as np
import json
from glob import glob
from tqdm import tqdm


def get_metric_name(task):
    if task in ['gpqa']:
        eval_task_type = 'choose'
        metric_name = 'acc'
        answer_field = "Correct Choice"
    elif task in ['aime', 'amc', 'math500']:
        eval_task_type = 'math'
        metric_name = 'math_equal'
        answer_field = 'answer'
    elif task in ['livecode']:
        eval_task_type = 'code'
        metric_name = 'pass@1'
        answer_field = 'public_test_cases'
    elif task in ['gaia', 'bamboogle']:
        eval_task_type = 'qa'
        metric_name = 'f1'
        answer_field = 'answer'
    else:
        raise NotImplementedError
    return metric_name, answer_field


if __name__ == '__main__':

    # model_alias = 'ds-llama-8b'
    model_alias = 'ds-qwen-7b'


    for ds in ['aime', 'amc', 'math500', 'gpqa', 'livecode', 'bamboogle']:
        direct_pred_log_dir = f'/fsx-comem/diwu0162/Search-o1/outputs/{ds}.{model_alias}.direct' if ds not in ['bamboogle'] else f'/fsx-comem/diwu0162/Search-o1/outputs/runs.qa/{ds}.{model_alias}.direct/'
        direct_pred_log_file = [x for x in glob(f'{direct_pred_log_dir}/*json') if 'metrics' not in x][-1]    # grabbing the latest log
        direct_pred_logs = {x['id']: x for x in json.load(open(direct_pred_log_file))}
        hint_pred_log_dir = f'/fsx-comem/diwu0162/Search-o1/outputs/{ds}.{model_alias}.directwithhint' if ds not in ['bamboogle'] else f'/fsx-comem/diwu0162/Search-o1/outputs/runs.qa/{ds}.{model_alias}.directwithhint/'
        hint_pred_log_file = [x for x in glob(f'{hint_pred_log_dir}/*json') if 'metrics' not in x][-1]    # grabbing the latest log
        hint_pred_logs = {x['id']: x for x in json.load(open(hint_pred_log_file))}

        metric_name, answer_field = get_metric_name(ds)
        assert len(hint_pred_logs) == len(direct_pred_logs)
        assert all(x in hint_pred_logs for x in direct_pred_logs)

        out_dir = f'diff_direct_pred_and_hint/{model_alias}/'
        os.makedirs(out_dir, exist_ok=True)
        out_file = f'{out_dir}/{ds}.json'

        out_data = []
        right_to_right, right_to_wrong, wrong_to_right, wrong_to_wrong, others = 0, 0, 0, 0, 0
        for idx in direct_pred_logs:
            direct_pred_entry = direct_pred_logs[idx]
            hint_pred_entry = hint_pred_logs[idx]
            direct_pred_metric = direct_pred_entry['Metrics'][metric_name]
            hint_pred_metric = hint_pred_entry['Metrics'][metric_name]
            if int(direct_pred_metric) == 1:
                if int(hint_pred_metric) == 1:
                    right_to_right += 1
                    perf_change_tag = 'right_to_right'
                else:
                    right_to_wrong += 1
                    perf_change_tag = 'right_to_wrong'
            elif int(direct_pred_metric) == 0:
                if int(hint_pred_metric) == 1:
                    wrong_to_right += 1
                    perf_change_tag = 'wrong_to_right'
                else:
                    wrong_to_wrong += 1
                    perf_change_tag = 'wrong_to_wrong'
            else:
                others += 1
                perf_change_tag = 'others'
            # if int(direct_pred_metric) != int(hint_pred_metric):
            out_data.append({
                'id': idx,
                'question': direct_pred_entry['Question'],
                'answer': direct_pred_entry[answer_field],
                'direct_pred': {
                    'prompt': direct_pred_entry['Question'],
                    'output': direct_pred_entry['Output'],
                    'pred_answer': direct_pred_entry['Pred_Answer'],
                    'metrics': direct_pred_entry['Metrics']
                },
                'hint_pred': {
                    'prompt': hint_pred_entry['Question'],
                    'output': hint_pred_entry['Output'],
                    'pred_answer': hint_pred_entry['Pred_Answer'],
                    'metrics': hint_pred_entry['Metrics']
                },
                'perf_change_tag': perf_change_tag,
            })
        json.dump(out_data, open(out_file, 'w'), indent=4)
        print('Task:', ds)
        print('\t', {'right_to_right': right_to_right, 'right_to_wrong': right_to_wrong, 'wrong_to_right': wrong_to_right, 'wrong_to_wrong': wrong_to_wrong, 'others': others}, sep='')
