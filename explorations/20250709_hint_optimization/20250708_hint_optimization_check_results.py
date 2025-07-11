import json
import sys
import numpy as np
from collections import Counter
from glob import glob


#def main(rollout_file, hint_dir):
def main(hint_dir, focused_qid=None):
    # # load rollout logs as reference
    # try:
    #     rollout_logs = json.load(open(rollout_file))
    # except:
    #     rollout_logs = [json.loads(line) for line in open(rollout_file).readlines()]
    # base_rollout_scores = {}
    # for entry in rollout_logs:
    #     try:
    #         base_rollout_scores[str(entry['item']['id'])] = np.mean(entry['perf_dict'][0]['metrics'])
    #     except:
    #         base_rollout_scores[str(entry['item']['id'])] = np.mean(entry['perf_dict']['0']['metrics'])
    # print(f'{len(base_rollout_scores)} items loaded from original rollout file')
    # print(base_rollout_scores)


    # grab hint logs and pick the latest for each qid
    hint_optimization_log_files_raw = glob(hint_dir + '/*')
    if focused_qid:
        hint_optimization_covered_qids = [str(focused_qid)]
    else:
        hint_optimization_covered_qids = set([str(x.split('/')[-1].split('_')[1].replace('questionid', '')) for x in hint_optimization_log_files_raw])
    hint_optimization_log_files_final = []
    for qid in sorted(hint_optimization_covered_qids):
        cur_qid_related_files = [x for x in hint_optimization_log_files_raw if f'_questionid{qid}_' in x]
        cur_qid_related_files.sort()
        hint_optimization_log_files_final.append(cur_qid_related_files[-1])
    assert len(hint_optimization_covered_qids) == len(hint_optimization_log_files_final)

    # parse and print results
    metrics = {}
    for cur_hint_file in hint_optimization_log_files_final:
        cur_hint_log = json.load(open(cur_hint_file))
        cur_qid = str(cur_hint_file.split('/')[-1].split('_')[1].replace('questionid', ''))
        print(cur_qid)
        # if str(cur_qid) in base_rollout_scores:
        #     print(f'\tBase rollout score without hint:', base_rollout_scores[str(cur_qid)])
        cur_metrics = {k: np.mean(cur_hint_log['hint_rollout_results'][k]['0']['metrics']) for k in cur_hint_log['hint_rollout_results']}
        cur_metrics['n_updates'] = sum([1 if x['hint_updated'] else 0 for x in cur_hint_log['hint_optimization_history'][1:]])
        print('\t', cur_metrics, sep='')
        # print('\t', {k: np.mean(cur_hint_log['hint_rollout_results'][k]['0']['hint']) for k in cur_hint_log['hint_rollout_results']}, sep='')
        print('\tInitial hint:', cur_hint_log['hint_optimization_history'][0]['hint'])
        print('\tFinal hint:', cur_hint_log['hint_optimization_history'][-1]['proposed_hint'] if cur_hint_log['hint_optimization_history'][-1]['hint_updated'] else cur_hint_log['hint_optimization_history'][-1]['prev_hint'])
        print('\n======================================\n')
        # update metrics
        for k, v in cur_metrics.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)
    print('Total questions evaluated:', len(hint_optimization_log_files_final))
    print({k: round(np.mean(v), 3) for k, v in metrics.items()})


if __name__ == '__main__':
    exp = '20250709_v2'
    teacher = 'gpt-4o'          # gpt-4o o3
    n_iters = 10
    task = 'gpqa'      # math500 bamboogle gpqa
    focused_qid = None      # None if need to check all

    hint_dir = f'/fsx-comem/diwu0162/Search-o1/explorations/20250709_hint_optimization/logs/{exp}/teacher{teacher}_{n_iters}iters/{task}/'
    
    # if task == 'bamboogle'
    # # rollout_file = '/fsx-comem/diwu0162/Search-o1/explorations/20250707_rollout_analysis/perf_dicts_wrong_only_math500_20250708-1641.json'
    # # rollout_file = '/fsx-comem/diwu0162/Search-o1/explorations/20250707_rollout_analysis/perf_dicts_wrong_only_gpqa_20250709-1517.jsonl'
    # rollout_file = '/fsx-comem/diwu0162/Search-o1/explorations/20250707_rollout_analysis/perf_dicts_wrong_only_bamboogle_20250708-0443.json'
    # assert task in rollout_file
    
    main(hint_dir, focused_qid)


    #     bamboogle: 56, 17