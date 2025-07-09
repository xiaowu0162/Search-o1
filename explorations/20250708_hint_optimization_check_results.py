import json
import sys
import numpy as np
from collections import Counter
from glob import glob


def main(rollout_file, hint_dir):
    # load rollout logs as reference
    try:
        rollout_logs = json.load(open(rollout_file))
    except:
        rollout_logs = [json.loads(line) for line in open(rollout_file).readlines()]
    base_rollout_scores = {}
    for entry in rollout_logs:
        try:
            base_rollout_scores[str(entry['item']['id'])] = np.mean(entry['perf_dict'][0]['metrics'])
        except:
            base_rollout_scores[str(entry['item']['id'])] = np.mean(entry['perf_dict']['0']['metrics'])
    print(f'{len(base_rollout_scores)} items loaded from original rollout file')
    print(base_rollout_scores)


    # grab hint logs and pick the latest for each qid
    hint_optimization_log_files_raw = glob(hint_dir + '/*')
    hint_optimization_covered_qids = set([str(x.split('/')[-1].split('_')[1].replace('questionid', '')) for x in hint_optimization_log_files_raw])
    hint_optimization_log_files_final = []
    for qid in hint_optimization_covered_qids:
        cur_qid_related_files = [x for x in hint_optimization_log_files_raw if f'_questionid{qid}_' in x]
        cur_qid_related_files.sort()
        hint_optimization_log_files_final.append(cur_qid_related_files[-1])
    assert len(hint_optimization_covered_qids) == len(hint_optimization_log_files_final)


    # parse and print results
    for cur_hint_file in hint_optimization_log_files_final:
        cur_hint_log = json.load(open(cur_hint_file))
        cur_qid = str(cur_hint_file.split('/')[-1].split('_')[1].replace('questionid', ''))
        print(cur_qid)
        if str(cur_qid) in base_rollout_scores:
            print(f'\tBase rollout score without hint:', base_rollout_scores[str(cur_qid)])
        print('\t', {k: np.mean(cur_hint_log['hint_rollout_results'][k]['0']['metrics']) for k in cur_hint_log['hint_rollout_results']}, sep='')



if __name__ == '__main__':
    task = 'gpqa'   # math500 bamboogle gpqa

    hint_dir = f'/fsx-comem/diwu0162/Search-o1/explorations/202507_hint_optimization_logs/{task}/'
    
    # rollout_file = '/fsx-comem/diwu0162/Search-o1/explorations/20250707_rollout_analysis/perf_dicts_wrong_only_math500_20250708-1641.json'
    rollout_file = '/fsx-comem/diwu0162/Search-o1/explorations/20250707_rollout_analysis/perf_dicts_correct_only_gpqa_20250708-1744.json'
    assert task in rollout_file
    
    main(rollout_file, hint_dir)