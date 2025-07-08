import json
import sys
import numpy as np
from collections import Counter


result_log = sys.argv[1]
logs = json.load(open(result_log))


failure_loc = []
counts = Counter([len(x['perf_dict']) for x in logs])
for k, v in counts.items():
    if k != 1:
        failure_loc += [k for _ in range(v)]
print(counts)
print('Average failure location of non 0-halt items:', np.mean(failure_loc))


avg_correctness = []
correctness_trends = []
for entry in logs:
    if len(entry['perf_dict']) == 1:
        continue
    cur_sample_correctness_trends = []
    for i_step in range(len(entry['perf_dict'])):
        cur_avg_correctness = np.mean(entry['perf_dict'][str(i_step)]['metrics'])
        cur_sample_correctness_trends.append(cur_avg_correctness)
    avg_correctness.append(np.mean(cur_sample_correctness_trends))
    slope, intercept = np.polyfit([x for x in range(len(entry['perf_dict']))], cur_sample_correctness_trends, 1)
    if slope > 0:
        trend = 'increasing'
    elif slope < 0:
        trend = 'decreasing'
    else:
        trend = 'consistent'
    correctness_trends.append(trend)
    print(trend, [round(x.item(), 2) for x in cur_sample_correctness_trends])
    
print('Average correctness:', np.mean(avg_correctness))
print('Correctness trends:', Counter(correctness_trends))
