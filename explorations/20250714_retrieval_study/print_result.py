import sys
import json
import numpy as np


in_log = sys.argv[1]
try:
    in_data = json.load(open(in_log))
except:
    in_data = [json.loads(line) for line in open(in_log).readlines()]

print(in_log)
# for metric_cared in ['recall', 'ndcg']:
for metric_cared in ['recall@5', 'recall@10']:
    print(f'\t{metric_cared}: {round(np.mean([x["retrieval_result"]["metrics"][metric_cared] for x in in_data]), 3)}')

    # for k in in_data[0]['retrieval_result']['metrics']:
        # if metric_cared in k.lower():
        #     print(f'\t{k}: {round(np.mean([x["retrieval_result"]["metrics"][k] for x in in_data]), 3)}')
   
