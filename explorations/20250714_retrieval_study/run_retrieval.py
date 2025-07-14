import os
import multiprocessing as mp
from functools import partial
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import argparse
from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer, util
# from openai import OpenAI
# from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import ndcg_score


def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--query_file', type=str, required=True)
    parser.add_argument('--corpus_file_pattern', type=str, required=True)
    parser.add_argument('--corpus_dir', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    
    # retrieval parameters
    parser.add_argument('--retriever', type=str, required=True, choices=['bm25'])
    parser.add_argument('--q_type', type=str, required=True, choices=['question', 'question_description', 'first_thought_steps'])
    parser.add_argument('--k_type', type=str, required=True, choices=['self', 'question', 'question_description', 'first_thought_steps'])
    # parser.add_argument('--v_type', type=str, required=True, choices=['hint'])   # no need to experiment for now
    parser.add_argument('--record_top_k', type=int, default=10)
    parser.add_argument('--report_metrics', action='store_true')
    # parser.add_argument('--n_workers', type=int, default=1)

    return parser.parse_args()


def recall_at_k(scores, labels, k, gt_label=1):
    sorted_indices = np.argsort(scores)[::-1]
    return int(any(labels[j] == gt_label for j in sorted_indices[:k])), sorted_indices


def ndcg_at_k(scores, labels, k, gt_label=1):
    top_k = np.argsort(scores)[::-1][:k]
    relevance = [1 if labels[j] == gt_label else 0 for j in top_k]
    ideal = sorted(relevance, reverse=True)
    return ndcg_score([ideal], [relevance]) if any(ideal) else 0.0


def main():
    args = parse_args()

    ##########################################################################
    # load corpus and build index
    ##########################################################################
    corpus_values, corpus_keys, corpus_labels = [], [], []
    # part 1: from original corpus
    corpus_files = glob(f'{args.corpus_dir}/{args.corpus_file_pattern}')
    for corpus_file in corpus_files:
        count = 0
        for line in open(corpus_file).readlines():
            entry = json.loads(line)
            # prepare value 
            corpus_values.append(entry['hint']['content']['hint'])
            # prepare key 
            if args.k_type == 'self':
                corpus_keys.append(corpus_values[-1])
            else:
                raise NotImplementedError    
            # prepare label        
            corpus_labels.append(0)
            count += 1
        print(f'Loaded {count} from {corpus_file}')

    # part 2: from retrieval file
    query_data = []    # query, extended label, data_structure
    try:
        query_data_raw = [json.loads(line) for line in open(args.query_file).readlines()]
    except:
        query_data_raw = json.load(open(args.query_file))
    for query_entry in query_data_raw:
        # prepare value 
        corpus_values.append(query_entry['hint']['content']['hint'])
        
        # prepare key 
        if args.k_type == 'self':
            corpus_keys.append(corpus_values[-1])
        else:
            raise NotImplementedError    

        # prepare label
        cur_chunk_extended_labels = [0 if x['question'] != query_entry['question'] else 1 for x in query_data_raw]
        assert sum(cur_chunk_extended_labels) == 1 and len(cur_chunk_extended_labels) == len(query_data_raw)

        # prepare query
        corpus_values.append(entry['hint']['content']['hint'])
        if args.q_type == 'question':
            cur_query = entry['question']
        else:
            raise NotImplementedError 
        
        # other data to log
        cur_out_data_structure = {
            'question': query_entry['question'],
            'hint': query_entry['hint']
        }
        
        query_data.append([cur_query, cur_chunk_extended_labels, cur_out_data_structure])

    print(f'Corpus building complete: {len(corpus_keys)} items from {len(corpus_files)} files.')


    ##########################################################################
    # run retrieval
    ##########################################################################
    tokenized_corpus = [doc.split(" ") for doc in corpus_values]
    bm25 = BM25Okapi(tokenized_corpus)
    out_data = []
    all_metrics = []
    for query, extended_labels, out_data_structure in tqdm(query_data, desc='Running retrieval'):
        scores = bm25.get_scores(query.split(" "))
        all_labels = corpus_labels + extended_labels
        # calculate metrics
        metrics = {}
        for k in [5, 10, 50, 100]:
            metrics[f'recall@{k}'], sorted_indices = recall_at_k(scores, all_labels, k)
            metrics[f'ndcg@{k}'] = ndcg_at_k(scores, all_labels, k)

        out_data_structure['retrieval_result'] = {
            'metrics': metrics,
            'top_items': [corpus_values[idx] for idx in sorted_indices[:args.record_top_k]]
        }
        all_metrics.append(metrics)
        out_data.append(out_data_structure)


    ##########################################################################
    # print scores and write retrieval results to file
    ##########################################################################
    print('Retrieval metrics:')
    for k in all_metrics[0]:
        print(f'\t{k}: {round(np.mean([x[k] for x in all_metrics]), 4)}')

    json.dump(out_data, open(args.out_file, 'w'), indent=4)
    print('Saved to', args.out_file)


if __name__ == '__main__':
    main()