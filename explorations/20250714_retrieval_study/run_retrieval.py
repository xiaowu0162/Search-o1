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
    parser.add_argument('--q_type', type=str, required=True, choices=['question', 'question_description', 'first_thought_steps', 'oracle_hint'])
    parser.add_argument('--k_type', type=str, required=True, choices=['self', 'question', 'question_description', 'first_thought_steps'])
    # parser.add_argument('--v_type', type=str, required=True, choices=['hint'])   # no need to experiment for now
    parser.add_argument('--record_top_k', type=int, default=10)
    parser.add_argument('--remove_oracle_from_top_k_record', action='store_true')
    parser.add_argument('--report_metrics', action='store_true')
    
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--debug_run', action='store_true')

    return parser.parse_args()


def recall_at_k(scores, labels, k, gt_label=1):
    sorted_indices = np.argsort(scores)[::-1]
    return int(any(labels[j] == gt_label for j in sorted_indices[:k])), sorted_indices


def ndcg_at_k(scores, labels, k, gt_label=1):
    top_k = np.argsort(scores)[::-1][:k]
    relevance = [1 if labels[j] == gt_label else 0 for j in top_k]
    ideal = sorted(relevance, reverse=True)
    return ndcg_score([ideal], [relevance]) if any(ideal) else 0.0



def batch_run_retrieval(query_data_chunk, args, tokenized_corpus, corpus_values, corpus_labels):
    bm25 = BM25Okapi(tokenized_corpus)
    cur_chunk_out_data = []
    for query, extended_labels, out_data_structure in tqdm(query_data_chunk, 
                                                           desc=f'Process {mp.current_process().name.split("-")[-1]}'):
        scores = bm25.get_scores(query.split(" "))
        all_labels = corpus_labels + extended_labels
        metrics = {}
        for k in [1, 3, 5, 10, 50, 100]:
            metrics[f'recall@{k}'], sorted_indices = recall_at_k(scores, all_labels, k)
            if k == 1:
                metrics[f'ndcg@{k}'] = metrics[f'recall@{k}']
            else:
                metrics[f'ndcg@{k}'] = ndcg_at_k(scores, all_labels, k)

        # save 
        if args.remove_oracle_from_top_k_record:
            top_items_to_save = [corpus_values[idx] for idx in sorted_indices[:args.record_top_k+10] if all_labels[idx] == 0][:args.record_top_k]
        else:
            top_items_to_save = [corpus_values[idx] for idx in sorted_indices[:args.record_top_k]]
        out_data_structure['retrieval_result'] = {
            'metrics': metrics,
            'top_items': top_items_to_save
        }
        cur_chunk_out_data.append(out_data_structure)
        # print('Query:', query.split(" "))
        # print('Top-k indices:', sorted_indices[:args.record_top_k])
        # print('Metrics:',metrics)
        
    return cur_chunk_out_data


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
            elif args.k_type == 'question':
                corpus_keys.append(entry['question'])
            elif args.k_type == 'question_description':
                corpus_keys.append(entry['hint']['content']['applicable_problems'] if entry['hint']['content']['applicable_problems'] else entry['question'])
            elif args.k_type == 'first_thought_steps':
                corpus_keys.append(' '.join(entry['teacher_thoughts'].split()[:100]))  # just a simple heuristic
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
    if args.debug_run:
        query_data_raw = query_data_raw[:50]
        print(f'Debug run: truncating to {len(query_data_raw)} items')
    for query_entry in query_data_raw:
        # cleaning hack for livecode
        query_entry['question'] = query_entry['question'].split('\n\nProblem Statement:\n')[-1]

        # prepare value 
        corpus_values.append(query_entry['hint']['content']['hint'])
        
        # prepare key 
        if args.k_type == 'self':
            corpus_keys.append(corpus_values[-1])
        elif args.k_type == 'question':
            corpus_keys.append(query_entry['question'])
        elif args.k_type == 'question_description':
            corpus_keys.append(query_entry['hint']['content']['applicable_problems'] if query_entry['hint']['content']['applicable_problems'] else query_entry['question'])
        elif args.k_type == 'first_thought_steps':
            corpus_keys.append(' '.join(query_entry['teacher_thoughts'].split()[:100]))  # just a simple heuristic
        else:
            raise NotImplementedError    

        # prepare label
        cur_chunk_extended_labels = [0 if x['question'] != query_entry['question'] else 1 for x in query_data_raw]
        assert sum(cur_chunk_extended_labels) == 1 and len(cur_chunk_extended_labels) == len(query_data_raw)

        # prepare query
        if args.q_type == 'question':
            cur_query = query_entry['question']
        elif args.q_type == 'question_description':
            cur_query = query_entry['hint']['content']['applicable_problems'] if query_entry['hint']['content']['applicable_problems'] else query_entry['question']
        elif args.q_type == 'first_thought_steps':
            cur_query = ' '.join(query_entry['teacher_thoughts'].split()[:100])  # just a simple heuristic
        elif args.q_type == 'oracle_hint':
            cur_query = query_entry['hint']['content']['hint']
        else:
            raise NotImplementedError 
        
        # other data to log
        cur_out_data_structure = {
            'question': query_entry['question'],
            'hint': query_entry['hint']
        }
        query_data.append([cur_query, cur_chunk_extended_labels, cur_out_data_structure])

    print(f'Corpus building complete: {len(corpus_keys)} items from {len(corpus_files)} files.')

    # ##########################################################################
    # # run retrieval (sequential)
    # ##########################################################################
    # tokenized_corpus = [doc.split(" ") for doc in corpus_values]
    # bm25 = BM25Okapi(tokenized_corpus)
    # out_data = []
    # all_metrics = []
    # for query, extended_labels, out_data_structure in tqdm(query_data, desc='Running retrieval'):
    #     scores = bm25.get_scores(query.split(" "))
    #     all_labels = corpus_labels + extended_labels
    #     # calculate metrics
    #     metrics = {}
    #     for k in [5, 10, 50, 100]:
    #         metrics[f'recall@{k}'], sorted_indices = recall_at_k(scores, all_labels, k)
    #         metrics[f'ndcg@{k}'] = ndcg_at_k(scores, all_labels, k)

    #     out_data_structure['retrieval_result'] = {
    #         'metrics': metrics,
    #         'top_items': [corpus_values[idx] for idx in sorted_indices[:args.record_top_k]]
    #     }
    #     all_metrics.append(metrics)
    #     out_data.append(out_data_structure)

    ##########################################################################
    # run retrieval (parallel)
    ##########################################################################
    # chunking data
    num_processes = args.n_workers
    query_data_chunked = []
    chunk_size = len(query_data) // num_processes
    remainder = len(query_data) % num_processes
    start = 0
    for i in range(num_processes):
        end = start + chunk_size + (1 if i < remainder else 0)
        query_data_chunked.append(query_data[start:end])
        start = end

    # workload
    tokenized_corpus = [[x for x in doc.split(" ") if x.strip() != ''] for doc in corpus_keys]
    
    # init workers and split work
    print('Setting num processes = {} with retriever {}'.format(num_processes, args.retriever))
    mp.set_start_method('spawn')
    pool = mp.Pool(num_processes)
    worker = partial(batch_run_retrieval, args=args, tokenized_corpus=tokenized_corpus,
                     corpus_values=corpus_values, corpus_labels=corpus_labels)

    # collect work and wrap up
    out_data = []
    for d in pool.imap_unordered(worker, query_data_chunked):
        out_data += d
    pool.close()

    ##########################################################################
    # print scores and write retrieval results to file
    ##########################################################################
    print('Retrieval metrics:')
    for k in out_data[0]['retrieval_result']['metrics']:
        print(f'\t{k}: {round(np.mean([x["retrieval_result"]["metrics"][k] for x in out_data]), 4)}')

    json.dump(out_data, open(args.out_file, 'w'), indent=4)
    print('Saved to', args.out_file)


if __name__ == '__main__':
    main()