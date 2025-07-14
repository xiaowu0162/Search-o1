#!/bin/bash                                                                                                                                                                                   

task=$1
q_type=$2       # 'question', 'question_description', 'first_thought_steps'
k_type=$3       # 'self', 'question', 'question_description', 'first_thought_steps'
retriever=bm25

out_dir='retrieval_logs'
out_file="${out_dir}/${task}_${retriever}_q=${q_type}_k=${k_type}_$(date +'%Y%m%d-%H%M').json"
mkdir -p $out_dir

python run_retrieval.py \
    --query_file "/fsx-comem/diwu0162/Search-o1/explorations/20250710_hint_eval_dataset/logs_hint_distillation_eval_task_${task}.jsonl" \
    --corpus_dir "/fsx-comem/diwu0162/Search-o1/explorations/20250710_openthoughts/" \
    --corpus_file_pattern 'logs_hint_distillation_openthoughts_shard_*jsonl' \
    --out_file $out_file \
    --retriever bm25 \
    --q_type $q_type \
    --k_type $k_type \
    --record_top_k 10 \
    --report_metrics

