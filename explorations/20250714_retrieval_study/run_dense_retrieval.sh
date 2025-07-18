#!/bin/bash                                                                                                                                                                                   

task=$1
q_type=$2       # 'question', 'question_description', 'first_thought_steps', 'oracle_hint'
k_type=$3       # 'self', 'question', 'question_description', 'first_thought_steps'
retriever=$4    # contriever
workers=$5      # 25

if [[ $retriever == "reasonir" || $retriever == "qwen8b" ]]; then
    maxlen=1024
else
    maxlen=500
fi


out_dir='retrieval_logs'
# out_file="${out_dir}/${task}_${retriever}_q=${q_type}_k=${k_type}_$(date +'%Y%m%d-%H%M').json"
out_file="${out_dir}/${task}_${retriever}_q=${q_type}_k=${k_type}.json"
mkdir -p $out_dir

python run_retrieval.py \
    --query_file "/fsx-comem/diwu0162/Search-o1/explorations/20250714_retrieval_study/embeddings/eval_datasets/logs_hint_distillation_eval_task_${task}.jsonl.embed.${retriever}.npz" \
    --corpus_dir "/fsx-comem/diwu0162/Search-o1/explorations/20250714_retrieval_study/embeddings/ot3_${retriever}_${maxlen}/" \
    --corpus_file_pattern "logs_hint_distillation_openthoughts_shard_*.npz" \
    --out_file $out_file \
    --retriever ${retriever} \
    --q_type $q_type \
    --k_type $k_type \
    --record_top_k 10 \
    --n_workers ${workers} \
    --remove_oracle_from_top_k_record --report_metrics  #     --debug_run


# batch run 
if [ $subset == "xxx" ]; then
    # export CUDA_VISIBLE_DEVICES=0
    # export task=bamboogle
    export n_workers=1
    for retriever in contriever reasonir qwen8b; do
        bash run_dense_retrieval.sh $task question first_thought_steps $retriever $n_workers
        bash run_dense_retrieval.sh $task question question $retriever $n_workers
        bash run_dense_retrieval.sh $task question_description question_description $retriever $n_workers
        bash run_dense_retrieval.sh $task question_description self $retriever $n_workers
    done
fi