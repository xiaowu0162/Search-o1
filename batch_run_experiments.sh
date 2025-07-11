#!/bin/bash

model_name=$1 # "Qwen/QwQ-32B"  deepseek-ai/DeepSeek-R1-Distill-Llama-8B  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
server_base=$2 # "http://localhost:8001/v1"
subset=$3

# dataset sizes
# amc - 40
# aime - 30
# math500 - 500
# livecode_1to4 - 112
# bamboogle - 125
# musique/nq/triviaqa/hotpotqa/2wiki - 500


##############################################################################
# Baseline 1 Direct Generation 
##############################################################################

if [ $subset == "0" ]; then
    python scripts/run_direct_gen.py --dataset_name amc --split test --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base} 
fi
if [ $subset == "1" ]; then
    python scripts/run_direct_gen.py --dataset_name aime --split test --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
fi
if [ $subset == "2" ]; then
    python scripts/run_direct_gen.py --dataset_name math500 --split test --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
fi 
if [ $subset == "3" ]; then
    python scripts/run_direct_gen.py --dataset_name livecode --split test_1to4 --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
fi 
if [ $subset == "4" ]; then
    python scripts/run_direct_gen.py --dataset_name gpqa --split diamond --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base} 
fi

# QA tasks (relateively fast to run)
if [ $subset == "5" ]; then
    python scripts/run_direct_gen.py --dataset_name bamboogle --split test --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
    for ds in nq triviaqa; do
        python scripts/run_direct_gen.py --dataset_name $ds --split test_first500 --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
    done
    for ds in musique hotpotqa 2wiki; do
        python scripts/run_direct_gen.py --dataset_name $ds --split dev_first500 --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
    done
fi 


##############################################################################
# Baseline 2 Vanilla RAG (retrieve once with the original question)
##############################################################################
if [ $subset == "xxx" ]; then
    python scripts/run_naive_rag.py --dataset_name gpqa --split diamond \
        --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base} \
        --serper_subscription_key_file 'serper_api_key.txt' \
        --use_jina True --jina_api_key 'jina_api_key.txt'     
fi


##############################################################################
# Baseline 3 RAG Agent (LRM writes query and checks full pages if needed)
##############################################################################

if [ $subset == "yyy" ]; then
    python scripts/run_rag_agent.py --dataset_name gpqa --split diamond \
        --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base} \
        --serper_subscription_key_file 'serper_api_key.txt' \
        --use_jina True --jina_api_key 'jina_api_key.txt'       # --subset_num 1
fi


##############################################################################
# Search-o1
##############################################################################

if [ $subset == "zzz" ]; then
    python scripts/run_search_o1.py --dataset_name gpqa --split diamond \
        --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base} \
        --serper_subscription_key_file 'serper_api_key.txt' \
        --use_jina True --jina_api_key 'jina_api_key.txt'       # --subset_num 1
fi