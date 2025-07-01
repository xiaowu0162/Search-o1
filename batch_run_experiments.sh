#!/bin/bash

model_name="Qwen/QwQ-32B"
server_base="http://localhost:8001/v1"
subset=$1


##############################################################################
# Baseline 1 Direct Generation 
##############################################################################

if [ $subset == "1" ]; then
    for ds in amc aime math500; do 
        python scripts/run_direct_gen.py --dataset_name $ds --split test --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
    done
    python scripts/run_direct_gen.py --dataset_name bamboogle --split test --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
fi

if [ $subset == "2" ]; then
    python scripts/run_direct_gen.py --dataset_name livecode --split test_1to4 --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
    python scripts/run_direct_gen.py --dataset_name musique --split dev_first500 --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
fi 

if [ $subset == "3" ]; then
    for ds in nq triviaqa; do
        python scripts/run_direct_gen.py --dataset_name $ds --split test_first500 --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
    done
fi 

if [ $subset == "4" ]; then
    for ds in hotpotqa 2wiki; do
        python scripts/run_direct_gen.py --dataset_name $ds --split dev_first500 --model_path ${model_name} --use_openai_inference --openai_server_base ${server_base}
    done
fi 