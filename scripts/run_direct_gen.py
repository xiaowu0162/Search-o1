import json
import torch
import os, time
from tqdm import tqdm
from openai import OpenAI
from generation_utils import run_generate_with_backoff, OPENAI_REQUEST_TIMEOUT
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from evaluate import run_evaluation
from prompts import (
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run direct generation for various datasets and models.")
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True, 
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle', 'medmcqa', 'pubhealth'],
        help="Name of the dataset to use."
    )
    parser.add_argument(
        '--split', 
        type=str, 
        required=True, 
        choices=['test', 'diamond', 'main', 'extended', 'test_first500', 'dev_first500', 'test_1to4', 'test_1to6'],
        help="Dataset split to use."
    )
    parser.add_argument(
        '--subset_num', 
        type=int, 
        default=-1, 
        help="Number of examples to process. Defaults to all if not specified."
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help="Path to the pre-trained model."
    )
    # API-based inference backend instead of create an vllm model inside the script
    parser.add_argument(
        '--use_openai_inference',
        action='store_true' 
    )
    parser.add_argument(
        '--openai_server_base', 
        type=str,
        default=None,
    )
    parser.add_argument(
        '--openai_organization', 
        type=str,
        default=None,
    )
    parser.add_argument(
        '--openai_api_key', 
        type=str,
        default=None,
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.7, 
        help="Sampling temperature."
    )
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.8, 
        help="Top-p sampling parameter."
    )
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=20, 
        help="Top-k sampling parameter."
    )
    parser.add_argument(
        '--repetition_penalty', 
        type=float, 
        default=None, 
        help="Repetition penalty. If not set, defaults based on the model."
    )
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=32768, 
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )
    # For explorations
    parser.add_argument(
        '--self_segment_reasoning', 
        action='store_true', 
        help="Self-segment the reasoning trajectory."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens

    # exploration
    if args.self_segment_reasoning:
        assert 'qwq' in model_path.lower()
    
    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower() else 1.0
    
    # Paths to datasets
    if dataset_name == 'math500':
        data_path = f'./data/MATH500/{split}.json'
    elif dataset_name == 'gpqa':
        data_path = f'./data/GPQA/{split}.json'
    elif dataset_name == 'aime':
        data_path = f'./data/AIME/{split}.json'
    elif dataset_name == 'amc':
        data_path = f'./data/AMC/{split}.json'
    elif dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
        data_path = f'./data/QA_Datasets/{dataset_name}_{split}.json'
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    if 'qwq' in model_path.lower():
        model_short_name = 'qwq'
    elif 'deepseek' in model_path.lower():
        if 'llama-8b' in model_path.lower():
            model_short_name = 'ds-llama-8b'
        elif 'qwen-7b' in model_path.lower():
            model_short_name = 'ds-qwen-7b'
        elif 'qwen-32b' in model_path.lower():
            model_short_name = 'ds-qwen-32b'
    elif 'sky-t1' in model_path.lower():
        model_short_name = 'sky-t1'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')

    if model_short_name in ['qwq', 'ds-llama-8b', 'ds-qwen-7b', 'ds-qwen-32b', 'sky-t1']:
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.{model_short_name}.direct'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.{model_short_name}.direct'
    else:
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.direct'
    os.makedirs(output_dir, exist_ok=True)
    
    if args.use_openai_inference:
        if args.openai_server_base:
            # use local emulators such as vllm
            llm = OpenAI(base_url=args.openai_server_base, api_key="EMPTY", timeout=OPENAI_REQUEST_TIMEOUT)
        else:
            # use openai
            assert args.openai_api_key is not None
            if args.openai_organization:
                llm = OpenAI(api_key=args.openai_api_key, organization=args.openai_organization)
            else:
                llm = OpenAI(api_key=args.openai_api_key)
    else:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.95,
        )
    
    # Load data
    with open(data_path, mode='r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)
    
    # prepare input
    input_list = []
    for item in filtered_data:
        question = item['Question']
        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq', self_segment_reasoning=args.self_segment_reasoning)
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif dataset_name in ['math500', 'aime', 'amc']:
            if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq', self_segment_reasoning=args.self_segment_reasoning)
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name in ['gpqa']:
            if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq', self_segment_reasoning=args.self_segment_reasoning)
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)
            
        elif dataset_name == 'livecode':
            question_title = item.get('question_title', '')
            if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq', self_segment_reasoning=args.self_segment_reasoning)
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            user_prompt = ""  # Default to empty if dataset not matched
        if args.self_segment_reasoning:
            prompt = [
                # {"role": "system", "content": "When you are thinking, i.e., between <think> </think>, after you finish each thinking step (generally a few sentences), you must write a mark <break> before you proceed. This is crucial and helps us process your response. Do not write <break> outside thinking but instead write them between thinking start (<think>) and thinking end (</think>)"},
                # {"role": "system", "content": "Within your internal thinking process, i.e., between <think> </think>, separate each thinking step with a special mark ###. This is crucial and helps us process your response. Note that this format requirement is for your internal thinking after <think> <think>. not in the response."},
                # 
                {"role": "system", "content": "Within your internal thinking process, i.e., between <think> </think>, separate each thinking steps with a special mark <step>. This is crucial and helps us process your response. Note that this format requirement is for your internal thinking after <think> <think>, not in the response. <think> This is the format I should follow. <step> Step 1 ... step 1 ends. <step> Step 2 ... step 2 ends. <step> Step 3 ... step 3 ends.</think>"},
                # {"role": "system", "content": 'Within your internal thinking process, i.e., between <think> </think>, separate each thinking step with three newlines instead of two, i.e., when you want to write "\n\n", write "\n\n\n" instead. This is crucial and helps us process your response. Note that this format requirement is for your internal thinking between <think> <think>, not in the response. \n\n'},
                {"role": "user", "content": user_prompt}
            ]
        else:
            prompt = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)
    
    if subset_num != -1:
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]
    
    # Set default max_tokens if not provided
    if max_tokens is None:
        if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
            if dataset_name in ['aime', 'amc', 'livecode']:
                max_tokens = 32768
            else:
                max_tokens = 25600
        else:
            max_tokens = 3096
    
    t_start = time.time()
    # Generate model outputs
    if args.use_openai_inference:
        # batch inference with local server
        bsz = len(input_list)   # 40
        pbar = tqdm(total=len(input_list))
        output_list = []
        for i_b in range(0, len(input_list), bsz):
            prompt_batch = input_list[i_b:i_b+bsz]
            batch_outputs = run_generate_with_backoff(
                llm,
                model=model_path,
                prompt=prompt_batch,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=OPENAI_REQUEST_TIMEOUT,
                extra_body={
                    'top_k': top_k,
                    'repetition_penalty': repetition_penalty
                }
            )
            output_list.extend(batch_outputs.choices)
            pbar.update(len(prompt_batch))
        pbar.close() 
    else:
        output_list = llm.generate(
            input_list, 
            sampling_params=SamplingParams(
                max_tokens=max_tokens, 
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k, 
                repetition_penalty=repetition_penalty,
            )
        )
    total_time = time.time() - t_start
    
    # Run evaluation
    run_evaluation(
        filtered_data, 
        input_list, 
        output_list, 
        dataset_name, 
        output_dir, 
        total_time, 
        split,
    )


if __name__ == "__main__":
    main()
