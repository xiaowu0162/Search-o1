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


def truncate_text(x, tokenizer, truncate_length):
    return tokenizer.decode(tokenizer.encode(x, add_special_tokens=False)[:truncate_length])


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
    parser.add_argument(
        '--self_segment_reasoning_hijacking', 
        action='store_true', 
        help="Self-segment the reasoning trajectory via thought hijacking."
    )
    # RAG with hint
    parser.add_argument(
        '--retrieved_hint_file', 
        type=str,
        required=True,
        help="Retrieved hint file."
    )
    parser.add_argument(
        '--retrieval_exp_name', 
        type=str,
        required=True,
        help="Identifier for retrieval experiment that generates the hint file."
    )
    parser.add_argument(
        '--augmentation_strategy', 
        type=str,
        required=True,
        choices=['direct-in-thought', 'summ-then-in-thought']
    )
    parser.add_argument(
        '--keep_hint_topk', 
        type=int,
        required=True,
    )
    return parser.parse_args()


def format_question_in_prompt(args, item, model_path, dataset_name, question):
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
        raise NotImplementedError
        # user_prompt = ""  # Default to empty if dataset not matched
    return user_prompt


def build_hint_summarization_prompt(question, retrieved_top_k_hints, tokenizer):
    # prompt = f'You are given a question and a few problem solving templates that might be relevant. Carefully think about the given question and then write a hint for the question. Your hint should be brief. Highlight the important steps and overall thinking direction instead of giving out the final answer. You may use information from the templates to help you if there are relevant. If the templates are not helpful, you do not need to use them. Start your hint with ### Hint.'
    # prompt = f'You are an expert tutor. You are given a question and a few problem solving templates that might be relevant. Write a high-level hint for how to approach the question. Highlight the important steps or thinking directions. Write the general insight and do not go into the details of solving the problem. Do not write the final answer. You may borrow information from the templates if they are helpful. Start your hint with ### Hint.\n\n\n### Question to solve:\n{question}'
    # prompt = f'You are an expert tutor. You are given a question and a few problem solving templates that might be relevant. Write a high-level plan for how to approach the question. Highlight the important steps or thinking directions. Write the general insight and do not go into the details of solving the problem. Do not write the final answer. You may borrow information from the templates if they are helpful. Start your plan with ### Plan.\n\n\n### Question to solve:\n{question}'
    # prompt = f'You are an expert tutor. You are given a question and a few problem solving templates that might be relevant. Combine the templates to write a brief hint for the question. Do not give out the final answer. Start your hint with ### Hint.\n\n\n### Question that you need to write hint for:\n{question}'
    # prompt = f'You are an expert tutor. You are given a question and a few problem solving templates that might be relevant. Based on your knowledge, rewrite the templates so that it can help a student approach the problem. The template should be high-level and does not contain specific solutions. Start your rewrite with ### Template.\n\n\n### Question that you need to write template for:\n{question}'
    prompt = f'Write a solution sketch for this problem. Use the provided solution templates if they are helpful. The solution sketch should be high-level and very short.\n\n\n### Question that you need to write solution sketch for:\n{question}'
    for i_hint, hint_text in enumerate(retrieved_top_k_hints):
        truncated_hint_text = tokenizer.decode(tokenizer.encode(hint_text, add_special_tokens=False)[:1000])
        prompt += f'\n\n\n### Template {i_hint+1}\n{truncated_hint_text}'
    # prompt += f'\n\n\nNow, write the high-level plan for approaching the given question.'
    return prompt


def process_question_instruct_add_hint(task, instruction, hint_augmentation_mode):
    if hint_augmentation_mode in ['direct-in-thought', 'summ-then-in-thought']:
        if task in ['gpqa']:
            instruction_new = instruction.replace('Please answer the following multiple-choice question.', 'Please answer the following multiple-choice question. Hints might be provided during your question answering wrapped within [hint] and [end of hint]. If you see hints, try to leverage them to guide your thinking process.')
        elif task in ['aime', 'amc', 'math500']:
            instruction_new = instruction.replace('Please answer the following math question.', 'Please answer the following math question. Hints might be provided during your question answering wrapped within [hint] and [end of hint]. If you see hints, try to leverage them to guide your thinking process.')
        elif task in ['livecode']:
            instruction_new = instruction.replace('Generate a correct Python program that passes all tests for the given problem.', 'Generate a correct Python program that passes all tests for the given problem. Hints might be provided during your question answering wrapped within [hint] and [end of hint]. If you see hints, try to leverage them to guide your thinking process.')
        elif task in ['bamboogle']:
            instruction_new = instruction.replace('Please answer the following question.', 'Please answer the following question. Hints might be provided during your question answering wrapped within [hint] and [end of hint]. If you see hints, try to leverage them to guide your thinking process.')
        else:
            raise NotImplementedError
        assert instruction_new != instruction
    else:
        raise NotImplementedError
    return instruction_new


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
    if args.self_segment_reasoning or args.self_segment_reasoning_hijacking:
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
        elif 'llama-70b' in model_path.lower():
            model_short_name = 'ds-llama-70b'
        else:
            raise NotImplementedError
    elif 'sky-t1' in model_path.lower():
        model_short_name = 'sky-t1'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')

    # Set default max_tokens if not provided
    max_hint_length = 2000
    if max_tokens is None:
        if 'qwq' in model_path.lower() or 'deepseek' in model_path.lower() or 'sky-t1' in model_path.lower():
            if dataset_name in ['aime', 'amc', 'livecode']:
                max_tokens = 32768
            else:
                max_tokens = 25600
        else:
            max_tokens = 3096
    if 'deepseek' in model_path.lower() and ('llama' in model_path.lower() or 'qwen' in model_path.lower()):
        max_tokens = 11000

    # if model_short_name in ['qwq', 'ds-llama-8b', 'ds-qwen-7b', 'ds-qwen-32b', 'sky-t1']:
    #     if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
    #         output_dir = f'./outputs/{dataset_name}.{model_short_name}.raghint/{args.retrieval_exp_name}/'
    #     else:
    #         output_dir = f'./outputs/runs.qa/{dataset_name}.{model_short_name}.raghint/{args.retrieval_exp_name}/'
    # else:
    output_dir = f'./outputs/runs.raghint/{args.augmentation_strategy}_{args.keep_hint_topk}hints/{args.retrieval_exp_name}/{dataset_name}.{model_short_name}/'
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
        if subset_num != -1:
            filtered_data = filtered_data[:subset_num]
            print(f'Warning: only using the first {args.subset_num} data points.')
    
    # load hints
    try:
        retrieved_hints = [json.loads(line) for line in open(args.retrieved_hint_file).readlines()]
    except:
        retrieved_hints = json.load(open(args.retrieved_hint_file))
    if subset_num == -1:
        assert len(filtered_data) == len(retrieved_hints)
    question_to_retrieved_hints = {x['question']: x['retrieval_result']['top_items'] for x in retrieved_hints}
    question_to_retrieved_hints = {k.split('Generate a correct Python program that passes all tests for the given problem. You should provide your final code within a Python code block using triple backticks (```python\nYOUR_CODE\n```).')[-1].lower().strip(): v for k, v in question_to_retrieved_hints.items()}  # patch for livecode

    # optionally, further process the hints before augmentation
    question_to_summarized_hints = {}
    if args.augmentation_strategy == 'summ-then-in-thought':
        summ_workload = []
        for item in filtered_data:
            question = item['Question']
            user_prompt = format_question_in_prompt(args, item, model_path, dataset_name, question)
            extracted_q_lower = user_prompt.split('\n\nQuestion:\n')[-1].lower()
            extracted_q_lower = extracted_q_lower.split('Generate a correct Python program that passes all tests for the given problem. You should provide your final code within a Python code block using triple backticks (```python\nYOUR_CODE\n```).'.lower())[-1].lower().strip()  # for livecode
            extracted_q_lower = extracted_q_lower.split('\n\nproblem statement:\n')[-1].lower().strip()  # for livecode
            retrieved_top_k_hints = question_to_retrieved_hints[extracted_q_lower][:args.keep_hint_topk]
            hint_summarization_prompt = build_hint_summarization_prompt(question, retrieved_top_k_hints, tokenizer)
            hint_summarization_prompt = tokenizer.apply_chat_template([{"role": "user", "content": hint_summarization_prompt}], tokenize=False, add_generation_prompt=True)   # <think> added here


            # maybe disable thinking for the hint processing model?
            hint_summarization_prompt = hint_summarization_prompt.replace('<think>', '### Brief Solution Sketch\n\n')


            # hint_summarization_prompt += 'Okay, so my task is to write a high-level hint for the question without giving out the answer. Hmm'
            summ_workload.append([extracted_q_lower, retrieved_top_k_hints, hint_summarization_prompt])

        hints_summ_outputs = run_generate_with_backoff(llm, model=model_path, prompt=[x[-1] for x in summ_workload], 
                                                       max_tokens=3000, temperature=temperature, top_p=top_p, timeout=OPENAI_REQUEST_TIMEOUT,
                                                       extra_body={
                                                           'top_k': top_k,
                                                           'repetition_penalty': repetition_penalty})
        for i_item, summ_response in enumerate(hints_summ_outputs.choices):
            extracted_q_lower, retrieved_top_k_hints, hint_summarization_prompt = summ_workload[i_item]
            hint_summ_output = summ_response.text
            if '</think>' in hint_summ_output:
                hint_summarization_thinking_process = hint_summ_output.split('</think>')[0].strip()
                hint_summary = hint_summ_output.split('</think>')[-1].strip()
            else:
                hint_summarization_thinking_process = ''
                hint_summary = hint_summ_output.strip()
            hint_summary = hint_summary.split('### Plan')[-1].split('### Hint')[-1].strip().split('### Template')[-1].strip()
            hint_summary = hint_summary.split('### Answer')[0].split('### Final Answer')[0].split('**Answer**')[0].split('**Final Answer**')[0]
            hint_summary = hint_summary.replace('<think>', '').strip()
            hint_summary = tokenizer.decode(tokenizer.encode(hint_summary, add_special_tokens=False)[:300]) # truncate
            # print(json.dumps({
            #     'question': extracted_q_lower,
            #     # 'hints_to_summarize': retrieved_top_k_hints,
            #     # 'hint_summarization_thinking_process': hint_summarization_thinking_process,
            #     'hint_summary': hint_summary,
            # }, indent=4))
            # print('\n\n=============================================\n\n')
            question_to_summarized_hints[extracted_q_lower] = hint_summary

    # prepare final inference input
    input_list = []
    for item in filtered_data:
        question = item['Question']
        user_prompt = format_question_in_prompt(args, item, model_path, dataset_name, question)
        
        # Injecting hints part 1 (change instruction)
        user_prompt_processed = process_question_instruct_add_hint(dataset_name, user_prompt, hint_augmentation_mode=args.augmentation_strategy)
        
        if args.self_segment_reasoning:
            prompt = [
                # {"role": "system", "content": "When you are thinking, i.e., between <think> </think>, after you finish each thinking step (generally a few sentences), you must write a mark <break> before you proceed. This is crucial and helps us process your response. Do not write <break> outside thinking but instead write them between thinking start (<think>) and thinking end (</think>)"},
                # {"role": "system", "content": "Within your internal thinking process, i.e., between <think> </think>, separate each thinking step with a special mark ###. This is crucial and helps us process your response. Note that this format requirement is for your internal thinking after <think> <think>. not in the response."},
                # 
                {"role": "system", "content": "Within your internal thinking process, i.e., between <think> </think>, separate each thinking steps with a special mark <step>. This is crucial and helps us process your response. Note that this format requirement is for your internal thinking after <think> <think>, not in the response. <think> This is the format I should follow. <step> Step 1 ... step 1 ends. <step> Step 2 ... step 2 ends. <step> Step 3 ... step 3 ends.</think>"},
                # {"role": "system", "content": 'Within your internal thinking process, i.e., between <think> </think>, separate each thinking step with three newlines instead of two, i.e., when you want to write "\n\n", write "\n\n\n" instead. This is crucial and helps us process your response. Note that this format requirement is for your internal thinking between <think> <think>, not in the response. \n\n'},
                {"role": "user", "content": user_prompt_processed}
            ]
        else:
            prompt = [{"role": "user", "content": user_prompt_processed}]

        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        # Injecting hints part 2 (thought hijacking with hint)
        extracted_q_lower = user_prompt.split('\n\nQuestion:\n')[-1].lower()
        extracted_q_lower = extracted_q_lower.split('Generate a correct Python program that passes all tests for the given problem. You should provide your final code within a Python code block using triple backticks (```python\nYOUR_CODE\n```).'.lower())[-1].lower().strip()  # for livecode
        extracted_q_lower = extracted_q_lower.split('\n\nproblem statement:\n')[-1].lower().strip()  # for livecode
        if args.augmentation_strategy == 'direct-in-thought':
            retrieved_hints = question_to_retrieved_hints[extracted_q_lower]
            if args.keep_hint_topk == 1:
                hint_formatted = retrieved_hints[0]
            else:
                hint_formatted = retrieved_hints[0]
                for i_hint, hint_text in enumerate(retrieved_hints[1:args.keep_hint_topk]):
                    truncated_hint_text = tokenizer.decode(tokenizer.encode(hint_text, add_special_tokens=False)[:500])   # truncate to a reasonable length...
                    hint_formatted += f'\n\nAn alternative approach to explore. {truncated_hint_text}'
            hint_formatted_truncated = truncate_text(hint_formatted, tokenizer, max_hint_length)
            hint_str = f'[hint] {hint_formatted_truncated} [end of hint]\n\nOkay,'
        elif args.augmentation_strategy == 'summ-then-in-thought':
            hint_formatted = question_to_summarized_hints[extracted_q_lower]
            hint_formatted_truncated = truncate_text(hint_formatted, tokenizer, max_hint_length)
            hint_str = f'[hint] {hint_formatted_truncated} [end of hint]\n\nOkay,'
        else:
            raise NotImplementedError
        prompt = prompt + ' ' + hint_str

        if args.self_segment_reasoning_hijacking:
            # add hijacking thoughts
            prompt += 'Okay, before I start working on the question, I should make sure that I separate each step in my thinking with a step token. <step> Okay'

        input_list.append(prompt)
    
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
