import re
import os
import sys
import json
import string
from tqdm import tqdm
import torch
import numpy as np
from openai import OpenAI
from collections import Counter
import backoff
from transformers import AutoTokenizer

OPENAI_REQUEST_TIMEOUT = 60 * 60 * 24
BATCH_SIZE = 1024  # batch size for each OpenAI reaquest


@backoff.on_exception(backoff.constant, Exception, interval=5)
def run_generate_with_backoff(client, **kwargs):
    """Retry wrapper around the completion call."""
    return client.completions.create(**kwargs)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------



def build_hint_prompt(question: str, teacher_answer: str, teacher_thought_str: str, tokenizer) -> str:
    """Compose the single prompt string that will be sent to the model."""

    if len(tokenizer.encode(teacher_thought_str)) > 20000:
        teacher_thought_str = tokenizer.decode(tokenizer.encode(teacher_thought_str, add_special_tokens=False)[:20000])
    prompt = (
        "You are an expert tutor. Given a question, a final answer written by the teacher, "
        "and a long thinking process written by a teacher, write a brief hint that can help "
        "yourself approach similar questions without revealing the answer or any intermediate "
        "results. The hint should outline the steps to solve these general questions and the "
        "general strategy. The hint should also highlight key points in thinking to expedite "
        "problem solving and avoid common traps. Utilize and pay special attention to the places "
        "where the teacher also gets confused or spends too much time. Start by outlining the "
        "general problem under a section ### Applicable Problems . Then, start your hint on a new line by ### Hint ."
        "Inside the hint, you must first re-state the general problem setting that the hint can apply to. "
        "Then, use a first person perspective just like you are the student, e.g., say something like 'For problems like X, I should...'."
        "\n\n\n### Question:\n{question}\n\n\n### Teacher's Answer:\n{teacher_answer}\n\n\n### Teacher's Thinking:\n{teacher_thought_str}"
        "\n\n\nNow, analyze the question, answer, and teacher's thought and write your hint. Make sure your hint helps "
        "approach similar questions without revealing the answer or any intermediate results. "
    ).format(question=question, teacher_answer=teacher_answer, teacher_thought_str=teacher_thought_str)

    return tokenizer.apply_chat_template([
        {"role": "user", "content": prompt}
    ], tokenize=False, add_generation_prompt=True)


def parse_hint_response(raw_text: str):
    """Extract structured data from the raw response text."""
    if '</think>' in raw_text:
        hint_derivation_thinking, hint_body = raw_text.split('</think>', 1)
        hint_derivation_thinking = hint_derivation_thinking.strip()
        hint_body = hint_body.strip()
    else:
        hint_derivation_thinking = ''
        hint_body = raw_text.strip()

    result = {
        'hint_derivation_thinking': hint_derivation_thinking,
        'hint': '',
        'applicable_problems': ''
    }

    if '### Hint' in hint_body:
        before, _, after = hint_body.partition('### Hint')
        result['hint'] = after.strip()
        result['applicable_problems'] = before.replace('### Applicable Problems', '').strip()
    else:
        result['hint'] = hint_body
    return result


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    shard = sys.argv[1]
    port = sys.argv[2] # 8100

    teacher_model_name = 'Qwen/QwQ-32B'

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY", timeout=OPENAI_REQUEST_TIMEOUT)
    print('Started openai client')
    print(client.models.list())

    # for shard in ['05']:
    in_file_name = f'/fsx-comem/diwu0162/OpenThoughts3/data/train_full/train_full_1.2m.jsonl.{shard}.part.qdedup.jsonl'
    logs = [json.loads(line) for line in open(in_file_name).readlines()]
    out_file_name = f'logs_hint_distillation_openthoughts_shard_{shard}.new.jsonl'   # .new just to prevent being overwritten by old runs

    with open(out_file_name, 'w') as out_f, tqdm(total=len(logs), desc="Generating hints") as pbar:
        total_count = 0
        # Iterate through the log entries in *batches* instead of one-by-one
        for start_idx in range(0, len(logs), BATCH_SIZE):
            batch_entries = logs[start_idx:start_idx + BATCH_SIZE]

            # ------------------------------------------------------------------
            # 1) Build prompts for the entire batch
            # ------------------------------------------------------------------
            prompts, batch_meta = [], []

            for entry in batch_entries:
                # Handle the two possible formats in the log file
                question_field = entry.get('Question', '')
                if question_field:
                    question = question_field.split('\n\nQuestion:\n')[-1]
                    question = question.replace('\n\n<|im_end|>\n<|im_start|>assistant\n<think>\n', '').strip()
                else:
                    question = entry['conversations'][0]['value']

                # print(question)
                # print(entry.keys())
                output_field = entry.get('Output', '')
                if not output_field:
                    output_field = entry['conversations'][1]['value']
                if '</think>' in output_field:
                    thoughts, answer = output_field.split('</think>', 1)
                    thoughts = thoughts.replace('<think>', '').strip()
                    answer = answer.strip()
                else:
                    thoughts = output_field.replace('<think>', '').strip()
                    answer = 'answer unknown due to thinking unfinished'

                prompts.append(build_hint_prompt(question, answer, thoughts, tokenizer))
                batch_meta.append({
                    'question': question,
                    'teacher_answer': answer,
                    'teacher_thoughts': thoughts,
                })

            # ------------------------------------------------------------------
            # 2) Issue *one* API call for the whole batch
            # ------------------------------------------------------------------
            response = run_generate_with_backoff(
                client,
                model=teacher_model_name,
                prompt=prompts,  # list[str]: batched prompts
                n=1,
                temperature=0.7,
                top_p=0.8,
                max_tokens=10000,
                timeout=OPENAI_REQUEST_TIMEOUT,
                extra_body={
                    'top_k': 20,
                    'include_stop_str_in_output': True,
                    'repetition_penalty': 1.05,
                }
            )

            # ------------------------------------------------------------------
            # 3) Post-process each individual completion
            # ------------------------------------------------------------------
            if len(response.choices) != len(batch_meta):
                raise ValueError("Mismatch between request and response length.")

            for meta, choice in zip(batch_meta, response.choices):
                raw_hint_text = choice.text
                structured_hint = parse_hint_response(raw_hint_text)

                out_entry = {
                    'question': meta['question'],
                    'teacher_answer': meta['teacher_answer'],
                    'teacher_thoughts': meta['teacher_thoughts'],
                    'hint': {
                        'hint_model': teacher_model_name,
                        'hint_text_raw': raw_hint_text,
                        'content': structured_hint,
                    }
                }
                print(json.dumps(out_entry), file=out_f)
                total_count += 1
                if total_count % 1000 == 0:
                    # periodic flushing to reduce the load to the disk
                    out_f.flush()

            # Update progress bar by the number of processed examples
            pbar.update(len(batch_entries))

    print(f"All done! Batched hints written to {out_file_name}")
