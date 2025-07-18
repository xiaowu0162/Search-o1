#!/usr/bin/env python3
"""
parallel_encode.py

Encode a JSONL corpus on multiple GPUs.

Usage
-----
python embed_corpus.py \
    --input  corpus.jsonl \
    --output corpus_encoded.jsonl \
    --model  sentence-transformers/all-MiniLM-L6-v2 \
    --batch-size 256

Assumptions
-----------
* Each line of `input` is a JSON object and **must contain**:
    - `"id"`   : unique identifier (becomes `entry_name` in the output)
    - `"text"` : string to encode
  If your file has different field names or you need to combine multiple
  fields into one string, adapt `_extract_text`.
* Script is run on a node with ≥ 1 CUDA‑visible GPU.
"""

import argparse, json, os, sys, math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from torch.multiprocessing import Process, Queue, set_start_method


##############################################################################
# Helper: adapt here if your JSON schema differs
##############################################################################
def _extract_text(line_id, js_line):
    """
    Returns a 5‑tuple:
        (entry_name, question, question_description, thought, hint)

    Expected JSON keys per line:
        id, question, question_description, thought, hint
    (Rename below if your keys differ.)
    """
    try:
        return (
            str(line_id),
            js_line["question"],
            js_line["hint"]['content']['applicable_problems'],
            js_line["teacher_thoughts"],
            js_line["hint"]['content']['hint'],
        )
    except KeyError as e:
        raise KeyError(f"Missing field: {e.args[0]}")


##############################################################################
# Worker
##############################################################################
def _truncate_text(x, tokenizer, truncate_length):
    return tokenizer.decode(tokenizer.encode(x, add_special_tokens=False)[:truncate_length])


def _encode_worker(
    gpu_id: int,
    model_name: str,
    batch_size: int,
    shard,
    out_q,
):
    torch.cuda.set_device(gpu_id)
    model = SentenceTransformer(model_name, device=f"cuda:{gpu_id}", 
                                trust_remote_code=True, model_kwargs={"torch_dtype": "auto"})
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # unpack shard
    if shard:
        (names,
         q_texts,
         qd_texts,
         t_texts,
         h_texts) = zip(*shard)
    else:
        names = q_texts = qd_texts = t_texts = h_texts = []

    # TODO: model-specific handlings
    if model_name in ['facebook/contriever']:
        max_length = 500
    elif model_name in ['Qwen/Qwen3-Embedding-8B', 'reasonir/ReasonIR-8B']:
        max_length = 1024
    else:
        raise NotImplementedError
    q_texts_trunc = [_truncate_text(x, tokenizer, max_length) for x in q_texts]
    qd_texts_trunc = [_truncate_text(x, tokenizer, max_length) for x in qd_texts]
    t_texts_trunc = [_truncate_text(x, tokenizer, max_length) for x in t_texts]
    h_texts_trunc = [_truncate_text(x, tokenizer, max_length) for x in h_texts]
    print(f'GPU {gpu_id} step 0 truncations done.', flush=True)

    # four independent encoding passes
    # if gpu_id == 0:
    #     print('[INFO] Encoding step 1: questions')
    q_embs  = model.encode(q_texts_trunc, batch_size=batch_size,
                           show_progress_bar=(gpu_id == 0),
                           convert_to_numpy=True).astype(np.float32)
    print(f'[INFO] GPU {gpu_id} step 1/4 done.', flush=True)
    
    # if gpu_id == 0:
    #     print('[INFO] Encoding step 2: question descriptions')
    qd_embs = model.encode(qd_texts_trunc, batch_size=batch_size,
                           show_progress_bar=False,
                           convert_to_numpy=True).astype(np.float32)
    print(f'GPU {gpu_id} step 2/4 done.', flush=True)
    
    # if gpu_id == 0:
    #     print('[INFO] Encoding step 3: start part of thoughts')
    t_embs  = model.encode(t_texts_trunc, batch_size=batch_size,
                           show_progress_bar=False,
                           convert_to_numpy=True).astype(np.float32)
    print(f'[INFO] GPU {gpu_id} step 3/4 done.', flush=True)
    
    # if gpu_id == 0:
    #     print('[INFO] Encoding step 4: hints')
    h_embs  = model.encode(h_texts_trunc, batch_size=batch_size,
                           show_progress_bar=False,
                           convert_to_numpy=True).astype(np.float32)
    print(f'[INFO] GPU {gpu_id} step 4/4 done.', flush=True)

    # enqueue one item per entry
    for i, name in enumerate(names):
        out_q.put((
            name,
            q_texts[i],  qd_texts[i], t_texts[i], h_texts[i],
            q_embs[i].tolist(), qd_embs[i].tolist(),
            t_embs[i].tolist(),  h_embs[i].tolist()
        ))


##############################################################################
# Main
##############################################################################
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi‑GPU SentenceTransformer encoder")
    p.add_argument("--input", required=True, type=Path,)
    p.add_argument("--output", required=True, type=Path,)
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass  # already set in interactive environments

    args = _parse_args()

    # --------------------------------------------------------------------- #
    # Load corpus into memory (feel free to stream if very large)
    # --------------------------------------------------------------------- #
    try:
        in_data = [json.loads(line) for line in open(args.input).readlines()]
    except:
        in_data = json.load(open(args.input))

    if not in_data:
        print("No valid lines found in input.", file=sys.stderr)
        sys.exit(1)

    corpus = []
    for i, entry in enumerate(tqdm(in_data, desc='Preparing data')):
        corpus.append(_extract_text(i, entry))

    # --------------------------------------------------------------------- #
    # Shard corpus evenly across GPUs
    # --------------------------------------------------------------------- #
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA devices visible!")

    shard_size = math.ceil(len(corpus) / n_gpus)
    shards = [
        corpus[i * shard_size : (i + 1) * shard_size] for i in range(n_gpus)
    ]

    # --------------------------------------------------------------------- #
    # Spawn workers
    # --------------------------------------------------------------------- #
    out_q: Queue = Queue()
    procs: List[Process] = []

    for gpu_id, shard in enumerate(shards):
        if not shard:  # handle fewer lines than GPUs
            continue
        p = Process(
            target=_encode_worker,
            args=(gpu_id, args.model, args.batch_size, shard, out_q),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # --------------------------------------------------------------------- #
    # Gather results and write to NPZ  (4 fields, 4 embeddings)
    # --------------------------------------------------------------------- #
    names = []
    q_texts, qd_texts, t_texts, h_texts = [], [], [], []
    q_embs,  qd_embs,  t_embs,  h_embs  = [], [], [], []

    finished      = 0
    total_records = len(corpus)

    while finished < total_records:
        (name,
        qtxt, qdtxt, ttxt, htxt,
        qemb, qdemb, temb, hemb) = out_q.get()

        names.append(name)
        q_texts.append(qtxt);  qd_texts.append(qdtxt)
        t_texts.append(ttxt);  h_texts.append(htxt)

        q_embs.append(qemb);   qd_embs.append(qdemb)
        t_embs.append(temb);   h_embs.append(hemb)

        finished += 1
        if finished % 5000 == 0 or finished == total_records:
            print(f"[INFO] {finished}/{total_records} encoded → buffer", file=sys.stderr)

    # convert to ndarray / object arrays
    entry_name_arr  = np.array(names,     dtype=object)
    q_text_arr      = np.array(q_texts,   dtype=object)
    qd_text_arr     = np.array(qd_texts,  dtype=object)
    t_text_arr      = np.array(t_texts,   dtype=object)
    h_text_arr      = np.array(h_texts,   dtype=object)

    q_emb_arr  = np.asarray(q_embs,  dtype=np.float32)
    qd_emb_arr = np.asarray(qd_embs, dtype=np.float32)
    t_emb_arr  = np.asarray(t_embs,  dtype=np.float32)
    h_emb_arr  = np.asarray(h_embs,  dtype=np.float32)

    out_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        out_path,
        entry_name=entry_name_arr,
        question=q_text_arr,
        question_description=qd_text_arr,
        thought=t_text_arr,
        hint=h_text_arr,
        question_embedding=q_emb_arr,
        question_description_embedding=qd_emb_arr,
        thought_embedding=t_emb_arr,
        hint_embedding=h_emb_arr,
    )
    print(f"[INFO] NPZ written to {out_path}", file=sys.stderr)


    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
