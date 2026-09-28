#!/usr/bin/env python3
"""Encode a .jsonl file (field "text") into a uint16 token shard (.npy).

Memory-efficient: documents are encoded in batches and stored as compact
numpy chunks (2 bytes/token) instead of a giant Python list, so hundreds of
millions of tokens fit comfortably in RAM."""
import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


def load_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def encode_file(input_path, tokenizer_path, out_path, batch_docs=2000, add_eos=True):
    sp = load_tokenizer(tokenizer_path)
    assert sp.get_piece_size() < 65536, "vocab must fit in uint16"
    eos_id = sp.eos_id()
    chunks, n_docs, n_tokens = [], 0, 0

    def flush(texts):
        nonlocal n_tokens
        if not texts:
            return
        for ids in sp.encode(texts, out_type=int):
            if add_eos:
                ids = ids + [eos_id]
            arr = np.asarray(ids, dtype=np.uint16)
            chunks.append(arr)
            n_tokens += len(arr)

    buf = []
    with open(input_path) as f:
        for line in f:
            buf.append(json.loads(line)["text"])
            n_docs += 1
            if len(buf) >= batch_docs:
                flush(buf)
                buf = []
    flush(buf)
    arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return n_docs, len(arr), arr.nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/nova50_tokenizer.model")
    ap.add_argument("--out", required=True, help="Output shard path (.npy)")
    ap.add_argument("--batch_docs", type=int, default=2000)
    args = ap.parse_args()
    n_docs, n_tokens, nbytes = encode_file(args.input, args.tokenizer, args.out, args.batch_docs)
    print(f"{n_docs} documents -> {n_tokens:,} tokens -> {args.out} ({nbytes/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
