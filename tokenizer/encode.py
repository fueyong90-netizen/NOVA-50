#!/usr/bin/env python3
"""Encode un fichier .jsonl (champ "text") en IDs de tokens, sauvegardés en
shards binaires (.npy) que le training loop peut mémoire-mapper."""
import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


def load_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/nova50_tokenizer.model")
    ap.add_argument("--out", required=True, help="Chemin du shard .npy de sortie")
    ap.add_argument("--add_eos", action="store_true", default=True)
    args = ap.parse_args()

    sp = load_tokenizer(args.tokenizer)
    eos_id = sp.eos_id()

    all_ids = []
    n_docs = 0
    with open(args.input) as f:
        for line in f:
            rec = json.loads(line)
            ids = sp.encode(rec["text"], out_type=int)
            if args.add_eos:
                ids = ids + [eos_id]
            all_ids.extend(ids)
            n_docs += 1

    arr = np.array(all_ids, dtype=np.uint16)  # vocab < 65536 -> uint16 suffit
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, arr)
    print(f"{n_docs} documents -> {len(arr):,} tokens -> {args.out} ({arr.nbytes/1024:.1f} KB)")


if __name__ == "__main__":
    main()
