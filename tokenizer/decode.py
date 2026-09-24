#!/usr/bin/env python3
"""Décode une liste d'IDs (JSON sur stdin/argument ou shard .npy) en texte."""
import argparse
import json

import numpy as np
import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/nova50_tokenizer.model")
    ap.add_argument("--ids", help="Liste d'IDs séparés par des virgules, ex: 12,45,7")
    ap.add_argument("--npy", help="Chemin d'un shard .npy à décoder (premiers 200 tokens)")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    elif args.npy:
        ids = np.load(args.npy)[:200].tolist()
    else:
        raise SystemExit("Fournir --ids ou --npy")

    print(sp.decode(ids))


if __name__ == "__main__":
    main()
