#!/usr/bin/env python3
"""Split train/val/test (par hash, pour éviter les fuites) + écriture de shards
texte bruts (la tokenisation en shards binaires se fait après entraînement du
tokenizer, voir tokenizer/encode.py)."""
import argparse
import hashlib
import json
from pathlib import Path


def split_bucket(h: str, val_frac: float, test_frac: float) -> str:
    # Split déterministe basé sur le hash du texte -> pas de fuite train/val/test
    # même si le script est relancé, et pas de dépendance à l'ordre des lignes.
    v = int(h[:8], 16) / 0xFFFFFFFF
    if v < test_frac:
        return "test"
    if v < test_frac + val_frac:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/processed")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--test_frac", type=float, default=0.05)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    splits = {"train": [], "val": [], "test": []}

    for f in sorted(in_dir.glob("*.clean.jsonl")):
        with open(f) as fin:
            for line in fin:
                rec = json.loads(line)
                split = split_bucket(rec["hash"], args.val_frac, args.test_frac)
                splits[split].append(rec)

    for split, records in splits.items():
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w") as fout:
            for rec in records:
                fout.write(json.dumps(rec) + "\n")
        print(f"{split}: {len(records)} exemples -> {out_path}")


if __name__ == "__main__":
    main()
