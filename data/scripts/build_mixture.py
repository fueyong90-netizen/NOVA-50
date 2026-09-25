#!/usr/bin/env python3
"""Builds a data mixture (section 8) by sampling from the per-bucket cleaned
files according to the proportions in configs/data_mix.yaml, using only the
TRAIN split hashes (so mixtures never touch val/test) — reuses the same
train.jsonl records, regrouped by bucket."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

random.seed(7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_mix_config", default="configs/data_mix.yaml")
    ap.add_argument("--mixture", required=True, help="Name of a mixture in data_mix.yaml")
    ap.add_argument("--train_jsonl", default="data/processed/train.jsonl")
    ap.add_argument("--n_examples", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.data_mix_config) as f:
        mix_cfg = yaml.safe_load(f)
    proportions = mix_cfg["mixtures"][args.mixture]

    by_bucket = defaultdict(list)
    with open(args.train_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            bucket = rec["source"].split("/")[-1] if "/" in rec.get("source", "") else "general_text"
            by_bucket[bucket].append(rec)

    out_records = []
    for bucket, frac in proportions.items():
        pool = by_bucket.get(bucket, [])
        n_take = int(round(args.n_examples * frac))
        if not pool:
            continue
        # sample with replacement if the toy pool is smaller than requested
        sampled = [random.choice(pool) for _ in range(n_take)] if n_take > len(pool) else random.sample(pool, n_take)
        out_records.extend(sampled)

    random.shuffle(out_records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Mixture '{args.mixture}': {len(out_records)} examples -> {args.out}")
    print(f"Composition: { {b: proportions[b] for b in proportions} }")


if __name__ == "__main__":
    main()
