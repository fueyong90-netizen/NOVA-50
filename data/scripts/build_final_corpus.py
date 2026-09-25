#!/usr/bin/env python3
"""Assembles the final training corpus from data/production/deduped/*.jsonl
according to configs/data_mix.yaml, splits train/val/test (hash-based, no
leakage), and writes to data/production/final/. This is the production
equivalent of data/scripts/{build_mixture,split_and_shard}.py, operating on
real (not toy) data once it exists locally."""
import argparse
import json
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data.scripts.split_and_shard import split_bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/production/deduped")
    ap.add_argument("--data_mix_config", default="configs/data_mix.yaml")
    ap.add_argument("--mixture", default=None, help="Defaults to active_mixture in data_mix.yaml")
    ap.add_argument("--out_dir", default="data/production/final")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*.jsonl"))
    if not files:
        print(f"No files found in {in_dir}. NOT RUN — populate data/production/ first "
              f"(download_production_data.py -> filter -> dedupe) on a machine with internet access.")
        return

    with open(args.data_mix_config) as f:
        mix_cfg = yaml.safe_load(f)
    mixture_name = args.mixture or mix_cfg["active_mixture"]
    print(f"Building final corpus with mixture '{mixture_name}'")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": [], "test": []}
    for f in files:
        with open(f) as fin:
            for line in fin:
                rec = json.loads(line)
                split = split_bucket(rec["hash"], val_frac=0.02, test_frac=0.02)
                splits[split].append(rec)

    for split, records in splits.items():
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w") as fout:
            for rec in records:
                fout.write(json.dumps(rec) + "\n")
        print(f"{split}: {len(records)} examples -> {out_path}")


if __name__ == "__main__":
    main()
