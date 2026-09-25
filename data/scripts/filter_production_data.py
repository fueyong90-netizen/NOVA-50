#!/usr/bin/env python3
"""Quality filtering for production data (length bounds, language heuristic,
minimum alphabetic ratio to drop garbage/boilerplate). Real, runnable filter
— operates on whatever data/production/raw/*.jsonl actually exists."""
import argparse
import json
import re
from pathlib import Path

MIN_CHARS, MAX_CHARS = 20, 20000
MIN_ALPHA_RATIO = 0.6


def passes_filter(text: str) -> bool:
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(len(text), 1) < MIN_ALPHA_RATIO:
        return False
    if re.search(r"(.)\1{20,}", text):  # runaway repeated character -> likely garbage
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/production/raw")
    ap.add_argument("--out_dir", default="data/production/filtered")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.jsonl"))
    if not files:
        print(f"No files found in {in_dir} — run data/scripts/download_production_data.py first "
              f"on a machine with internet access. NOT RUN.")
        return

    for f in files:
        n_in = n_kept = 0
        out_path = out_dir / f.name
        with open(f) as fin, open(out_path, "w") as fout:
            for line in fin:
                n_in += 1
                rec = json.loads(line)
                if passes_filter(rec["text"]):
                    fout.write(json.dumps(rec) + "\n")
                    n_kept += 1
        print(f"[{f.name}] {n_in} -> {n_kept} kept")


if __name__ == "__main__":
    main()
