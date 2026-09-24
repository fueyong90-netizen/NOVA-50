#!/usr/bin/env python3
"""Nettoyage + normalisation + déduplication (section 7 du brief).
RAW DATA -> validation -> nettoyage -> normalisation -> filtrage -> dédup
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

MIN_CHARS = 5


def clean_text(t: str) -> str:
    t = t.strip()
    t = re.sub(r"[ \t]+", " ", t)      # normaliser les espaces
    t = re.sub(r"\n{3,}", "\n\n", t)   # normaliser les sauts de ligne multiples
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    stats = {}

    for jsonl_file in sorted(in_dir.glob("*.jsonl")):
        bucket = jsonl_file.stem
        n_in = n_kept = n_dropped_short = n_dropped_dupe = 0
        out_path = out_dir / f"{bucket}.clean.jsonl"
        with open(jsonl_file) as fin, open(out_path, "w") as fout:
            for line in fin:
                n_in += 1
                rec = json.loads(line)
                text = clean_text(rec["text"])
                if len(text) < MIN_CHARS:
                    n_dropped_short += 1
                    continue
                h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    n_dropped_dupe += 1
                    continue
                seen_hashes.add(h)
                rec["text"] = text
                rec["hash"] = h
                fout.write(json.dumps(rec) + "\n")
                n_kept += 1
        stats[bucket] = {
            "n_in": n_in, "n_kept": n_kept,
            "n_dropped_short": n_dropped_short, "n_dropped_duplicate": n_dropped_dupe,
        }
        print(f"[{bucket}] in={n_in} kept={n_kept} dropped_short={n_dropped_short} "
              f"dropped_dup={n_dropped_dupe}")

    with open(out_dir / "CLEAN_STATS.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
