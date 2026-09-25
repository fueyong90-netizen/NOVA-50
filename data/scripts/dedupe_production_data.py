#!/usr/bin/env python3
"""Deduplicates production data. Reuses the same exact-hash approach as
data/scripts/clean_and_dedupe.py, plus a fuzzy MinHash pass for near-duplicates
(needed at web scale, unlike the tiny toy corpus)."""
import argparse
import hashlib
import json
from pathlib import Path

try:
    from datasketch import MinHash, MinHashLSH
    HAVE_MINHASH = True
except ImportError:
    HAVE_MINHASH = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/production/filtered")
    ap.add_argument("--out_dir", default="data/production/deduped")
    ap.add_argument("--fuzzy", action="store_true",
                     help="Also run MinHash near-dup detection (requires `pip install datasketch`)")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.jsonl"))
    if not files:
        print(f"No files found in {in_dir}. NOT RUN (run filter_production_data.py first).")
        return

    seen_hashes = set()
    lsh = MinHashLSH(threshold=0.85, num_perm=64) if (args.fuzzy and HAVE_MINHASH) else None
    if args.fuzzy and not HAVE_MINHASH:
        print("WARNING: datasketch not installed, falling back to exact-hash dedup only "
              "(pip install datasketch --break-system-packages for fuzzy dedup)")

    idx = 0
    for f in files:
        n_in = n_kept = 0
        out_path = out_dir / f.name
        with open(f) as fin, open(out_path, "w") as fout:
            for line in fin:
                n_in += 1
                rec = json.loads(line)
                h = hashlib.sha256(rec["text"].encode()).hexdigest()
                if h in seen_hashes:
                    continue
                if lsh is not None:
                    mh = MinHash(num_perm=64)
                    for tok in rec["text"].split():
                        mh.update(tok.encode())
                    if lsh.query(mh):
                        continue
                    lsh.insert(f"doc{idx}", mh)
                seen_hashes.add(h)
                rec["hash"] = h
                fout.write(json.dumps(rec) + "\n")
                n_kept += 1
                idx += 1
        print(f"[{f.name}] {n_in} -> {n_kept} kept")


if __name__ == "__main__":
    main()
