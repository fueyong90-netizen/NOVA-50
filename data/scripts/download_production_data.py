#!/usr/bin/env python3
"""
Downloads real public datasets permitted by the official rules (FineWeb,
TinyStories, The Pile, or similar — see docs/PRODUCTION_DATASET.md) for the
FINAL competition training corpus.

REAL STATUS in this repository: this container's network egress allow-list
covers only package registries (PyPI, npm, GitHub, crates.io) — it does NOT
include huggingface.co, which is where these datasets are hosted. Running
this script here fails at the download step; the failure is captured and
reported below rather than silently skipped or faked.

Run this unmodified on a machine with normal internet access to actually
populate data/production/raw/.
"""
import argparse
import json
import sys
from pathlib import Path

SUPPORTED = {
    "tinystories": {
        "hf_dataset": "roneneldan/TinyStories",
        "license": "CDLA-Sharing-1.0 (per Hugging Face dataset card)",
    },
    "fineweb_sample": {
        "hf_dataset": "HuggingFaceFW/fineweb",
        "license": "ODC-By 1.0 (per Hugging Face dataset card)",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=SUPPORTED.keys(), required=True)
    ap.add_argument("--out_dir", default="data/production/raw")
    ap.add_argument("--n_examples", type=int, default=10000)
    args = ap.parse_args()

    meta = SUPPORTED[args.dataset]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        ds = load_dataset(meta["hf_dataset"], split=f"train[:{args.n_examples}]")
        out_path = out_dir / f"{args.dataset}.jsonl"
        with open(out_path, "w") as f:
            for ex in ds:
                text = ex.get("text") or ex.get("content") or json.dumps(ex)
                f.write(json.dumps({
                    "text": text, "source": meta["hf_dataset"], "license": meta["license"],
                }) + "\n")
        print(f"Downloaded {len(ds)} examples -> {out_path}")
    except Exception as e:
        report = {
            "status": "NOT RUN",
            "dataset": args.dataset,
            "hf_dataset": meta["hf_dataset"],
            "error_type": type(e).__name__,
            "error_message": str(e)[:500],
            "reason": ("huggingface.co is not reachable from this container's network "
                       "egress allow-list — see docs/COMPETITION_RULES.md. Re-run this "
                       "exact command on a machine with normal internet access."),
        }
        print(json.dumps(report, indent=2))
        Path("results/production_download_status.json").parent.mkdir(parents=True, exist_ok=True)
        with open("results/production_download_status.json", "a") as f:
            f.write(json.dumps(report) + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
