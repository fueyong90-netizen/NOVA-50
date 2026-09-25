#!/usr/bin/env python3
"""Estimates compute (FLOPs) for a given run from REAL run metadata
(experiments/<run>/train_log.jsonl + the model config), using the standard
6*N*D approximation (Kaplan et al. 2020: forward+backward FLOPs per token
~= 6 * non-embedding parameters). Clearly separates MEASURED from ESTIMATED."""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.layers import ModelConfig
from model.transformer import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--train_log", required=True, help="experiments/.../train_log.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.model_config) as f:
        raw = yaml.safe_load(f)
    raw.pop("name", None)
    cfg = ModelConfig(**raw)
    model = build_model(cfg)
    counts = model.count_parameters()
    total_params = counts["total_trainable"]
    non_embed_params = total_params - counts["embedding"]  # standard convention for the 6N approximation

    tokens_seen = 0
    elapsed_sec = 0.0
    with open(args.train_log) as f:
        for line in f:
            rec = json.loads(line)
            if "tokens_per_sec" in rec and "elapsed_sec" in rec:
                tokens_seen = rec["tokens_per_sec"] * rec["elapsed_sec"]
                elapsed_sec = rec["elapsed_sec"]

    estimated_flops = 6 * non_embed_params * tokens_seen  # Kaplan et al. approximation

    report = {
        "model_config": args.model_config,
        "train_log": args.train_log,
        "MEASURED": {
            "total_trainable_parameters": total_params,
            "wall_clock_seconds": round(elapsed_sec, 2),
            "tokens_processed": int(tokens_seen),
        },
        "ESTIMATED": {
            "method": "6 * non_embedding_parameters * tokens_processed (Kaplan et al. 2020 approximation)",
            "non_embedding_parameters": non_embed_params,
            "approximate_flops": estimated_flops,
            "approximate_pflop_days": estimated_flops / 1e15 / 86400,
        },
    }
    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
