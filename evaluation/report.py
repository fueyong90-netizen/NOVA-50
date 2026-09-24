#!/usr/bin/env python3
"""Agrège perplexité + reasoning + contamination en un rapport JSON/Markdown
unique, à partir des résultats RÉELLEMENT mesurés (pas de valeurs inventées)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_and_capture(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--out", default="results/evaluation_report.md")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    lines = ["# Evaluation Report\n"]
    lines.append(f"- Checkpoint: `{args.checkpoint}`\n- Model config: `{args.model_config}`\n")

    ppl_out, rc = run_and_capture([
        sys.executable, str(root / "evaluation/perplexity.py"),
        "--checkpoint", args.checkpoint, "--model_config", args.model_config,
        "--shard", "data/processed/shards/test.npy",
    ])
    lines.append("## Perplexity (test split)\n```\n" + ppl_out + "```\n")

    reason_out, rc2 = run_and_capture([
        sys.executable, str(root / "evaluation/reasoning.py"),
        "--checkpoint", args.checkpoint, "--model_config", args.model_config,
    ])
    lines.append("## Reasoning (arithmetic exact-match, maison — pas officiel)\n```\n" + reason_out + "```\n")

    contam_out, rc3 = run_and_capture([sys.executable, str(root / "evaluation/contamination.py")])
    lines.append("## Contamination check\n```\n" + contam_out + "```\n")

    bench_out, _ = run_and_capture([sys.executable, str(root / "evaluation/benchmarks.py")])
    lines.append("## Official benchmarks\n```\n" + bench_out + "```\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Rapport écrit -> {args.out}")


if __name__ == "__main__":
    main()
