#!/usr/bin/env python3
"""Given a SHORT real profiling run (e.g. 30-50 steps), computes how many training
steps fit in a session-time budget (e.g. Kaggle's ~9h GPU limit) at the model's
measured throughput. Never guesses a tokens/sec number: it reads it from an actual
experiments/<run>/train_log.jsonl produced by training/train.py.

Usage:
  python training/train.py --config configs/train_production_nova.yaml   # let it run ~5 min, then Ctrl-C
  python scripts/estimate_training_budget.py \
      --train_log experiments/nova/production_run/train_log.jsonl --hours 8.5
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_log", required=True)
    ap.add_argument("--hours", type=float, required=True, help="Available session wall-clock time")
    ap.add_argument("--safety_margin", type=float, default=0.9,
                    help="Fraction of the budget to actually target (leaves room for eval/save/variance)")
    args = ap.parse_args()

    steps, tps = [], []
    with open(args.train_log) as f:
        for line in f:
            rec = json.loads(line)
            if "tokens_per_sec" in rec:
                steps.append(rec["step"]); tps.append(rec["tokens_per_sec"])
    if len(steps) < 2:
        raise SystemExit(f"Not enough log entries in {args.train_log} yet — let the profiling run "
                          f"produce at least 2 log_every intervals first.")

    measured_tps = tps[-1]  # steady-state throughput, not the noisy first few steps
    sec_per_step = 1.0 / (measured_tps / (steps[-1] and 1))  # placeholder, corrected below
    # more direct: seconds/step from consecutive log entries
    with open(args.train_log) as f:
        recs = [json.loads(l) for l in f if "elapsed_sec" in l]
    d_step = recs[-1]["step"] - recs[-2]["step"]
    d_time = recs[-1]["elapsed_sec"] - recs[-2]["elapsed_sec"]
    sec_per_step = d_time / max(d_step, 1)

    budget_sec = args.hours * 3600 * args.safety_margin
    max_steps = int(budget_sec / sec_per_step)

    print(f"Measured (last interval): {sec_per_step:.4f} sec/step, {measured_tps:,.0f} tokens/sec")
    print(f"Budget: {args.hours}h x {args.safety_margin} safety margin = {budget_sec/3600:.2f}h usable")
    print(f"-> Recommended total_steps for configs/train_production_nova.yaml: {max_steps:,}")
    print(f"   (~{max_steps * sec_per_step / 3600:.2f}h wall clock at the measured rate)")


if __name__ == "__main__":
    main()
