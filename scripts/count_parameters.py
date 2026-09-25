#!/usr/bin/env python3
"""
Vérifie le nombre de paramètres entraînables d'une configuration de modèle
contre le budget du challenge (50 000 000). Échoue (exit code != 0) si
la limite est dépassée. Voir section 3 du brief.

Usage:
    python scripts/count_parameters.py --config configs/model_nova.yaml
    python scripts/count_parameters.py --config configs/model_baseline.yaml
    python scripts/count_parameters.py --config configs/model_nova.yaml --json results/parameter_report.json --md results/parameter_report.md
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.layers import ModelConfig
from model.transformer import build_model

BUDGET = 50_000_000


def load_config(path: str) -> ModelConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw.pop("name", None)
    return ModelConfig(**raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--json", default=None, help="Chemin de sortie du rapport JSON")
    ap.add_argument("--md", default=None, help="Chemin de sortie du rapport Markdown")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    counts = model.count_parameters()
    total = counts["total_trainable"]
    status = "PASS" if total <= BUDGET else "FAIL"

    print(f"\nConfig: {args.config}")
    print("Trainable parameters\n")
    print(f"Embedding:      {counts['embedding']:>12,}")
    print(f"Attention:      {counts['attention']:>12,}")
    print(f"FFN:            {counts['ffn']:>12,}")
    print(f"Normalization:  {counts['normalization']:>12,}")
    print(f"Output head:    {counts['output_head']:>12,}  "
          f"{'(tied with embedding — 0 new params)' if cfg.weight_tying else ''}")
    print(f"Other:          {counts['other']:>12,}")
    print(f"\nTOTAL (unique trainable tensors): {total:,}")
    print(counts["weight_tying_note"])
    conservative = counts["conservative_total_if_head_untied"]
    print(f"Conservative total (if tied head were counted separately): {conservative:,}")
    print(f"\n50M LIMIT: {BUDGET:,}")
    print(f"STATUS: {status}  ({100*total/BUDGET:.2f}% of budget, "
          f"{100*conservative/BUDGET:.2f}% under the conservative accounting)\n")

    report = {
        "config_file": args.config,
        "config_name": getattr(cfg, "name", None),
        "counts": counts,
        "budget": BUDGET,
        "percentage_of_budget": round(100 * total / BUDGET, 4),
        "status": status,
    }

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.md, "w") as f:
            f.write(f"# Parameter Budget Report — {args.config}\n\n")
            f.write("| Bucket | Count |\n|---|---|\n")
            for k in ["embedding", "attention", "ffn", "normalization", "output_head", "other"]:
                f.write(f"| {k} | {counts[k]:,} |\n")
            f.write(f"| **TOTAL (unique trainable tensors)** | **{total:,}** |\n")
            f.write(f"| Conservative total (tied head counted separately) | {counts['conservative_total_if_head_untied']:,} |\n\n")
            f.write(f"> {counts['weight_tying_note']}\n\n")
            f.write(f"- Budget: {BUDGET:,}\n")
            f.write(f"- Percentage used: {report['percentage_of_budget']}%\n")
            f.write(f"- Status: **{status}**\n")

    if status == "FAIL":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
