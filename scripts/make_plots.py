#!/usr/bin/env python3
"""Génère les graphiques (section 20 du brief) à partir des VRAIS logs JSONL
produits par training/train.py. N'invente aucune donnée."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(path):
    steps_tr, train_loss, steps_val, val_loss, lrs, tps = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "train_loss" in rec:
                steps_tr.append(rec["step"]); train_loss.append(rec["train_loss"])
                lrs.append(rec.get("lr")); tps.append(rec.get("tokens_per_sec"))
            if "val_loss" in rec and "train_loss" not in rec:
                steps_val.append(rec["step"]); val_loss.append(rec["val_loss"])
    return steps_tr, train_loss, steps_val, val_loss, lrs, tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                     help="Paires label=chemin_vers_train_log.jsonl")
    ap.add_argument("--out_dir", default="results/plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pair in args.runs:
        label, path = pair.split("=", 1)
        st, tl, sv, vl, lrs, tps = read_log(path)
        ax.plot(st, tl, label=f"{label} (train)")
        if sv:
            ax.plot(sv, vl, "--", label=f"{label} (val)")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training / validation loss (données réelles)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=140)
    print(f"-> {out_dir/'loss_curves.png'}")

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for pair in args.runs:
        label, path = pair.split("=", 1)
        st, tl, sv, vl, lrs, tps = read_log(path)
        ax2.plot(st, lrs, label=label)
    ax2.set_xlabel("step"); ax2.set_ylabel("learning rate")
    ax2.set_title("Learning rate schedule (warmup + cosine)")
    ax2.legend(); fig2.tight_layout()
    fig2.savefig(out_dir / "lr_schedule.png", dpi=140)
    print(f"-> {out_dir/'lr_schedule.png'}")

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    for pair in args.runs:
        label, path = pair.split("=", 1)
        st, tl, sv, vl, lrs, tps = read_log(path)
        ax3.plot(st, tps, label=label)
    ax3.set_xlabel("step"); ax3.set_ylabel("tokens/sec (CPU, cumulatif)")
    ax3.set_title("Débit d'entraînement mesuré")
    ax3.legend(); fig3.tight_layout()
    fig3.savefig(out_dir / "throughput.png", dpi=140)
    print(f"-> {out_dir/'throughput.png'}")


if __name__ == "__main__":
    main()
