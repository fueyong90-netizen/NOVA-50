#!/usr/bin/env python3
"""Lance une expérience complète (entraînement + éval) à partir d'un fichier
YAML, et journalise config/seed/commit git/version dataset/matériel dans
experiments/<name>/EXPERIMENT_INFO.json (section 12 du brief)."""
import argparse
import json
import subprocess
import sys
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.hardware import detect_hardware


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN (pas de dépôt git initialisé ou git indisponible)"


def dataset_version(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        p = Path(p)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config d'entraînement (ex: configs/train_toy_nova.yaml)")
    args = ap.parse_args()

    with open(args.config) as f:
        tcfg = yaml.safe_load(f)

    run_dir = Path(tcfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "config_file": args.config,
        "config": tcfg,
        "seed": tcfg["seed"],
        "git_commit": git_commit(),
        "dataset_version_sha256_16": dataset_version([tcfg["train_shard"], tcfg["val_shard"]]),
        "hardware": detect_hardware(),
    }
    with open(run_dir / "EXPERIMENT_INFO.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Info expérience -> {run_dir/'EXPERIMENT_INFO.json'}")

    print("Lancement de l'entraînement...")
    subprocess.run([sys.executable, str(ROOT / "training/train.py"), "--config", args.config], check=True)

    best_ckpt = run_dir / "best.ckpt"
    if best_ckpt.exists():
        print("Lancement de l'évaluation...")
        subprocess.run([
            sys.executable, str(ROOT / "evaluation/report.py"),
            "--checkpoint", str(best_ckpt),
            "--model_config", tcfg["model_config"],
            "--out", str(run_dir / "evaluation_report.md"),
        ], check=True)


if __name__ == "__main__":
    main()
