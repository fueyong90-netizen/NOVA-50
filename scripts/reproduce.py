#!/usr/bin/env python3
"""Exécute (ou explique) la chaîne complète de reproduction (section 21):
install -> dataset -> tokenizer -> modèle -> vérif paramètres -> training ->
évaluation -> plots -> rapport. Chaque étape est un vrai appel de script,
pas une simulation."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

STEPS = [
    ("Préparer le corpus jouet", [PY, "data/scripts/prepare_toy_corpus.py"]),
    ("Nettoyer + dédupliquer", [PY, "data/scripts/clean_and_dedupe.py"]),
    ("Split train/val/test", [PY, "data/scripts/split_and_shard.py"]),
    ("Entraîner le tokenizer", [PY, "tokenizer/train_tokenizer.py", "--vocab_size", "2000"]),
    ("Encoder train", [PY, "tokenizer/encode.py", "--input", "data/processed/train.jsonl",
                        "--out", "data/processed/shards/train.npy"]),
    ("Encoder val", [PY, "tokenizer/encode.py", "--input", "data/processed/val.jsonl",
                      "--out", "data/processed/shards/val.npy"]),
    ("Encoder test", [PY, "tokenizer/encode.py", "--input", "data/processed/test.jsonl",
                       "--out", "data/processed/shards/test.npy"]),
    ("Vérifier le budget de paramètres (baseline)",
     [PY, "scripts/count_parameters.py", "--config", "configs/model_baseline.yaml"]),
    ("Vérifier le budget de paramètres (NOVA-50)",
     [PY, "scripts/count_parameters.py", "--config", "configs/model_nova.yaml",
      "--json", "results/parameter_report.json", "--md", "results/parameter_report.md"]),
    ("Entraîner NOVA-50 (démo miniature)", [PY, "scripts/run_experiment.py", "--config", "configs/train_toy_nova.yaml"]),
    ("Entraîner baseline (démo miniature)", [PY, "scripts/run_experiment.py", "--config", "configs/train_toy_baseline.yaml"]),
    ("Entraîner NOVA-50 shared (ablation)", [PY, "scripts/run_experiment.py", "--config", "configs/train_toy_nova_shared.yaml"]),
    ("Générer les graphiques", [PY, "scripts/make_plots.py", "--runs",
                                 "NOVA-50=experiments/nova/toy_run/train_log.jsonl",
                                 "baseline=experiments/baseline/toy_run/train_log.jsonl",
                                 "NOVA-50-shared=experiments/ablations/param_sharing_run/train_log.jsonl"]),
    ("Vérifier l'absence de fuite train/test", [PY, "evaluation/contamination.py"]),
    ("Lancer les tests", [PY, "-m", "pytest", "-q"]),
]


def main():
    for name, cmd in STEPS:
        print(f"\n=== {name} ===")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"ÉCHEC à l'étape: {name} (exit code {r.returncode})")
            sys.exit(r.returncode)
    print("\nReproduction complète terminée avec succès.")


if __name__ == "__main__":
    main()
