# REPRODUCIBILITY.md

## Environnement testé
- Python 3.12.3
- torch (CPU-only, aucun GPU disponible dans l'environnement de développement)
- OS: Linux (conteneur Ubuntu 24.04)
- Voir `requirements.txt` pour les versions exactes des dépendances

## Étapes (automatisées par `scripts/reproduce.py`)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/reproduce.py
```

Ce script exécute dans l'ordre :
1. Génération du corpus (jouet, voir `docs/DATASET.md` — à remplacer par un
   vrai téléchargeur avant la soumission finale)
2. Nettoyage + déduplication
3. Split train/val/test (déterministe par hash, sans fuite)
4. Entraînement du tokenizer BPE
5. Encodage des 3 splits en shards `.npy`
6. Vérification du budget de paramètres pour `model_baseline.yaml` et
   `model_nova.yaml` (échoue avec exit code ≠ 0 si > 50M)
7. Trois runs d'entraînement de démonstration (baseline, NOVA-50, NOVA-50
   avec partage de paramètres) sur les configs `*_tiny_demo.yaml`
8. Génération des graphiques à partir des logs réels
9. Vérification de contamination train/test
10. `pytest -q`

## Seeds
Chaque config d'entraînement fixe `seed: 42` (voir `configs/train_toy_*.yaml`).
`torch.manual_seed` est appelé avant la construction du modèle et avant le
premier batch. Le test `tests/test_config_and_repro.py::test_same_seed_same_init`
vérifie que deux modèles construits avec le même seed ont des poids
initiaux identiques.

## Ce qui N'EST PAS garanti reproductible bit-à-bit
- Les opérations CPU multi-thread de PyTorch ne garantissent pas un résultat
  strictement déterministe à la dernière décimale d'une exécution à l'autre.
- Aucun test de déterminisme bit-exact du forward/backward n'a été ajouté
  (seulement l'identité des poids initiaux).

## Reprise d'un entraînement interrompu
```bash
python training/train.py --config configs/train_toy_nova.yaml   # (avec resume: true dans le YAML)
```
`training/checkpoint.py` sauvegarde `last.ckpt` à chaque évaluation
(`eval_every` steps) ; relancer avec `resume: true` reprend exactement au
`step` sauvegardé (modèle + optimiseur + scheduler).

## Entraîner les configs pleine échelle (50M)
Les configs `configs/model_baseline.yaml` et `configs/model_nova.yaml` sont
les architectures candidates réelles pour la compétition (14,97M et 32,94M
paramètres respectivement — voir `results/parameter_report.md`). Elles n'ont
**pas** été entraînées dans cet environnement (pas de GPU, pas de vrai
corpus à grande échelle disponible ici). Pour les entraîner réellement :

1. Remplacer le corpus jouet dans `data/raw/` par de vraies données (voir
   `docs/DATASET.md`, section "Sources à compléter").
2. Créer un fichier `configs/train_full_nova.yaml` sur le modèle de
   `configs/train_toy_nova.yaml`, en pointant `model_config` vers
   `configs/model_nova.yaml` et vers les shards du vrai corpus.
3. `python scripts/run_experiment.py --config configs/train_full_nova.yaml`
   sur une machine avec GPU (le code détecte et utilise CUDA automatiquement,
   voir `training/hardware.py`).
