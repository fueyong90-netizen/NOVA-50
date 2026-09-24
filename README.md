# NOVA-50 — Parameter-Efficient Reasoning Language Model

> Soumission pour le **Global Innovation Build Challenge V2, Track 01: TECH**.
> ⚠️ Les règles officielles n'ont pas pu être vérifiées en ligne — voir
> [docs/COMPETITION_RULES.md](docs/COMPETITION_RULES.md). Tout ce qui suit
> respecte l'hypothèse la plus stricte : **≤ 50 000 000 paramètres
> entraînables, embeddings et output head inclus.**

## Overview
NOVA-50 est un modèle de langage causal construit **from scratch** (aucun
poids pré-entraîné, aucune distillation), avec une architecture optimisée
pour maximiser la capacité de langage/raisonnement sous une contrainte
stricte de paramètres.

## Research Question
> How much reasoning and language modeling capability can we extract from a
> deliberately tiny language model under a strict 50M trainable-parameter
> budget?

## Key Idea
Au lieu d'un simple decoder-only Transformer standard, NOVA-50 combine des
techniques connues pour être efficientes en paramètres : RMSNorm, RoPE,
Grouped Query Attention, SwiGLU, weight tying, et une option de partage de
poids entre blocs (tout ceci est **mesuré**, pas supposé meilleur — voir
[docs/ABLATIONS.md](docs/ABLATIONS.md)).

## Architecture
Une seule implémentation (`model/transformer.py`, `model/layers.py`) sert à
la fois MODEL A (baseline) et MODEL B (NOVA-50) — seule la configuration
change, ce qui garantit une comparaison à code égal.

| | MODEL A — baseline | MODEL B — NOVA-50 |
|---|---|---|
| Norm | LayerNorm | RMSNorm |
| Position | apprise (aucune, `use_rope: false`) | RoPE |
| Attention | MHA classique | GQA (4 têtes KV pour 8 têtes Q) |
| FFN | GELU, 2 matrices | SwiGLU, 3 matrices |
| Weight tying | non | oui |
| Config | `configs/model_baseline.yaml` | `configs/model_nova.yaml` |

## Parameter Budget
Voir `results/parameter_report.md` / `.json` (générés par
`scripts/count_parameters.py`, qui échoue avec exit code ≠ 0 si le budget est
dépassé — testé automatiquement par `tests/test_parameter_budget.py`).

| Config | Total paramètres entraînables | % du budget | Statut |
|---|---|---|---|
| `model_baseline.yaml` | voir sortie de la commande ci-dessous | — | PASS |
| `model_nova.yaml` | voir sortie de la commande ci-dessous | — | PASS |

```bash
python scripts/count_parameters.py --config configs/model_baseline.yaml
python scripts/count_parameters.py --config configs/model_nova.yaml
```

## Dataset
**Corpus jouet synthétique** — voir [docs/DATASET.md](docs/DATASET.md) pour
les raisons (pas d'accès réseau général dans l'environnement de
développement) et les vraies statistiques mesurées. **À remplacer avant
soumission finale** par un vrai corpus public sous licence permissive.

## Training
Boucle d'entraînement PyTorch maison (`training/train.py`) : AdamW, warmup +
cosine decay, gradient clipping, gradient accumulation, checkpointing/resume,
détection matérielle automatique (CPU/GPU). **Exécuté réellement dans cet
environnement en mode CPU** sur des configurations miniatures de démonstration
(`configs/model_*_tiny_demo.yaml`) — voir `docs/ABLATIONS.md` pour les
résultats mesurés et leurs limites. L'entraînement pleine échelle
(`model_nova.yaml`, 50M params, vrai corpus) n'a **pas** été exécuté ici
(NOT RUN — nécessite un GPU et un vrai corpus).

## Evaluation
- `evaluation/perplexity.py` — perplexité réellement mesurée sur les splits val/test
- `evaluation/reasoning.py` — exact-match arithmétique maison (**pas un
  benchmark officiel** — aucun n'a pu être identifié, voir
  `docs/COMPETITION_RULES.md`)
- `evaluation/contamination.py` — vérifie l'absence de fuite train/test (par hash)
- `evaluation/benchmarks.py` — point d'entrée pour les benchmarks officiels,
  actuellement UNKNOWN / NOT RUN

## Ablation Studies
Voir [docs/ABLATIONS.md](docs/ABLATIONS.md) — 2 expériences réellement
exécutées (baseline vs NOVA-50 ; partage de paramètres), 2 NON EXÉCUTÉES
(curriculum, data mixture), toutes honnêtement documentées avec leurs limites.

## Results
**NOT YET RUN À PLEINE ÉCHELLE.** Les seuls résultats disponibles sont ceux
des runs miniatures de démonstration dans `docs/ABLATIONS.md` et
`results/plots/`. Aucun chiffre de performance à l'échelle de la compétition
n'est affirmé.

## Installation
```bash
git clone <repo>
cd NOVA-50
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Quick Start
```bash
python scripts/reproduce.py
```
Exécute tout le pipeline de bout en bout sur le corpus jouet (voir
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) pour le détail étape par étape et
comment brancher un vrai corpus à la place).

## Training (commande individuelle)
```bash
python scripts/run_experiment.py --config configs/train_toy_nova.yaml
```

## Evaluation (commande individuelle)
```bash
python evaluation/report.py --checkpoint experiments/nova/toy_run/best.ckpt \
    --model_config configs/model_nova_tiny_demo.yaml --out results/evaluation_report.md
```

## Inference
```bash
python inference/generate.py --checkpoint experiments/nova/toy_run/best.ckpt \
    --model_config configs/model_nova_tiny_demo.yaml \
    --prompt "Question: What is 4 plus 5? Answer:" --max_new_tokens 20
```

## Web Demo
```bash
python inference/server.py --checkpoint experiments/nova/toy_run/best.ckpt \
    --model_config configs/model_nova_tiny_demo.yaml --port 8000
# puis ouvrir web/index.html dans un navigateur
```

## Reproducibility
Voir [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Hardware
Testé exclusivement en **mode CPU** (aucun GPU disponible dans
l'environnement de développement). `training/hardware.py` détecte
automatiquement CUDA/bf16/fp16 et adapte batch size / précision / grad
accumulation en conséquence — non testé en conditions GPU réelles.

## Limitations
- Corpus d'entraînement = données synthétiques jouets, pas un vrai corpus web-scale
- Modèles de démonstration effectivement entraînés = 150K-350K paramètres, pas 50M
- Aucun benchmark officiel du challenge identifié/exécuté
- Ablations curriculum et data-mixture non exécutées
- Un seul seed par expérience (pas d'écart-type)
- Jamais testé sur GPU

## Competition Compliance
Voir [docs/COMPETITION_RULES.md](docs/COMPETITION_RULES.md) : toutes les
règles non vérifiables sont marquées `UNKNOWN`. La seule contrainte traitée
comme confirmée est le budget de 50M paramètres entraînables, appliquée et
testée automatiquement (`tests/test_parameter_budget.py`).

## Built With
Python, PyTorch (CPU), SentencePiece, NumPy, Matplotlib, pytest — voir
`submission/built_with.md`.

## License
UNKNOWN — à préciser par l'utilisateur avant publication/soumission.
