# NOVA-50 — Technical Report

## Abstract
NOVA-50 est un modèle de langage causal construit from scratch sous une
contrainte stricte de 50 000 000 paramètres entraînables, pour le Global
Innovation Build Challenge V2 (Track 01: TECH). Ce rapport documente
l'architecture, le pipeline de données, la méthode d'entraînement, et les
expériences réellement exécutées — en distinguant explicitement les
résultats mesurés des étapes non exécutées (`NOT RUN`), conformément à la
contrainte "no fake results" du brief.

## Introduction
Les modèles de langage à grande échelle nécessitent des budgets de calcul et
de paramètres considérables. Ce projet explore combien de capacité de
modélisation du langage peut être extraite d'un modèle délibérément minuscule
sous un budget de 50M paramètres.

## Research Question
> How much reasoning and language modeling capability can we extract from a
> deliberately tiny language model under a strict 50M trainable-parameter
> budget?

## Related Design Choices
Choix architecturaux évalués (littérature) : RMSNorm (Zhang & Sennrich 2019),
RoPE (Su et al. 2021), Grouped Query Attention (Ainslie et al. 2023), SwiGLU
(Shazeer 2020), weight tying (Press & Wolf 2017). Ces techniques sont connues
pour améliorer l'efficacité paramétrique dans la littérature générale ; leur
effet spécifique sur NOVA-50 est mesuré, pas supposé (voir section Ablations).

## Architecture
Voir `model/layers.py` et `model/transformer.py`. Une seule classe
`GPTModel` paramétrée par `ModelConfig` implémente à la fois MODEL A
(baseline) et MODEL B (NOVA-50), garantissant une comparaison à code
identique. Détails dans `README.md#architecture`.

## Parameter Budget
| Bucket | NOVA-50 (`model_nova.yaml`) |
|---|---|
| Embedding | 4,096,000 |
| Attention | 7,864,320 |
| FFN | 20,966,400 |
| Normalization | 10,752 |
| Output head | 0 (weight-tied avec l'embedding) |
| **Total** | **32,937,472** (65.87% du budget de 50M) |

Baseline (`model_baseline.yaml`) : 14,974,080 paramètres (29.95% du budget —
voir `results/parameter_report_baseline.md`). Les deux configurations sont sous le budget
avec une marge confortable, laissant de la place pour ajuster `dim`/`n_layers`
une fois un vrai corpus disponible.

## Dataset
Corpus jouet synthétique (gabarits paramétrés, ~19 500 tokens d'entraînement)
— voir `docs/DATASET.md`. **Ce n'est pas le corpus de soumission finale.**

## Data Processing
`data/scripts/{prepare_toy_corpus,clean_and_dedupe,split_and_shard}.py` :
génération → nettoyage/normalisation → déduplication exacte (SHA-256) →
split déterministe par hash (évite les fuites train/val/test — vérifié par
`evaluation/contamination.py`, 0 fuite détectée sur ce run).

## Training Method
AdamW (weight decay séparé, pas de decay sur biais/norm), warmup linéaire
puis cosine decay, gradient clipping (norme 1.0), gradient accumulation,
checkpointing + resume. Implémentation dans `training/`.

## Curriculum
Défini dans `configs/curriculum.yaml` (4 phases). **NOT RUN** :
`training/train.py` ne consomme pas encore la mixture de données de façon
dynamique par phase — l'intégration reste à faire avant de pouvoir comparer
curriculum vs mixture constante.

## Experiments
Toutes exécutées sur des configurations miniatures de démonstration
(`configs/model_*_tiny_demo.yaml`, 150K-350K paramètres, PAS les 32,9M de
NOVA-50) car aucun GPU n'est disponible dans l'environnement de développement
et le corpus disponible est un corpus jouet de ~20K tokens.

| Run | Paramètres | Val loss (step 300) | Tokens/s (CPU) |
|---|---|---|---|
| baseline_tiny_demo | 355,584 | 4.3266 | ~31,200 |
| nova50_tiny_demo | 218,176 | 4.6931 | ~26,600 |
| nova50_shared_tiny_demo | 218,176 | 4.6647 | ~17,300 |

(Chiffres extraits directement de `experiments/*/toy_run*/train_log.jsonl`,
non arrondis artificiellement au-delà de la précision affichée par le
logger.)

## Ablations
Voir `docs/ABLATIONS.md` pour l'analyse complète, hypothèses et limites.
Résumé : (A) baseline vs NOVA-50 — comparaison biaisée par une différence de
paramètres (355K vs 218K), non concluante en l'état. (B) partage de
paramètres à budget égal — léger avantage mesuré pour la version partagée
(4.6647 vs 4.6931), non significatif statistiquement (seed unique). (C)
curriculum et (D) data mixture — NOT RUN.

## Evaluation
- Perplexité sur test : mesurée par `evaluation/perplexity.py` (voir
  `results/evaluation_report_nova.md` pour le run NOVA-50 démo — perplexité
  élevée, cohérente avec un modèle minuscule entraîné 300 pas sur 20K tokens)
- Reasoning arithmétique (exact-match maison, PAS un benchmark officiel) :
  0% sur ce checkpoint démo — attendu, vu la taille du modèle/corpus
- Benchmarks officiels du challenge : **UNKNOWN / NOT RUN** (règles non
  identifiées, voir `docs/COMPETITION_RULES.md`)

## Efficiency
Débit mesuré en CPU pur : 17,000-31,000 tokens/sec pour des modèles de
150-350K paramètres sur cette machine. Ce chiffre n'est PAS extrapolable
linéairement à un modèle de 32,9M paramètres (33-150x plus de calcul par
token) ni à un GPU.

## Limitations
- Corpus jouet, pas de données réelles à l'échelle nécessaire pour un vrai
  pré-entraînement
- Modèles réellement entraînés = fraction du budget final (218K-355K vs 15-33M)
- Un seul seed par expérience
- Aucun test sur GPU
- Aucun benchmark officiel identifié/exécuté
- Curriculum learning et data-mixture non testés

## Future Work
1. Obtenir un vrai corpus (respectant les licences) et relancer le pipeline
   avec les configs pleine échelle sur GPU.
2. Implémenter la consommation dynamique de `curriculum.yaml` dans
   `training/train.py`.
3. Multi-seed pour les ablations, avec test statistique simple.
4. Brancher les benchmarks officiels dès qu'identifiés.

## Conclusion
Le pipeline complet (données → tokenizer → modèle → entraînement →
évaluation → démo) est implémenté et **réellement fonctionnel de bout en
bout**, vérifié par 23 tests automatisés et 3 runs d'entraînement réels.
Les résultats de performance à l'échelle de la compétition restent `NOT RUN`
et nécessitent un vrai corpus et du calcul GPU, indisponibles dans cet
environnement de développement.
