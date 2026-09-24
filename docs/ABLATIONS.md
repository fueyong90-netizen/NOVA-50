# ABLATIONS.md

⚠️ Toutes les expériences ci-dessous ont été **réellement exécutées** dans cet
environnement (CPU, `torch` 2.14, aucun GPU disponible), sur le **corpus
jouet synthétique** décrit dans `docs/DATASET.md` (≈19 500 tokens
d'entraînement) avec des modèles **miniatures** (configs `*_tiny_demo.yaml`,
150K-350K paramètres, pas les configs finales à 50M). Les tendances observées
ici sont donc indicatives du bon fonctionnement du pipeline, **pas** des
conclusions généralisables à l'échelle de la compétition — refaire ces
mêmes expériences avec les configs `model_baseline.yaml` / `model_nova.yaml`
sur un vrai corpus est nécessaire avant de tirer des conclusions définitives.

## A. Baseline vs NOVA-50 (architecture)

**Hypothèse** : RMSNorm + RoPE + GQA + SwiGLU + weight tying (NOVA-50)
convergent au moins aussi vite que LayerNorm + positions apprises + MHA +
GELU (baseline) à budget de paramètres comparable.

| | baseline_tiny_demo | nova50_tiny_demo |
|---|---|---|
| Paramètres entraînables | 355,584 | 218,176 |
| Val loss finale (step 300) | **4.3266** | 4.6931 |
| Config | `configs/model_baseline_tiny_demo.yaml` | `configs/model_nova_tiny_demo.yaml` |
| Log | `experiments/baseline/toy_run/train_log.jsonl` | `experiments/nova/toy_run/train_log.jsonl` |

**Résultat mesuré** : sur cette mini-expérience, le baseline atteint une
loss de validation plus basse — mais il a aussi **63% de paramètres en
plus** (weight tying + GQA réduisent volontairement la taille de NOVA-50).
À budget de paramètres strictement égal, la comparaison serait différente ;
ce n'est pas ce qui a été testé ici.

**Interprétation honnête** : ce résultat ne permet PAS de conclure que
l'architecture baseline est "meilleure". Il montre surtout qu'avec ~350K vs
~218K paramètres et seulement 300 pas sur 20K tokens, la différence de
capacité brute domine l'effet des choix architecturaux. Une comparaison
académiquement valide nécessite d'égaliser le nombre de paramètres (ou
d'entraîner NOVA-50 avec un `dim` plus grand pour combler l'écart) — **à
refaire avant de tirer une conclusion pour le rapport final.**

## B. Parameter sharing (section 5 du brief)

**Hypothèse** : à nombre de paramètres égal, répéter un même bloc de poids
plusieurs fois (profondeur "virtuelle" accrue) peut égaler ou dépasser un
nombre de couches indépendantes plus petit.

| | nova50_tiny_demo (2 couches indépendantes) | nova50_shared_tiny_demo (4 couches logiques, 2 blocs distincts) |
|---|---|---|
| Paramètres entraînables | 218,176 | 218,176 (identique — c'est le point de la comparaison) |
| Val loss finale (step 300) | 4.6931 | **4.6647** |
| Log | `experiments/nova/toy_run/train_log.jsonl` | `experiments/ablations/param_sharing_run/train_log.jsonl` |

**Résultat mesuré** : à paramètres strictement égaux, la version avec
partage de poids (profondeur logique 4, 2 blocs distincts réutilisés) obtient
une loss de validation légèrement meilleure que la version à 2 couches
indépendantes.

**Interprétation honnête** : l'écart (4.6647 vs 4.6931) est faible et mesuré
sur un run unique, un seed unique, un corpus jouet minuscule — **pas
significatif statistiquement** en l'état (pas de plusieurs seeds, pas de
test statistique). Il est cohérent avec l'intuition que la profondeur
logique aide même sans paramètres supplémentaires, mais ne le prouve pas de
façon rigoureuse. Pour une conclusion robuste : plusieurs seeds, plus de
pas d'entraînement, un vrai corpus.

## C. Curriculum learning vs mixture constante (section 9)

**Statut : NON EXÉCUTÉ.** `configs/curriculum.yaml` définit un curriculum en
4 phases et une mixture de comparaison, mais aucune expérience comparative
n'a été lancée dans cette session (nécessite plusieurs runs d'entraînement
plus longs). `training/train.py` ne consomme pas encore directement
`curriculum.yaml` — l'intégration (changer la mixture de données en cours
de run) reste à implémenter avant de pouvoir exécuter cette ablation.

## D. Data mixture (section 8)

**Statut : NON EXÉCUTÉ.** Une seule mixture (`baseline_uniform`) a été
utilisée pour générer le corpus jouet effectivement utilisé dans tous les
runs ci-dessus. La mixture alternative (`code_heavy_experimental`) définie
dans `configs/data_mix.yaml` n'a pas été testée.

## Ce qui reste à faire avant soumission
1. Relancer A et B avec les configs pleine échelle (`model_baseline.yaml`,
   `model_nova.yaml`) sur un vrai corpus, à budget de paramètres comparable.
2. Répéter chaque expérience sur au moins 3 seeds pour un écart-type.
3. Implémenter et exécuter les ablations C (curriculum) et D (data mixture).
4. Une fois les vraies règles du challenge connues (`docs/COMPETITION_RULES.md`),
   remplacer `evaluation/reasoning.py` par les benchmarks officiels.
