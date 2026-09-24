# Judging Alignment Matrix

| Criterion | Evidence | File | Metric |
|---|---|---|---|
| Perplexity & Accuracy | Perplexité mesurée sur split test (checkpoint démo) | `results/evaluation_report_nova.md` | Perplexité (voir fichier — élevée, cohérente avec un modèle jouet) |
| Reasoning | Exact-match arithmétique maison (pas un benchmark officiel) | `evaluation/reasoning.py`, `results/evaluation_report_nova.md` | 0% sur checkpoint démo (NOT the final model) |
| Training Efficiency | Débit CPU mesuré | `experiments/*/toy_run*/train_log.jsonl` | 17K-31K tokens/s (modèles 150-350K params, CPU) |
| Innovation | Ablation parameter-sharing exécutée à budget de paramètres égal | `docs/ABLATIONS.md` section B | Val loss 4.6647 (partagé) vs 4.6931 (indépendant) |
| Documentation & Demo | README, rapport technique, démo web fonctionnelle | `README.md`, `docs/TECHNICAL_REPORT.md`, `web/index.html` | 23/23 tests pytest passent |

Note : les benchmarks officiels du challenge n'ont pas pu être identifiés
(voir `docs/COMPETITION_RULES.md`) — cette matrice sera mise à jour dès que
les critères réels seront connus.
