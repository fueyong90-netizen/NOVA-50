# Judging Alignment Matrix — Track 01

| Criterion | Evidence | Metric | Artifact | Result |
|---|---|---|---|---|
| Perplexity & Accuracy | Official WikiText-103 perplexity | Perplexity | `evaluation/wikitext103.py`, `results/benchmarks/wikitext103_report.md` | **NOT RUN** — network-gated (see docs/COMPETITION_RULES.md); toy-corpus perplexity available in `results/evaluation_report_nova.md` for pipeline validation only |
| Reasoning Performance | Official lm-evaluation-harness suite | HellaSwag / ARC-Easy / PIQA / WinoGrande accuracy | `evaluation/run_lm_eval.py`, `results/benchmarks/lm_eval_report.md` | **NOT RUN** — same network constraint; harness integration verified working up to the data-download step |
| Training Efficiency | Measured hardware + compute | tokens/sec, wall-clock, estimated FLOPs | `results/hardware_report.json`, `results/compute_report_*.json` | CPU-only, 17K-31K tok/s on 150-350K-param demo models; full-scale run NOT RUN |
| Innovation | Parameter-sharing ablation, matched-baseline comparison, real curriculum integration | Val loss deltas across 3 seeds | `docs/ABLATIONS.md` | Shared-weight variant beats independent layers on 3/3 seeds (4.6551±0.0147 vs 4.6730±0.0207); NOVA-50 beats matched-size baseline (3.7859 vs 4.3266) |
| Documentation & Demo | README, technical report, working web demo | Test pass rate | `README.md`, `docs/TECHNICAL_REPORT.md`, `web/index.html` | 26/26 pytest tests pass |

Two of five criteria (Perplexity & Accuracy, Reasoning Performance) cannot
be scored with official numbers from this repository as submitted from this
development environment — see `docs/COMPETITION_RULES.md` for the exact,
verifiable reason (network egress allow-list) and the precise command to
produce real numbers on a machine with normal internet access.
