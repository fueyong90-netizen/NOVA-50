# NOVA-50 — Technical Report

## Abstract
NOVA-50 is a causal language model trained from scratch under a strict
50,000,000 trainable-parameter budget, for the Global Innovation Build
Challenge V2 (Track 01: TECH, rules verified 2026-09-24 at
gibc-v2.devpost.com/rules). This report documents the architecture, data
pipeline, training method, and every experiment actually executed —
distinguishing measured results from `NOT RUN` items per the competition's
no-fake-results requirement. The official benchmark suite (HellaSwag,
ARC-Easy, PIQA, WinoGrande, WikiText-103) could not be executed in this
development environment (network/GPU constraints) and is reported as such.

## Research Question
How much reasoning and language-modeling capability can be extracted from a
deliberately tiny language model under a strict 50M trainable-parameter
budget?

## Motivation
Frontier LLMs require compute and parameter budgets far beyond individual or
small-team reach. This project studies the trade-offs available at the
opposite end of the spectrum.

## Architecture
`model/layers.py` / `model/transformer.py` implement one parameterizable
`GPTModel` used for both the baseline and NOVA-50, so both share identical
code and differ only by configuration (`configs/model_baseline.yaml` vs.
`configs/model_nova.yaml`): RMSNorm vs. LayerNorm, RoPE vs. learned
positions, GQA vs. standard MHA, SwiGLU vs. GELU, weight tying vs. none.

## Parameter Budget
| | NOVA-50 | Baseline |
|---|---|---|
| Embedding | 4,096,000 | 2,560,000 |
| Attention | 7,864,320 | 3,276,800 |
| FFN | 20,966,400 | 6,566,400 |
| Normalization | 10,752 | 10,880 |
| Output head | 0 (tied) | 2,560,000 |
| **Total** | **32,937,472 (65.87%)** | **14,974,080 (29.95%)** |
| Conservative (tied head counted separately) | 37,033,472 (74.07%) | 14,974,080 (29.95%) |

Both pass the budget with margin. `scripts/search_architectures.py` explored
6 candidates from 32.9M up to a deliberately-oversized 56.7M to map out where
the budget ceiling actually is (two candidates FAIL — real arithmetic, not
adjusted to fit); see `results/architecture_search.md`. The current 32.9M
architecture was kept as the submission because it is the only candidate
that has actually been exercised (trained, tested) — swapping to a larger,
untested candidate without evidence it performs better would violate the
no-fake-results principle.

## Tokenizer
SentencePiece BPE, trained from scratch (`tokenizer/train_tokenizer.py`, no
external/proprietary tokenizer). A real vocabulary-size study was run on the
toy corpus:

| Vocab size | Compression (chars/token) | Feasible on toy corpus? |
|---|---|---|
| 500 | 3.17 | yes |
| 1,000 | 4.15 | yes |
| 2,000 | 4.33 | yes (used for all demo runs) |
| 4,000 | — | **no** — SentencePiece error: "Vocabulary size too high (4000). Please set it to a value <= 3424." |
| 8,000 / 16,000 / 24,000 / 32,000 | — | **NOT RUN** — toy corpus has too little unique subword content; requires the production corpus (docs/PRODUCTION_DATASET.md, NOT RUN in this environment) |

Diminishing returns are already visible between 1,000 and 2,000 (4.15 →
4.33). The final production vocabulary size (`configs/model_nova.yaml`
currently specifies 8,000) should be re-validated once real data is
available — this has not been done.

## Dataset
Toy synthetic corpus for tests/demo (`docs/DATASET.md`); production pipeline
implemented but NOT RUN (`docs/PRODUCTION_DATASET.md`) due to network
constraints (huggingface.co unreachable from this container).

## Data Filtering
`data/scripts/clean_and_dedupe.py` (toy): normalization + exact SHA-256
dedup. `data/scripts/filter_production_data.py` (production, implemented,
not yet exercised on real data): length bounds, alphabetic-ratio filter,
runaway-character heuristic. `dedupe_production_data.py` adds optional
MinHash near-duplicate detection for web-scale data.

## Training
Custom PyTorch loop (`training/train.py`): AdamW with decoupled weight decay,
linear warmup + cosine decay, gradient clipping, gradient accumulation,
checkpoint/resume, automatic CPU/GPU detection and batch-size/precision
adaptation (`training/hardware.py`), configurable DataLoader workers and
`pin_memory` when CUDA is available.

## Curriculum
`training/curriculum.py` builds a step→shard schedule from
`configs/curriculum.yaml` and `train.py` switches the active training shard
mid-run accordingly — verified live in training logs (phase transitions are
printed and logged). This is a real integration, not a config file that sits
unused.

## Experimental Design
All experiments below ran on miniature configs (`*_tiny_demo.yaml` /
`*_matched_demo.yaml`, 150K-350K params) on the toy corpus — no GPU, no
production data available in this environment.

## Baseline (fair comparison)
A second NOVA-50 variant (`nova50_matched_demo`, 346,984 params) was built
specifically to match `baseline_tiny_demo` (355,584 params) within 2.4%,
correcting the earlier unmatched comparison. Result: NOVA-50 reaches val
loss **3.7859** vs. baseline's **4.3266** at matched size (single run/seed —
see Ablations).

## Ablations
See `docs/ABLATIONS.md` in full. Summary:
- **A. Matched-parameter baseline vs. NOVA-50:** NOVA-50 wins (3.7859 vs.
  4.3266), single seed.
- **B. Parameter sharing, 3 seeds:** shared-weight variant beats
  independent-layer variant on 3/3 seeds (mean 4.6551 ± 0.0147 vs. 4.6730 ±
  0.0207).
- **C. Curriculum vs. fixed mixture:** curriculum reaches the lowest val loss
  of three conditions (4.8732 vs. 4.9150 / 4.9059), single seed, real phase
  switching verified in logs.
- **D. Data mixture (fixed):** no meaningful difference detected between
  `baseline_uniform` and `code_heavy_experimental` at this token scale — an
  honest null result, not a forced conclusion.

## Evaluation
- Official: **NOT RUN.** `evaluation/run_lm_eval.py` correctly integrates
  lm-evaluation-harness (bug found and fixed during integration: a read-only
  `device` property collision) and fails precisely at Hugging Face Hub data
  download (`results/benchmarks/lm_eval_report.md`, real captured
  `FileNotFoundError`). `evaluation/wikitext103.py` fails the same way at the
  canonical S3 / HF source for WikiText-103
  (`results/benchmarks/wikitext103_report.md`).
- Internal-only: `evaluation/reasoning.py` (arithmetic exact-match) is
  explicitly labeled **INTERNAL DEBUGGING BENCHMARK — NOT A GIBC BENCHMARK**
  and never reported as an official score.
- `evaluation/perplexity.py` on the toy test split: pipeline-validation
  only, never presented as the WikiText-103 number.

## Efficiency
Measured CPU throughput for the demo models: 17,000-31,000 tokens/sec
(150-350K-parameter models, 1 logical CPU core). `scripts/compute_report.py`
separates MEASURED wall-clock/tokens from ESTIMATED FLOPs (6×N×D
approximation, Kaplan et al. 2020) — e.g. ≈1.66×10¹¹ FLOPs for the 300-step
NOVA-50 demo run. This does not extrapolate linearly to the 32.9M-parameter
model or to GPU hardware.

## Limitations
- No official benchmark results (network-gated, not implementation-gated)
- No full-scale training run (no GPU, no production corpus)
- Toy-corpus-only demo experiments, 1-3 seeds
- Vocabulary-size study limited to what the toy corpus supports (≤3,424)
- Data-mixture ablation inconclusive at this token scale

## Reproducibility
See `REPRODUCIBILITY.md`. Every command in this report was actually run in
this repository during development.

## AI Assistance Disclosure
Claude was used throughout development for implementation, debugging (e.g.
diagnosing the lm-eval `device` property bug above), test writing, and
documentation. All code was executed and verified in this repository.

## Conclusion
The full pipeline (data → tokenizer → model → training → curriculum →
evaluation → demo) is implemented and demonstrably functional end to end —
26 automated tests pass, 8+ real training runs completed, and every claim in
this report traces to an artifact in `results/` or `experiments/`. What
remains is exactly what network/GPU access unlocks: the official benchmark
suite and a full-scale training run, both fully coded and ready to execute
elsewhere.
