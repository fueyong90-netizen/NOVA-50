# ABLATIONS.md

All experiments below were **actually executed** in this environment (CPU
only, `torch` 2.14, no GPU available — see `results/hardware_report.json`)
on the **toy synthetic corpus** described in `docs/DATASET.md` (~19,500-20,200
training tokens depending on the mixture) with **miniature models**
(configs `*_tiny_demo.yaml` / `*_matched_demo.yaml`, 150K-350K parameters —
not the final `configs/model_nova.yaml` at 32.9M). Trends here validate that
the pipeline works correctly end-to-end; they are not claimed as
conclusions that hold at competition scale until re-run with real data and
the final architecture.

## A. Fair baseline comparison (matched parameter count)

**Fixed in this revision:** the original A/B comparison used models of very
different sizes (355,584 vs 218,176 params). This is now corrected with a
second NOVA-50 variant deliberately sized to match the baseline as closely
as achievable with clean config values.

| | baseline_tiny_demo | nova50_matched_demo |
|---|---|---|
| Trainable parameters | 355,584 | **346,984** (2.4% smaller — closest match found) |
| Val loss, step 300 | 4.3266 | **3.7859** |
| Config | `configs/model_baseline_tiny_demo.yaml` | `configs/model_nova_matched_demo.yaml` |
| Log | `experiments/baseline/toy_run/train_log.jsonl` | `experiments/nova/matched_run/train_log.jsonl` |

**Result:** at closely matched parameter count, NOVA-50's architecture
(RMSNorm, RoPE, GQA, SwiGLU, weight tying) reaches a **noticeably lower**
validation loss than the baseline (LayerNorm, learned positions, MHA, GELU).
This is a single run, single seed, on a tiny synthetic corpus — a real
signal, but not yet a statistically robust claim (see Limitations below).

*(The earlier unmatched comparison — 355,584 vs 218,176 params — is kept in
git history for transparency but is no longer the reported comparison,
since it conflated architecture with raw capacity.)*

## B. Parameter sharing, 3 seeds (section 5/14)

**Hypothesis:** at equal parameter count, reusing the same weight block
across more logical layers (depth without extra parameters) can match or
beat independent layers.

| Seed | Independent (2 layers) | Shared (4 logical layers, 2 distinct blocks) |
|---|---|---|
| 42 | 4.6931 | 4.6647 |
| 123 | 4.6517 | 4.6382 |
| 2026 | 4.6741 | 4.6624 |
| **Mean ± std** | **4.6730 ± 0.0207** | **4.6551 ± 0.0147** |

Both configs have exactly 218,176 trainable parameters (verified by
`scripts/count_parameters.py`). The shared variant wins on **3/3 seeds**.

**Interpretation:** consistent direction across 3 independent seeds is a
real (if small, ~0.4% relative) signal that logical depth via weight reuse
helps at fixed parameter count in this toy setting. Still only 3 seeds and
a tiny corpus — worth re-validating at full scale before relying on it for
the final architecture choice.

## C. Curriculum learning vs. fixed mixture (section 9/11) — now actually implemented and run

`training/curriculum.py` + `training/train.py` now consume
`configs/curriculum.yaml` dynamically: the active training shard switches
mid-run according to configured phases (verified live in the training log,
e.g. `[curriculum] step 165: switching to phase 'phase3_reasoning_code_math'`).

| Run | Final val loss (step 300) |
|---|---|
| Fixed mixture (`baseline_uniform` for the whole run) | 4.9150 |
| Fixed mixture (`code_heavy_experimental` for the whole run) | 4.9059 |
| **Curriculum (4 phases, switching mixtures)** | **4.8732** |

Logs: `experiments/ablations/fixed_mixture_run/`,
`experiments/ablations/fixed_code_heavy_run/`, `experiments/ablations/curriculum_run/train_log.jsonl`.

**Result:** the curriculum run reaches the lowest validation loss of the
three, on this toy setup, single seed. Consistent with the curriculum
hypothesis, not proof of it at scale.

## D. Data mixture (section 8) — now actually run

See table in C above: `baseline_uniform` (general-text-heavy) and
`code_heavy_experimental` (35% code) perform almost identically as *fixed*
mixtures here (4.9150 vs 4.9059) — no meaningful difference detected at this
tiny scale/token budget. This is itself an honest (if unexciting) result:
with only ~20K tokens, there isn't enough signal to distinguish mixtures.

## What remains genuinely untested
1. All of the above at the final 32.9M-parameter architecture, on real data,
   with a GPU — **NOT RUN** (no GPU / no real corpus in this environment,
   see `docs/COMPETITION_RULES.md`).
2. Official benchmarks (HellaSwag, ARC-Easy, PIQA, WinoGrande,
   WikiText-103) — **NOT RUN**, see `results/benchmarks/`.
3. More than 3 seeds; a proper significance test.
