# Architecture Search — parameter/memory arithmetic (NOT a training result)

All rows use the same NOVA-50 architectural family (RMSNorm, RoPE, GQA, SwiGLU, weight tying) as `configs/model_nova.yaml`; only size varies, so this isolates the effect of scale. Parameter counts and memory estimates below are EXACT arithmetic (not measured on hardware). Benchmark/perplexity performance for each candidate is **NOT RUN** — that requires real training data and a GPU, not available in this development environment (see `docs/COMPETITION_RULES.md`). The current submission keeps `nova50_current_32.9M` as the selected architecture because it is the only candidate that has actually been exercised end-to-end (demo training, tests) in this repository; picking a bigger candidate without evidence it trains/evaluates better would contradict the "no fake results" rule.

| label | params | % of 50M budget | status | est. fp32 memory (GB, no activations) |
|---|---|---|---|---|
| nova50_current_32.9M | 32,937,472 | 65.87% | PASS | 0.527 |
| nova50_wider_40M | 40,009,536 | 80.02% | PASS | 0.64 |
| nova50_deeper_42M | 44,473,856 | 88.95% | PASS | 0.712 |
| nova50_wide_deep_45M | 47,089,728 | 94.18% | PASS | 0.753 |
| nova50_max_48M | 54,682,240 | 109.36% | FAIL | 0.875 |
| nova50_ceiling_49M | 56,745,600 | 113.49% | FAIL | 0.908 |

## NOT RUN for every candidate above
- Actual training loss / validation loss
- Perplexity
- HellaSwag / ARC-Easy / PIQA / WinoGrande
- WikiText-103 perplexity
- Wall-clock training time on real hardware
