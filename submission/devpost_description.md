# NOVA-50 — Devpost Description (draft, ready to paste)

## Inspiration
Track 01 asks how much language and reasoning capability can be built under
a strict 50-million-parameter budget — a constraint that forces genuine
architectural efficiency rather than scale.

## What it does
NOVA-50 is a causal language model trained entirely from scratch (no
pretrained weights, no fine-tuning, no distillation), using RMSNorm, RoPE,
Grouped Query Attention, SwiGLU, and weight tying — compared head-to-head
against a standard baseline Transformer built with the exact same code, so
the comparison isolates architecture from implementation quirks.

## How we built it
A single configurable PyTorch model implementation serves both the baseline
and NOVA-50. An automatic parameter-budget checker
(`scripts/count_parameters.py`) fails the build if the 50M limit is
exceeded, enforced by an automated test. A real curriculum-learning
scheduler switches training data mixtures mid-run. The full pipeline —
tokenizer, data cleaning/dedup, training, evaluation, and a local web
demo — is covered by 26 automated tests.

## Challenges we ran into
- The development container had no GPU and no general network access
  (only package registries), which blocked downloading both the official
  benchmark datasets (HellaSwag, ARC-Easy, PIQA, WinoGrande, WikiText-103)
  and a production-scale training corpus. Every relevant script is fully
  implemented and was actually run — the captured failures are real network
  errors, documented rather than hidden.
- Integrating lm-evaluation-harness surfaced a real bug (a read-only
  `device` property collision in our model wrapper), fixed during development.

## Accomplishments that we're proud of
- A genuinely end-to-end, tested pipeline — not a mockup
- An automatic, test-enforced parameter budget check
- Honest ablations, including a null result (data-mixture comparison showed
  no significant difference at the tested scale) reported as such rather
  than massaged into a positive finding

## What we learned
At matched parameter count on our (small, synthetic) test setup, the
NOVA-50 architectural choices outperformed the baseline; a parameter-sharing
variant beat independent layers on 3/3 seeds; and curriculum learning beat
fixed-mixture training in our single real comparison. None of this is yet
validated at competition scale.

## What's next for NOVA-50
Run the full-scale training (`configs/model_nova.yaml`, real corpus, GPU)
and the official benchmark suite on a machine with normal internet access —
every script needed for this already exists in the repository.
