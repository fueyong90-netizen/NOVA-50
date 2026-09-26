# COMPETITION_RULES.md — Global Innovation Build Challenge V2, Track 01: TECH

**Rules checked and verified on 2026-09-24** by directly fetching the
official page at <https://gibc-v2.devpost.com/rules>. This replaces the
earlier version of this document, which could not locate the official page
and marked everything `UNKNOWN`. The URL was provided by the user and
independently confirmed by fetching it — its content is treated as
authoritative because it resolves to the real Devpost domain, not because it
was asserted in a message.

## Event dates
Hackathon opens: July 11, 2026. Submission deadline: October 1, 2026, 11:45pm
Taipei time (UTC+8). Judging: Oct 1–7, 2026.

## Track 01 (TECH) — technical requirements

| Rule | Interpretation | Consequence for NOVA-50 |
|---|---|---|
| ≤ 50,000,000 total trainable parameters, **including token embeddings and the output head** | Matches the working assumption already used in this repo | No change needed to `scripts/count_parameters.py`'s definition |
| Repo must include a script or printed output showing the parameter count, plus the model config | Already satisfied | `scripts/count_parameters.py` + `configs/model_nova.yaml` |
| Model must be trained **from scratch** — no pretrained init, no fine-tuning, no distillation | NOVA-50 already trains from random init | No pretrained weights used anywhere in this repo |
| Standard frameworks/libraries allowed (PyTorch, JAX, TF, HF transformers, tokenizer libs) | "From scratch" = weights/training run, not tooling | PyTorch + SentencePiece are fine, must be credited in Built With |
| Public datasets allowed (FineWeb, TinyStories, The Pile, similar) | Final training data can be real public data | Toy synthetic corpus must be clearly labeled as such, not as the final training set |
| No hosted inference API (incl. sponsor Featherless perk) as the model itself | N/A | NOVA-50 is a locally-trained, locally-served model — compliant |
| README must report training hardware, total training time, approximate compute | Training Efficiency is a scored criterion | Must be filled in with **real measured values only** — see `scripts/hardware_report.py`, `scripts/compute_report.py` |
| Evaluation: HellaSwag, ARC-Easy, PIQA, WinoGrande via lm-evaluation-harness, plus WikiText-103 held-out perplexity | Official, mandatory benchmark suite | See `evaluation/run_lm_eval.py`, `evaluation/wikitext103.py` — execution status documented per-run under `results/benchmarks/` |

## Submission requirements (all tracks)
1. Devpost project description
2. Public source repo (GitHub/GitLab/Bitbucket/HF) with a README (setup, prerequisites, how to run)
3. Demo video, 2–5 min, YouTube/Vimeo/Youku, unlisted OK, **not private**, English audio or subtitles
4. Built With: every technology/framework/library/API/dataset/hardware component
5. Team info: real full names, each teammate needs a Devpost account
6. ≥ 3 screenshots (interface, pipeline, training curves, or build)

## Originality / third-party code / AI disclosure
- Project must be built during the hackathon window (July 11 – Oct 1, 2026)
- Third-party open-source code, frameworks, and public datasets are allowed and must be credited
- **AI coding assistants are allowed with no restrictions** (Cursor, Claude Code, Copilot, ChatGPT, v0, etc.) but **must be disclosed** in Built With and the README, noting which parts were AI-assisted. Undisclosed use = misrepresentation.
- Repository state at the deadline is what gets judged

## Language and access
- Repository must be public and unrestricted
- **All descriptions, documentation, and video demonstrations must be in English**

## Judging criteria (Track 01 only)
Perplexity & Accuracy, Reasoning Performance, Training Efficiency, Innovation, Documentation & Demo.

## Direct consequences already applied in this repository
1. `docs/*.md` and `submission/*.md` rewritten in English (this file included)
2. `README.md` and `submission/built_with.md` now include an **AI Assistance
   Disclosure** section
3. `evaluation/run_lm_eval.py` and `evaluation/wikitext103.py` implement the
   real required benchmark interfaces — see those files and
   `results/benchmarks/` for what could actually be executed in this
   development environment vs. what is marked `NOT RUN`
4. The pre-existing arithmetic exact-match check is relabeled
   **"INTERNAL DEBUGGING BENCHMARK — NOT A GIBC BENCHMARK"** and is no longer
   presented as an official evaluation result

## What remains genuinely unresolved
This development container has no GPU and a network egress allow-list that
does **not** include huggingface.co or other general dataset hosts (only
package registries: PyPI, npm, GitHub, crates.io). Concretely:
- The official benchmark datasets (HellaSwag, ARC-Easy, PIQA, WinoGrande,
  WikiText-103) could not be downloaded through the standard `lm-eval`/`datasets`
  path from inside this container — see `results/benchmarks/` for the exact
  error captured when this was attempted.
- Full-scale training of `configs/model_nova.yaml` (32.9M params) on a real
  corpus has not been run here for the same reason (no GPU, no large corpus).
- A follow-up attempt (2026-09-26) specifically searched for legitimate
  GitHub-hosted mirrors as an alternate route (`github.com` and
  `raw.githubusercontent.com` are on the allow-list). A real MIT-licensed
  mirror of HellaSwag was located (`rowanz/hellaswag`), but its data file
  returns a genuine 404 directly from GitHub's own raw-content CDN, and a
  genuine `451 x-disabled: true` from `github.com` itself via the `/raw/`
  redirect — both confirmed as GitHub-side responses, not this sandbox's
  proxy. No other legitimate mirror for ARC-Easy/PIQA/WinoGrande/WikiText-103
  was found within a reasonable search effort. See
  `FINAL_COMPETITION_READINESS.md` for the full investigation and the
  concrete next steps (Kaggle Notebooks, free GPU + internet access,
  recommended) that require the user's own account.

These are infrastructure/environment limitations, not unknowns about the
rules — the rules themselves are fully verified and documented above.
