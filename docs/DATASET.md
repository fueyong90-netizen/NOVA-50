# DATASET.md — Toy Corpus (tests / CI / demo runs only)

## Status
This document covers the **synthetic toy corpus** used for unit tests, CI,
and the small demo/ablation training runs referenced throughout this repo.
It is explicitly **not** the final competition training corpus — see
[PRODUCTION_DATASET.md](PRODUCTION_DATASET.md) for that (status: NOT RUN,
network-gated in this environment).

## Why a synthetic corpus exists at all
This development container has no general network access (only PyPI/npm/
GitHub/crates.io package registries — see `docs/COMPETITION_RULES.md`), so
downloading a real web-scale corpus is not possible here. `data/scripts/
prepare_toy_corpus.py` generates template-parameterized (numbers, names,
operations vary) English text across 6 categories, so the full pipeline —
cleaning, dedup, split, tokenizer, training, evaluation — can be exercised
and tested end to end without network access.

## Composition (measured, from `data/raw/MANIFEST.json` / `data/processed/CLEAN_STATS.json`)
- 6 categories: general_text, technical_text, mathematics, code, reasoning, documentation
- ~1,900 raw examples generated; after exact-hash dedup, ~1,400+ unique examples remain
  (numbers vary slightly per generation seed — always computed, never hand-set)
- Deterministic hash-based train/val/test split (95/5/5), verified leak-free
  by `evaluation/contamination.py`
- Tokenized training set: ~19,500-20,200 tokens depending on which mixture
  (`configs/data_mix.yaml`) is used — orders of magnitude below what a real
  pretraining run needs

## Cleaning
`data/scripts/clean_and_dedupe.py`: whitespace/newline normalization, drops
text under 5 characters, exact SHA-256 deduplication. No fuzzy/near-duplicate
detection here (added for production data in `dedupe_production_data.py`,
since near-duplicates are common at web scale but not in this template-based
toy generator).

## Split
`data/scripts/split_and_shard.py`: hash-based split (not line-order-based),
so it is stable across re-runs and resistant to accidental leakage.

## Known limitations
- Synthetic corpus, not real-world text
- No language-ID filtering (all English by construction)
- Small vocabulary (2,000 for the toy tokenizer vs. the final tokenizer —
  see the vocabulary-size study in `docs/TECHNICAL_REPORT.md`)
- Never used to report an official benchmark or WikiText-103 number
