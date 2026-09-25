# PRODUCTION_DATASET.md

## Status: NOT RUN in this development environment

The scripts in `data/scripts/{download_production_data,filter_production_data,
dedupe_production_data,build_final_corpus}.py` implement a complete, real
pipeline for building the final competition training corpus from public
datasets explicitly permitted by the official rules. They are fully
functional Python — the download step specifically fails in *this*
container because huggingface.co is outside the network egress allow-list
(only PyPI/npm/GitHub/crates.io package registries are reachable — see
`docs/COMPETITION_RULES.md`). The exact captured error is in
`results/production_download_status.json`.

**To populate real data:** run `python data/scripts/download_production_data.py
--dataset tinystories` (or `fineweb_sample`) unmodified on any machine with
normal internet access, then `filter_production_data.py`,
`dedupe_production_data.py`, and `build_final_corpus.py` in sequence.

## Candidate sources (permitted by the official rules — "FineWeb, TinyStories, The Pile and similar")

| Dataset | Source URL | License | Retrieval date | Approx. size (if downloaded) | Status |
|---|---|---|---|---|---|
| TinyStories | https://huggingface.co/datasets/roneneldan/TinyStories | CDLA-Sharing-1.0 (per HF dataset card) | NOT RUN | NOT RUN | NOT RUN |
| FineWeb (sample) | https://huggingface.co/datasets/HuggingFaceFW/fineweb | ODC-By 1.0 (per HF dataset card) | NOT RUN | NOT RUN | NOT RUN |

No hash/version pin exists yet since no download has actually succeeded from
this environment — `build_final_corpus.py` computes and stores a SHA-256 per
retained example (see its `hash` field) automatically once real data flows
through it.

## Filtering method (implemented, exercised only once real input exists)
`filter_production_data.py`: length bounds (20–20,000 characters), minimum
alphabetic-character ratio (0.6) to drop boilerplate/garbage, and a
runaway-repeated-character heuristic. `dedupe_production_data.py`: exact
SHA-256 dedup plus an optional MinHash near-duplicate pass (`--fuzzy`,
requires `datasketch`).

## Relationship to the toy corpus
`data/scripts/prepare_toy_corpus.py` (synthetic, template-generated) remains
in the repo specifically for unit tests and CI (`pytest -q` must not depend
on network access or GPU). It is never used as a stand-in for this
production pipeline in any documentation or result.
