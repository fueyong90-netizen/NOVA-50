# Built With

## Languages & core framework
- Python 3.12
- PyTorch 2.14.0 (CPU build in this development environment) — custom
  implementation of RMSNorm, RoPE, Grouped Query Attention, SwiGLU, and the
  training loop (no HF `Trainer`)

## Tokenization & data
- SentencePiece — BPE tokenizer trained from scratch on project data
- NumPy — memory-mapped token shards
- `datasets` / `lm-eval` (PyPI `lm-eval` 0.4.13) — official benchmark harness
  integration (see Evaluation below)
- `datasketch` (optional) — MinHash near-duplicate detection for production data

## Evaluation
- lm-evaluation-harness — HellaSwag, ARC-Easy, PIQA, WinoGrande (integration
  implemented and tested; task-data download NOT RUN in this environment —
  see `results/benchmarks/lm_eval_report.md`)
- WikiText-103 — custom evaluator (`evaluation/wikitext103.py`); NOT RUN for
  the same network-access reason

## Tooling
- Matplotlib — training-curve plots generated from real logs
- pytest — 26 automated tests
- Python stdlib `http.server` — local inference API (no Flask/FastAPI dependency)
- HTML/CSS/JS (vanilla) — static web demo

## Datasets
- Toy synthetic corpus (`data/scripts/prepare_toy_corpus.py`) — generated
  locally, template-based, used only for tests/CI/demo runs
- Production dataset targets (NOT downloaded in this environment):
  TinyStories (`roneneldan/TinyStories`, CDLA-Sharing-1.0) and FineWeb
  (`HuggingFaceFW/fineweb`, ODC-By 1.0) — see `docs/PRODUCTION_DATASET.md`

## Hardware actually used
- CPU: Intel(R) Xeon(R) Processor @ 2.10GHz (1 logical core visible to the container)
- RAM: 3.91 GB
- GPU: none available in this development environment

## AI tools used (disclosed per official rules)
- **Claude (Anthropic)** — used throughout development for code generation,
  debugging (including diagnosing a real `lm-evaluation-harness` integration
  bug), test generation, and documentation. All code was reviewed and
  executed in this repository; the team is responsible for the submitted
  implementation.
