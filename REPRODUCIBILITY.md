# REPRODUCIBILITY.md

## Summary
The pipeline below has been actually executed end to end in the development
environment (CPU-only container, no general network access) and works.
`scripts/reproduce.py` automates the toy-corpus version of it.

## Step by step

1. **Install**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Toy data (for tests/CI/demo runs only — see docs/DATASET.md)**
   ```bash
   python data/scripts/prepare_toy_corpus.py
   python data/scripts/clean_and_dedupe.py
   python data/scripts/split_and_shard.py
   ```

3. **Production data (requires internet access this container does not have — see docs/PRODUCTION_DATASET.md)**
   ```bash
   python data/scripts/download_production_data.py --dataset tinystories
   python data/scripts/filter_production_data.py
   python data/scripts/dedupe_production_data.py
   python data/scripts/build_final_corpus.py
   ```

4. **Tokenizer**
   ```bash
   python tokenizer/train_tokenizer.py --vocab_size 2000   # toy corpus vocab; see docs/TECHNICAL_REPORT.md for the vocab-size study
   python tokenizer/inspect_vocab.py
   ```

5. **Encode to shards**
   ```bash
   python tokenizer/encode.py --input data/processed/train.jsonl --out data/processed/shards/train.npy
   python tokenizer/encode.py --input data/processed/val.jsonl   --out data/processed/shards/val.npy
   python tokenizer/encode.py --input data/processed/test.jsonl  --out data/processed/shards/test.npy
   ```

6. **Parameter budget check** (fails if > 50M)
   ```bash
   python scripts/count_parameters.py --config configs/model_baseline.yaml
   python scripts/count_parameters.py --config configs/model_nova.yaml \
       --json results/parameter_report.json --md results/parameter_report.md
   ```

7. **Training** (toy demo; swap in `configs/model_nova.yaml` + a production
   shard on a GPU machine for the real, full-scale run)
   ```bash
   python scripts/run_experiment.py --config configs/train_toy_nova_matched.yaml
   python scripts/run_experiment.py --config configs/train_toy_baseline.yaml
   python scripts/run_experiment.py --config configs/train_toy_curriculum.yaml
   ```

8. **Evaluation**
   ```bash
   python evaluation/report.py --checkpoint experiments/nova/matched_run/best.ckpt \
       --model_config configs/model_nova_matched_demo.yaml
   python evaluation/contamination.py
   python evaluation/run_lm_eval.py --checkpoint <ckpt> --model_config <cfg>      # NOT RUN here, see README
   python evaluation/wikitext103.py --checkpoint <ckpt> --model_config <cfg>      # NOT RUN here, see README
   ```

9. **Reports**
   ```bash
   python scripts/hardware_report.py
   python scripts/compute_report.py --model_config configs/model_nova_tiny_demo.yaml \
       --train_log experiments/nova/toy_run/train_log.jsonl
   python scripts/search_architectures.py
   ```

10. **Tests, audit, security**
    ```bash
    pytest -q
    python scripts/competition_audit.py
    python scripts/security_scan.py
    ```

## Seeding and determinism
- Default seed 42 (`configs/train_toy_*.yaml`), applied via `torch.manual_seed`
  before model construction and training.
- `tests/test_config_and_repro.py::test_same_seed_same_init` verifies
  identical initialization for a fixed seed.
- The parameter-sharing ablation was re-run with seeds 42, 123, 2026 (see
  `docs/ABLATIONS.md`) — results are consistent in direction across all three.
- Train/val/test split is deterministic (hash-based, not order-based).
- **Not guaranteed bit-identical across different machines/BLAS backends.**

## What is NOT reproducible as-is in this environment
- Full-scale results (32.9M params, production data, GPU) — never executed here.
- `results/benchmarks/*` — NOT RUN, network-gated (see docs/COMPETITION_RULES.md).
- Anything marked `NOT RUN` or `UNKNOWN` anywhere in the documentation.

## Environment actually used
- OS: Ubuntu 24.04 (sandboxed container)
- Python: 3.12
- PyTorch: 2.14.0, CPU build (`torch.cuda.is_available() == False`)
- No general network access (only PyPI/npm/GitHub/crates.io package registries)
