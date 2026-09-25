#!/usr/bin/env python3
"""
Evaluates perplexity on the held-out WikiText-103 test split, as required by
Track 01. Implementation reuses evaluation/perplexity.py's math (cross
entropy averaged token-weighted -> exp) but re-tokenizes real WikiText-103
raw text through the NOVA-50 tokenizer.

REAL STATUS in this repository: the canonical WikiText-103 distribution
(s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip) and
its Hugging Face mirror (huggingface.co/datasets/Salesforce/wikitext) are
both outside this container's network egress allow-list (see
docs/COMPETITION_RULES.md). This script implements the full evaluation
pipeline and will run correctly if pointed at a local copy of the raw
WikiText-103 test file — it reports NOT RUN rather than fabricate a
perplexity value.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.layers import ModelConfig
from model.transformer import build_model
from training.checkpoint import load_checkpoint

CANONICAL_SOURCE = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
EXPECTED_TEST_FILE = "wiki.test.raw"


def compute_perplexity_on_text(model, sp, text: str, seq_len: int, stride: int = None):
    """Standard sliding-window perplexity (Hugging Face-style), token-weighted."""
    stride = stride or seq_len
    ids = sp.encode(text, out_type=int)
    total_nll, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, max(len(ids) - 1, 1), stride):
            chunk = ids[start:start + seq_len + 1]
            if len(chunk) < 2:
                break
            x = torch.tensor([chunk[:-1]], dtype=torch.long)
            y = torch.tensor([chunk[1:]], dtype=torch.long)
            _, loss = model(x, y)
            n = y.numel()
            total_nll += loss.item() * n
            total_tokens += n
    avg_nll = total_nll / max(total_tokens, 1)
    return math.exp(avg_nll), avg_nll, total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/nova50_tokenizer.model")
    ap.add_argument("--wikitext103_test_file", default=None,
                     help=f"Path to a local '{EXPECTED_TEST_FILE}' file. "
                          f"Not available in this environment — see module docstring.")
    ap.add_argument("--out_json", default="results/benchmarks/wikitext103_report.json")
    ap.add_argument("--out_md", default="results/benchmarks/wikitext103_report.md")
    args = ap.parse_args()

    import datetime
    common = {
        "checkpoint": args.checkpoint,
        "model_config": args.model_config,
        "expected_source": CANONICAL_SOURCE,
        "expected_split": "test",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if args.wikitext103_test_file is None or not Path(args.wikitext103_test_file).exists():
        report = {
            "status": "NOT RUN",
            **common,
            "reason": (
                f"Could not download WikiText-103 in this environment: neither the canonical "
                f"S3 source ({CANONICAL_SOURCE}) nor huggingface.co are reachable from this "
                f"container's network egress allow-list (only PyPI/npm/GitHub package "
                f"registries are permitted — see docs/COMPETITION_RULES.md). Re-run this exact "
                f"script with --wikitext103_test_file pointing at a local wiki.test.raw on a "
                f"machine with normal internet access."
            ),
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        with open(args.out_md, "w") as f:
            f.write("# WikiText-103 Perplexity Report\n\n")
            f.write(f"- Status: **NOT RUN**\n- Reason: {report['reason']}\n")
        print(json.dumps(report, indent=2))
        sys.exit(0)

    with open(args.model_config) as f:
        raw = yaml.safe_load(f)
    raw.pop("name", None)
    cfg = ModelConfig(**raw)
    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    text = Path(args.wikitext103_test_file).read_text()
    ppl, avg_nll, n_tokens = compute_perplexity_on_text(model, sp, text, seq_len=cfg.max_seq_len)

    report = {
        "status": "RUN", **common,
        "tokenizer": args.tokenizer,
        "n_tokens_evaluated": n_tokens,
        "avg_cross_entropy": avg_nll,
        "perplexity": ppl,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    with open(args.out_md, "w") as f:
        f.write("# WikiText-103 Perplexity Report\n\n")
        f.write(f"- Status: **RUN**\n- Perplexity: {ppl:.4f}\n- Tokens evaluated: {n_tokens:,}\n"
                f"- Checkpoint: `{args.checkpoint}`\n- Date: {report['date']}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
