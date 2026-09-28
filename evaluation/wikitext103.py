#!/usr/bin/env python3
"""
WikiText-103 held-out perplexity (Track 01 requirement).

Data     : official WikiText-103 *raw* TEST split (Salesforce/wikitext, wikitext-103-raw-v1),
           rows concatenated as-is (no lower-casing, "@-@" artefacts kept). `--download`
           fetches it from Hugging Face; otherwise pass a local `wiki.test.raw`.
Method   : sliding-window perplexity (window = model context, stride = context/2), every
           token scored exactly once with >= context/2 tokens of left context (except at
           the start). The text is prefixed with EOS, matching training-time packing.
Reported : token-level perplexity (depends on OUR tokenizer), word-level perplexity
           (exp(total NLL / whitespace words), tokenizer-independent, comparable across
           models) and bits-per-byte.
"""
import argparse
import datetime
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch
import yaml
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.layers import ModelConfig
from model.transformer import build_model
from training.checkpoint import load_checkpoint

SOURCE = "https://huggingface.co/datasets/Salesforce/wikitext (config wikitext-103-raw-v1, split test)"


def sliding_window_nll(model, token_ids, eos_id, window, stride, device="cpu"):
    """Returns (total_nll, n_scored_tokens). Every token in `token_ids` is scored once."""
    toks = [eos_id] + list(token_ids)
    n_targets = len(toks) - 1
    scored, total_nll, b = 0, 0.0, 0
    model.eval()
    with torch.no_grad():
        while scored < n_targets:
            e = min(b + window, n_targets)
            x = torch.tensor([toks[b:e]], dtype=torch.long, device=device)
            y = torch.tensor(toks[b + 1:e + 1], dtype=torch.long, device=device)
            new = e - scored
            logits, _ = model(x)
            lp = torch.log_softmax(logits[0, -new:].float(), dim=-1)
            total_nll -= float(lp.gather(1, y[-new:, None]).sum())
            scored = e
            if e == n_targets:
                break
            b += stride
    return total_nll, n_targets


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/nova50_tokenizer.model")
    ap.add_argument("--wikitext103_test_file", default="data/eval_text/wiki.test.raw")
    ap.add_argument("--download", action="store_true", help="Fetch the official test split from Hugging Face")
    ap.add_argument("--max_chars", type=int, default=None, help="Evaluate only a prefix slice (smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_json", default="results/benchmarks/wikitext103_report.json")
    ap.add_argument("--out_md", default="results/benchmarks/wikitext103_report.md")
    args = ap.parse_args()

    test_file = Path(args.wikitext103_test_file)
    common = {"checkpoint": args.checkpoint, "model_config": args.model_config, "tokenizer": args.tokenizer,
              "source": SOURCE, "split": "test", "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "command": "python evaluation/wikitext103.py " + " ".join(sys.argv[1:])}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    if args.download and not test_file.exists():
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("".join(ds["text"]))
        except Exception as e:
            report = {"status": "NOT RUN", **common, "error": f"{type(e).__name__}: {str(e)[:500]}"}
            Path(args.out_json).write_text(json.dumps(report, indent=2))
            Path(args.out_md).write_text(f"# WikiText-103 Perplexity\n\nStatus: **NOT RUN** — {report['error']}\n")
            print(json.dumps(report, indent=2)); sys.exit(0)

    if not test_file.exists():
        report = {"status": "NOT RUN", **common, "error": f"{test_file} not found (use --download)"}
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        Path(args.out_md).write_text(f"# WikiText-103 Perplexity\n\nStatus: **NOT RUN** — {report['error']}\n")
        print(json.dumps(report, indent=2)); sys.exit(0)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    with open(args.model_config) as f:
        raw = yaml.safe_load(f)
    raw.pop("name", None)
    cfg = ModelConfig(**raw)
    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model)
    model.to(device)
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    text = test_file.read_text()
    if args.max_chars:
        text = text[:args.max_chars]
    ids = sp.encode(text, out_type=int)
    window, stride = cfg.max_seq_len, cfg.max_seq_len // 2
    nll, n_tok = sliding_window_nll(model, ids, sp.eos_id(), window, stride, device)
    n_words, n_bytes = len(text.split()), len(text.encode("utf-8"))

    report = {
        "status": "RUN", **common, "smoke_test": args.max_chars is not None,
        "test_file": str(test_file), "test_file_sha256": sha256_file(test_file),
        "preprocessing": "raw text as distributed (rows concatenated), tokenized by SentencePiece "
                         "(newlines -> whitespace), EOS prefix, sliding window stride=context/2",
        "window": window, "stride": stride, "n_characters": len(text), "n_words_whitespace": n_words,
        "n_tokens_scored": n_tok, "total_nll_nats": nll,
        "avg_cross_entropy_per_token": nll / n_tok,
        "token_perplexity": math.exp(nll / n_tok),
        "word_perplexity": math.exp(nll / n_words),
        "bits_per_byte": nll / n_bytes / math.log(2),
        "trainable_parameters": model.count_parameters()["total_trainable"],
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "checkpoint_sha256": sha256_file(args.checkpoint),
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    if not report["smoke_test"]:
        Path(args.out_json).with_name("wikitext103_final.json").write_text(json.dumps(report, indent=2))
    md = [f"# WikiText-103 Perplexity Report\n",
          f"- Status: **RUN**" + ("  (SMOKE TEST slice — not a final result)" if report["smoke_test"] else ""),
          f"- Split: test ({SOURCE})", f"- Checkpoint: `{args.checkpoint}` (sha256 `{report['checkpoint_sha256'][:16]}…`)",
          f"- Tokenizer: `{args.tokenizer}`  |  window {window}, stride {stride}",
          f"- Tokens scored: {n_tok:,}  |  words: {n_words:,}  |  bytes: {n_bytes:,}\n",
          "| metric | value |", "|---|---|",
          f"| token-level perplexity | {report['token_perplexity']:.3f} |",
          f"| word-level perplexity | {report['word_perplexity']:.3f} |",
          f"| bits per byte | {report['bits_per_byte']:.4f} |",
          f"\nDate (UTC): {common['date_utc']}  |  device: {device}  |  command: `{common['command']}`"]
    Path(args.out_md).write_text("\n".join(md) + "\n")
    if not report["smoke_test"]:
        Path(args.out_md).with_name("wikitext103_final.txt").write_text("\n".join(md) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "n_tokens_scored", "token_perplexity", "word_perplexity", "bits_per_byte"]}, indent=2))


if __name__ == "__main__":
    main()
