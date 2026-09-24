#!/usr/bin/env python3
"""Génération de texte en ligne de commande à partir d'un checkpoint entraîné."""
import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.layers import ModelConfig
from model.transformer import build_model
from training.checkpoint import load_checkpoint


def load_model(checkpoint, model_config):
    with open(model_config) as f:
        raw = yaml.safe_load(f)
    raw.pop("name", None)
    cfg = ModelConfig(**raw)
    model = build_model(cfg)
    load_checkpoint(checkpoint, model)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/nova50_tokenizer.model")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    model, cfg = load_model(args.checkpoint, args.model_config)

    ids = sp.encode(args.prompt, out_type=int)
    idx = torch.tensor([ids], dtype=torch.long)

    t0 = time.time()
    out = model.generate(
        idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, eos_token_id=sp.eos_id(),
    )
    elapsed = time.time() - t0
    n_new = out.shape[1] - idx.shape[1]

    text = sp.decode(out[0].tolist())
    print(text)
    print(f"\n[{n_new} tokens générés en {elapsed:.2f}s -> {n_new/max(elapsed,1e-6):.1f} tok/s]")


if __name__ == "__main__":
    main()
