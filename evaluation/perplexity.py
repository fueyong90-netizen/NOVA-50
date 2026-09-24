#!/usr/bin/env python3
"""Calcule la perplexité d'un checkpoint sur un shard donné (val ou test)."""
import argparse
import math
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.layers import ModelConfig
from model.transformer import build_model
from training.dataset import TokenShardDataset
from training.checkpoint import load_checkpoint


def compute_perplexity(model, loader, device):
    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            n = y.numel()
            total_loss += loss.item() * n
            total_tokens += n
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss), avg_loss, total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    with open(args.model_config) as f:
        raw = yaml.safe_load(f)
    raw.pop("name", None)
    cfg = ModelConfig(**raw)
    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model)

    ds = TokenShardDataset(args.shard, seq_len=cfg.max_seq_len)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)
    ppl, avg_loss, n_tokens = compute_perplexity(model, loader, "cpu")
    print(f"Shard: {args.shard}")
    print(f"Tokens évalués: {n_tokens:,}")
    print(f"Loss moyenne (cross-entropy): {avg_loss:.4f}")
    print(f"Perplexité: {ppl:.4f}")


if __name__ == "__main__":
    main()
