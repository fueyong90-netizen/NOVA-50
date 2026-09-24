#!/usr/bin/env python3
"""
Évaluation "raisonnement" par exact-match sur des questions arithmétiques
simples générées de la même manière que le bucket 'mathematics'/'reasoning'
du corpus jouet (data/scripts/prepare_toy_corpus.py).

IMPORTANT : ceci n'est PAS un benchmark de raisonnement standard/officiel
(type GSM8K, ARC, etc.) — aucun benchmark officiel n'a pu être identifié
faute d'accès aux règles du challenge (voir docs/COMPETITION_RULES.md).
Ce script sert de suite d'évaluation minimale, honnête et reproductible,
à remplacer par les benchmarks officiels dès qu'ils seront connus.
"""
import argparse
import random
import re
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.layers import ModelConfig
from model.transformer import build_model
from training.checkpoint import load_checkpoint
import sentencepiece as spm

random.seed(123)  # seed différent du corpus d'entraînement -> questions non vues telles quelles


def gen_arithmetic_questions(n):
    qs = []
    for _ in range(n):
        a, b = random.randint(1, 50), random.randint(1, 50)
        op = random.choice(["plus", "minus"])
        if op == "plus":
            q, ans = f"Question: What is {a} plus {b}? Answer:", a + b
        else:
            q, ans = f"Question: What is {a} minus {b}? Answer:", a - b
        qs.append((q, ans))
    return qs


def extract_number(text):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/nova50_tokenizer.model")
    ap.add_argument("--n_questions", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=6)
    args = ap.parse_args()

    with open(args.model_config) as f:
        raw = yaml.safe_load(f)
    raw.pop("name", None)
    cfg = ModelConfig(**raw)
    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    questions = gen_arithmetic_questions(args.n_questions)
    correct = 0
    for q, expected in questions:
        ids = sp.encode(q, out_type=int)
        idx = torch.tensor([ids], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=0.1, top_k=5)
        generated = sp.decode(out[0].tolist())
        pred = extract_number(generated[len(sp.decode(ids)):])
        is_correct = (pred == expected)
        correct += int(is_correct)

    acc = correct / len(questions)
    print(f"Exact-match arithmétique sur {len(questions)} questions : {correct}/{len(questions)} = {acc:.1%}")
    print("NOTE: benchmark maison sur corpus jouet, PAS un benchmark officiel du challenge (UNKNOWN).")
    return acc


if __name__ == "__main__":
    main()
