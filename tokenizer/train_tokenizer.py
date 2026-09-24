#!/usr/bin/env python3
"""Entraîne un tokenizer BPE (SentencePiece) sur le corpus d'entraînement local.
Aucune dépendance à un tokenizer propriétaire ou à un LLM externe."""
import argparse
import json
from pathlib import Path

import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/train.jsonl")
    ap.add_argument("--out_dir", default="tokenizer/artifacts")
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--model_type", default="bpe", choices=["bpe", "unigram"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SentencePiece veut un fichier texte brut (une "phrase"/exemple par ligne)
    txt_path = out_dir / "_train_text.txt"
    n_lines = 0
    with open(args.input) as fin, open(txt_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            fout.write(rec["text"].replace("\n", " <nl> ") + "\n")
            n_lines += 1

    model_prefix = str(out_dir / "nova50_tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(txt_path),
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
        user_defined_symbols=["<nl>"],
    )

    stats = {
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "n_training_lines": n_lines,
        "special_tokens": {"pad": 0, "unk": 1, "bos": 2, "eos": 3, "newline_token": "<nl>"},
        "corpus": args.input,
        "corpus_note": "Corpus jouet synthétique (voir docs/DATASET.md), pas le corpus final",
    }
    with open(out_dir / "TOKENIZER_INFO.json", "w") as f:
        json.dump(stats, f, indent=2)

    txt_path.unlink()  # nettoyage du fichier temporaire
    print(f"Tokenizer entraîné -> {model_prefix}.model / .vocab")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
