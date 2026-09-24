#!/usr/bin/env python3
# Renommé depuis inspect.py -> inspect_vocab.py : le nom "inspect.py" entre en
# conflit avec le module standard `inspect` (importé par numpy) quand ce
# script est exécuté directement (Python ajoute son propre dossier à sys.path).
"""Affiche des statistiques sur le tokenizer entraîné : taille du vocabulaire,
tokens spéciaux, quelques tokens du vocabulaire, taux de compression mesuré."""
import argparse
import json

import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/nova50_tokenizer.model")
    ap.add_argument("--sample_text", default="data/processed/val.jsonl")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    print(f"Vocab size: {sp.get_piece_size()}")
    print(f"pad={sp.pad_id()} unk={sp.unk_id()} bos={sp.bos_id()} eos={sp.eos_id()}")
    print("\n10 premiers tokens du vocabulaire:")
    for i in range(10):
        print(f"  {i}: {sp.id_to_piece(i)!r}")

    # Taux de compression (chars / tokens) mesuré sur un échantillon réel
    n_chars = n_tokens = n_lines = 0
    with open(args.sample_text) as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"]
            ids = sp.encode(text, out_type=int)
            n_chars += len(text)
            n_tokens += len(ids)
            n_lines += 1
    print(f"\nÉchantillon: {n_lines} lignes, {n_chars} caractères, {n_tokens} tokens")
    print(f"Compression: {n_chars/max(n_tokens,1):.2f} caractères/token")


if __name__ == "__main__":
    main()
