#!/usr/bin/env python3
"""Trains several tokenizers at different vocab sizes on the SAME real corpus and
measures compression (chars/token) + resulting shard size, so the final vocab size
is picked on evidence instead of assumed. Writes results/vocab_size_study.{json,md}.

Usage (on the real production corpus, needs enough unique subword content):
  python tokenizer/vocab_size_study.py --input data/production/final/train.jsonl \
      --sizes 8000 16000 24000 32000
"""
import argparse
import json
import tempfile
from pathlib import Path

import sentencepiece as spm


def train_and_measure(input_jsonl, vocab_size, sample_for_eval, model_type="bpe"):
    with tempfile.TemporaryDirectory() as td:
        txt_path = Path(td) / "corpus.txt"
        n_lines = 0
        with open(input_jsonl) as fin, open(txt_path, "w") as fout:
            for line in fin:
                fout.write(json.loads(line)["text"].replace("\n", " <nl> ") + "\n")
                n_lines += 1
        prefix = str(Path(td) / "tok")
        try:
            spm.SentencePieceTrainer.train(
                input=str(txt_path), model_prefix=prefix, vocab_size=vocab_size,
                model_type=model_type, character_coverage=0.9995,
                pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
                user_defined_symbols=["<nl>"], input_sentence_size=2_000_000, shuffle_input_sentence=True,
            )
        except RuntimeError as e:
            return {"vocab_size": vocab_size, "status": "FAILED", "error": str(e)[:300], "n_training_lines": n_lines}

        sp = spm.SentencePieceProcessor()
        sp.load(prefix + ".model")
        n_chars = n_tokens = n_eval_lines = 0
        with open(sample_for_eval) as f:
            for line in f:
                text = json.loads(line)["text"]
                ids = sp.encode(text, out_type=int)
                n_chars += len(text)
                n_tokens += len(ids)
                n_eval_lines += 1
        return {
            "vocab_size": vocab_size, "status": "OK", "n_training_lines": n_lines,
            "eval_lines": n_eval_lines, "eval_chars": n_chars, "eval_tokens": n_tokens,
            "chars_per_token": round(n_chars / max(n_tokens, 1), 4),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Real training corpus (jsonl, field 'text')")
    ap.add_argument("--eval_file", default=None, help="Defaults to --input itself if no held-out sample given")
    ap.add_argument("--sizes", type=int, nargs="+", default=[8000, 16000, 24000, 32000])
    ap.add_argument("--out_json", default="results/vocab_size_study.json")
    ap.add_argument("--out_md", default="results/vocab_size_study.md")
    args = ap.parse_args()
    eval_file = args.eval_file or args.input

    rows = [train_and_measure(args.input, v, eval_file) for v in args.sizes]
    for r in rows:
        print(r)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if r["status"] == "OK"]
    best = max(ok, key=lambda r: r["chars_per_token"]) if ok else None
    md = ["# Vocabulary Size Study\n", f"Corpus: `{args.input}`\n",
          "| vocab_size | status | chars/token | eval tokens |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['vocab_size']:,} | {r['status']} | "
                  f"{r.get('chars_per_token', '-')} | {r.get('eval_tokens', '-')} |")
    if best:
        md.append(f"\n**Best measured compression: vocab_size={best['vocab_size']:,} "
                  f"({best['chars_per_token']} chars/token).** Diminishing returns beyond this "
                  f"point should be checked before committing to a larger vocab (it also costs "
                  f"parameters: vocab_size x dim x 2, counted against the 50M budget via weight tying).")
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(f"\n-> {args.out_json}\n-> {args.out_md}")


if __name__ == "__main__":
    main()
