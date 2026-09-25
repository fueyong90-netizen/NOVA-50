#!/usr/bin/env python3
"""
Checks whether the TRAINING corpus contains text that overlaps with the
official evaluation benchmarks (HellaSwag, ARC-Easy, PIQA, WinoGrande,
WikiText-103), using an n-gram overlap heuristic (a real, standard technique
— e.g. the method used in the GPT-3 paper's contamination appendix), not an
exact-hash check (benchmark text is rarely verbatim in training corpora).

STATUS IN THIS REPOSITORY: the official benchmark datasets could not be
downloaded in this development environment (see docs/COMPETITION_RULES.md —
huggingface.co is not reachable from this container's network egress
allow-list). This script is fully implemented and will run correctly once
the benchmark text files are available locally; until then it reports
NOT RUN rather than a fabricated "no contamination found".
"""
import argparse
import json
import sys
from pathlib import Path

N_GRAM = 13  # standard choice in GPT-3-style contamination checks


def ngrams(tokens, n):
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def check_overlap(train_texts, eval_texts, n=N_GRAM):
    train_grams = set()
    for t in train_texts:
        train_grams |= ngrams(t.lower().split(), n)

    contaminated = 0
    for t in eval_texts:
        eg = ngrams(t.lower().split(), n)
        if eg & train_grams:
            contaminated += 1
    return contaminated, len(eval_texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/processed/train.jsonl")
    ap.add_argument("--eval_text_file", default=None,
                     help="Plain text file with one eval example per line "
                          "(HellaSwag/ARC-Easy/PIQA/WinoGrande/WikiText-103 text). "
                          "Not available in this environment — see module docstring.")
    ap.add_argument("--out", default="results/benchmarks/contamination_report.json")
    args = ap.parse_args()

    if args.eval_text_file is None or not Path(args.eval_text_file).exists():
        report = {
            "status": "NOT RUN",
            "reason": (
                "Official benchmark text (HellaSwag/ARC-Easy/PIQA/WinoGrande/"
                "WikiText-103) could not be downloaded in this environment "
                "(huggingface.co not reachable from the network egress "
                "allow-list). See docs/COMPETITION_RULES.md."
            ),
            "method_if_run": f"{N_GRAM}-gram overlap between training corpus and eval set text "
                              f"(standard technique, see GPT-3 paper appendix C)",
            "excluded_by_design": "The toy synthetic corpus used for demo training "
                                   "(data/scripts/prepare_toy_corpus.py) contains no real-world "
                                   "text at all, so it cannot overlap with any real benchmark by construction.",
            "limitations": [
                "n-gram overlap catches near-duplicates but not paraphrased contamination",
                "requires the exact eval-set text used by lm-evaluation-harness for a valid check",
            ],
        }
        print(json.dumps(report, indent=2))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(0)

    # Real path, exercised once eval text is actually available:
    with open(args.train_jsonl) as f:
        train_texts = [json.loads(line)["text"] for line in f]
    with open(args.eval_text_file) as f:
        eval_texts = [line.strip() for line in f if line.strip()]

    contaminated, total = check_overlap(train_texts, eval_texts)
    report = {
        "status": "RUN",
        "n_gram": N_GRAM,
        "eval_examples_checked": total,
        "eval_examples_with_overlap": contaminated,
        "contamination_rate": round(contaminated / max(total, 1), 4),
    }
    print(json.dumps(report, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
