#!/usr/bin/env python3
"""
Benchmark contamination check + decontamination (streaming, low memory).

Method (standard 13-gram overlap, as in the GPT-3 paper appendix):
  1. Hash every 13-gram of every evaluation text (HellaSwag, ARC-Easy, PIQA,
     WinoGrande, WikiText-103 test) -> small in-memory dict.
  2. Stream the TRAINING corpus document by document and look up its 13-grams.
  3. Report, per benchmark, how many eval examples share >= 1 13-gram with the
     training data, and optionally write a decontaminated copy of the corpus with
     every flagged document removed.

Text is lower-cased and stripped of punctuation before hashing. Limitations: catches
verbatim / near-verbatim leakage only (not paraphrases); eval examples shorter than
N words cannot be detected (counted in `too_short`).

Usage:
  python evaluation/benchmark_contamination.py --eval_dir data/eval_text \
      --train_jsonl data/production/deduped/fineweb_edu.jsonl \
      --decontaminate_out data/production/decontaminated/fineweb_edu.jsonl
"""
import argparse
import datetime
import json
import re
import sys
from pathlib import Path

N_GRAM = 13
_PUNCT = re.compile(r"[^\w\s]")


def norm_words(text):
    return _PUNCT.sub(" ", text.lower()).split()


def build_eval_index(eval_dir, n=N_GRAM):
    """Returns (gram_hash -> global example idx), meta list [(bench, ex_id)], stats."""
    index, meta, stats = {}, [], {}
    for path in sorted(Path(eval_dir).glob("*.jsonl")):
        bench = path.stem
        st = {"n_examples": 0, "too_short": 0}
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                gidx = len(meta)
                meta.append((bench, ex["id"]))
                st["n_examples"] += 1
                got_gram = False
                for t in ex["texts"]:
                    w = norm_words(t)
                    for i in range(len(w) - n + 1):
                        index.setdefault(hash(" ".join(w[i:i + n])), gidx)
                        got_gram = True
                if not got_gram:
                    st["too_short"] += 1
        stats[bench] = st
    return index, meta, stats


def scan_corpus(train_jsonl, index, n=N_GRAM, out_path=None):
    hit_examples, n_docs, n_flagged = set(), 0, 0
    fout = open(out_path, "w") if out_path else None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(train_jsonl) as f:
        for line in f:
            n_docs += 1
            rec = json.loads(line)
            w = norm_words(rec["text"])
            flagged = False
            for i in range(len(w) - n + 1):
                g = index.get(hash(" ".join(w[i:i + n])))
                if g is not None:
                    hit_examples.add(g)
                    flagged = True
            if flagged:
                n_flagged += 1
            elif fout:
                fout.write(line)
    if fout:
        fout.close()
    return hit_examples, n_docs, n_flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="data/eval_text")
    ap.add_argument("--train_jsonl", default="data/production/deduped/fineweb_edu.jsonl")
    ap.add_argument("--decontaminate_out", default=None,
                    help="If set, write the training corpus minus every flagged document here")
    ap.add_argument("--n", type=int, default=N_GRAM)
    ap.add_argument("--out", default="results/benchmarks/contamination_report.json")
    args = ap.parse_args()

    if not Path(args.eval_dir).exists() or not list(Path(args.eval_dir).glob("*.jsonl")):
        report = {"status": "NOT RUN", "reason": f"no eval text found in {args.eval_dir} "
                  "(run evaluation/dump_eval_text.py first, needs huggingface.co access)"}
        print(json.dumps(report, indent=2))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        sys.exit(0)

    index, meta, stats = build_eval_index(args.eval_dir, args.n)
    print(f"Indexed {len(index):,} eval {args.n}-grams from {len(meta):,} examples")
    hits, n_docs, n_flagged = scan_corpus(args.train_jsonl, index, args.n, args.decontaminate_out)

    per_bench = {}
    for b, st in stats.items():
        per_bench[b] = dict(st, contaminated=0)
    for g in hits:
        per_bench[meta[g][0]]["contaminated"] += 1
    for b, st in per_bench.items():
        st["contamination_rate"] = round(st["contaminated"] / max(st["n_examples"], 1), 5)

    report = {
        "status": "RUN", "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": f"{args.n}-gram overlap, lower-cased, punctuation stripped, streaming",
        "train_corpus": args.train_jsonl, "train_documents_scanned": n_docs,
        "train_documents_flagged": n_flagged,
        "decontaminated_corpus": args.decontaminate_out,
        "train_documents_kept": (n_docs - n_flagged) if args.decontaminate_out else None,
        "per_benchmark": per_bench,
        "limitations": ["verbatim/near-verbatim only, paraphrases not detected",
                        f"eval examples shorter than {args.n} words cannot be detected (see too_short)"],
    }
    print(json.dumps(report, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
