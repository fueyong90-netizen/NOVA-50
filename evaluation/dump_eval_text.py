#!/usr/bin/env python3
"""Dumps the text of the official evaluation sets to data/eval_text/*.jsonl so the
training data can be checked (and decontaminated) against them.

Task text is taken from lm-evaluation-harness itself (the exact prompts/choices it
scores), and WikiText-103 from the official Hugging Face copy. One JSON object per
evaluation example: {"id": i, "texts": ["<prompt> <choice_1>", "<prompt> <choice_2>", ...]}
(prompt + each candidate answer, so a leaked "question + correct answer" is detectable).
Needs internet access to huggingface.co.
"""
import argparse
import json
from pathlib import Path

TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande"]


def extract_examples(task):
    """task: an lm-eval task-like object (eval_docs, doc_to_text, doc_to_choice)."""
    examples = []
    for i, doc in enumerate(task.eval_docs):
        prompt = str(task.doc_to_text(doc))
        try:
            choices = task.doc_to_choice(doc)
        except Exception:
            choices = []
        texts = [f"{prompt} {c}".replace("\n", " ") for c in choices] or [prompt.replace("\n", " ")]
        examples.append({"id": i, "texts": texts})
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/eval_text")
    ap.add_argument("--skip_wikitext", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager()
    for name in TASKS:
        try:
            task = get_task_dict([name], tm)[name]
            ex = extract_examples(task)
            with open(out / f"{name}.jsonl", "w") as f:
                for e in ex:
                    f.write(json.dumps(e) + "\n")
            summary[name] = {"status": "OK", "n_examples": len(ex)}
        except Exception as e:  # keep going: report per-task status honestly
            summary[name] = {"status": "FAILED", "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(name, summary[name])

    if not args.skip_wikitext:
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
            n = 0
            with open(out / "wikitext103_test.jsonl", "w") as f:
                for i, row in enumerate(ds):
                    line = row["text"].strip()
                    if line and not line.startswith("="):  # skip empty lines and section headings
                        f.write(json.dumps({"id": i, "texts": [line]}) + "\n")
                        n += 1
            summary["wikitext103_test"] = {"status": "OK", "n_examples": n}
        except Exception as e:
            summary["wikitext103_test"] = {"status": "FAILED", "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print("wikitext103_test", summary["wikitext103_test"])

    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
