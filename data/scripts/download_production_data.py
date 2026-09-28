#!/usr/bin/env python3
"""
Downloads real public datasets permitted by the official Track 01 rules
(FineWeb / TinyStories / The Pile "and similar") into data/production/raw/.

Supported sources
  tinystories       roneneldan/TinyStories                       (CDLA-Sharing-1.0)
  fineweb_edu       HuggingFaceFW/fineweb-edu, config sample-10BT (ODC-By 1.0)
                    -> the educational-quality subset of FineWeb; streamed, so only
                       the first --n_examples documents are ever downloaded.

Every run appends a provenance record (source URL, license, retrieval date,
document count, SHA-256 of the produced file) to data/production/raw/SOURCES.json.
Requires internet access to huggingface.co (NOT available in the sandbox used to
write this repo — it was run on Kaggle; see docs/PRODUCTION_DATASET.md).
"""
import argparse
import datetime
import hashlib
import json
import sys
from itertools import islice
from pathlib import Path

SOURCES = {
    "tinystories": {
        "hf_dataset": "roneneldan/TinyStories", "config": None,
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "license": "CDLA-Sharing-1.0 (per Hugging Face dataset card)", "streaming": False,
    },
    "fineweb_edu": {
        "hf_dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
        "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        "license": "ODC-By 1.0 (per Hugging Face dataset card)", "streaming": True,
    },
}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=SOURCES.keys(), required=True)
    ap.add_argument("--out_dir", default="data/production/raw")
    ap.add_argument("--n_examples", type=int, default=10000)
    args = ap.parse_args()

    meta = SOURCES[args.dataset]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.jsonl"

    try:
        from datasets import load_dataset
        if meta["streaming"]:
            ds = load_dataset(meta["hf_dataset"], name=meta["config"], split="train", streaming=True)
            it = islice(ds, args.n_examples)
        else:
            ds = load_dataset(meta["hf_dataset"], split=f"train[:{args.n_examples}]")
            it = iter(ds)
        n = 0
        with open(out_path, "w") as f:
            for ex in it:
                text = ex.get("text") or ex.get("content")
                if not text:
                    continue
                f.write(json.dumps({"text": text, "source": meta["hf_dataset"],
                                    "license": meta["license"]}) + "\n")
                n += 1
        print(f"Downloaded {n} examples -> {out_path}")
    except Exception as e:
        report = {"status": "NOT RUN", "dataset": args.dataset, "hf_dataset": meta["hf_dataset"],
                  "error_type": type(e).__name__, "error_message": str(e)[:500],
                  "reason": "huggingface.co not reachable from this machine (enable Internet / run elsewhere)."}
        print(json.dumps(report, indent=2))
        Path("results").mkdir(exist_ok=True)
        with open("results/production_download_status.json", "a") as f:
            f.write(json.dumps(report) + "\n")
        sys.exit(1)

    record = {
        "dataset": args.dataset, "hf_dataset": meta["hf_dataset"], "config": meta["config"],
        "source_url": meta["url"], "license": meta["license"],
        "retrieval_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_documents": n, "output_file": str(out_path), "sha256": sha256_file(out_path),
        "size_mb": round(out_path.stat().st_size / 1e6, 1),
    }
    manifest = out_dir / "SOURCES.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    records = [r for r in records if r["dataset"] != args.dataset] + [record]
    manifest.write_text(json.dumps(records, indent=2))
    print(f"Provenance recorded -> {manifest}")


if __name__ == "__main__":
    main()
