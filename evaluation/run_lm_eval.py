#!/usr/bin/env python3
"""
Official Track 01 benchmark runner: HellaSwag, ARC-Easy, PIQA, WinoGrande through
lm-evaluation-harness (zero-shot), against a NOVA-50 checkpoint.

The harness loads the tasks itself; this file only provides the model adapter
(`NovaLMEvalWrapper`): a from-scratch NOVA-50 checkpoint + its SentencePiece
tokenizer exposed through lm-eval's `LM` interface (log-likelihood scoring,
batched, GPU-aware). Nothing is re-implemented locally.

Conventions of the adapter (documented for reviewers):
  * every scored sequence starts with the EOS token, mimicking the way documents
    are packed during training (documents are separated by EOS);
  * sequences longer than the model context are left-truncated;
  * newlines in prompts ("Question: ...\\nAnswer:") are treated as whitespace by the
    tokenizer, exactly as in the training corpus.

Outputs: results/benchmarks/lm_eval_results.json (+ lm_eval_final.json/.txt, lm_eval_report.md)
Needs internet access to huggingface.co the first time (task datasets).
"""
import argparse
import datetime
import json
import platform
import sys
import traceback
from pathlib import Path

import torch
import yaml
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.layers import ModelConfig
from model.transformer import build_model
from training.checkpoint import load_checkpoint

TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande"]


def build_lm_eval_wrapper(model, sp, device="cpu", batch_size=32):
    from lm_eval.api.model import LM

    max_len = model.cfg.max_seq_len
    eos = sp.eos_id()

    class NovaLMEvalWrapper(LM):
        def __init__(self):
            super().__init__()
            self.model = model
            self.sp = sp
            self._device = device
            self.batch_size_ = batch_size

        def tok_encode(self, string):
            return self.sp.encode(string, out_type=int)

        def _score_batch(self, items):
            """items: list of (full_ids, cont_len). Returns list of (logprob, is_greedy)."""
            L = max(len(f) - 1 for f, _ in items)
            inp = torch.zeros(len(items), L, dtype=torch.long)
            for r, (full, _) in enumerate(items):
                inp[r, :len(full) - 1] = torch.tensor(full[:-1])
            with torch.no_grad():
                logits, _ = self.model(inp.to(self._device))
            out = []
            for r, (full, cont_len) in enumerate(items):
                n_in = len(full) - 1
                lp = torch.log_softmax(logits[r, n_in - cont_len:n_in].float(), dim=-1)
                tgt = torch.tensor(full[-cont_len:], device=lp.device)
                tok_lp = lp.gather(1, tgt[:, None]).squeeze(1)
                out.append((float(tok_lp.sum()), bool((lp.argmax(-1) == tgt).all())))
            return out

        def loglikelihood(self, requests, disable_tqdm=False):
            prepared = []
            for i, req in enumerate(requests):
                context, continuation = req.args
                ctx_ids, cont_ids = self.tok_encode(context), self.tok_encode(continuation)
                if not cont_ids:
                    prepared.append((i, None, 0))
                    continue
                full = [eos] + ctx_ids + cont_ids
                if len(full) > max_len + 1:  # left-truncate the context
                    full = full[-(max_len + 1):]
                prepared.append((i, full, min(len(cont_ids), len(full) - 1)))
            results = [None] * len(prepared)
            todo = sorted([p for p in prepared if p[1] is not None], key=lambda p: -len(p[1]))
            for i, full, cl in prepared:
                if full is None:
                    results[i] = (0.0, True)
            for s in range(0, len(todo), self.batch_size_):
                chunk = todo[s:s + self.batch_size_]
                scored = self._score_batch([(f, cl) for _, f, cl in chunk])
                for (i, _, _), r in zip(chunk, scored):
                    results[i] = r
            return results

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError("Not needed for the required Track 01 tasks")

        def generate_until(self, requests, disable_tqdm=False):
            raise NotImplementedError("Not needed for the required Track 01 tasks (log-likelihood tasks only)")

    return NovaLMEvalWrapper()


def load_nova(checkpoint, model_config, device):
    with open(model_config) as f:
        raw = yaml.safe_load(f)
    raw.pop("name", None)
    cfg = ModelConfig(**raw)
    model = build_model(cfg)
    load_checkpoint(checkpoint, model)
    return model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/nova50_tokenizer.model")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--limit", type=int, default=None, help="Limit examples per task (smoke tests only)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--include_path", default=None, help="Extra lm-eval task YAML folder (offline tests)")
    ap.add_argument("--out_json", default="results/benchmarks/lm_eval_results.json")
    ap.add_argument("--out_md", default="results/benchmarks/lm_eval_report.md")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, cfg = load_nova(args.checkpoint, args.model_config, device)
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    n_params = model.count_parameters()["total_trainable"]

    meta = {
        "checkpoint": args.checkpoint, "model_config": args.model_config, "tokenizer": args.tokenizer,
        "trainable_parameters": n_params, "tasks_requested": args.tasks, "limit": args.limit,
        "num_fewshot": 0, "device": device, "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": "python evaluation/run_lm_eval.py " + " ".join(sys.argv[1:]),
        "smoke_test": args.limit is not None,
    }
    try:
        import lm_eval
        meta["lm_eval_version"] = lm_eval.__version__
    except Exception:
        meta["lm_eval_version"] = "UNKNOWN"

    table = ""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.tasks import TaskManager
        lm = build_lm_eval_wrapper(model, sp, device, args.batch_size)
        tm = TaskManager(include_path=args.include_path) if args.include_path else None
        res = simple_evaluate(model=lm, tasks=args.tasks, num_fewshot=0, limit=args.limit, task_manager=tm)
        report = {"status": "RUN", **meta, "results": res["results"],
                  "n_samples": res.get("n-samples"), "versions": res.get("versions")}
        try:
            from lm_eval.utils import make_table
            table = make_table(res)
        except Exception:
            table = json.dumps(res["results"], indent=2, default=str)
    except Exception as e:
        report = {"status": "NOT RUN", **meta, "error_type": type(e).__name__,
                  "error_message": str(e)[:2000],
                  "traceback_tail": "".join(traceback.format_exception(e)[-6:])}

    print(json.dumps(report, indent=2, default=str)[:3500])
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str))

    md = ["# lm-evaluation-harness Report — Track 01 official benchmarks\n",
          f"- Status: **{report['status']}**" + ("  (SMOKE TEST — not a final result)" if meta["smoke_test"] else ""),
          f"- Checkpoint: `{args.checkpoint}`  |  config: `{args.model_config}`  |  tokenizer: `{args.tokenizer}`",
          f"- Trainable parameters: {n_params:,}",
          f"- lm-eval version: {meta['lm_eval_version']}  |  zero-shot  |  device: {device} ({meta['gpu']})",
          f"- Date (UTC): {meta['date_utc']}", f"- Command: `{meta['command']}`\n"]
    if report["status"] == "RUN":
        md.append("```\n" + table + "\n```")
        if report.get("n_samples"):
            md.append("\nExamples evaluated: " + ", ".join(
                f"{t}={v.get('effective', v) if isinstance(v, dict) else v}" for t, v in report["n_samples"].items()))
        if not meta["smoke_test"]:
            (out_json.parent / "lm_eval_final.json").write_text(json.dumps(report, indent=2, default=str))
            (out_json.parent / "lm_eval_final.txt").write_text(table)
    else:
        md.append(f"**NOT RUN.** `{report['error_type']}: {report['error_message'][:400]}`")
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(f"\n-> {out_json}\n-> {args.out_md}")


if __name__ == "__main__":
    main()
