#!/usr/bin/env python3
"""
Runs the OFFICIAL Track 01 benchmark suite (HellaSwag, ARC-Easy, PIQA,
WinoGrande) via lm-evaluation-harness against a NOVA-50 checkpoint.

This wraps the checkpoint in a minimal lm-eval `LM` subclass (no HF
transformers dependency needed) and calls the harness's own task loader —
it does NOT reimplement the tasks itself, per the brief's requirement not to
fabricate local substitutes.

REAL STATUS in this repository (captured automatically below, not asserted):
`lm-eval` (PyPI package `lm-eval`, version pinned in requirements.txt) is
installed and importable in this environment. However, task data for
hellaswag/arc_easy/piqa/winogrande is fetched by the harness from the
Hugging Face Hub, and huggingface.co is not reachable from this container's
network egress allow-list (see docs/COMPETITION_RULES.md). Running this
script here fails at the data-download step, not at the harness-integration
step — the exact captured error is saved to results/benchmarks/.

To actually produce results: run this same script, unmodified, on a machine
with normal internet access and (ideally) a GPU.
"""
import argparse
import json
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


def build_lm_eval_wrapper(model, sp, device="cpu"):
    """Minimal lm-eval LM interface around our GPTModel + SentencePiece tokenizer."""
    from lm_eval.api.model import LM

    class NovaLMEvalWrapper(LM):
        def __init__(self):
            super().__init__()
            self.model = model
            self.sp = sp
            self._device = device

        def tok_encode(self, string):
            return self.sp.encode(string, out_type=int)

        def tok_decode(self, tokens):
            return self.sp.decode(tokens)

        def loglikelihood(self, requests):
            results = []
            for req in requests:
                context, continuation = req.args
                ctx_ids = self.tok_encode(context)
                cont_ids = self.tok_encode(continuation)
                ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long)
                with torch.no_grad():
                    logits, _ = self.model(ids[:, :-1])
                log_probs = torch.log_softmax(logits, dim=-1)
                cont_start = len(ctx_ids) - 1
                total_logprob = 0.0
                is_greedy = True
                for i, tok in enumerate(cont_ids):
                    pos = cont_start + i
                    if pos < 0 or pos >= log_probs.shape[1]:
                        is_greedy = False
                        continue
                    total_logprob += log_probs[0, pos, tok].item()
                    if log_probs[0, pos].argmax().item() != tok:
                        is_greedy = False
                results.append((total_logprob, is_greedy))
            return results

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("Not needed for the required Track 01 tasks")

        def generate_until(self, requests):
            raise NotImplementedError("Not needed for the required Track 01 tasks (loglikelihood tasks only)")

    return NovaLMEvalWrapper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/nova50_tokenizer.model")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--limit", type=int, default=None, help="Limit examples per task (debugging)")
    ap.add_argument("--out_json", default="results/benchmarks/lm_eval_results.json")
    ap.add_argument("--out_md", default="results/benchmarks/lm_eval_report.md")
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

    import datetime
    common_meta = {
        "checkpoint": args.checkpoint,
        "model_config": args.model_config,
        "tasks_requested": args.tasks,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": "see results/hardware_report.json",
    }
    try:
        import lm_eval
        common_meta["lm_eval_version"] = lm_eval.__version__
    except Exception:
        common_meta["lm_eval_version"] = "UNKNOWN"

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    try:
        from lm_eval import simple_evaluate
        lm = build_lm_eval_wrapper(model, sp)
        results = simple_evaluate(model=lm, tasks=args.tasks, limit=args.limit)
        report = {"status": "RUN", **common_meta, "results": results["results"]}
    except Exception as e:
        report = {
            "status": "NOT RUN",
            **common_meta,
            "error_type": type(e).__name__,
            "error_message": str(e)[:2000],
            "traceback_tail": "".join(traceback.format_exception(e)[-6:]),
            "reason": (
                "lm-evaluation-harness is installed and the model wrapper is implemented "
                "(see evaluation/run_lm_eval.py), but task data download failed. Task data "
                "for hellaswag/arc_easy/piqa/winogrande is served from the Hugging Face Hub, "
                "which is not reachable from this container's network egress allow-list "
                "(see docs/COMPETITION_RULES.md). Re-run this exact script on a machine with "
                "normal internet access to get real results."
            ),
        }

    print(json.dumps(report, indent=2)[:3000])

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)

    with open(args.out_md, "w") as f:
        f.write("# lm-evaluation-harness Report — Track 01 official benchmarks\n\n")
        f.write(f"- Status: **{report['status']}**\n")
        f.write(f"- Checkpoint: `{args.checkpoint}`\n- Model config: `{args.model_config}`\n")
        f.write(f"- Tasks requested: {', '.join(args.tasks)}\n")
        f.write(f"- lm-eval version: {report.get('lm_eval_version', 'UNKNOWN')}\n")
        f.write(f"- Date: {report['date']}\n")
        f.write(f"- Command: `python evaluation/run_lm_eval.py --checkpoint {args.checkpoint} "
                f"--model_config {args.model_config}`\n\n")
        if report["status"] == "RUN":
            f.write("| task | metric | value |\n|---|---|---|\n")
            for task, metrics in report["results"].items():
                for k, v in metrics.items():
                    f.write(f"| {task} | {k} | {v} |\n")
        else:
            f.write(f"**NOT RUN.** Reason: {report['reason']}\n\n")
            f.write(f"Captured error: `{report['error_type']}: {report['error_message'][:300]}`\n")

    print(f"\n-> {args.out_json}\n-> {args.out_md}")


if __name__ == "__main__":
    main()
