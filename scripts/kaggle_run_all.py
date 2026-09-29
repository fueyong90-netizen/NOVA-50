#!/usr/bin/env python3
"""
One-shot, resumable orchestrator for running the NOVA-50 pipeline on a fresh
Kaggle GPU session: data -> tokenizer -> shards -> decontamination -> short
profiling -> full training launch.

Designed to be safe to re-run: every phase checks whether its own output
already exists (and looks valid) before redoing work, so if a Kaggle session
disconnects mid-training, re-running this exact script resumes instead of
starting over (training itself resumes via train_production_nova.yaml's
`resume: true`).

Usage (from the repo root, i.e. /kaggle/working/NOVA-50):
    python scripts/kaggle_run_all.py --hours 8

Every step's real stdout is streamed live and also written under
results/kaggle_run_all/<step>.log so nothing is hidden if something fails.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG_DIR = ROOT / "results" / "kaggle_run_all"


def run(step_name, cmd, cwd=ROOT, check=True):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step_name}.log"
    print(f"\n{'=' * 70}\nSTEP: {step_name}\n$ {' '.join(cmd)}\n{'=' * 70}")
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()
    if check and proc.returncode != 0:
        print(f"\n!!! STEP '{step_name}' FAILED (exit {proc.returncode}) — see {log_path}")
        sys.exit(proc.returncode)
    return proc.returncode


def phase_done(marker: Path) -> bool:
    return marker.exists()


def mark_done(marker: Path, info=None):
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(info or {}, indent=2))


def pick_best_vocab_size(study_json_path, min_size=None):
    """Picks the OK candidate with the highest measured chars/token (best
    compression). Never picks a FAILED size. Returns None if all failed."""
    rows = json.loads(Path(study_json_path).read_text())
    ok = [r for r in rows if r["status"] == "OK"]
    if min_size:
        ok = [r for r in ok if r["vocab_size"] >= min_size] or ok
    if not ok:
        return None
    return max(ok, key=lambda r: r["chars_per_token"])["vocab_size"]


def patch_model_vocab_size(config_path, vocab_size):
    text = Path(config_path).read_text()
    new_text, n = re.subn(r"(?m)^vocab_size:\s*\d+", f"vocab_size: {vocab_size}", text)
    assert n == 1, f"expected exactly one vocab_size: line in {config_path}, found {n}"
    Path(config_path).write_text(new_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tinystories", choices=["tinystories", "fineweb_edu"])
    ap.add_argument("--n_examples", type=int, default=50000)
    ap.add_argument("--vocab_sizes", type=int, nargs="+", default=[8000, 16000, 24000, 32000])
    ap.add_argument("--hours", type=float, required=True, help="Remaining GPU session time, e.g. 8")
    ap.add_argument("--safety_margin", type=float, default=0.85)
    ap.add_argument("--profile_seconds", type=int, default=180)
    ap.add_argument("--skip_lm_eval", action="store_true", help="Skip official benchmarks after training")
    args = ap.parse_args()

    markers = ROOT / "results" / "kaggle_run_all" / "markers"
    prod_final = ROOT / "data/production/final"
    train_jsonl = prod_final / "train.jsonl"

    # --- 1. Production data ---
    if not train_jsonl.exists():
        run("download", [PY, "data/scripts/download_production_data.py",
                          "--dataset", args.dataset, "--n_examples", str(args.n_examples)])
        run("filter", [PY, "data/scripts/filter_production_data.py"])
        run("dedupe", [PY, "data/scripts/dedupe_production_data.py"])
        run("build_final_corpus", [PY, "data/scripts/build_final_corpus.py"])
    else:
        print(f"[skip] production data already present: {train_jsonl}")

    # --- 2. Vocabulary size study + tokenizer ---
    vs_marker = markers / "vocab_and_tokenizer.json"
    tok_model = ROOT / "tokenizer/artifacts/nova50_tokenizer.model"
    if phase_done(vs_marker):
        chosen_vocab = json.loads(vs_marker.read_text())["chosen_vocab_size"]
        print(f"[skip] tokenizer already trained at vocab_size={chosen_vocab}")
    else:
        run("vocab_size_study", [PY, "tokenizer/vocab_size_study.py",
                                  "--input", str(train_jsonl), "--sizes", *map(str, args.vocab_sizes)])
        chosen_vocab = pick_best_vocab_size(ROOT / "results/vocab_size_study.json")
        if chosen_vocab is None:
            print("!!! Every candidate vocab size FAILED on this corpus (too little unique text). "
                  "Falling back to 8000 and letting SentencePiece cap it automatically if needed.")
            chosen_vocab = 8000
        print(f"Chosen vocab_size = {chosen_vocab} (best measured chars/token)")
        run("train_tokenizer", [PY, "tokenizer/train_tokenizer.py",
                                 "--input", str(train_jsonl), "--vocab_size", str(chosen_vocab)])
        patch_model_vocab_size(ROOT / "configs/model_nova.yaml", chosen_vocab)
        mark_done(vs_marker, {"chosen_vocab_size": chosen_vocab})

    run("param_budget_check", [PY, "scripts/count_parameters.py", "--config", "configs/model_nova.yaml",
                                "--json", "results/parameter_report.json", "--md", "results/parameter_report.md"])

    # --- 3. Encode shards ---
    shards_marker = markers / "shards.json"
    shard_dir = ROOT / "data/processed/shards"
    if not phase_done(shards_marker):
        for split in ["train", "val", "test"]:
            run(f"encode_{split}", [PY, "tokenizer/encode.py",
                                    "--input", str(prod_final / f"{split}.jsonl"),
                                    "--out", str(shard_dir / f"production_{split}_raw.npy")])
        mark_done(shards_marker)
    else:
        print("[skip] shards already encoded")

    # --- 4. Decontamination against official benchmarks ---
    decontam_marker = markers / "decontam.json"
    final_train_shard = shard_dir / "production_train.npy"
    if not phase_done(decontam_marker):
        run("dump_eval_text", [PY, "evaluation/dump_eval_text.py"], check=False)  # may be NOT RUN if HF is blocked
        eval_dir = ROOT / "data/eval_text"
        decontam_out = prod_final / "train_decontam.jsonl"
        if eval_dir.exists() and list(eval_dir.glob("*.jsonl")):
            run("contamination_check", [PY, "evaluation/benchmark_contamination.py",
                                        "--eval_dir", str(eval_dir), "--train_jsonl", str(train_jsonl),
                                        "--decontaminate_out", str(decontam_out)])
            run("encode_decontam_train", [PY, "tokenizer/encode.py", "--input", str(decontam_out),
                                          "--out", str(final_train_shard)])
        else:
            print("[warn] no eval text available (HF unreachable?) — training shard is NOT decontaminated. "
                  "Re-run this script once internet/HF access is confirmed to fix this before the real run.")
            run("copy_shard_as_is", [PY, "-c",
                f"import shutil; shutil.copy('{shard_dir / 'production_train_raw.npy'}', '{final_train_shard}')"])
        for split in ["val", "test"]:
            raw = shard_dir / f"production_{split}_raw.npy"
            if raw.exists() and not (shard_dir / f"production_{split}.npy").exists():
                raw.rename(shard_dir / f"production_{split}.npy")
        mark_done(decontam_marker)
    else:
        print("[skip] decontamination already done")

    # --- 5. Short profiling run -> real step budget for THIS session ---
    profile_dir = ROOT / "experiments/nova/production_profile"
    if not (profile_dir / "train_log.jsonl").exists():
        run("profile_training", [PY, "-c", f"""
import subprocess, sys, time
p = subprocess.Popen([sys.executable, 'training/train.py', '--config', 'configs/train_profile_nova.yaml'])
time.sleep({args.profile_seconds})
p.terminate(); p.wait()
"""], check=False)
    log = profile_dir / "train_log.jsonl"
    if not log.exists():
        print(f"[retry] no log after {args.profile_seconds}s (slow/CPU-only machine?) — "
              f"retrying once with {args.profile_seconds * 3}s")
        run("profile_training_retry", [PY, "-c", f"""
import subprocess, sys, time
p = subprocess.Popen([sys.executable, 'training/train.py', '--config', 'configs/train_profile_nova.yaml'])
time.sleep({args.profile_seconds * 3})
p.terminate(); p.wait()
"""], check=False)
    if not log.exists():
        print(f"!!! Still no log in {profile_dir} after two attempts — check "
              f"results/kaggle_run_all/profile_training*.log by hand, this usually means "
              f"training/train.py itself errored (not just 'slow').")
        sys.exit(1)

    budget = subprocess.run([PY, "scripts/estimate_training_budget.py", "--train_log", str(log),
                             "--hours", str(args.hours), "--safety_margin", str(args.safety_margin)],
                            cwd=ROOT, capture_output=True, text=True)
    print(budget.stdout)
    m = re.search(r"total_steps.*?:\s*([\d,]+)", budget.stdout)
    if not m:
        print("!!! Could not parse recommended total_steps from estimator output; defaulting to 2000")
        total_steps = 2000
    else:
        total_steps = int(m.group(1).replace(",", ""))
    print(f"-> Using total_steps={total_steps} for the full run")

    prod_cfg_text = (ROOT / "configs/train_production_nova.yaml").read_text()
    prod_cfg_text = re.sub(r"(?m)^total_steps:.*", f"total_steps: {total_steps}", prod_cfg_text)
    prod_cfg_text = re.sub(r"^train_shard:.*", f"train_shard: data/processed/shards/production_train.npy", prod_cfg_text, flags=re.M)
    prod_cfg_text = re.sub(r"^val_shard:.*", f"val_shard: data/processed/shards/production_val.npy", prod_cfg_text, flags=re.M)
    (ROOT / "configs/train_production_nova.yaml").write_text(prod_cfg_text)

    # --- 6. Full training (resumable: safe to re-run this whole script after a disconnect) ---
    run("full_training", [PY, "training/train.py", "--config", "configs/train_production_nova.yaml"])

    best_ckpt = ROOT / "experiments/nova/production_run/best.ckpt"
    if not args.skip_lm_eval and best_ckpt.exists():
        run("lm_eval_smoke", [PY, "evaluation/run_lm_eval.py", "--checkpoint", str(best_ckpt),
                              "--model_config", "configs/model_nova.yaml", "--tasks", "hellaswag", "--limit", "20"],
            check=False)
        run("lm_eval_full", [PY, "evaluation/run_lm_eval.py", "--checkpoint", str(best_ckpt),
                             "--model_config", "configs/model_nova.yaml"], check=False)
        run("wikitext103", [PY, "evaluation/wikitext103.py", "--checkpoint", str(best_ckpt),
                            "--model_config", "configs/model_nova.yaml", "--download"], check=False)
        run("hardware_report", [PY, "scripts/hardware_report.py"], check=False)
        run("compute_report", [PY, "scripts/compute_report.py", "--model_config", "configs/model_nova.yaml",
                               "--train_log", "experiments/nova/production_run/train_log.jsonl",
                               "--out", "results/compute_report_production.json"], check=False)

    print("\n" + "=" * 70)
    print("ALL PHASES ATTEMPTED. Check results/kaggle_run_all/*.log for any step marked NOT RUN/FAILED,")
    print("and results/benchmarks/ for the official benchmark outputs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
