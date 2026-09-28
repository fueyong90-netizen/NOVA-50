"""Regression tests for the AMP/precision paths and checkpoint resume added when
preparing the project for real GPU training (Kaggle)."""
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _toy_config(tmp_path, run_name, **overrides):
    with open(ROOT / "configs/train_toy_nova.yaml") as f:
        cfg = yaml.safe_load(f)
    defaults = dict(total_steps=8, eval_every=4, log_every=2, warmup_steps=2,
                     run_dir=str(tmp_path / run_name))
    defaults.update(overrides)
    cfg.update(defaults)
    p = tmp_path / f"{run_name}.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def _run(config_path):
    r = subprocess.run([sys.executable, str(ROOT / "training/train.py"), "--config", str(config_path)],
                        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r


def test_fp32_precision_runs_and_writes_summary(tmp_path):
    cfg = _toy_config(tmp_path, "fp32run", precision="fp32")
    _run(cfg)
    summary = json.loads((tmp_path / "fp32run" / "run_summary.json").read_text())
    assert summary["precision"] == "fp32"
    assert summary["status"] == "COMPLETED"
    assert summary["tokens_processed_total"] > 0
    assert summary["from_scratch"] is True


def test_fp16_precision_with_grad_scaler_runs(tmp_path):
    # On CPU this still exercises the GradScaler(enabled=False) code path without crashing;
    # correctness of the CUDA path is exercised for real on the GPU machine (see README).
    cfg = _toy_config(tmp_path, "fp16run", precision="fp16")
    _run(cfg)
    summary = json.loads((tmp_path / "fp16run" / "run_summary.json").read_text())
    assert summary["precision"] == "fp16"


def test_resume_continues_step_count_and_accumulates_tokens(tmp_path):
    cfg = _toy_config(tmp_path, "resume_run", total_steps=4, resume=True)
    _run(cfg)
    first = json.loads((tmp_path / "resume_run" / "run_summary.json").read_text())
    assert first["steps_completed"] == 4

    cfg2 = _toy_config(tmp_path, "resume_run", total_steps=8, resume=True)
    _run(cfg2)
    second = json.loads((tmp_path / "resume_run" / "run_summary.json").read_text())
    assert second["steps_completed"] == 8
    assert second["n_sessions"] == 2
    assert second["tokens_processed_total"] > first["tokens_processed_total"]


def test_checkpoint_write_is_atomic_no_tmp_left_behind(tmp_path):
    cfg = _toy_config(tmp_path, "atomicrun")
    _run(cfg)
    run_dir = tmp_path / "atomicrun"
    assert (run_dir / "best.ckpt").exists()
    assert not any(run_dir.glob("*.tmp"))
