#!/usr/bin/env python3
"""
Boucle d'entraînement PyTorch pour NOVA-50 / baseline (section 10 du brief).

Supporte : gradient accumulation, gradient clipping, AdamW, LR scheduler avec
warmup, checkpointing + resume, validation périodique, logging JSONL, seed,
early stopping optionnel, throughput (tokens/sec), mémoire.

Usage:
    python training/train.py --config configs/train_toy.yaml
"""
import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.layers import ModelConfig
from model.transformer import build_model
from training.dataset import TokenShardDataset
from training.optimizer import build_optimizer
from training.scheduler import build_lr_lambda
from training.checkpoint import save_checkpoint, load_checkpoint
from training.logger import JSONLLogger
from training.hardware import detect_hardware, choose_device_and_precision
from training.curriculum import build_phase_schedule, shard_for_step


def load_model_config(path: str) -> ModelConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw.pop("name", None)
    return ModelConfig(**raw)


def evaluate(model, val_loader, device, max_batches=20, amp_ctx=None):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if amp_ctx is not None:
                with amp_ctx():
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Fichier YAML de config d'entraînement")
    args = ap.parse_args()

    with open(args.config) as f:
        tcfg = yaml.safe_load(f)

    seed = tcfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    hw_info = detect_hardware()
    hw_choice = choose_device_and_precision(hw_info, tcfg.get("device", "auto"))
    device = hw_choice["device"]
    print(f"Hardware: {hw_info}")
    print(f"Choice: {hw_choice}")

    # --- Mixed precision (fp16 + GradScaler on T4/P100, bf16 on Ampere+, fp32 otherwise) ---
    precision = tcfg.get("precision", "auto")
    if precision == "auto":
        precision = hw_choice["precision"]
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    def amp_ctx():
        return torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp)

    scaler_enabled = use_amp and precision == "fp16"
    try:
        scaler = torch.amp.GradScaler(device, enabled=scaler_enabled)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled and device == "cuda")
    print(f"Precision: {precision} (autocast={use_amp}, grad_scaler={scaler_enabled})")
    model_cfg = load_model_config(tcfg["model_config"])
    model = build_model(model_cfg).to(device)
    param_counts = model.count_parameters()
    print(f"Paramètres entraînables: {param_counts['total_trainable']:,}")
    assert param_counts["total_trainable"] <= 50_000_000, "Budget de 50M dépassé — abandon."

    train_ds = TokenShardDataset(tcfg["train_shard"], seq_len=model_cfg.max_seq_len)
    val_ds = TokenShardDataset(tcfg["val_shard"], seq_len=model_cfg.max_seq_len)

    batch_size = tcfg.get("batch_size", hw_choice["recommended_batch_size"])
    grad_accum = tcfg.get("grad_accum", hw_choice["recommended_grad_accum"])
    num_workers = min(tcfg.get("num_workers", 0), hw_choice["recommended_num_workers"])
    pin_memory = (device == "cuda")

    def make_loader(ds):
        g = torch.Generator()
        g.manual_seed(seed)  # reproducible shuffling
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=pin_memory, generator=g,
        )

    train_loader = make_loader(train_ds)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    total_steps = tcfg["total_steps"]

    # --- Curriculum learning (section 11): real, step-driven shard switching ---
    curriculum_schedule = None
    loader_cache = {tcfg["train_shard"]: train_loader}
    if tcfg.get("curriculum_config"):
        curriculum_schedule = build_phase_schedule(
            tcfg["curriculum_config"], tcfg["mixture_shards"], total_steps
        )
        print("Curriculum phases:")
        for s, e, shard, name in curriculum_schedule:
            print(f"  [{s:>5}-{e:>5}) {name} -> {shard}")

    optimizer = build_optimizer(model, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.01))
    lr_lambda = build_lr_lambda(tcfg.get("warmup_steps", 10), total_steps, tcfg.get("min_lr_ratio", 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run_dir = Path(tcfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(run_dir / "train_log.jsonl")

    start_step = 0
    prev_tokens, prev_elapsed, n_sessions = 0, 0.0, 1
    best_val = float("inf")
    ckpt_path = run_dir / "last.ckpt"
    if tcfg.get("resume", False) and ckpt_path.exists():
        start_step, extra = load_checkpoint(ckpt_path, model, optimizer, scheduler)
        prev_tokens = extra.get("tokens_seen_total", 0)
        prev_elapsed = extra.get("elapsed_total_sec", 0.0)
        n_sessions = extra.get("n_sessions", 1) + 1
        best_val = extra.get("best_val", float("inf"))
        print(f"Resumed at step {start_step} (tokens so far: {prev_tokens:,}, session #{n_sessions})")

    model.train()
    train_iter = iter(train_loader)
    current_shard = tcfg["train_shard"]
    patience = tcfg.get("early_stopping_patience", None)
    bad_evals = 0
    start_utc = datetime.datetime.now(datetime.timezone.utc)
    t0 = time.time()
    tokens_seen = 0  # this session only
    eval_batches = tcfg.get("eval_batches", 20)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def ckpt_extra(val_loss):
        return {
            "val_loss": val_loss, "model_config": tcfg["model_config"],
            "tokens_seen_total": prev_tokens + tokens_seen,
            "elapsed_total_sec": prev_elapsed + (time.time() - t0),
            "n_sessions": n_sessions, "best_val": best_val, "precision": precision, "seed": seed,
        }

    step = start_step
    while step < total_steps:
        if curriculum_schedule is not None:
            target_shard, phase_name = shard_for_step(curriculum_schedule, step)
            if target_shard != current_shard:
                print(f"  [curriculum] step {step}: switching to phase '{phase_name}' ({target_shard})")
                if target_shard not in loader_cache:
                    new_ds = TokenShardDataset(target_shard, seq_len=model_cfg.max_seq_len)
                    loader_cache[target_shard] = make_loader(new_ds)
                train_loader = loader_cache[target_shard]
                train_iter = iter(train_loader)
                current_shard = target_shard

        optimizer.zero_grad(set_to_none=True)
        accum_loss = torch.zeros((), device=device)
        for _ in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with amp_ctx():
                _, loss = model(x, y)
            scaler.scale(loss / grad_accum).backward()
            accum_loss += loss.detach() / grad_accum  # no GPU sync per micro-batch
            tokens_seen += x.numel()

        scaler.unscale_(optimizer)  # needed so clipping sees the true gradient scale
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

        if step % tcfg.get("log_every", 5) == 0 or step == total_steps:
            elapsed = time.time() - t0
            loss_val = float(accum_loss)
            peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
            logger.log(
                step=step, train_loss=loss_val,
                lr=scheduler.get_last_lr()[0],
                grad_norm=float(grad_norm),
                tokens_per_sec=tokens_seen / max(elapsed, 1e-6),
                elapsed_sec=prev_elapsed + elapsed,
                tokens_total=prev_tokens + tokens_seen,
                peak_mem_gb=peak_gb,
                shard=current_shard,
            )
            print(f"step {step}/{total_steps} loss={loss_val:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} grad_norm={float(grad_norm):.3f} "
                  f"tok/s={tokens_seen / max(elapsed, 1e-6):,.0f}")

        if step % tcfg.get("eval_every", 20) == 0 or step == total_steps:
            val_loss = evaluate(model, val_loader, device, max_batches=eval_batches, amp_ctx=amp_ctx)
            logger.log(step=step, val_loss=val_loss)
            print(f"  [eval] step {step} val_loss={val_loss:.4f}")
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step, extra=ckpt_extra(val_loss))
            if improved:
                bad_evals = 0
                save_checkpoint(run_dir / "best.ckpt", model, optimizer, scheduler, step,
                                 extra=ckpt_extra(val_loss))
            else:
                bad_evals += 1
                if patience is not None and bad_evals >= patience:
                    print(f"Early stopping à step {step} (val_loss n'améliore plus depuis {patience} évals)")
                    break

    logger.close()
    end_utc = datetime.datetime.now(datetime.timezone.utc)
    session_sec = time.time() - t0
    total_tokens = prev_tokens + tokens_seen
    total_sec = prev_elapsed + session_sec
    import math
    summary = {
        "status": "COMPLETED" if step >= total_steps else "STOPPED_EARLY",
        "start_utc": start_utc.isoformat(), "end_utc": end_utc.isoformat(),
        "n_sessions": n_sessions,
        "wall_clock_sec_this_session": round(session_sec, 1),
        "wall_clock_sec_total": round(total_sec, 1),
        "wall_clock_hours_total": round(total_sec / 3600, 3),
        "steps_completed": step, "steps_configured": total_steps,
        "tokens_processed_total": int(total_tokens),
        "avg_tokens_per_sec_total": round(total_tokens / max(total_sec, 1e-6), 1),
        "micro_batch_size": batch_size, "grad_accum": grad_accum,
        "sequence_length": model_cfg.max_seq_len,
        "effective_batch_tokens": batch_size * grad_accum * model_cfg.max_seq_len,
        "learning_rate_peak": tcfg["lr"], "weight_decay": tcfg.get("weight_decay", 0.01),
        "warmup_steps": tcfg.get("warmup_steps", 10), "optimizer": "AdamW",
        "scheduler": "linear warmup + cosine decay",
        "precision": precision, "seed": seed,
        "best_val_loss": best_val,
        "best_val_perplexity": math.exp(best_val) if best_val < 50 else None,
        "peak_cuda_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None,
        "hardware": hw_info,
        "parameters": param_counts,
        "model_config_file": tcfg["model_config"], "train_config_file": args.config,
        "train_shard": tcfg["train_shard"],
        "from_scratch": True,
    }
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Training finished. Best val_loss={best_val:.4f}. Checkpoints + run_summary.json in {run_dir}")


if __name__ == "__main__":
    main()
