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
import sys
import time
from pathlib import Path

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


def evaluate(model, val_loader, device, max_batches=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
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

    torch.manual_seed(tcfg["seed"])

    hw_info = detect_hardware()
    hw_choice = choose_device_and_precision(hw_info, tcfg.get("device", "auto"))
    device = hw_choice["device"]
    print(f"Hardware: {hw_info}")
    print(f"Choix: {hw_choice}")

    model_cfg = load_model_config(tcfg["model_config"])
    model = build_model(model_cfg).to(device)
    param_counts = model.count_parameters()
    print(f"Paramètres entraînables: {param_counts['total_trainable']:,}")
    assert param_counts["total_trainable"] <= 50_000_000, "Budget de 50M dépassé — abandon."

    train_ds = TokenShardDataset(tcfg["train_shard"], seq_len=model_cfg.max_seq_len)
    val_ds = TokenShardDataset(tcfg["val_shard"], seq_len=model_cfg.max_seq_len)

    batch_size = tcfg.get("batch_size", hw_choice["recommended_batch_size"])
    grad_accum = tcfg.get("grad_accum", hw_choice["recommended_grad_accum"])
    num_workers = tcfg.get("num_workers", 0)
    pin_memory = (device == "cuda")

    def make_loader(ds):
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=pin_memory,
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
    ckpt_path = run_dir / "last.ckpt"
    if tcfg.get("resume", False) and ckpt_path.exists():
        start_step, _ = load_checkpoint(ckpt_path, model, optimizer, scheduler)
        print(f"Reprise à step {start_step}")

    model.train()
    train_iter = iter(train_loader)
    current_shard = tcfg["train_shard"]
    best_val = float("inf")
    patience = tcfg.get("early_stopping_patience", None)
    bad_evals = 0
    t0 = time.time()
    tokens_seen = 0

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
        accum_loss = 0.0
        for _ in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / grad_accum).backward()
            accum_loss += loss.item() / grad_accum
            tokens_seen += x.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
        optimizer.step()
        scheduler.step()
        step += 1

        if step % tcfg.get("log_every", 5) == 0 or step == total_steps:
            elapsed = time.time() - t0
            logger.log(
                step=step, train_loss=accum_loss,
                lr=scheduler.get_last_lr()[0],
                grad_norm=float(grad_norm),
                tokens_per_sec=tokens_seen / max(elapsed, 1e-6),
                elapsed_sec=elapsed,
                shard=current_shard,
            )
            print(f"step {step}/{total_steps} loss={accum_loss:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} grad_norm={float(grad_norm):.3f}")

        if step % tcfg.get("eval_every", 20) == 0 or step == total_steps:
            val_loss = evaluate(model, val_loader, device)
            logger.log(step=step, val_loss=val_loss)
            print(f"  [eval] step {step} val_loss={val_loss:.4f}")
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step,
                             extra={"val_loss": val_loss, "model_config": tcfg["model_config"]})
            if val_loss < best_val:
                best_val = val_loss
                bad_evals = 0
                save_checkpoint(run_dir / "best.ckpt", model, optimizer, scheduler, step,
                                 extra={"val_loss": val_loss, "model_config": tcfg["model_config"]})
            else:
                bad_evals += 1
                if patience is not None and bad_evals >= patience:
                    print(f"Early stopping à step {step} (val_loss n'améliore plus depuis {patience} évals)")
                    break

    logger.close()
    print(f"Entraînement terminé. Meilleur val_loss={best_val:.4f}. "
          f"Checkpoints dans {run_dir}")


if __name__ == "__main__":
    main()
