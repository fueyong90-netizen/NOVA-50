"""Sauvegarde / reprise de checkpoints (modèle + optimiseur + scheduler + état
d'entraînement), pour que `train.py --resume` reprenne exactement où il en était."""
import torch
from pathlib import Path


def save_checkpoint(path, model, optimizer, scheduler, step, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["step"], ckpt.get("extra", {})
