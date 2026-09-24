"""Construction de l'optimiseur AdamW avec weight decay séparé (pas de decay
sur les biais et les poids de normalisation, convention standard GPT)."""
import torch


def build_optimizer(model, lr: float, weight_decay: float, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)
