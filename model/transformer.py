"""
Implémentation du modèle de langage causal.

Une seule classe `GPTModel` paramétrée par `ModelConfig` sert à la fois pour
MODEL A (baseline) et MODEL B (NOVA-50) : ce qui les distingue, ce sont les
valeurs de configuration (voir configs/model_baseline.yaml et
configs/model_nova.yaml), pas le code. Cela garantit une comparaison
équitable (même code d'entraînement, mêmes chemins).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import ModelConfig, TransformerBlock, RMSNorm, LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)

        if cfg.param_sharing:
            assert cfg.n_layers % cfg.shared_group_size == 0, (
                f"n_layers ({cfg.n_layers}) doit être un multiple de "
                f"shared_group_size ({cfg.shared_group_size}) pour le partage de paramètres"
            )
            n_distinct = cfg.shared_group_size
            distinct_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_distinct)])
            # Chaque position de la pile réutilise un bloc distinct de façon cyclique
            # (schéma "recurrent transformer" / universal-transformer-like).
            self.blocks = nn.ModuleList(
                [distinct_blocks[i % n_distinct] for i in range(cfg.n_layers)]
            )
            self._distinct_blocks = distinct_blocks  # référence pour éviter le garbage collection
        else:
            self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        Norm = RMSNorm if cfg.norm_type == "rmsnorm" else LayerNorm
        self.norm_f = Norm(cfg.dim)

        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight  # même tenseur physique

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = idx.shape
        assert t <= self.cfg.max_seq_len, (
            f"Séquence de longueur {t} > max_seq_len={self.cfg.max_seq_len}"
        )
        x = self.tok_embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                mask = cum_probs - probs > top_p
                sorted_logits[mask] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(
                    1, sorted_idx, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
        return idx

    # ----------------------------------------------------------------- #
    def count_parameters(self) -> dict:
        """Décompte détaillé des paramètres entraînables, sans double-compter
        les tenseurs physiquement partagés (weight tying)."""
        seen: set[int] = set()
        buckets = {
            "embedding": 0,
            "attention": 0,
            "ffn": 0,
            "normalization": 0,
            "output_head": 0,
            "other": 0,
        }

        def add(bucket: str, p: torch.nn.Parameter):
            if id(p) in seen:
                return  # déjà compté (ex: weight tying) -> ne pas compter deux fois
            seen.add(id(p))
            buckets[bucket] += p.numel()

        # Embedding
        add("embedding", self.tok_embeddings.weight)

        # Blocs (uniquement les blocs distincts si param_sharing)
        blocks_to_count = (
            self._distinct_blocks if self.cfg.param_sharing else self.blocks
        )
        for block in blocks_to_count:
            for p in block.attn.parameters():
                add("attention", p)
            for p in block.ffn.parameters():
                add("ffn", p)
            add("normalization", block.norm1.weight)
            if hasattr(block.norm1, "bias"):
                add("normalization", block.norm1.bias)
            add("normalization", block.norm2.weight)
            if hasattr(block.norm2, "bias"):
                add("normalization", block.norm2.bias)

        add("normalization", self.norm_f.weight)
        if hasattr(self.norm_f, "bias"):
            add("normalization", self.norm_f.bias)

        # Output head (compte 0 params neufs si weight tying, car déjà vu comme embedding)
        add("output_head", self.lm_head.weight)

        total = sum(buckets.values())
        buckets["total_trainable"] = total
        return buckets


def build_model(cfg: ModelConfig) -> GPTModel:
    return GPTModel(cfg)
