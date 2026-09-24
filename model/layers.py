"""
Couches réutilisables pour MODEL A (baseline) et MODEL B (NOVA-50).

Toutes les couches sont implémentées à la main (pas de dépendance à un
bloc Transformer pré-construit) pour garder un contrôle total sur le
nombre de paramètres et pouvoir tester chaque brique isolément.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """RMSNorm (Zhang & Sennrich, 2019). Moins de paramètres que LayerNorm
    (pas de biais, pas de moyenne), et un seul vecteur de poids par couche."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class LayerNorm(nn.Module):
    """LayerNorm standard (avec biais), utilisé par MODEL A (baseline) pour
    servir de point de comparaison face à RMSNorm dans NOVA-50."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


# --------------------------------------------------------------------------- #
# RoPE (Rotary Position Embeddings)
# --------------------------------------------------------------------------- #
def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None):
    assert head_dim % 2 == 0, "head_dim doit être pair pour RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin  # chacun (seq_len, head_dim/2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (batch, heads, seq_len, head_dim)"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    out = torch.stack([rot_x1, rot_x2], dim=-1).flatten(-2)
    return out


# --------------------------------------------------------------------------- #
# Attention causale (avec option Grouped Query Attention)
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        use_rope: bool = True,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads doit être multiple de n_kv_heads (GQA)"
        assert dim % n_heads == 0
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // self.n_kv_heads  # =1 => attention standard (MHA)
        self.use_rope = use_rope
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if use_rope:
            cos, sin = build_rope_cache(max_seq_len, self.head_dim, base=rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope_cos[:t], self.rope_sin[:t]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.n_rep > 1:  # GQA: répéter les têtes K/V pour matcher les têtes Q
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.wo(out)


# --------------------------------------------------------------------------- #
# Feed-forward
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    """FFN de type SwiGLU (Shazeer, 2020) : gate SiLU + projection, 3 matrices.
    À budget de paramètres égal, le hidden_dim doit être réduit (~2/3) par
    rapport à un MLP classique à 2 matrices pour compenser la 3e matrice."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GELUFFN(nn.Module):
    """MLP classique à 2 matrices avec GELU, utilisé par MODEL A (baseline)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# --------------------------------------------------------------------------- #
# Bloc Transformer générique
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        Norm = RMSNorm if cfg.norm_type == "rmsnorm" else LayerNorm
        self.norm1 = Norm(cfg.dim)
        self.attn = CausalSelfAttention(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            use_rope=cfg.use_rope,
            dropout=cfg.dropout,
            max_seq_len=cfg.max_seq_len,
        )
        self.norm2 = Norm(cfg.dim)
        if cfg.ffn_type == "swiglu":
            self.ffn = SwiGLU(cfg.dim, cfg.ffn_hidden_dim, dropout=cfg.dropout)
        else:
            self.ffn = GELUFFN(cfg.dim, cfg.ffn_hidden_dim, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: Optional[int] = None  # None => MHA classique ; < n_heads => GQA
    ffn_hidden_dim: int = 683         # ~ (8/3)*dim pour SwiGLU à param-count comparable à un MLP 4x
    max_seq_len: int = 512
    dropout: float = 0.0
    norm_type: str = "rmsnorm"        # "rmsnorm" | "layernorm"
    ffn_type: str = "swiglu"          # "swiglu" | "gelu"
    use_rope: bool = True
    weight_tying: bool = True
    param_sharing: bool = False       # NOVA-50 : partage de poids entre blocs
    shared_group_size: int = 1        # nb de blocs distincts si param_sharing=True
                                       # (n_layers doit être un multiple de ce nombre)
    init_std: float = 0.02
