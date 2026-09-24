"""Tests: chargement de config YAML, et reproductibilité (même seed -> mêmes poids initiaux)."""
import yaml
import torch
from pathlib import Path
from model.layers import ModelConfig
from model.transformer import build_model

ROOT = Path(__file__).resolve().parent.parent


def test_all_model_configs_load():
    for p in sorted((ROOT / "configs").glob("model_*.yaml")):
        with open(p) as f:
            raw = yaml.safe_load(f)
        raw.pop("name", None)
        cfg = ModelConfig(**raw)
        model = build_model(cfg)
        assert sum(p.numel() for p in model.parameters()) > 0


def test_same_seed_same_init():
    cfg = ModelConfig(vocab_size=20, dim=8, n_layers=1, n_heads=2, n_kv_heads=2,
                       ffn_hidden_dim=16, max_seq_len=4, weight_tying=True)
    torch.manual_seed(123)
    m1 = build_model(cfg)
    torch.manual_seed(123)
    m2 = build_model(cfg)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)
