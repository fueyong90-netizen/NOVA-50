"""Tests for the judge-proof parameter accounting (conservative total, tied-head note)."""
from model.layers import ModelConfig
from model.transformer import build_model


def test_conservative_total_equals_total_when_untied():
    cfg = ModelConfig(vocab_size=50, dim=16, n_layers=1, n_heads=2, n_kv_heads=2,
                       ffn_hidden_dim=32, max_seq_len=8, weight_tying=False)
    model = build_model(cfg)
    counts = model.count_parameters()
    assert counts["conservative_total_if_head_untied"] == counts["total_trainable"]


def test_conservative_total_exceeds_total_when_tied():
    cfg = ModelConfig(vocab_size=50, dim=16, n_layers=1, n_heads=2, n_kv_heads=2,
                       ffn_hidden_dim=32, max_seq_len=8, weight_tying=True)
    model = build_model(cfg)
    counts = model.count_parameters()
    assert counts["conservative_total_if_head_untied"] == counts["total_trainable"] + counts["embedding"]
    assert counts["output_head"] == 0


def test_both_final_configs_pass_even_conservatively():
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for name in ["model_baseline.yaml", "model_nova.yaml"]:
        with open(root / "configs" / name) as f:
            raw = yaml.safe_load(f)
        raw.pop("name", None)
        cfg = ModelConfig(**raw)
        model = build_model(cfg)
        counts = model.count_parameters()
        assert counts["conservative_total_if_head_untied"] <= 50_000_000, (
            f"{name} exceeds budget under conservative accounting"
        )
