"""Tests: forward/backward pass, causal masking, RoPE, attention, config."""
import torch
from model.layers import ModelConfig, CausalSelfAttention, build_rope_cache, apply_rope
from model.transformer import build_model


def small_cfg(**overrides):
    base = dict(vocab_size=50, dim=32, n_layers=2, n_heads=4, n_kv_heads=4,
                ffn_hidden_dim=64, max_seq_len=16, dropout=0.0,
                norm_type="rmsnorm", ffn_type="swiglu", use_rope=True,
                weight_tying=True, param_sharing=False, shared_group_size=1)
    base.update(overrides)
    return ModelConfig(**base)


def test_forward_shapes():
    cfg = small_cfg()
    model = build_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is None


def test_forward_with_loss():
    cfg = small_cfg()
    model = build_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, y)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_backward_pass_produces_gradients():
    cfg = small_cfg()
    model = build_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_with_grad > 0


def test_causal_masking_future_tokens_dont_leak():
    """Changer un token futur ne doit pas changer les logits des positions antérieures."""
    cfg = small_cfg(dropout=0.0)
    model = build_model(cfg)
    model.eval()
    x1 = torch.randint(0, cfg.vocab_size, (1, 8))
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size  # modifie seulement le DERNIER token
    with torch.no_grad():
        l1, _ = model(x1)
        l2, _ = model(x2)
    # Les logits des positions 0..6 ne doivent pas dépendre du token en position 7
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5)


def test_rope_changes_representation_with_position():
    cos, sin = build_rope_cache(seq_len=8, head_dim=8)
    x = torch.randn(1, 1, 8, 8)
    rotated = apply_rope(x, cos, sin)
    assert rotated.shape == x.shape
    # positions différentes -> rotations différentes -> pas identique à l'entrée
    assert not torch.allclose(rotated, x)


def test_gqa_attention_runs():
    attn = CausalSelfAttention(dim=32, n_heads=4, n_kv_heads=2, use_rope=True, max_seq_len=16)
    x = torch.randn(2, 8, 32)
    out = attn(x)
    assert out.shape == x.shape


def test_weight_tying_shares_tensor():
    cfg = small_cfg(weight_tying=True)
    model = build_model(cfg)
    assert model.lm_head.weight is model.tok_embeddings.weight


def test_param_sharing_reduces_distinct_blocks():
    cfg = small_cfg(n_layers=4, param_sharing=True, shared_group_size=2)
    model = build_model(cfg)
    assert len(model._distinct_blocks) == 2
    assert len(model.blocks) == 4
    # bloc 0 et bloc 2 doivent être le MÊME objet (poids partagés)
    assert model.blocks[0] is model.blocks[2]
    assert model.blocks[1] is model.blocks[3]


def test_generate_produces_correct_length():
    cfg = small_cfg()
    model = build_model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 3))
    out = model.generate(idx, max_new_tokens=5, temperature=1.0)
    assert out.shape[1] == 3 + 5
