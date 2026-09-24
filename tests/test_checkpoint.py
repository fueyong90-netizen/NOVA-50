"""Test: sauvegarde/chargement de checkpoint restaure exactement l'état du modèle."""
import torch
from model.layers import ModelConfig
from model.transformer import build_model
from training.checkpoint import save_checkpoint, load_checkpoint
from training.optimizer import build_optimizer


def test_checkpoint_roundtrip(tmp_path):
    cfg = ModelConfig(vocab_size=50, dim=16, n_layers=1, n_heads=2, n_kv_heads=2,
                       ffn_hidden_dim=32, max_seq_len=8, weight_tying=True)
    model = build_model(cfg)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.0)

    x = torch.randint(0, cfg.vocab_size, (2, 4))
    y = torch.randint(0, cfg.vocab_size, (2, 4))
    _, loss = model(x, y)
    loss.backward()
    opt.step()

    before = {k: v.clone() for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "test.ckpt"
    save_checkpoint(ckpt_path, model, opt, None, step=42)

    model2 = build_model(cfg)
    step, extra = load_checkpoint(ckpt_path, model2)
    assert step == 42
    for k in before:
        assert torch.allclose(before[k], model2.state_dict()[k]), f"Mismatch on {k}"
