import glob
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.layers import ModelConfig
from model.transformer import build_model

BUDGET = 50_000_000


def _all_model_configs():
    return sorted(glob.glob(str(ROOT / "configs" / "model_*.yaml")))


@pytest.mark.parametrize("config_path", _all_model_configs())
def test_model_under_budget(config_path):
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw.pop("name", None)
    cfg = ModelConfig(**raw)
    model = build_model(cfg)
    counts = model.count_parameters()
    total = counts["total_trainable"]
    assert total <= BUDGET, (
        f"{config_path}: {total:,} paramètres entraînables > budget {BUDGET:,}"
    )


def test_at_least_one_config_found():
    assert len(_all_model_configs()) >= 2, "Aucune config de modèle trouvée dans configs/"
