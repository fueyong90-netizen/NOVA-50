"""Tests du chargement de dataset depuis un shard .npy."""
import numpy as np
import pytest
from training.dataset import TokenShardDataset


def test_dataset_windows(tmp_path):
    arr = np.arange(100, dtype=np.uint16)
    path = tmp_path / "shard.npy"
    np.save(path, arr)

    ds = TokenShardDataset(str(path) if str(path).endswith(".npy") else str(path) + ".npy", seq_len=10)
    x, y = ds[0]
    assert x.shape == (10,)
    assert y.shape == (10,)
    # y doit être x décalé d'un cran
    assert (y[:-1] == x[1:]).all()


def test_dataset_raises_if_too_small(tmp_path):
    arr = np.arange(5, dtype=np.uint16)
    path = tmp_path / "tiny.npy"
    np.save(path, arr)
    with pytest.raises(ValueError):
        TokenShardDataset(str(path), seq_len=10)
