"""Dataset qui mémoire-mappe un shard .npy de tokens et sert des fenêtres
(input, target) de longueur fixe pour l'entraînement causal."""
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenShardDataset(Dataset):
    def __init__(self, npy_path: str, seq_len: int):
        self.data = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len
        n_windows = max(0, (len(self.data) - 1) // seq_len)
        self.n_windows = n_windows
        if n_windows == 0:
            raise ValueError(
                f"{npy_path}: seulement {len(self.data)} tokens, "
                f"pas assez pour une fenêtre de {seq_len+1} (corpus jouet trop petit ? "
                f"réduire seq_len dans la config)."
            )

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y
