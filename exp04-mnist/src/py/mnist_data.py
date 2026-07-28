from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class Quadratic(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class MnistDataset(Dataset):
    def __init__(self, fpath_h5: str, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> None:
        with h5py.File(fpath_h5, "r") as h5:
            X      = np.asarray(h5["X"])  # (N, 784) float32
            self.y = np.asarray(h5["y"])  # (N,) int64
        self.classes = [str(i) for i in range(10)]
        self.mean = X.mean(axis=0) if mean is None else mean
        self.std  = (X.std(axis=0) + 1e-6) if std is None else std
        self.X = ((X - self.mean) / self.std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.X[idx]), self.y[idx]


def eval_acc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            correct += (model(X_batch).argmax(1) == y_batch).sum().item()
            total   += len(y_batch)
    return correct / total
