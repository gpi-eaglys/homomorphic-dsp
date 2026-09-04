"""
MNIST feature loading, shared by exp05's scripts.

Mirrors exp03/exp04's `MnistDataset`: reads the repo-wide `build/fea/mnist-*.h5`
artifacts and applies per-feature (mean, std) standardisation. Test-set statistics
are always the *train* set's — pass them in via `mean=`/`std=`.
"""

import os
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import BLD_DIR

FEA_DIR = os.path.join(BLD_DIR, "fea")

_EXTRACT_HINT = (
    "MNIST features not found: {path}\n"
    "Generate them once via exp03's extractor (repo-wide artifact, not exp05-specific):\n"
    "    PYTHONPATH=../exp03-mnist/src/py .venv/bin/python "
    "../exp03-mnist/src/py/extract_features.py"
)


def fea_path(split: str) -> str:
    """Path of the h5 features for `split` ('train' or 'test'); errors if absent."""
    fpath = os.path.join(FEA_DIR, f"mnist-{split}.h5")
    if not os.path.exists(fpath):
        raise FileNotFoundError(_EXTRACT_HINT.format(path=fpath))
    return fpath


class MnistDataset(Dataset):
    def __init__(
        self,
        fpath_h5: str,
        mean:  Optional[np.ndarray] = None,
        std:   Optional[np.ndarray] = None,
        limit: Optional[int] = None,
    ) -> None:
        with h5py.File(fpath_h5, "r") as h5:
            X      = np.asarray(h5["X"])  # (N, 784) float32
            self.y = np.asarray(h5["y"])  # (N,) int64
        if limit is not None:
            X, self.y = X[:limit], self.y[:limit]
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
