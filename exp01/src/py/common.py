import logging
import os
from typing import Optional

import h5py
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EXP_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
BLD_DIR    = os.path.join(EXP_DIR, "build")
REPO_DIR   = os.path.abspath(os.path.join(EXP_DIR, ".."))
META_CSV   = os.path.join(REPO_DIR, "assets/esc-50/ESC-50-master/meta/esc50.csv")

LOG = logging.getLogger(__name__)


class Esc10Dataset(Dataset):
    def __init__(self) -> None:
        self._df = pl.read_csv(META_CSV).filter(pl.col("esc10"))
        self.classes = sorted(self._df["category"].unique().to_list())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def load_features(self, fpath_h5: str) -> None:
        self.X = self.y = None
        with h5py.File(fpath_h5, "r") as f:
            keys = list(f.keys())
            self.X = np.stack([f[k][:].mean(axis=0) for k in keys])  # mean-pool frames -> (n, bins)
            self.y = np.array([
                self.class_to_idx[self._df.filter(pl.col("filename") == k + ".wav")["category"][0]]
                for k in keys
            ], dtype=np.int64)

        mean = self.X.mean(axis=0)
        std  = self.X.std(axis=0) + 1e-6
        self.X = ((self.X - mean) / std).astype(np.float32)
        self.mean, self.std = mean, std

        LOG.info("Loaded %d samples from %s", len(self.y), os.path.basename(fpath_h5))
        for i, c in enumerate(self.classes):
            LOG.info("  %2d  %-20s  %d", i, c, int((self.y == i).sum()))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.X[idx]), self.y[idx]
