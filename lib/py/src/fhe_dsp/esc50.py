import logging
import os
from collections.abc import Generator
from typing import Optional

import h5py
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

LOG = logging.getLogger(__name__)


class Esc50Dataset(Dataset):
    def __init__(self, esc50_metafile: str, esc10: bool) -> None:
        self._df = pl.read_csv(esc50_metafile).filter(pl.col("esc10")) if esc10 else pl.read_csv(esc50_metafile)
        self.classes = sorted(self._df["category"].unique().to_list())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self._dpath_asset_root = os.path.normpath(os.path.join(esc50_metafile, "..", ".."))
        self._audio_dir = os.path.join(self._dpath_asset_root, "audio")


    def audio_dir(self) -> str:
        return self._audio_dir

    def iter_audio(self) -> Generator[str, None, None]:
        """Yield absolute paths to audio files present on disk. Skips missing files."""
        log = logging.getLogger(__name__)
        for filename in self._df["filename"]:
            path = os.path.join(self._audio_dir, filename)
            if os.path.isfile(path):
                yield path
            else:
                log.warning("Cannot find audio: %s", path)


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
