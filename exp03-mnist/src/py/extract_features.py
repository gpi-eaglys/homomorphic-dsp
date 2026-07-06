"""
Extract pixel features from downloaded MNIST gz files and save to build/fea/.

Output files:
    mnist-train.h5  — 60000 samples
    mnist-test.h5   —  10000 samples

H5 layout per file:
    X: (N, 784) float32 — raw pixels scaled to [0, 1]
    y: (N,)     int64   — class labels 0–9
"""

import gzip
import logging
import os
import struct

import h5py
import numpy as np

from common import BLD_DIR, MNIST_ROOT

LOG = logging.getLogger(__name__)

_SPLITS = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test":  ("t10k-images-idx3-ubyte.gz",  "t10k-labels-idx1-ubyte.gz"),
}


def _load_images(fpath_gz: str) -> np.ndarray:
    with gzip.open(fpath_gz, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        assert magic == 0x803
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, h * w).astype(np.float32) / 255.0


def _load_labels(fpath_gz: str) -> np.ndarray:
    with gzip.open(fpath_gz, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x801
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def main() -> None:
    fea_dir = os.path.join(BLD_DIR, "fea")
    os.makedirs(fea_dir, exist_ok=True)

    for split, (img_file, lbl_file) in _SPLITS.items():
        X = _load_images(os.path.join(MNIST_ROOT, img_file))
        y = _load_labels(os.path.join(MNIST_ROOT, lbl_file))
        fpath_h5 = os.path.join(fea_dir, f"mnist-{split}.h5")
        with h5py.File(fpath_h5, "w") as f:
            f.create_dataset("X", data=X)
            f.create_dataset("y", data=y)
        LOG.info("Saved %d samples -> %s", len(y), fpath_h5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    main()
