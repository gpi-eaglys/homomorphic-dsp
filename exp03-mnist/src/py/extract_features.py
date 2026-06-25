"""
Download MNIST test set and save pixel features to build/fea/mnist.h5.

H5 layout:
    X: (10000, 784) float32 — raw pixels scaled to [0, 1]
    y: (10000,)    int64   — class labels 0–9
"""

import gzip
import logging
import os
import struct
import urllib.request

import h5py
import numpy as np

from common import BLD_DIR, MNIST_ROOT

LOG = logging.getLogger(__name__)

_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "images": "t10k-images-idx3-ubyte.gz",
    "labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(fname: str, dest_dir: str) -> str:
    fpath = os.path.join(dest_dir, fname)
    if not os.path.isfile(fpath):
        url = _BASE + fname
        LOG.info("Downloading %s ...", url)
        urllib.request.urlretrieve(url, fpath)
    return fpath


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
    os.makedirs(MNIST_ROOT, exist_ok=True)
    fea_dir = os.path.join(BLD_DIR, "fea")
    os.makedirs(fea_dir, exist_ok=True)

    X = _load_images(_download(_FILES["images"], MNIST_ROOT))
    y = _load_labels(_download(_FILES["labels"], MNIST_ROOT))

    fpath_h5 = os.path.join(fea_dir, "mnist.h5")
    with h5py.File(fpath_h5, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
    LOG.info("Saved %d samples -> %s", len(y), fpath_h5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    main()
