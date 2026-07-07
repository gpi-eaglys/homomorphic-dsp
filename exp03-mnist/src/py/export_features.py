"""
Convert build/fea/mnist-test.h5 to build/fea/mnist-test.txt for the C++ inference binary.

Each output line:  <idx>  <f0>  <f1>  ...  <f783>
Features are raw (un-normalized) pixel values; the C++ binary applies
mean/std normalization from ckks_model.json.
"""

import logging
import os

import h5py

from common import BLD_DIR

LOG = logging.getLogger(__name__)


def export_h5(fpath_h5: str, fpath_out: str) -> None:
    if os.path.isfile(fpath_out):
        LOG.info("Skipping (already exists): %s", os.path.basename(fpath_out))
        return

    with h5py.File(fpath_h5, "r") as f:
        X = f["X"][:]  # (N, 784) float32

    with open(fpath_out, "w") as out:
        for i, vec in enumerate(X):
            out.write(f"{i:05d}  " + "  ".join(f"{v:.10f}" for v in vec) + "\n")

    LOG.info("Exported %d samples -> %s", len(X), os.path.basename(fpath_out))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    export_h5(os.path.join(BLD_DIR, "fea", "mnist-test.h5"), os.path.join(BLD_DIR, "fea", "mnist-test.txt"))
