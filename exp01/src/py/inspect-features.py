import os
import logging

import h5py
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EXP_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
BLD_DIR    = os.path.join(EXP_DIR, "build")
REPO_DIR   = os.path.abspath(os.path.join(EXP_DIR, ".."))
META_CSV   = os.path.join(REPO_DIR, "assets/esc-50/ESC-50-master/meta/esc50.csv")

LOG = logging.getLogger(__name__)


def load_features(fpath_h5: str) -> np.ndarray:
    """Load all features from HDF5, mean-pool frames, return (n_samples, n_channels)."""
    with h5py.File(fpath_h5, "r") as f:
        return np.stack([f[k][:].mean(axis=0) for k in f.keys()])


def find_and_inspect_all(search_dir: str = BLD_DIR):
    """Find all .h5 files under search_dir and run plot_channel_histograms on each."""
    h5_files = [
        os.path.join(root, fname)
        for root, _, files in os.walk(search_dir)
        for fname in files
        if fname.endswith(".h5")
    ]
    if not h5_files:
        LOG.warning("No .h5 files found under %s", search_dir)
        return
    LOG.info("Found %d .h5 file(s)", len(h5_files))
    for fpath_h5 in h5_files:
        plot_channel_histograms(fpath_h5)


def plot_channel_histograms(fpath_h5: str):
    stem = os.path.splitext(os.path.basename(fpath_h5))[0]
    dpath_out = os.path.join(os.path.dirname(fpath_h5), f"hist-{stem}")

    X = load_features(fpath_h5)
    n_samples, n_channels = X.shape
    LOG.info("Loaded %d samples x %d channels from %s", n_samples, n_channels, fpath_h5)

    os.makedirs(dpath_out, exist_ok=True)
    sns.set_theme(style="darkgrid")

    for ch in range(n_channels):
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(X[:, ch], bins=30, kde=True, ax=ax)
        ax.set_title(f"Channel {ch}  (mean={X[:, ch].mean():.3f}, std={X[:, ch].std():.3f})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        fpath_png = os.path.join(dpath_out, f"ch{ch:03d}.png")
        fig.tight_layout()
        fig.savefig(fpath_png, dpi=100)
        plt.close(fig)
        LOG.debug("Saved %s", os.path.relpath(fpath_png, BLD_DIR))

    LOG.info("Saved %d plots to %s", n_channels, os.path.relpath(dpath_out, BLD_DIR))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("PIL").setLevel(logging.WARNING)
    find_and_inspect_all()

