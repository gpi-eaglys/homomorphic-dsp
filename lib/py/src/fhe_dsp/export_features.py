"""
Utilities for exporting precomputed features from .h5 files to plain text.

For each <feat_root>/<stem>.h5 writes <feat_root>/<stem>.txt where every line is:
    <sample_key>  <f0>  <f1>  ...  <f_{bins-1}>
Features are mean-pooled over time frames and written raw (un-normalized) so the
C++ binary can apply its own normalization from ckks_model.json.
"""

import logging
import os

import h5py

LOG = logging.getLogger(__name__)


def export_h5(fpath_h5: str) -> None:
    fpath_out = os.path.splitext(fpath_h5)[0] + ".txt"
    if os.path.isfile(fpath_out):
        LOG.info("Skipping (already exists): %s", os.path.basename(fpath_out))
        return

    with h5py.File(fpath_h5, "r") as f:
        keys = sorted(f.keys())
        rows = [(k, f[k][:].mean(axis=0)) for k in keys]

    with open(fpath_out, "w") as out:
        for key, vec in rows:
            out.write(key + "  " + "  ".join(f"{v:.10f}" for v in vec) + "\n")

    LOG.info("Exported %d samples -> %s", len(rows), os.path.basename(fpath_out))


def export_all(feat_root: str) -> None:
    if not os.path.isdir(feat_root):
        LOG.warning("Feature directory not found: %s", feat_root)
        return

    h5_files = sorted(
        os.path.join(feat_root, f)
        for f in os.listdir(feat_root)
        if f.endswith(".h5")
    )
    if not h5_files:
        LOG.warning("No .h5 files found under %s", feat_root)
        return

    LOG.info("Found %d .h5 file(s)", len(h5_files))
    for fpath_h5 in h5_files:
        LOG.info("--- %s ---", os.path.basename(fpath_h5))
        export_h5(fpath_h5)
