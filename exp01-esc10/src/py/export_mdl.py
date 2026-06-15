"""
export_mdl.py  --  Export a trained MLP to a flat JSON readable by fhe_server.cpp.

For each trained model in build/mdl/mlp-*/  it writes  ckks_model.json  next to
model.pt.  The C++ server reads this file; no other dependency is needed.

Weight layout (Halevi-Shoup diagonal encoding):
    W @ x  =  sum_i  diag_i  *  rotate(x, i)
where diag_i[j] = W_padded[j, (j+i) % dim].
All matrices are zero-padded to (dim x dim), dim = next power-of-2 >= max(input, hidden, classes).
"""

import json
import os
import logging

import numpy as np
import torch

from common import BLD_DIR, ESC50_FPATH_META
from fhe_dsp.esc50 import Esc50Dataset
from train_mlp import MLP

MDL_ROOT   = os.path.join(BLD_DIR, "mdl")
FEAT_ROOT  = os.path.join(BLD_DIR, "fea")

LOG = logging.getLogger(__name__)


def next_pow2(n: int) -> int:
    return 1 << (max(n, 1) - 1).bit_length()


def diagonals(W: np.ndarray, dim: int) -> list[list[float]]:
    """Halevi-Shoup diagonals of W padded to (dim x dim)."""
    Wp = np.zeros((dim, dim))
    Wp[:W.shape[0], :W.shape[1]] = W
    idx = np.arange(dim)
    return [Wp[idx, (idx + i) % dim].tolist() for i in range(dim)]


def pad(v: np.ndarray, dim: int) -> list[float]:
    out = np.zeros(dim)
    out[:len(v)] = v
    return out.tolist()


def _layer_params(layer):
    return layer.weight.detach().numpy(), layer.bias.detach().numpy()


def export_model(dpath_mdl: str) -> None:
    with open(os.path.join(dpath_mdl, "meta.json")) as f:
        meta = json.load(f)

    feat_stem = os.path.basename(dpath_mdl).removeprefix("mlp-")
    fpath_h5  = os.path.join(FEAT_ROOT, f"{feat_stem}.h5")
    if not os.path.isfile(fpath_h5):
        LOG.warning("Feature file not found, skipping: %s", fpath_h5)
        return

    ds = Esc50Dataset(ESC50_FPATH_META, esc10=True)
    ds.load_features(fpath_h5)

    model = MLP(input_dim=meta["input_dim"], hidden=meta["hidden"], num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(os.path.join(dpath_mdl, "model.pt"), weights_only=True))
    model.eval()

    W1, b1 = _layer_params(model.fc1)
    W2, b2 = _layer_params(model.fc2)
    W3, b3 = _layer_params(model.fc3)

    dim = next_pow2(max(meta["input_dim"], meta["hidden"], len(meta["classes"])))
    LOG.info("packed_dim = %d", dim)

    doc = {
        "input_dim":   meta["input_dim"],
        "hidden":      meta["hidden"],
        "num_classes": len(meta["classes"]),
        "packed_dim":  dim,
        "classes":     meta["classes"],
        "mean":        pad(ds.mean, dim),
        "std":         pad(ds.std,  dim),
        "W1_diag":     diagonals(W1, dim),
        "b1":          pad(b1, dim),
        "W2_diag":     diagonals(W2, dim),
        "b2":          pad(b2, dim),
        "W3_diag":     diagonals(W3, dim),
        "b3":          pad(b3, dim),
    }

    fpath_out = os.path.join(dpath_mdl, "ckks_model.json")
    with open(fpath_out, "w") as f:
        json.dump(doc, f)
    LOG.info("Exported -> %s  (packed_dim=%d)", os.path.relpath(fpath_out, BLD_DIR), dim)


def export_all(mdl_root: str = MDL_ROOT) -> None:
    if not os.path.isdir(mdl_root):
        LOG.warning("No models found under %s", mdl_root)
        return

    mdl_dirs = [
        os.path.join(mdl_root, d)
        for d in sorted(os.listdir(mdl_root))
        if d.startswith("mlp-") and os.path.isfile(os.path.join(mdl_root, d, "model.pt"))
    ]
    if not mdl_dirs:
        LOG.warning("No models found under %s", mdl_root)
        return

    LOG.info("Found %d model(s)", len(mdl_dirs))
    for dpath_mdl in mdl_dirs:
        LOG.info("--- %s ---", os.path.basename(dpath_mdl))
        export_model(dpath_mdl)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    export_all()
