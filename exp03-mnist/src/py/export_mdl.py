"""
Export trained MLP to CKKS-compatible JSON (Halevi-Shoup diagonal encoding).
Mean/std are read from meta.json saved during training — no dataset reload needed.
"""

import json
import logging
import os

import numpy as np
import torch

from common import BLD_DIR
from train_mlp import MLP

LOG = logging.getLogger(__name__)


def next_pow2(n: int) -> int:
    return 1 << (max(n, 1) - 1).bit_length()


def diagonals(W: np.ndarray, dim: int) -> list[list[float]]:
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

    fpath_out = os.path.join(dpath_mdl, "ckks_model.json")
    if os.path.isfile(fpath_out):
        LOG.info("Skipping (already exported): %s", fpath_out)
        return

    model = MLP(input_dim=meta["input_dim"], hidden=meta["hidden"], num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(os.path.join(dpath_mdl, "model.pt"), weights_only=True, map_location="cpu"))
    model.eval()

    W1, b1 = _layer_params(model.fc1)
    W2, b2 = _layer_params(model.fc2)
    W3, b3 = _layer_params(model.fc3)

    dim = next_pow2(max(meta["input_dim"], meta["hidden"], len(meta["classes"])))
    LOG.info("packed_dim = %d", dim)

    mean = np.array(meta["mean"])
    std  = np.array(meta["std"])

    doc = {
        "input_dim":   meta["input_dim"],
        "hidden":      meta["hidden"],
        "num_classes": len(meta["classes"]),
        "packed_dim":  dim,
        "classes":     meta["classes"],
        "mean":        pad(mean, dim),
        "std":         pad(std,  dim),
        "W1_diag":     diagonals(W1, dim),
        "b1":          pad(b1, dim),
        "W2_diag":     diagonals(W2, dim),
        "b2":          pad(b2, dim),
        "W3_diag":     diagonals(W3, dim),
        "b3":          pad(b3, dim),
    }

    with open(fpath_out, "w") as f:
        json.dump(doc, f)
    LOG.info("Exported -> %s  (packed_dim=%d)", fpath_out, dim)


def export_all(mdl_root: str) -> None:
    if not os.path.isdir(mdl_root):
        LOG.warning("No models found under %s", mdl_root)
        return

    mdl_dirs = sorted(
        root
        for root, _, files in os.walk(mdl_root)
        if "model.pt" in files
    )
    if not mdl_dirs:
        LOG.warning("No models found under %s", mdl_root)
        return

    LOG.info("Found %d model(s)", len(mdl_dirs))
    for dpath_mdl in mdl_dirs:
        LOG.info("--- %s ---", os.path.basename(dpath_mdl))
        export_model(dpath_mdl)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    export_all(os.path.join(BLD_DIR, "mdl", "exp03"))
