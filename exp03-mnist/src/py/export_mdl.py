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
from train_mlp import ACTIVATION_MAP, MLP

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

    layers     = meta["layers"]  # full spec: [input, hidden..., output]
    activation = ACTIVATION_MAP[meta.get("activation", "GELU")]
    dropout    = meta.get("dropout", 0.0)
    model = MLP(layers=layers, activation=activation, dropout=dropout)
    model.load_state_dict(torch.load(os.path.join(dpath_mdl, "model.pt"), weights_only=True, map_location="cpu"))
    model.eval()

    # Extract (W, b) for each Linear layer in order
    linear_layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    layer_params  = [_layer_params(l) for l in linear_layers]

    all_dims = layers
    hidden   = layers[1:-1]
    dim = next_pow2(max(all_dims))
    LOG.info("packed_dim = %d", dim)

    mean = np.array(meta["mean"])
    std  = np.array(meta["std"])

    doc = {
        "input_dim":   layers[0],
        "hidden":      hidden,
        "num_classes": layers[-1],
        "packed_dim":  dim,
        "classes":     meta["classes"],
        "mean":        pad(mean, dim),
        "std":         pad(std,  dim),
    }
    for i, (W, b) in enumerate(layer_params):
        doc[f"W{i+1}_diag"] = diagonals(W, dim)
        doc[f"b{i+1}"]      = pad(b, dim)

    with open(fpath_out, "w") as f:
        json.dump(doc, f)
    LOG.info("Exported -> %s  (packed_dim=%d)", fpath_out, dim)


def _is_ckks_exportable(dpath_mdl: str) -> bool:
    meta_path = os.path.join(dpath_mdl, "meta.json")
    if not os.path.isfile(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    layers = meta.get("layers")
    return meta.get("activation") == "Quadratic" and layers is not None and len(layers) == 3


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

    # Only Quadratic (polynomial, CKKS-evaluable) single-hidden-layer MLPs are exportable.
    exportable = [d for d in mdl_dirs if _is_ckks_exportable(d)]
    LOG.info("Found %d model(s), %d exportable (Quadratic, single hidden layer)", len(mdl_dirs), len(exportable))
    for dpath_mdl in exportable:
        LOG.info("--- %s ---", os.path.basename(dpath_mdl))
        export_model(dpath_mdl)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    export_all(os.path.join(BLD_DIR, "mdl", "exp03"))
