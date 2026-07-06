import hashlib
import itertools
import json
import logging
import os
import socket

import mlflow
import torch
import torch.nn as nn

from common import BLD_DIR
from train_mlp import Quadratic
from train_cnn import ExperimentConfig, train

LOG = logging.getLogger(__name__)

EXPERIMENT_NAME = "exp03-mnist-cnn"

# kernel_size and fc_layers are fixed (not swept) to keep the grid a
# manageable size — conv_channels already covers topology depth/width.
KERNEL_SIZE = 3
FC_LAYERS   = [128, 10]

# See grid_search_mlp.py's HASHED_PARAMS comment for the value-vs-attribute
# distinction this guards against. "pool" is included because it's the
# CKKS-relevant axis (max vs avg) — see train_cnn.is_ckks_compatible().
HASHED_PARAMS = ("conv_channels", "activation", "pool", "lr", "dropout")

GRID = {
    "conv_channels": [
        [1, 16, 32],       # 2 conv layers
        [1, 32, 64],       # 2 conv layers, wider
        [1, 16, 32, 64],   # 3 conv layers, deeper
    ],
    "activation": [nn.GELU, nn.ReLU, nn.Sigmoid, Quadratic],
    "pool":       ["max", "avg"],
    "lr":         [1e-3, 5e-3],
    "dropout":    [0.0, 0.2, 0.4],
}


def _schema_hash(param_names) -> str:
    return hashlib.sha256(json.dumps(sorted(param_names)).encode()).hexdigest()[:12]


def _param_hash(conv_channels: list[int], activation, pool: str, lr: float, dropout: float) -> str:
    key = json.dumps({
        "conv_channels": conv_channels,
        "activation":    activation.__name__,
        "pool":          pool,
        "lr":            lr,
        "dropout":       dropout,
    }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _already_finished(param_hash: str, schema_hash: str) -> bool:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=(
            f"tags.param_hash = '{param_hash}' and tags.schema_hash = '{schema_hash}' "
            "and status = 'FINISHED'"
        ),
    )
    return not runs.empty


def run_grid() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available (torch.cuda.is_available() is False) — running "
            f"{len(list(itertools.product(*GRID.values())))} grid search configs on "
            "CPU would be far too slow. Check `torch.__version__` / `torch.version.cuda` "
            "— `uv sync` reverts torch to the CPU build pinned by lib/py/pyproject.toml. "
            "Reinstall with: uv pip install \"torch==2.6.0+cu124\" "
            "--index-url https://download.pytorch.org/whl/cu124 --python .venv/bin/python"
        )

    if set(GRID.keys()) != set(HASHED_PARAMS):
        raise ValueError(
            f"GRID keys {sorted(GRID.keys())} != HASHED_PARAMS {sorted(HASHED_PARAMS)}. "
            "Adding/removing a search axis requires deliberately updating "
            "HASHED_PARAMS and _param_hash() together — see the comment above GRID."
        )
    schema_hash = _schema_hash(HASHED_PARAMS)

    fea_dir  = os.path.join(BLD_DIR, "fea")
    mdl_root = os.path.join(BLD_DIR, "mdl", "exp03")
    run_id   = f"{socket.gethostname()}-{os.getpid()}"

    keys   = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    total  = len(combos)
    LOG.info("Grid search: %d runs (schema_hash=%s)", total, schema_hash)

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        param_hash = _param_hash(params["conv_channels"], params["activation"], params["pool"],
                                  params["lr"], params["dropout"])

        if _already_finished(param_hash, schema_hash):
            LOG.info("--- Run %d/%d (param_hash=%s) already FINISHED — skipping ---", i, total, param_hash)
            continue

        cfg = ExperimentConfig(
            conv_channels = params["conv_channels"],
            kernel_size   = KERNEL_SIZE,
            fc_layers     = FC_LAYERS,
            activation    = params["activation"],
            pool          = params["pool"],
            lr            = params["lr"],
            dropout       = params["dropout"],
            patience      = 500,
            min_acc       = 0.95,
            run_id        = run_id,
            param_hash    = param_hash,
            schema_hash   = schema_hash,
        )
        act_name = params["activation"].__name__
        LOG.info("--- Run %d/%d  param_hash=%s  conv_channels=%s  act=%s  pool=%s  lr=%s  dropout=%s ---",
                 i, total, param_hash, cfg.conv_channels, act_name, cfg.pool, cfg.lr, cfg.dropout)
        try:
            train(
                fpath_train_h5=os.path.join(fea_dir, "mnist-train.h5"),
                fpath_test_h5 =os.path.join(fea_dir, "mnist-test.h5"),
                mdl_root      =mdl_root,
                cfg           =cfg,
            )
        except Exception:
            LOG.exception("Run %d/%d failed — skipping", i, total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(BLD_DIR, "mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_grid()
