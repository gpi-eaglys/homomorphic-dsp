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
from train_mlp import ACTIVATION_MAP, ExperimentConfig, Quadratic, train

LOG = logging.getLogger(__name__)

EXPERIMENT_NAME = "exp03-mnist-mlp"

# HASHED_PARAMS are _param_hash()-ed for each running combination. 
# Adding new *values* to any of these (e.g. another dropout or layers entry)
# is always safe — each grid point's param_hash depends only on its own
# values, not on GRID's shape are untouched. 
# Adding a *new axis* to GRID (a key not in this tuple) will FAIL due to  schema-hash check at the top of run_grid().
HASHED_PARAMS = ("layers", "activation", "lr", "dropout")

GRID = {
    "layers": [
        [784,  10],  # no hidden
        [784,  784, 10],  # single hidden
        [784,  256, 128, 64, 10],  # went 98%
        [784,  512, 256, 128, 64, 10], 
        [784, 1024, 512, 256, 128, 64, 10],  # gets wider
    ],
    "activation": [nn.GELU, nn.ReLU, nn.Sigmoid, Quadratic],
    "lr":         [1e-3, 5e-3],
    "dropout":    [0.0, 0.1, 0.2, 0.3, 0.4],
}


def _schema_hash(param_names) -> str:
    return hashlib.sha256(json.dumps(sorted(param_names)).encode()).hexdigest()[:12]


def _param_hash(layers: list[int], activation, lr: float, dropout: float) -> str:
    key = json.dumps({
        "layers":     layers,
        "activation": activation.__name__,
        "lr":         lr,
        "dropout":    dropout,
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
        param_hash = _param_hash(params["layers"], params["activation"], params["lr"], params["dropout"])

        if _already_finished(param_hash, schema_hash):
            LOG.info("--- Run %d/%d (param_hash=%s) already FINISHED — skipping ---", i, total, param_hash)
            continue

        cfg = ExperimentConfig(
            layers      = params["layers"],
            activation  = params["activation"],
            lr          = params["lr"],
            dropout     = params["dropout"],
            patience    = 500,
            min_acc     = 0.95,
            run_id      = run_id,
            param_hash  = param_hash,
            schema_hash = schema_hash,
        )
        act_name = params["activation"].__name__
        LOG.info("--- Run %d/%d  param_hash=%s  layers=%s  act=%s  lr=%s  dropout=%s ---",
                 i, total, param_hash, cfg.layers, act_name, cfg.lr, cfg.dropout)
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

