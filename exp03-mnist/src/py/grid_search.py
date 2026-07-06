import itertools
import logging
import os
import sys

import mlflow
import torch.nn as nn

from common import BLD_DIR
from train_mlp import ACTIVATION_MAP, ExperimentConfig, Quadratic, train

LOG = logging.getLogger(__name__)

GRID = {
    "layers": [
        [784, 1024, 512, 256, 128, 64, 10],  # gets wider
        [784,  784, 10],  # single hidden
        [784,  10],  # no hidden
        [784,  256, 128, 64, 10],  # went 98%
        [784,  512, 256, 128, 64, 10],
    ],
    "activation": [nn.GELU, nn.ReLU, nn.Sigmoid, Quadratic],
    "lr":         [1e-3, 5e-3],
    "dropout":    [0.0, 0.1, 0.2, 0.3, 0.4],
}


def run_grid() -> None:
    fea_dir  = os.path.join(BLD_DIR, "fea")
    mdl_root = os.path.join(BLD_DIR, "mdl", "exp03")

    keys   = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    total  = len(combos)
    LOG.info("Grid search: %d runs", total)

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        cfg = ExperimentConfig(
            layers     = params["layers"],
            activation = params["activation"],
            lr         = params["lr"],
            dropout    = params["dropout"],
            patience   = 500,
            min_acc    = 0.95,
        )
        act_name = params["activation"].__name__
        LOG.info("--- Run %d/%d  layers=%s  act=%s  lr=%s  dropout=%s ---",
                 i, total, cfg.layers, act_name, cfg.lr, cfg.dropout)
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
    mlflow.set_experiment("CKKS - exp03 - MNIST")
    run_grid()
