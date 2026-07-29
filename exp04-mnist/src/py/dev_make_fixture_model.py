"""
Dev-only fixture: dumps an *untrained* Quadratic+avg-pool CNN in the same
build/mdl/exp04/<run>/<epoch>/{model.pt,meta.json} layout real training
produces, so export_mdl.py's directory walk and plaintext self-validator can
be exercised before the live grid search (grid_search_cnn.py) has trained a
real Quadratic-activation model. Not part of the train->export->infer
workflow — superseded by a real model once one exists.
"""

import json
import logging
import os

import torch

from common import BLD_DIR
from grid_search_cnn import FC_LAYERS, KERNEL_SIZE
from mnist_data import MnistDataset, Quadratic
from train_cnn import CNN, ExperimentConfig, _cfg_to_dict

LOG = logging.getLogger(__name__)

CONV_CHANNELS = [1, 16, 32]
FIXTURE_HASH  = "devfixture001"


def make_fixture(mdl_root: str) -> str:
    train_ds = MnistDataset(os.path.join(BLD_DIR, "fea", "mnist-train.h5"))

    cfg = ExperimentConfig(
        conv_channels=CONV_CHANNELS, kernel_size=KERNEL_SIZE, fc_layers=FC_LAYERS,
        activation=Quadratic, pool="avg", param_hash=FIXTURE_HASH,
    )
    model = CNN(conv_channels=cfg.conv_channels, kernel_size=cfg.kernel_size, fc_layers=cfg.fc_layers,
                activation=cfg.activation, pool=cfg.pool, dropout=cfg.dropout)
    model.eval()

    dpath_mdl = os.path.join(mdl_root, f"cnn-mnist_{FIXTURE_HASH}", "e0000")
    os.makedirs(dpath_mdl, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(dpath_mdl, "model.pt"))
    with open(os.path.join(dpath_mdl, "meta.json"), "w") as f:
        json.dump({
            **_cfg_to_dict(cfg),
            "feat":    "mnist",
            "classes": train_ds.classes,
            "mean":    train_ds.mean.tolist(),
            "std":     train_ds.std.tolist(),
        }, f, indent=2)

    LOG.info("Fixture model written -> %s", dpath_mdl)
    return dpath_mdl


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    make_fixture(os.path.join(BLD_DIR, "mdl", "exp04-dev-fixture"))
