import json
import logging
import os
from dataclasses import dataclass
from typing import Callable

import mlflow
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from train_mlp import ACTIVATION_MAP, MnistDataset, Quadratic, _eval_acc

EPOCHS = 100000
SEED   = 42

LOG = logging.getLogger(__name__)
torch.manual_seed(SEED)

# AvgPool2d is a linear op (CKKS-compatible); MaxPool2d requires comparisons
# (not CKKS-compatible). Only "avg"-pool configs should ever be exported to
# CKKS — see is_ckks_compatible() below.
POOL_MAP: dict[str, Callable[[int], nn.Module]] = {
    "max": nn.MaxPool2d,
    "avg": nn.AvgPool2d,
}


@dataclass
class ExperimentConfig:
    conv_channels:       list[int]  # e.g. [1, 16, 32] — first entry is input channels (1 for grayscale MNIST)
    kernel_size:         int
    fc_layers:           list[int]  # e.g. [128, 10] — last entry must be num_classes
    activation:          Callable[[], nn.Module]
    pool:                str   = "max"  # "max" or "avg" — see POOL_MAP
    dropout:             float = 0.0
    lr:                  float = 5e-3
    batch:               int   = 64
    min_acc:             float = 0.97
    patience:            int   = 150
    patience_train_acc:  int   = 50
    # metadata only — not used by the model or training loop
    run_id:              str   = ""
    param_hash:          str   = ""
    schema_hash:         str   = ""


def is_ckks_compatible(cfg: ExperimentConfig) -> bool:
    return cfg.pool == "avg" and cfg.activation.__name__ != "ReLU"


class CNN(nn.Module):
    def __init__(self, conv_channels: list[int], kernel_size: int, fc_layers: list[int],
                 activation: Callable[[], nn.Module], pool: str = "max", dropout: float = 0.0) -> None:
        super().__init__()
        pool_cls = POOL_MAP[pool]
        conv_mods: list[nn.Module] = []
        for i in range(len(conv_channels) - 1):
            conv_mods.append(nn.Conv2d(conv_channels[i], conv_channels[i + 1], kernel_size, padding=kernel_size // 2))
            conv_mods.append(activation())
            conv_mods.append(pool_cls(2))
            if dropout > 0.0:
                conv_mods.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*conv_mods)

        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, conv_channels[0], 28, 28)).numel()

        fc_mods: list[nn.Module] = []
        dims = [flat_dim] + fc_layers
        for i in range(len(dims) - 1):
            fc_mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                fc_mods.append(activation())
                if dropout > 0.0:
                    fc_mods.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fc_mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], 1, 28, 28)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


def _cfg_to_dict(cfg: ExperimentConfig) -> dict:
    return {
        "conv_channels":      cfg.conv_channels,
        "kernel_size":        cfg.kernel_size,
        "fc_layers":          cfg.fc_layers,
        "activation":         cfg.activation.__name__,
        "pool":               cfg.pool,
        "dropout":            cfg.dropout,
        "lr":                 cfg.lr,
        "batch":              cfg.batch,
        "min_acc":            cfg.min_acc,
        "patience":           cfg.patience,
        "patience_train_acc": cfg.patience_train_acc,
        "run_id":             cfg.run_id,
        "param_hash":         cfg.param_hash,
        "schema_hash":        cfg.schema_hash,
    }


def train(fpath_train_h5: str, fpath_test_h5: str, mdl_root: str, cfg: ExperimentConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MnistDataset(fpath_train_h5)
    test_ds  = MnistDataset(fpath_test_h5, mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch, shuffle=False)

    dpath_run = os.path.join(mdl_root, f"cnn-mnist_{cfg.param_hash}")
    os.makedirs(dpath_run, exist_ok=True)

    assert cfg.conv_channels[0] == 1, \
        f"Input channel mismatch: MNIST is grayscale (1 channel), config has {cfg.conv_channels[0]}"

    model = CNN(conv_channels=cfg.conv_channels, kernel_size=cfg.kernel_size, fc_layers=cfg.fc_layers,
                activation=cfg.activation, pool=cfg.pool, dropout=cfg.dropout).to(device)
    LOG.info("train=%d  test=%d  device=%s  conv_channels=%s  fc_layers=%s  pool=%s",
             len(train_ds), len(test_ds), device, cfg.conv_channels, cfg.fc_layers, cfg.pool)

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run():
        mlflow.set_tags({"run_id": cfg.run_id, "param_hash": cfg.param_hash, "schema_hash": cfg.schema_hash})
        mlflow.log_params({
            "conv_channels": cfg.conv_channels,
            "kernel_size":   cfg.kernel_size,
            "fc_layers":     cfg.fc_layers,
            "activation":    cfg.activation.__name__,
            "pool":          cfg.pool,
            "dropout":       cfg.dropout,
            "lr":            cfg.lr,
            "batch":         cfg.batch,
        })

        config_path = os.path.join(dpath_run, "config.json")
        with open(config_path, "w") as f:
            json.dump(_cfg_to_dict(cfg), f, indent=2)
        mlflow.log_artifact(config_path)

        best_test_acc = 0.0
        best_epoch    = 0
        n_worse = 0
        n_perfect_train = 0

        for epoch in range(EPOCHS):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y_batch)
                correct    += (logits.argmax(1) == y_batch).sum().item()
                total      += len(y_batch)

            train_acc  = correct / total
            train_loss = total_loss / total
            test_acc   = _eval_acc(model, test_loader, device)

            mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}, step=epoch)

            if train_acc >= 1.0:
                n_perfect_train += 1
                if n_perfect_train >= cfg.patience_train_acc:
                    LOG.info("epoch %4d train_acc 100%% for %d consecutive epochs — stopping", epoch, cfg.patience_train_acc)
                    break
            else:
                n_perfect_train = 0

            if test_acc > 0.99999 or epoch % 10 == 0 or epoch == EPOCHS - 1:
                LOG.info("epoch %4d  loss %.4f  train_acc %.5f  test_acc %.5f", epoch, train_loss, train_acc, test_acc)

            if test_acc <= best_test_acc:
                n_worse += 1
                if n_worse >= cfg.patience:
                    break
            else:
                n_worse = 0
                best_test_acc = test_acc
                best_epoch    = epoch
                LOG.info("epoch %4d  loss %.4f  train_acc %.5f  test_acc %.5f <- BEST", epoch, train_loss, train_acc, test_acc)
                if test_acc > cfg.min_acc:
                    dname_mdl = f"e{epoch:04d}"
                    dpath_mdl = os.path.join(dpath_run, dname_mdl)
                    os.makedirs(dpath_mdl, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(dpath_mdl, "model.pt"))
                    meta_path = os.path.join(dpath_mdl, "meta.json")
                    with open(meta_path, "w") as f:
                        json.dump({
                            **_cfg_to_dict(cfg),
                            "feat":     "mnist",
                            "classes":  train_ds.classes,
                            "mean":     train_ds.mean.tolist(),
                            "std":      train_ds.std.tolist(),
                        }, f, indent=2)
                    mlflow.log_artifact(meta_path)
                if test_acc > 0.9999999:
                    break

        result_path = os.path.join(dpath_run, "result.json")
        with open(result_path, "w") as f:
            json.dump({"best_test_acc": best_test_acc, "best_epoch": best_epoch}, f, indent=2)
        mlflow.log_artifact(result_path)
