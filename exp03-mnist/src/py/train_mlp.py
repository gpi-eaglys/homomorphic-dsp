import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import h5py
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset

from common import BLD_DIR

EPOCHS = 100000
SEED   = 42

LOG = logging.getLogger(__name__)
torch.manual_seed(SEED)


@dataclass
class ExperimentConfig:
    layers:             list[int]
    activation:         Callable[[], nn.Module]
    dropout:            float = 0.0
    lr:                 float = 5e-3
    batch:              int   = 64
    min_acc:            float = 0.97
    patience:           int   = 150
    patience_train_acc: int   = 50


class Quadratic(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


ACTIVATION_MAP: dict[str, Callable[[], nn.Module]] = {
    "GELU":      nn.GELU,
    "ReLU":      nn.ReLU,
    "Sigmoid":   nn.Sigmoid,
    "Quadratic": Quadratic,
    "Tanh":      nn.Tanh,
}

CONFIGS: dict[str, ExperimentConfig] = {
    "baseline":   ExperimentConfig(layers=[784, 64, 32, 32, 10],        activation=nn.GELU),
    "wide":       ExperimentConfig(layers=[784, 256, 256, 10],           activation=nn.GELU),
    "relu-deep":  ExperimentConfig(layers=[784, 512, 256, 128, 10],      activation=nn.ReLU),
    "tanh-small": ExperimentConfig(layers=[784, 64, 32, 10],             activation=nn.Tanh),
    "dropout":    ExperimentConfig(layers=[784, 512, 256, 128, 10],      activation=nn.ReLU, dropout=0.3),
}


class MLP(nn.Module):
    def __init__(self, layers: list[int], activation: Callable[[], nn.Module], dropout: float = 0.0) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(activation())
                if dropout > 0.0:
                    mods.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MnistDataset(Dataset):
    def __init__(self, fpath_h5: str, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> None:
        with h5py.File(fpath_h5, "r") as h5:
            X      = np.asarray(h5["X"])  # (N, 784) float32
            self.y = np.asarray(h5["y"])  # (N,) int64
        self.classes = [str(i) for i in range(10)]
        self.mean = X.mean(axis=0) if mean is None else mean
        self.std  = (X.std(axis=0) + 1e-6) if std is None else std
        self.X = ((X - self.mean) / self.std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.X[idx]), self.y[idx]


def _eval_acc(model: MLP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            correct += (model(X_batch).argmax(1) == y_batch).sum().item()
            total   += len(y_batch)
    return correct / total


def train(fpath_train_h5: str, fpath_test_h5: str, mdl_root: str, cfg: ExperimentConfig, run_tags: Optional[dict] = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MnistDataset(fpath_train_h5)
    test_ds  = MnistDataset(fpath_test_h5, mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch, shuffle=False)

    os.makedirs(mdl_root, exist_ok=True)

    assert train_ds.X.shape[1] == cfg.layers[0], \
        f"Input dim mismatch: data has {train_ds.X.shape[1]}, config has {cfg.layers[0]}"

    model = MLP(layers=cfg.layers, activation=cfg.activation, dropout=cfg.dropout).to(device)
    LOG.info("train=%d  test=%d  device=%s  layers=%s", len(train_ds), len(test_ds), device, cfg.layers)

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run():
        if run_tags:
            mlflow.set_tags(run_tags)
        mlflow.log_params({
            "layers":     cfg.layers,
            "activation": cfg.activation.__name__,
            "dropout":    cfg.dropout,
            "lr":         cfg.lr,
            "batch":      cfg.batch,
        })

        best_test_acc = 0.0
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
                LOG.info("epoch %4d  loss %.4f  train_acc %.5f  test_acc %.5f <- BEST", epoch, train_loss, train_acc, test_acc)
                if test_acc > cfg.min_acc:
                    dname_mdl = f"mlp-mnist_e{epoch:04d}_acc={test_acc:.3f}"
                    dpath_mdl = os.path.join(mdl_root, dname_mdl)
                    os.makedirs(dpath_mdl, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(dpath_mdl, "model.pt"))
                    meta_path = os.path.join(dpath_mdl, "meta.json")
                    with open(meta_path, "w") as f:
                        json.dump({
                            "feat":       "mnist",
                            "classes":    train_ds.classes,
                            "layers":     cfg.layers,
                            "activation": cfg.activation.__name__,
                            "dropout":    cfg.dropout,
                            "mean":       train_ds.mean.tolist(),
                            "std":        train_ds.std.tolist(),
                        }, f, indent=2)
                    mlflow.log_artifact(meta_path)
                if test_acc > 0.9999999:
                    break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(BLD_DIR, "mlflow.db"))
    mlflow.set_experiment("CKKS - exp03 - MNIST")
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if cfg_name not in CONFIGS:
        print(f"Unknown config '{cfg_name}'. Available: {list(CONFIGS)}")
        sys.exit(1)
    cfg = CONFIGS[cfg_name]
    fea_dir = os.path.join(BLD_DIR, "fea")
    train(
        fpath_train_h5=os.path.join(fea_dir, "mnist-train.h5"),
        fpath_test_h5 =os.path.join(fea_dir, "mnist-test.h5"),
        mdl_root      =os.path.join(BLD_DIR, "mdl", "exp03"),
        cfg           =cfg,
    )
