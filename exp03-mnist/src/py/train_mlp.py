import csv
import json
import logging
import os
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from common import BLD_DIR

HIDDEN       = 64
EPOCHS       = 100000
LR           = 1e-3
BATCH        = 32
SEED         = 42
MIN_ACC      = 0.9
PATIENCE     = 100
DEVSET_SIZE  = 800   # held-out samples for evaluation (10% of MNIST test set)

LOG = logging.getLogger(__name__)
torch.manual_seed(SEED)


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden)
        self.act1 = SquareActivation()
        self.fc2  = nn.Linear(hidden, hidden)
        self.act2 = SquareActivation()
        self.fc3  = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class MnistDataset(Dataset):
    def __init__(self, fpath_h5: str) -> None:
        with h5py.File(fpath_h5, "r") as f:
            X = f["X"][:]       # (N, 784) float32
            self.y = f["y"][:]  # (N,) int64
        self.classes = [str(i) for i in range(10)]
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0) + 1e-6
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


def train(fpath_h5: str, mdl_root: str, devset_size: int = DEVSET_SIZE) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = os.path.splitext(os.path.basename(fpath_h5))[0]
    ds = MnistDataset(fpath_h5)

    train_ds, dev_ds = random_split(
        ds, [len(ds) - devset_size, devset_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH, shuffle=False)

    run_dir = os.path.join(mdl_root, "run_" + datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(run_dir, exist_ok=True)

    model = MLP(input_dim=784, hidden=HIDDEN, num_classes=10).to(device)
    LOG.info("run_dir=%s  train=%d  dev=%d  feature='%s'  device=%s",
             run_dir, len(train_ds), len(dev_ds), feat, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    with open(os.path.join(run_dir, "batches.csv"), "w", newline="") as batch_fh, \
         open(os.path.join(run_dir, "epochs.csv"),  "w", newline="") as epoch_fh:

        bw = csv.writer(batch_fh)
        ew = csv.writer(epoch_fh)
        bw.writerow(["step", "epoch", "loss", "train_acc"])
        ew.writerow(["epoch", "train_loss", "train_acc", "dev_acc"])

        step = 0
        best_dev_acc = 0.0
        n_worse = 0

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

                batch_loss = loss.item()
                batch_acc  = (logits.argmax(1) == y_batch).float().mean().item()
                bw.writerow([step, epoch, f"{batch_loss:7.3f}", f"{batch_acc:.6f}"])
                batch_fh.flush()

                total_loss += batch_loss * len(y_batch)
                correct    += (logits.argmax(1) == y_batch).sum().item()
                total      += len(y_batch)
                step       += 1

            train_acc  = correct / total
            train_loss = total_loss / total
            dev_acc    = _eval_acc(model, dev_loader, device)

            ew.writerow([epoch, f"{train_loss:7.3f}", f"{train_acc:.6f}", f"{dev_acc:.6f}"])
            epoch_fh.flush()

            if dev_acc > 0.99999 or epoch % 10 == 0 or epoch == EPOCHS - 1:
                LOG.info("epoch %4d  loss %.4f  train_acc %.5f  dev_acc %.5f", epoch, train_loss, train_acc, dev_acc)

            if dev_acc <= best_dev_acc:
                n_worse += 1
                if n_worse >= PATIENCE:
                    break
            else:
                n_worse = 0
                best_dev_acc = dev_acc
                if dev_acc > MIN_ACC:
                    dname_mdl = f"mlp-{feat}_e{epoch:04d}_acc={dev_acc:.3f}"
                    LOG.info("epoch %4d  loss %.4f  train_acc %.5f  dev_acc %.5f <- BEST (saving: %s)", epoch, train_loss, train_acc, dev_acc, dname_mdl)
                    dpath_mdl = os.path.join(run_dir, dname_mdl)
                    os.makedirs(dpath_mdl, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(dpath_mdl, "model.pt"))
                    with open(os.path.join(dpath_mdl, "meta.json"), "w") as f:
                        json.dump({
                            "feat":      feat,
                            "classes":   ds.classes,
                            "input_dim": 784,
                            "hidden":    HIDDEN,
                            "mean":      ds.mean.tolist(),
                            "std":       ds.std.tolist(),
                        }, f, indent=2)
                if dev_acc > 0.9999999:
                    break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    train(
        fpath_h5 = os.path.join(BLD_DIR, "fea", "mnist.h5"),
        mdl_root = os.path.join(BLD_DIR, "mdl", "exp03"),
    )

