import json
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fhe_dsp.esc50 import Esc50Dataset

HIDDEN = 64
EPOCHS = 10000
LR     = 1e-3
BATCH  = 32
SEED   = 42

LOG = logging.getLogger(__name__)
torch.manual_seed(SEED)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

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
        self.log_activations = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _log(name: str, t: torch.Tensor) -> torch.Tensor:
            if self.log_activations:
                LOG.debug("%-12s  min=%8.3f  max=%8.3f  std=%8.3f", name, t.min().item(), t.max().item(), t.std().item())
            return t

        x = _log("input",  x)
        x = _log("fc1",    self.fc1(x))
        x = _log("act1",   self.act1(x))
        x = _log("fc2",    self.fc2(x))
        x = _log("act2",   self.act2(x))
        x = _log("fc3",    self.fc3(x))
        return x


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_mlp(ds: Esc50Dataset, feat: str, mdl_root: str, min_acc: float, patience: int) -> MLP:
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

    input_dim = ds.X.shape[1]
    model = MLP(input_dim=input_dim, hidden=HIDDEN, num_classes=len(ds.classes))
    LOG.info("Model: input=%d hidden=%d classes=%d feature='%s'", input_dim, HIDDEN, len(ds.classes), feat)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(mdl_root, exist_ok=True)

    accs = [0.0]
    n_worse = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        acc = correct / total
        loss = total_loss / total

        if acc == 1.0 or epoch % 50 == 0 or epoch == EPOCHS - 1:
            LOG.info("epoch %4d  loss %.4f  acc %.3f", epoch, loss, acc)

        dname_mdl = f"mlp_{feat}_e{epoch:04d}_acc={acc:.3f}"
        if acc <= accs[-1]:
            n_worse += 1
            if n_worse >= patience:
                break
        else:
            n_worse = 0
            accs.append(acc)
            if acc > min_acc:
                LOG.info("epoch %4d  loss %.4f  acc %.3f <- BEST (%s)", epoch, loss, acc, dname_mdl)
                dpath_mdl = os.path.join(mdl_root, dname_mdl)
                os.makedirs(dpath_mdl, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(dpath_mdl, "model.pt"))
                with open(os.path.join(dpath_mdl, "meta.json"), "w") as f:
                    json.dump({"feat": feat, "classes": ds.classes, "input_dim": input_dim, "hidden": HIDDEN}, f, indent=2)
                # LOG.info("Saved -> %s", dpath_mdl)
            if acc == 1.0:
                break

    return model


def train_all(search_dir: str, mdl_root: str, esc50_metafile: str, esc10: bool,
              h5_prefix: str, min_acc: float = 0.9, patience: int = 20) -> None:
    h5_files = [
        os.path.join(root, fname)
        for root, _, files in os.walk(search_dir)
        for fname in files
        if fname.startswith(h5_prefix) and fname.endswith(".h5")
    ]
    if not h5_files:
        LOG.warning("No .h5 files found under %s", search_dir)
        return

    LOG.info("Found %d .h5 file(s)", len(h5_files))
    for fpath_h5 in h5_files:
        ds = Esc50Dataset(esc50_metafile, esc10=esc10)
        ds.load_features(fpath_h5)
        feat = os.path.splitext(os.path.basename(fpath_h5))[0]
        train_mlp(ds, feat, mdl_root, min_acc=min_acc, patience=patience)
