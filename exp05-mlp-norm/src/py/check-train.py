"""
Fast smoke test for exp05: wire up data -> model -> training loop and confirm the
whole thing learns. Not the real trainer — no MLflow, no early stopping, no model
dumping; those belong in `train_mlp_norm.py` (cf. exp03's `train_mlp.py`).

Defaults finish in well under a minute even on CPU (8k train samples, 3 epochs).

Usage:
    PYTHONPATH=src/py .venv/bin/python src/py/check-train.py
    PYTHONPATH=src/py .venv/bin/python src/py/check-train.py --norm batch --activation Quadratic
    PYTHONPATH=src/py .venv/bin/python src/py/check-train.py --all-norms      # compare none/batch/layer
"""

import argparse
import logging
import time

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from mlp_norm import ACTIVATION_MAP, NORM_MODES, MlpNorm, is_ckks_compatible
from mnist_data import MnistDataset, eval_acc, fea_path

LOG  = logging.getLogger(__name__)
SEED = 42

# a smoke test that learns nothing is a broken smoke test, not a bad model
MIN_TEST_ACC = 0.80


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers",      type=int, nargs="+", default=[784, 256, 10])
    p.add_argument("--activation",  choices=sorted(ACTIVATION_MAP), default="ReLU")
    p.add_argument("--norm",        choices=NORM_MODES, default="none")
    p.add_argument("--all-norms",   action="store_true", help="run every norm mode and print a comparison")
    p.add_argument("--dropout",     type=float, default=0.0)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch",       type=int,   default=64)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--n-train",     type=int,   default=8000, help="train subset size (0 = all 60000)")
    p.add_argument("--n-test",      type=int,   default=2000, help="test subset size (0 = all 10000)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--min-acc",     type=float, default=MIN_TEST_ACC, help="fail below this test accuracy")
    return p.parse_args()


def run_one(args: argparse.Namespace, norm: str, train_ds: MnistDataset, test_ds: MnistDataset) -> float:
    torch.manual_seed(SEED)
    device = torch.device(args.device)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    model = MlpNorm(
        layers     = args.layers,
        activation = ACTIVATION_MAP[args.activation],
        norm       = norm,
        dropout    = args.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    n_param = sum(p.numel() for p in model.parameters())
    LOG.info("norm=%-5s  layers=%s  act=%s  params=%d  ckks=%s  device=%s",
             norm, args.layers, args.activation, n_param,
             is_ckks_compatible(args.activation, norm), device)

    t0 = time.time()
    test_acc = 0.0
    for epoch in range(args.epochs):
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

        test_acc = eval_acc(model, test_loader, device)
        LOG.info("  epoch %2d  loss %.4f  train_acc %.4f  test_acc %.4f",
                 epoch, total_loss / total, correct / total, test_acc)

    LOG.info("  norm=%-5s done in %.1fs  test_acc=%.4f", norm, time.time() - t0, test_acc)
    return test_acc


def main() -> None:
    args = parse_args()

    assert args.layers[0] == 784, f"MNIST input dim is 784, got {args.layers[0]}"
    assert args.layers[-1] == 10, f"MNIST has 10 classes, got {args.layers[-1]}"

    train_ds = MnistDataset(fea_path("train"), limit=args.n_train or None)
    test_ds  = MnistDataset(fea_path("test"), mean=train_ds.mean, std=train_ds.std, limit=args.n_test or None)
    LOG.info("train=%d  test=%d  input_dim=%d", len(train_ds), len(test_ds), train_ds.X.shape[1])

    modes = list(NORM_MODES) if args.all_norms else [args.norm]
    accs = {norm: run_one(args, norm, train_ds, test_ds) for norm in modes}

    LOG.info("--- summary (%d epochs, %d train samples) ---", args.epochs, len(train_ds))
    for norm, acc in accs.items():
        LOG.info("  %-5s  test_acc %.4f  ckks=%s", norm, acc, is_ckks_compatible(args.activation, norm))

    failed = {n: a for n, a in accs.items() if a < args.min_acc}
    if failed:
        raise SystemExit(f"FAIL: test_acc below --min-acc={args.min_acc}: "
                         + ", ".join(f"{n}={a:.4f}" for n, a in failed.items()))
    LOG.info("OK: all runs above --min-acc=%.2f", args.min_acc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    main()
