"""
Walk build/mdl/exp03 and, for each parameter setting (one directory per
param_hash, e.g. mlp-mnist_<hash>/ or cnn-mnist_<hash>/), find and print the
last (highest-epoch) model dump — the one saved from that setting's best
checkpoint, since checkpoints are only ever saved on a new best test_acc.
Settings that never crossed min_acc have no checkpoint; "-" is printed in
its place, but best_test_acc/config fields still show since those are
written unconditionally (result.json/config.json), independent of whether
a checkpoint was ever saved.

Pass --trim to additionally delete every non-best checkpoint dir for each
setting — keeps only the last (highest-accuracy) one, permanently removing
the earlier model.pt/meta.json dumps. Off by default; this is real deletion,
not reversible like MLflow's soft-delete.

Usage:
    .venv/bin/python exp03-mnist/src/py/trim-build-dir.py [--trim]
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys

from common import BLD_DIR

MDL_ROOT = os.path.join(BLD_DIR, "mdl", "exp03")

EPOCH_DIR_RE = re.compile(r"^e(\d+)$")

LOG = logging.getLogger(__name__)


def _checkpoints(dpath_run: str) -> list[tuple[int, str]]:
    checkpoints = []
    for name in os.listdir(dpath_run):
        m = EPOCH_DIR_RE.match(name)
        if m and os.path.isfile(os.path.join(dpath_run, name, "model.pt")):
            checkpoints.append((int(m.group(1)), name))
    checkpoints.sort()
    return checkpoints


def _load_json(fpath: str) -> dict:
    if not os.path.isfile(fpath):
        return {}
    with open(fpath) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trim", action="store_true",
                         help="Delete every non-best checkpoint dir per setting (keeps only the last one)")
    args = parser.parse_args()

    if not os.path.isdir(MDL_ROOT):
        print(f"No model dumps found under {MDL_ROOT}", file=sys.stderr)
        return

    rows = []
    deleted_count = 0
    for dname_run in sorted(os.listdir(MDL_ROOT)):
        dpath_run = os.path.join(MDL_ROOT, dname_run)
        if not os.path.isdir(dpath_run):
            continue

        checkpoints = _checkpoints(dpath_run)

        if args.trim:
            for _, name in checkpoints[:-1]:
                dpath_old = os.path.join(dpath_run, name)
                shutil.rmtree(dpath_old)
                LOG.info("deleted %s", dpath_old)
                deleted_count += 1
            continue

        last = checkpoints[-1][1] if checkpoints else "-"

        cfg    = _load_json(os.path.join(dpath_run, "config.json"))
        result = _load_json(os.path.join(dpath_run, "result.json"))

        best_test_acc = result.get("best_test_acc", float("nan"))
        param_hash    = cfg.get("param_hash", "?")
        activation    = cfg.get("activation", "?")
        dropout       = cfg.get("dropout", float("nan"))
        lr            = cfg.get("lr", float("nan"))
        layers        = cfg.get("layers", [])
        if len(layers) >= 2:
            layers = layers[1:-1]  # input/output layers are fixed and shared across all configs

        rows.append((best_test_acc, last, param_hash, activation, dropout, lr, layers))

    if args.trim:
        if deleted_count == 0:
            LOG.info("No directories were deleted")
        else:
            LOG.info("Deleted %d director%s", deleted_count, "y" if deleted_count == 1 else "ies")
        return

    # nan (no result.json yet — still-running/orphaned run) sorts last, not first
    rows.sort(key=lambda row: row[0] if row[0] == row[0] else -1.0, reverse=True)

    for i, (best_test_acc, last, param_hash, activation, dropout, lr, layers) in enumerate(rows, 1):
        serial = f"{i}."
        print(f"{serial:<5s}  {last:<6s}  {best_test_acc:.5f}  {param_hash:<12s}  {activation:<9s}  {dropout:.5f}  {lr:.5f}  {str(layers):<30s}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    main()
