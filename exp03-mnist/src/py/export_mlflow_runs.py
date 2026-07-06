"""
Export MLflow run params + best-epoch metrics for the exp03 MNIST experiment to CSV.

For each run, "best epoch" is the epoch with the highest test_acc; train_acc and
train_loss are taken from that same epoch (not their own best/final values).

Usage:
    .venv/bin/python exp03-mnist/src/py/export_mlflow_runs.py [output.csv]
"""

import logging
import os
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from common import BLD_DIR

LOG = logging.getLogger(__name__)

EXPERIMENT_NAME = "CKKS - exp03 - MNIST"


def _best_epoch_metrics(client: MlflowClient, run_id: str) -> dict:
    test_acc_hist = client.get_metric_history(run_id, "test_acc")
    if not test_acc_hist:
        return {"best_epoch": None, "test_acc": None, "train_acc": None, "train_loss": None}

    train_acc_by_step  = {m.step: m.value for m in client.get_metric_history(run_id, "train_acc")}
    train_loss_by_step = {m.step: m.value for m in client.get_metric_history(run_id, "train_loss")}

    best = max(test_acc_hist, key=lambda m: m.value)
    return {
        "best_epoch": best.step,
        "test_acc":   best.value,
        "train_acc":  train_acc_by_step.get(best.step),
        "train_loss": train_loss_by_step.get(best.step),
    }


def export_runs_csv(fpath_out: str) -> None:
    fpath_db = os.path.join(BLD_DIR, "mlflow.db")
    if not os.path.isfile(fpath_db):
        raise FileNotFoundError(f"Cannot find MLFlow DB at: {fpath_db}")
    mlflow.set_tracking_uri("sqlite:///" + fpath_db)
    client = MlflowClient()

    df = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    param_cols = [c for c in df.columns if c.startswith("params.")]
    df = df[["run_id", "status", "start_time", "end_time"] + param_cols].reset_index(drop=True)
    df.columns = [c.removeprefix("params.") for c in df.columns]

    best_df = pd.DataFrame([_best_epoch_metrics(client, run_id) for run_id in df["run_id"]])
    df = pd.concat([df, best_df], axis=1)

    df.to_csv(fpath_out, index=False)
    LOG.info("Wrote %d runs -> %s", len(df), fpath_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fpath_out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BLD_DIR, "exp03_runs.csv")
    export_runs_csv(fpath_out)
