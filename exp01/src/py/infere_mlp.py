import json
import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

from common import Esc10Dataset
from train_mlp import MLP

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EXP_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
BLD_DIR    = os.path.join(EXP_DIR, "build")
MDL_ROOT   = os.path.join(BLD_DIR, "mdl")
FEAT_ROOT  = os.path.join(BLD_DIR, "fea")

LOG = logging.getLogger(__name__)


def load_model(dpath_mdl: str) -> tuple[MLP, dict]:
    with open(os.path.join(dpath_mdl, "meta.json")) as f:
        meta = json.load(f)
    model = MLP(input_dim=meta["input_dim"], hidden=meta["hidden"], num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(os.path.join(dpath_mdl, "model.pt"), weights_only=True))
    model.eval()
    return model, meta


def infer(dpath_mdl: str) -> None:
    # derive feature file from model dir name: "mlp-esc10-mfb_40" -> "esc10-mfb_40.h5"
    feat = os.path.basename(dpath_mdl).removeprefix("mlp-")
    fpath_h5 = os.path.join(FEAT_ROOT, f"{feat}.h5")

    if not os.path.isfile(fpath_h5):
        LOG.warning("Feature file not found, skipping: %s", fpath_h5)
        return

    model, meta = load_model(dpath_mdl)
    classes = meta["classes"]
    LOG.info("Model: MLP-%s  classes=%d  input=%d", feat, len(classes), meta["input_dim"])

    ds = Esc10Dataset()
    ds.load_features(fpath_h5)

    activations: dict[str, np.ndarray] = {}
    def _hook(name: str):
        def fn(_, __, output):
            activations[name] = output.detach().cpu().numpy().flatten()
        return fn

    layer_order = ["fc1", "act1", "fc2", "act2", "fc3"]
    handles = [
        model.fc1.register_forward_hook(_hook("fc1")),
        model.act1.register_forward_hook(_hook("act1")),
        model.fc2.register_forward_hook(_hook("fc2")),
        model.act2.register_forward_hook(_hook("act2")),
        model.fc3.register_forward_hook(_hook("fc3")),
    ]

    model.log_activations = True
    X = torch.from_numpy(ds.X)
    with torch.no_grad():
        logits = model(X)
    model.log_activations = False

    for h in handles:
        h.remove()

    mdl_name = os.path.basename(dpath_mdl)
    dpath_plots = os.path.join(dpath_mdl, "activations")
    os.makedirs(dpath_plots, exist_ok=True)
    sns.set_theme(style="darkgrid")
    for i, layer_name in enumerate(layer_order):
        vals = activations.get(layer_name)
        if vals is None:
            continue
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(vals, bins=50, kde=True, ax=ax)
        ax.set_title(f"{mdl_name}  /  {layer_name}  "
                     f"(min={vals.min():.2f}  max={vals.max():.2f}  std={vals.std():.2f})")
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fpath_png = os.path.join(dpath_plots, f"{i+1:02d}_{layer_name}.png")
        fig.savefig(fpath_png, dpi=110)
        plt.close(fig)
        LOG.debug("Saved %s", os.path.relpath(fpath_png, BLD_DIR))
    preds = logits.argmax(dim=1).numpy()

    correct = (preds == ds.y).sum()
    total = len(ds.y)
    LOG.info("Accuracy: %d / %d  (%.1f%%)", correct, total, 100.0 * correct / total)

    for i, c in enumerate(classes):
        mask = ds.y == i
        if mask.sum() == 0:
            continue
        acc_c = (preds[mask] == i).sum() / mask.sum()
        LOG.info("  %-20s  %5.1f%%", c, 100.0 * acc_c)


def infer_all(mdl_root: str = MDL_ROOT) -> None:
    mdl_dirs = [
        os.path.join(mdl_root, d)
        for d in sorted(os.listdir(mdl_root))
        if d.startswith("mlp-") and os.path.isfile(os.path.join(mdl_root, d, "model.pt"))
    ] if os.path.isdir(mdl_root) else []

    if not mdl_dirs:
        LOG.warning("No trained models found under %s", mdl_root)
        return
    LOG.info("Found %d model(s)", len(mdl_dirs))
    for dpath_mdl in mdl_dirs:
        LOG.info("--- %s ---", os.path.basename(dpath_mdl))
        infer(dpath_mdl)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    infer_all()
