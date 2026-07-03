import csv
import json
import logging
import os

import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.realpath(__file__))
EXP_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
REPO_DIR    = os.path.abspath(os.path.join(EXP_DIR, ".."))
BLD_DIR     = os.path.join(REPO_DIR, "build")
RESULTS_DIR = os.path.join(BLD_DIR, "results")
MDL_DIR     = os.path.join(BLD_DIR, "mdl")
ESC50_META  = os.path.join(REPO_DIR, "assets/esc-50/ESC-50-master/meta/esc50.csv")

LOG = logging.getLogger(__name__)

MODES = ["plaintext", "fhe"]


def load_esc50_meta(fpath: str) -> tuple[dict, dict]:
    stem_to_target = {}
    target_to_category = {}
    with open(fpath) as f:
        for row in csv.DictReader(f):
            stem = os.path.splitext(row["filename"])[0]
            target = int(row["target"])
            stem_to_target[stem] = target
            target_to_category[target] = row["category"]
    return stem_to_target, target_to_category


def load_classes(feat: str, target_to_category: dict, esc10_only: bool = False) -> list:
    meta_path = os.path.join(MDL_DIR, f"mlp-{feat}", "meta.json")
    try:
        with open(meta_path) as f:
            return json.load(f)["classes"]
    except FileNotFoundError:
        categories = target_to_category.values()
        if esc10_only:
            esc10_targets = {0, 1, 10, 11, 12, 20, 21, 38, 40, 41}
            categories = [target_to_category[t] for t in sorted(esc10_targets)]
        return sorted(set(categories))


def evaluate(fpath: str, stem_to_target: dict, target_to_category: dict, classes: list) -> dict:
    category_to_idx = {c: i for i, c in enumerate(classes)}

    y_true, y_pred = [], []
    with open(fpath) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            stem = parts[0]
            scores = np.array([float(x) for x in parts[1:]])
            pred = int(np.argmax(scores))

            target = stem_to_target.get(stem)
            if target is None:
                LOG.warning("Stem not in CSV: %s", stem)
                continue
            category = target_to_category.get(target)
            true_idx = category_to_idx.get(category)
            if true_idx is None:
                LOG.warning("Category not in model classes: %s", category)
                continue

            y_true.append(true_idx)
            y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = (y_true == y_pred)

    per_class = {}
    for i, cls in enumerate(classes):
        mask = y_true == i
        per_class[cls] = float(correct[mask].mean()) if mask.sum() > 0 else None

    return {"overall": float(correct.mean()), "per_class": per_class, "n": len(y_true)}


def print_table(feat: str, results: dict) -> None:
    active_modes = [m for m in MODES if results.get(m) is not None]
    classes = list(next(r for r in results.values() if r)["per_class"].keys())

    print(f"\n{'='*56}")
    print(f"  {feat}")
    print(f"{'='*56}")
    print(f"{'Class':<22}" + "".join(f"{m:>16}" for m in active_modes))
    print(f"{'-'*22}" + "".join(f"{'-'*16}" for _ in active_modes))

    for cls in classes:
        row = f"{cls:<22}"
        for mode in active_modes:
            acc = results[mode]["per_class"][cls]
            row += f"{100*acc:>14.1f}%" if acc is not None else f"{'—':>15}"
        print(row)

    print(f"{'-'*22}" + "".join(f"{'-'*16}" for _ in active_modes))
    row = f"{'OVERALL':<22}"
    for mode in active_modes:
        row += f"{100*results[mode]['overall']:>14.1f}%"
    print(row)
    n = next(r["n"] for r in results.values() if r)
    print(f"  (n={n})")


def main() -> None:
    stem_to_target, target_to_category = load_esc50_meta(ESC50_META)

    plaintext_dir = os.path.join(RESULTS_DIR, "plaintext")
    if not os.path.isdir(plaintext_dir):
        LOG.error("Results dir not found: %s", plaintext_dir)
        return

    feats = sorted(
        os.path.splitext(fn)[0]
        for fn in os.listdir(plaintext_dir)
        if fn.endswith(".txt")
    )

    for feat in feats:
        classes = load_classes(feat, target_to_category)

        results = {}
        for mode in MODES:
            fpath = os.path.join(RESULTS_DIR, mode, f"{feat}.txt")
            if not os.path.isfile(fpath):
                LOG.warning("Missing result file: %s", fpath)
                results[mode] = None
                continue
            results[mode] = evaluate(fpath, stem_to_target, target_to_category, classes)

        print_table(feat, results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
