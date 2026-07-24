"""
Compute accuracy of a CKKS inference .hyp file against ground-truth MNIST
test labels.

.hyp line format (from infer_mlp / run-inference.sh): <id>\t<logit_0>\t...\t<logit_9>
Predicted class is argmax of the logits; ids are indices into build/fea/mnist-test.h5's "y".
"""

import argparse
import logging
import os

import h5py
import numpy as np

from common import BLD_DIR

LOG = logging.getLogger(__name__)


def eval_hyp(fpath_hyp: str, fpath_labels_h5: str) -> None:
    with h5py.File(fpath_labels_h5, "r") as f:
        y = f["y"][:]

    total, correct = 0, 0
    mismatches = []
    with open(fpath_hyp) as f:
        for line in f:
            idx_str, *logit_strs = line.rstrip("\n").split("\t")
            idx = int(idx_str)
            pred = int(np.argmax([float(v) for v in logit_strs]))
            true = int(y[idx])
            total += 1
            if pred == true:
                correct += 1
            else:
                mismatches.append((idx, true, pred))

    LOG.info("Evaluated %d sample(s) from %s", total, fpath_hyp)
    LOG.info("Accuracy: %d/%d = %.2f%%", correct, total, 100 * correct / total)
    if mismatches:
        LOG.info("First mismatches (id, true, pred): %s", mismatches[:10])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hyp", help="path to a .hyp file produced by infer_mlp")
    parser.add_argument("--labels", default=os.path.join(BLD_DIR, "fea", "mnist-test.h5"), help="path to the labeled dataset (default: build/fea/mnist-test.h5)")
    args = parser.parse_args()

    eval_hyp(args.hyp, args.labels)
