#!/usr/bin/env python3
"""
train.py  --  Offline (plaintext) training for the FHE acoustic-scene demo.

Pipeline:
  1. Load a 5-class subset of ESC-50.
  2. Extract MFCC feature vectors (fixed length -> one vector per clip).
  3. Train a tiny 2-layer MLP with SQUARE activation (h = (W1 x + b1)^2),
     because x^2 is the cheapest non-linearity to evaluate under CKKS.
  4. Export normalization stats + weights as plain text the C++ server reads.

We deliberately keep the network tiny and the activation polynomial so that
the homomorphic inference stays at multiplicative depth 3.
"""

import os
import json
import numpy as np

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
# 5 "natural soundscape" classes from ESC-50. Edit to taste.
CLASSES = ["rain", "sea_waves", "crackling_fire", "wind", "thunderstorm"]

N_MFCC      = 16          # MFCC coefficients -> input feature dimension
HIDDEN      = 16          # hidden units (keep == input dim to ease packing)
SR          = 22050       # resample rate
DURATION    = 5.0         # ESC-50 clips are 5s
SEED        = 0

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "model")

np.random.seed(SEED)


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def extract_features(wav_path):
    """One fixed-length MFCC-mean vector per clip."""
    import librosa
    y, _ = librosa.load(wav_path, sr=SR, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    # mean over time -> (N_MFCC,) ; simple + keeps the input small
    return mfcc.mean(axis=1)


def load_dataset(esc50_dir):
    """
    Expects standard ESC-50 layout:
        esc50_dir/meta/esc50.csv
        esc50_dir/audio/*.wav
    Returns X (n, N_MFCC), y (n,) with labels indexed into CLASSES.
    """
    import csv
    meta = os.path.join(esc50_dir, "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_dir, "audio")

    X, y = [], []
    with open(meta) as f:
        for row in csv.DictReader(f):
            cat = row["category"]
            if cat not in CLASSES:
                continue
            feat = extract_features(os.path.join(audio_dir, row["filename"]))
            X.append(feat)
            y.append(CLASSES.index(cat))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


# ----------------------------------------------------------------------
# Tiny MLP with square activation, trained by plain gradient descent
# ----------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_mlp(X, y, epochs=400, lr=0.05):
    n, d = X.shape
    k = len(CLASSES)
    rng = np.random.default_rng(SEED)

    # Small init keeps pre-activations in a range where x^2 behaves well.
    W1 = rng.normal(0, 0.3, size=(HIDDEN, d))
    b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, 0.3, size=(k, HIDDEN))
    b2 = np.zeros(k)

    Y = np.eye(k)[y]

    for ep in range(epochs):
        # forward
        z1 = X @ W1.T + b1          # (n, HIDDEN)
        h  = z1 ** 2                # square activation
        z2 = h @ W2.T + b2          # (n, k)
        p  = softmax(z2)

        loss = -np.mean(np.sum(Y * np.log(p + 1e-9), axis=1))

        # backward
        dz2 = (p - Y) / n           # (n, k)
        dW2 = dz2.T @ h
        db2 = dz2.sum(axis=0)
        dh  = dz2 @ W2              # (n, HIDDEN)
        dz1 = dh * (2 * z1)        # d/dz1 of z1^2
        dW1 = dz1.T @ X
        db1 = dz1.sum(axis=0)

        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

        if ep % 50 == 0 or ep == epochs - 1:
            acc = (p.argmax(1) == y).mean()
            print(f"epoch {ep:4d}  loss {loss:.4f}  train_acc {acc:.3f}")

    return W1, b1, W2, b2


# ----------------------------------------------------------------------
# Export: pad to power-of-two for clean rotations on the server
# ----------------------------------------------------------------------
def next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def diagonals(mat, dim):
    """
    Return the `dim` generalized diagonals of a (dim x dim) matrix,
    diag_i[j] = mat[j, (j+i) % dim].  This is the Halevi-Shoup encoding
    consumed by the server's rotate-and-sum matvec.
    """
    diags = []
    for i in range(dim):
        diags.append([mat[j][(j + i) % dim] for j in range(dim)])
    return diags


def export(W1, b1, W2, b2, mean, std, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    d   = W1.shape[1]          # input dim
    h   = W1.shape[0]          # hidden dim
    k   = W2.shape[0]          # classes
    dim = next_pow2(max(d, h, k))   # common square dimension for packing

    def pad_mat(m, rows, cols):
        out = np.zeros((rows, cols))
        out[:m.shape[0], :m.shape[1]] = m
        return out

    # Pad every matrix to (dim x dim) so rotations wrap cleanly.
    W1p = pad_mat(W1, dim, dim)
    W2p = pad_mat(W2, dim, dim)
    b1p = np.zeros(dim); b1p[:h] = b1
    b2p = np.zeros(dim); b2p[:k] = b2
    meanp = np.zeros(dim); meanp[:d] = mean
    stdp  = np.ones(dim);  stdp[:d]  = std   # ones avoid div-by-zero in pad region

    model = {
        "input_dim": int(d),
        "hidden_dim": int(h),
        "num_classes": int(k),
        "packed_dim": int(dim),
        "classes": CLASSES,
        "mean": meanp.tolist(),
        "std": stdp.tolist(),
        "W1_diag": diagonals(W1p, dim),
        "b1": b1p.tolist(),
        "W2_diag": diagonals(W2p, dim),
        "b2": b2p.tolist(),
    }
    path = os.path.join(out_dir, "model.json")
    with open(path, "w") as f:
        json.dump(model, f)
    print(f"\nExported model -> {path}")
    print(f"packed_dim = {dim}  (input {d}, hidden {h}, classes {k})")


# ----------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", help="path to ESC-50 root (meta/ + audio/)")
    ap.add_argument("--synthetic", action="store_true",
                    help="skip audio, train on synthetic separable data (smoke test)")
    args = ap.parse_args()

    if args.synthetic:
        print("Using synthetic data (smoke test, no audio needed).")
        n_per = 60
        k = len(CLASSES)
        centers = np.random.default_rng(SEED).normal(0, 3, size=(k, N_MFCC))
        X = np.vstack([centers[c] + np.random.normal(0, 1, size=(n_per, N_MFCC))
                       for c in range(k)])
        y = np.repeat(np.arange(k), n_per)
    else:
        if not args.esc50:
            raise SystemExit("Provide --esc50 PATH or use --synthetic")
        print("Extracting MFCC features from ESC-50 ...")
        X, y = load_dataset(args.esc50)
        print(f"Loaded {len(X)} clips across {len(CLASSES)} classes.")

    # normalize (client will apply the same mean/std before encrypting)
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-6
    Xn = (X - mean) / std

    W1, b1, W2, b2 = train_mlp(Xn, y)
    export(W1, b1, W2, b2, mean, std, MODEL_DIR)


if __name__ == "__main__":
    main()
