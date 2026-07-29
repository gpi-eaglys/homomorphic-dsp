"""
Export a trained CKKS-compatible (Quadratic activation, avg-pool) CNN to
ckks_model.json.

Homomorphic conv/pool never repacks the ciphertext: each channel is one
packed_dim=1024 ciphertext holding a *fixed* 28x28 physical buffer
(w_phys=28) for the whole network, regardless of channel count or how far
pooling has shrunk the logical resolution. A logical pixel (row, col) at
the current `stride` lives at physical slot row*stride*w_phys + col*stride
— pooling just doubles `stride` and floor-halves h_active/w_active instead
of compacting the ciphertext. This is safe because every later rotation
offset is a multiple of the current stride, so a rotation always maps an
active slot to another active slot; garbage sitting at non-active slots
can never leak into one.

AvgPool2d's 1/4 scale is never applied on its own (that would cost an
extra multiplicative level) — it's folded into the *weight* constants of
the next linear layer (next conv's per-term plaintext, or FC1's diagonal
values). It must never be folded into that layer's bias: the true pooled
input is 0.25*pooled_unscaled, so z = W_next @ (0.25*pooled_unscaled) +
bias_next = (0.25*W_next) @ pooled_unscaled + bias_next.

Because this bookkeeping (stride tracking, SAME-padding masks at the
*logical* grid edge, scale-fold placement) is easy to get subtly wrong,
export refuses to write ckks_model.json unless a pure-numpy simulator of
the exact same slot-level algorithm reproduces the real PyTorch model's
forward pass first — see _validate_via_simulator().
"""

import json
import logging
import os

import h5py
import numpy as np
import torch
import torch.nn as nn

from common import BLD_DIR
from mnist_data import Quadratic
from train_cnn import CNN

LOG = logging.getLogger(__name__)

W_PHYS = 28

ACTIVATION_MAP: dict = {"GELU": nn.GELU, "ReLU": nn.ReLU, "Sigmoid": nn.Sigmoid, "Quadratic": Quadratic}

# Duplicated verbatim from exp03-mnist/src/py/export_mdl.py — tiny, stable
# pure functions; not worth a cross-experiment import (this file already
# shares its own name with exp03's, which would collide on sys.path).
def next_pow2(n: int) -> int:
    return 1 << (max(n, 1) - 1).bit_length()


def diagonals(W: np.ndarray, dim: int) -> list[list[float]]:
    Wp = np.zeros((dim, dim))
    Wp[:W.shape[0], :W.shape[1]] = W
    idx = np.arange(dim)
    return [Wp[idx, (idx + i) % dim].tolist() for i in range(dim)]


def pad(v: np.ndarray, dim: int) -> list[float]:
    out = np.zeros(dim)
    out[:len(v)] = v
    return out.tolist()


def phys_addr(row: int, col: int, stride: int, w_phys: int = W_PHYS) -> int:
    return row * stride * w_phys + col * stride


def conv_mask(ky: int, kx: int, h_active: int, w_active: int, stride: int, dim: int, w_phys: int = W_PHYS) -> np.ndarray:
    """1 at physical slot phys_addr(row,col,stride) iff (row+ky,col+kx) is a valid *logical*
    position — i.e. SAME zero-padding at the logical grid edge, robust to floor-division
    pool drops shrinking h_active/w_active below the physical buffer size."""
    m = np.zeros(dim)
    for row in range(h_active):
        for col in range(w_active):
            if 0 <= row + ky < h_active and 0 <= col + kx < w_active:
                m[phys_addr(row, col, stride, w_phys)] = 1.0
    return m


def pool_rotations(stride: int, w_phys: int = W_PHYS) -> set:
    return {stride, stride * w_phys, stride * w_phys + stride}


def export_conv_layer(conv: nn.Conv2d, stride: int, h_active: int, w_active: int, dim: int,
                       pending_scale: float, w_phys: int = W_PHYS) -> tuple:
    """One term per (c_in, ky, kx): rotate x[c_in] once by `shift` (reused across every
    c_out below — rotation count is c_in*k*k, independent of c_out), then for each c_out
    multiply by a single plaintext = mask(ky,kx) * weight[c_out,c_in,ky,kx] * pending_scale
    — folding the SAME-padding mask and the deferred pool scale into the weight means each
    term costs exactly one EvalMult, same as a plain Linear layer."""
    W = conv.weight.detach().numpy()  # (c_out, c_in, k, k)
    b = conv.bias.detach().numpy()    # (c_out,)
    c_out, c_in, k, _ = W.shape
    half = k // 2

    terms = []
    rotations = set()
    for ci in range(c_in):
        for ky in range(-half, half + 1):
            for kx in range(-half, half + 1):
                shift = (ky * stride) * w_phys + kx * stride
                mask = conv_mask(ky, kx, h_active, w_active, stride, dim, w_phys)
                plaintexts = [(mask * (W[co, ci, ky + half, kx + half] * pending_scale)).tolist()
                              for co in range(c_out)]
                terms.append({"c_in_idx": ci, "shift": int(shift), "plaintexts": plaintexts})
                if shift != 0:
                    rotations.add(int(shift))

    layer_doc = {
        "c_in": c_in, "c_out": c_out, "kernel_size": k,
        "h_active": h_active, "w_active": w_active, "pool_stride_before": stride,
        "terms": terms, "bias": b.tolist(),  # bias is UNSCALED — pending_scale never touches bias
    }
    return layer_doc, rotations


def export_fc1(linear: nn.Linear, in_channels: int, h_active: int, w_active: int, stride: int,
                dim: int, pending_scale: float, w_phys: int = W_PHYS) -> tuple:
    """Flatten(1) gives column order c*h_active*w_active + row*w_active + col. Still
    C_last separate per-channel ciphertexts, so build C_last independent Halevi-Shoup
    matVecs: place W[out_idx, ...] only at this channel's true physical slot address and
    zero elsewhere (so leftover garbage at non-active slots is multiplied by 0 — harmless
    regardless of its value), then keep only the diagonals that are actually nonzero
    (measured, not assumed — the true sparsity is topology-dependent)."""
    W = linear.weight.detach().numpy()  # (fc_out, in_channels*h_active*w_active)
    b = linear.bias.detach().numpy()
    fc_out = W.shape[0]
    n_per_channel = h_active * w_active

    channels = []
    rotations = set()
    for ci in range(in_channels):
        cols = W[:, ci * n_per_channel:(ci + 1) * n_per_channel]  # (fc_out, h_active*w_active)
        Wp = np.zeros((dim, dim))
        k_idx = 0
        for row in range(h_active):
            for col in range(w_active):
                addr = phys_addr(row, col, stride, w_phys)
                Wp[:fc_out, addr] = cols[:, k_idx] * pending_scale
                k_idx += 1
        diags = diagonals(Wp, dim)
        sparse = [{"shift": i, "values": d} for i, d in enumerate(diags) if any(v != 0.0 for v in d)]
        channels.append({"c_in_idx": ci, "diagonals": sparse})
        rotations |= {d["shift"] for d in sparse if d["shift"] != 0}

    fc1_doc = {
        "in_channels": in_channels, "h_active": h_active, "w_active": w_active,
        "out_features": fc_out, "channels": channels, "bias": pad(b, dim),  # bias UNSCALED
    }
    return fc1_doc, rotations


def _hrot(x: np.ndarray, shift: int) -> np.ndarray:
    """Matches OpenFHE's EvalRotate: rotate(x, shift)[j] = x[(j+shift) % dim]."""
    return np.roll(x, -shift)


def _simulate(doc: dict, x0: np.ndarray) -> np.ndarray:
    """Pure-numpy replay of the exact slot-level algorithm infer_cnn.cpp will run on
    ciphertexts, operating on plain float arrays instead. Used only to self-validate
    export_mdl.py's geometry/masking/scale-folding logic against the real PyTorch model."""
    dim = doc["packed_dim"]
    x = {0: x0}
    for conv in doc["conv_layers"]:
        z = [np.zeros(dim) for _ in range(conv["c_out"])]
        for term in conv["terms"]:
            r = _hrot(x[term["c_in_idx"]], term["shift"])
            for co in range(conv["c_out"]):
                z[co] = z[co] + r * np.array(term["plaintexts"][co])
        stride = conv["pool_stride_before"]
        pooled = {}
        for co in range(conv["c_out"]):
            z[co] = (z[co] + conv["bias"][co]) ** 2  # Quadratic activation — applies to every conv layer
            pooled[co] = (z[co] + _hrot(z[co], stride) + _hrot(z[co], stride * doc["w_phys"])
                          + _hrot(z[co], stride * doc["w_phys"] + stride))  # unscaled avg-pool sum
        x = pooled

    fc1 = doc["fc1"]
    acc = np.zeros(dim)
    for ch in fc1["channels"]:
        xc = x[ch["c_in_idx"]]
        for d in ch["diagonals"]:
            acc = acc + _hrot(xc, d["shift"]) * np.array(d["values"])
    acc = acc + np.array(fc1["bias"])

    fc_dense = doc["fc_dense"]
    if fc_dense:
        acc = acc ** 2  # FC1 is a hidden layer (more linear layers follow) -> activated
        for i, layer in enumerate(fc_dense):
            W_diag = layer["W_diag"]
            z = np.zeros(dim)
            for shift in range(dim):
                z = z + _hrot(acc, shift) * np.array(W_diag[shift])
            z = z + np.array(layer["bias"])
            if i < len(fc_dense) - 1:  # activation on every FC layer except the final output layer
                z = z ** 2
            acc = z
    return acc[:doc["num_classes"]]


def _validate_via_simulator(doc: dict, model: nn.Module, mean: np.ndarray, std: np.ndarray,
                             n_samples: int = 20, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    """Combined relative+absolute tolerance (np.allclose semantics): the Quadratic
    activation repeatedly squares, so an untrained/large-weight model's logits can reach
    the thousands, at which point a fixed absolute tolerance would just be measuring
    float32-vs-float64 accumulation noise, not a real logic error — relative error is
    what actually distinguishes the two."""
    fpath_h5 = os.path.join(BLD_DIR, "fea", "mnist-test.h5")
    if not os.path.isfile(fpath_h5):
        raise RuntimeError(f"cannot validate export: {fpath_h5} not found — run exp03's extract_features.py first")

    with h5py.File(fpath_h5, "r") as h5:
        X = np.asarray(h5["X"])[:n_samples]  # raw (unnormalized) pixels, (n_samples, 784)

    dim = doc["packed_dim"]
    x_norm = (X - mean) / std
    model.eval()
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(x_norm.astype(np.float32))).numpy()

    for i in range(len(X)):
        x0 = np.zeros(dim)
        x0[:len(mean)] = x_norm[i]
        sim_logits = _simulate(doc, x0)
        if not np.allclose(sim_logits, torch_logits[i], rtol=rtol, atol=atol):
            max_err = np.max(np.abs(sim_logits - torch_logits[i]))
            raise RuntimeError(
                f"CKKS export self-check FAILED on sample {i}: max abs error {max_err:.6g} "
                f"(rtol={rtol}, atol={atol})\n  simulator: {sim_logits}\n  pytorch:   {torch_logits[i]}"
            )
    LOG.info("Export self-check passed: simulator matches PyTorch on %d samples (rtol=%g, atol=%g)", len(X), rtol, atol)


def export_model(dpath_mdl: str) -> None:
    with open(os.path.join(dpath_mdl, "meta.json")) as f:
        meta = json.load(f)

    fpath_out = os.path.join(dpath_mdl, "ckks_model.json")
    if os.path.isfile(fpath_out):
        LOG.info("Skipping (already exported): %s", fpath_out)
        return

    conv_channels = meta["conv_channels"]
    kernel_size   = meta["kernel_size"]
    fc_layers     = meta["fc_layers"]
    activation    = ACTIVATION_MAP[meta["activation"]]
    pool          = meta["pool"]
    dropout       = meta.get("dropout", 0.0)

    model = CNN(conv_channels=conv_channels, kernel_size=kernel_size, fc_layers=fc_layers,
                activation=activation, pool=pool, dropout=dropout)
    model.load_state_dict(torch.load(os.path.join(dpath_mdl, "model.pt"), weights_only=True, map_location="cpu"))
    model.eval()

    dim = next_pow2(W_PHYS * W_PHYS)
    LOG.info("packed_dim = %d", dim)

    convs      = [m for m in model.conv if isinstance(m, nn.Conv2d)]
    fc_linears = [m for m in model.fc  if isinstance(m, nn.Linear)]
    assert len(convs) == len(conv_channels) - 1, \
        f"expected {len(conv_channels) - 1} Conv2d layers, found {len(convs)}"

    stride, h_active, w_active, pending_scale = 1, W_PHYS, W_PHYS, 1.0
    conv_docs: list = []
    rotations: set = set()

    for conv in convs:
        layer_doc, term_rots = export_conv_layer(conv, stride, h_active, w_active, dim, pending_scale)
        conv_docs.append(layer_doc)
        rotations |= term_rots
        rotations |= pool_rotations(stride)
        h_active, w_active = h_active // 2, w_active // 2
        stride *= 2
        # pending_scale just got folded into this conv's weights above — reset before
        # accumulating the NEXT pool's own (separate) deferred scale. Must not compound
        # across layers: each pool's 0.25 is resolved exactly once, by the very next
        # linear layer, never carried further.
        pending_scale = 0.25

    fc1_doc, fc1_rots = export_fc1(fc_linears[0], convs[-1].out_channels, h_active, w_active, stride, dim, pending_scale)
    rotations |= fc1_rots

    fc_dense_docs: list = []
    for linear in fc_linears[1:]:
        W = linear.weight.detach().numpy()
        b = linear.bias.detach().numpy()
        Wp = np.zeros((dim, dim))
        Wp[:W.shape[0], :W.shape[1]] = W
        fc_dense_docs.append({"W_diag": diagonals(Wp, dim), "bias": pad(b, dim)})

    if fc_dense_docs:
        # matVec() rotates every i in 0..dim-1 unconditionally (exp03's convention) —
        # these keys are needed regardless of individual diagonals being all-zero.
        rotations |= set(range(1, dim))

    mean = np.array(meta["mean"])
    std  = np.array(meta["std"])

    doc = {
        "packed_dim": dim, "w_phys": W_PHYS, "num_classes": fc_layers[-1],
        "classes": meta["classes"], "mean": pad(mean, dim), "std": pad(std, dim),
        "rotations": sorted(rotations),
        "conv_layers": conv_docs, "fc1": fc1_doc, "fc_dense": fc_dense_docs,
    }

    _validate_via_simulator(doc, model, mean, std)

    with open(fpath_out, "w") as f:
        json.dump(doc, f)
    LOG.info("Exported -> %s  (packed_dim=%d, conv_layers=%d, fc_dense=%d, rotations=%d)",
              fpath_out, dim, len(conv_docs), len(fc_dense_docs), len(rotations))


def _is_ckks_exportable(dpath_mdl: str) -> bool:
    meta_path = os.path.join(dpath_mdl, "meta.json")
    if not os.path.isfile(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("activation") == "Quadratic" and meta.get("pool") == "avg"


def export_all(mdl_root: str) -> None:
    if not os.path.isdir(mdl_root):
        LOG.warning("No models found under %s", mdl_root)
        return

    mdl_dirs = sorted(
        root
        for root, _, files in os.walk(mdl_root)
        if "model.pt" in files
    )
    if not mdl_dirs:
        LOG.warning("No models found under %s", mdl_root)
        return

    exportable = [d for d in mdl_dirs if _is_ckks_exportable(d)]
    LOG.info("Found %d model(s), %d exportable (Quadratic activation, avg pool)", len(mdl_dirs), len(exportable))
    for dpath_mdl in exportable:
        LOG.info("--- %s ---", os.path.basename(dpath_mdl))
        export_model(dpath_mdl)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    export_all(os.path.join(BLD_DIR, "mdl", "exp04"))
