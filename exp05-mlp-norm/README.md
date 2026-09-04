# MNIST — MLP with normalisation layers

* exp05 studies what a **normalisation layer between `Linear` and the activation** buys
  a CKKS-exportable MLP
* companion to [exp03-mnist](../exp03-mnist) (plain MLP) and [exp04-mnist](../exp04-mnist) (CNN)
* status: **scaffolding only** — `src/py/check-train.py` (smoke test) and the model are in
  place; the real trainer / grid search / CKKS export are not written yet

## The question

`Quadratic` (`x^2`) is the only CKKS-cheap activation, and it is what limits depth in
exp03: each layer squares the activation scale, so a 3-hidden-layer `Quadratic` MLP
diverges where the same net with `ReLU` trains fine. Normalising the *pre-activations* is
the direct lever on that.

| norm mode | layer | CKKS cost |
|---|---|---|
| `none` | — | baseline, identical to exp03's MLP |
| `batch` | `nn.BatchNorm1d` | **free** — in eval mode it is a fixed per-feature affine map `y = γ(x−μ)/√(σ²+ε) + β`, so it folds into the preceding `Linear`'s `(W, b)` at export time |
| `layer` | `nn.LayerNorm` | **not exportable** — needs a per-sample mean, variance and inverse square root of the ciphertext; kept as a plaintext upper bound to see what `batch` gives up |

`is_ckks_compatible(activation, norm)` in `src/py/mlp_norm.py` encodes that rule
(`Quadratic` + `none`/`batch`), mirroring exp04's `train_cnn.py`.

## Layout

```
src/py/common.py       repo paths (BLD_DIR, MNIST_ROOT) — same as exp03/exp04
src/py/mnist_data.py   MnistDataset (per-feature mean/std) + eval_acc + fea_path
src/py/mlp_norm.py     MlpNorm: [Linear -> Norm -> Activation -> Dropout] x N -> Linear
src/py/check-train.py  fast smoke test (below)
scripts/run-check-train.sh
```

## 0. Extract features (shared, one-time)

`build/fea/mnist-{train,test}.h5` is a repo-wide artifact. If missing, generate via exp03:

```bash
PYTHONPATH=../exp03-mnist/src/py ../.venv/bin/python ../exp03-mnist/src/py/extract_features.py
```

That reads the raw idx gz files from `assets/mnist/` (gitignored).

## 1. Smoke test

Not the real trainer — no MLflow, no early stopping, no model dumping. It trains on an
8000-sample subset for 3 epochs (~1 s per run on CPU) purely to confirm data -> model ->
loop wiring, and exits non-zero if test accuracy lands below `--min-acc` (default 0.80).

```bash
./scripts/run-check-train.sh                      # ReLU, [784,256,10], norm=none
./scripts/run-check-train.sh --all-norms          # compare none / batch / layer
./scripts/run-check-train.sh --norm batch --activation Quadratic
```

### Observed on the first run (3 epochs, 8000 train / 2000 test samples)

Shallow `[784, 256, 10]` + `ReLU` — norm makes no difference, as expected:

| norm | test_acc |
|---|---|
| `none` | 0.9170 |
| `batch` | 0.9020 |
| `layer` | 0.9110 |

Deep `[784, 256, 128, 64, 10]` + `Quadratic` — this is the whole point of exp05:

| norm | test_acc | |
|---|---|---|
| `none` | 0.1535 | collapses — the exp03 depth wall |
| `batch` | 0.5025 | recovers substantially, and is **free under CKKS** |
| `layer` | 0.9135 | matches `ReLU`, but is **not CKKS-exportable** |

So at this (deliberately tiny) budget `batch` closes roughly half the gap. Whether it
closes more with real training length is the first thing the full trainer has to answer.
Note `--min-acc` will fail the `Quadratic` deep runs by design — pass `--min-acc 0` when
exploring those.

## Next steps

1. `train_mlp_norm.py` — the real trainer (MLflow, early stopping, dump to
   `build/mdl/exp05/`), mirroring exp03's `train_mlp.py`
2. `grid_search_mlp_norm.py` — sweep `layers` x `activation` x `norm` x `lr` x `dropout`
3. `fold_batchnorm()` in the exporter — collapse `BatchNorm1d` into the preceding
   `Linear`, with a numeric equivalence check as the correctness gate (cf. exp04's
   numpy simulator gate), then reuse exp03's Halevi-Shoup export and `infer_mlp`
