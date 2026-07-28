# MNIST - plaintext

* exp04 trains Convolutional Neural Network (CNN) models for the MNIST classification task
* companion to [exp03-mnist](../exp03-mnist), which does the same for MLPs
* CKKS export/encrypted inference (like exp03's `infer_mlp`) is not implemented here yet —
  homomorphic convolution is a separate, harder task

## GPU setup

`lib/py/pyproject.toml` pins `torch==2.4.0+cpu` for the whole workspace (needed by
exp01/exp02's `kaldifeat+cpu.torch2.4.0` dependency), so `uv sync` will revert torch
to the CPU-only build. Training here is far too slow on CPU — after any `uv sync`,
reinstall CUDA torch:

```bash
./scripts/install-torch+cu124.sh

# in details:
uv pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124 --python .venv/bin/python
```

`grid_search_cnn.py` refuses to start if `torch.cuda.is_available()` is `False`, so a
missed reinstall fails fast instead of quietly running on CPU.

## Workflow

### 0. Extract features (shared, one-time)
* `build/fea/mnist-train.h5` / `mnist-test.h5` are a repo-wide artifact, not exp04-specific
* if they don't already exist, generate them via exp03-mnist:

```bash
PYTHONPATH=../exp03-mnist/src/py .venv/bin/python ../exp03-mnist/src/py/extract_features.py
```

### 1. Run experiments
* run a grid search on CNN parameters of: `conv_channels`/`activation`/`pool`/`lr`/`dropout`
  (cf. `src/py/grid_search_cnn.py`)

```bash
./scripts/run-grid-search-cnn.sh --bg   # background, logs -> build/grid_search_cnn_<timestamp>.log
```

* each parameter combination is hashed
* model train out dir: `build/mdl/exp04/cnn-mnist_<param_hash>/`
* models are only dumped if they perform above `min_acc` on the test set
* upon completion, the best accuracy is written into `result.json`

### 2. Pick a model
* use MLflow's browser service (experiment `exp04-mnist-cnn`)
* or run the following script to review the models

```bash
./scripts/trim-build-dir.sh
```

* `AvgPool2d` is a linear op (CKKS-compatible); `MaxPool2d` requires comparisons (not
  CKKS-compatible) — see `is_ckks_compatible()` in `src/py/train_cnn.py`. Only `"avg"`-pool,
  `Quadratic`-activation configs would be candidates for a future CKKS export step.
