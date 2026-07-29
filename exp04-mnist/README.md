# MNIST - plaintext

* exp04 trains Convolutional Neural Network (CNN) models for the MNIST classification task
* companion to [exp03-mnist](../exp03-mnist), which does the same for MLPs
* CKKS export/encrypted inference (`export_mdl.py` / `infer_cnn`) mirrors exp03's, but
  homomorphic Conv2d + AvgPool2d needed a different packing scheme — see the "Workflow:
  train plain -> export -> infer under CKKS" section below and `src/py/export_mdl.py`'s
  module docstring for the design

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
  `Quadratic`-activation configs are exportable to CKKS (`export_mdl.py`'s
  `_is_ckks_exportable()` enforces this — `activation == "Quadratic" and pool == "avg"`).

* **caveat (as of this writing)**: the live `grid_search_cnn.py` sweep trains `GELU`,
  `ReLU`, `Sigmoid` before `Quadratic` for each `conv_channels` value, so no real
  `Quadratic`+`avg` model has finished training yet — steps 3/4 below have been verified
  against `src/py/dev_make_fixture_model.py`'s untrained fixture (random weights, same
  architecture), not real accuracy. Re-run steps 3/4 against a real model once the grid
  search reaches one.

## Workflow: train plain -> export -> infer under CKKS

### 3.a Export model
* export all CKKS-supported models using `export_mdl.py`
* models written into file `ckks_model.json` — next to the original model dir
* exporting means encoding the data:
  * homomorphic Conv2d/AvgPool2d never repack the ciphertext — each channel stays one
    `packed_dim`-length ciphertext holding a *fixed* 28x28 physical buffer for the whole
    network; pooling doubles a tracked `stride` and floor-halves the active grid instead
    of compacting slots (see the module docstring in `src/py/export_mdl.py` for the full
    rationale, including why `AvgPool2d`'s 1/4 scale is folded into the next layer's
    weights instead of applied on its own)
  * plaintext (no encryption); Halevi-Shoup diagonal packing (reused from exp03) for the
    flatten -> FC step
  * export refuses to write `ckks_model.json` unless a pure-numpy simulator of the exact
    same slot-level algorithm reproduces the real PyTorch model's forward pass first —
    this is the primary correctness gate for the conv/pool geometry and is **not
    optional**

```bash
PYTHONPATH=src/py .venv/bin/python src/py/export_mdl.py
```

* to exercise this pipeline before a real `Quadratic`+`avg` model exists, generate a dev
  fixture first (untrained, random weights — for pipeline development only):

```bash
PYTHONPATH=src/py .venv/bin/python src/py/dev_make_fixture_model.py
PYTHONPATH=src/py .venv/bin/python -c \
  "from export_mdl import export_all; export_all('build/mdl/exp04-dev-fixture')"
```

### 3.b Export features
* features are just the input pictures — same raw MNIST pixels regardless of MLP vs CNN
* reuse exp03's `export_features.py` unchanged, no exp04-specific copy needed

```bash
PYTHONPATH=../exp03-mnist/src/py .venv/bin/python ../exp03-mnist/src/py/export_features.py
```

### 4. Run inference
* runs inference on MNIST test data
* requires input:
  * encoded model, i.e., the output of `export_mdl.py`
  * encoded features, i.e., the output of `export_features.py`
* runs inference on encrypted data: encrypts data before inference, decrypts results
  for comparison

```bash
head -3 build/fea/mnist-test.txt | awk '{print $1}' > /tmp/ids.txt

./build/cmake-build-release/exp04-mnist/infer_cnn \
    build/mdl/exp04/cnn-mnist_<param_hash>/eNNNN/ckks_model.json \
    build/fea/mnist-test.txt \
    /tmp/ids.txt \
    /tmp/results.txt
```

Writes one line per sample to `results.txt`: `<id>\t<logit_0>\t...\t<logit_9>` — same
`.hyp` format as exp03's `infer_mlp`, so exp03's `eval_inference.py` works unchanged.
Rebuild after changing `src/cpp/infer_cnn.cpp` with `../scripts/build-cpp.sh` (from repo
root).

* **performance note**: any dense FC layer's Halevi-Shoup `matVec()` needs rotation keys
  for the full `1..packed_dim-1` range regardless of model size (same as exp03's
  `infer_mlp`) — `EvalRotateKeyGen` for ~1000 keys at this ring dimension alone takes
  minutes, before any per-sample forward pass. `export_mdl.py`'s flatten -> FC1 step
  keeps only the diagonals it measures to be actually nonzero (not a fixed assumed
  ratio — it's topology-dependent, typically a modest 1.2x-2.5x reduction for this
  grid's configs, not more), but conv-layer term plaintexts are stored densely, so
  `ckks_model.json` can be tens to hundreds of MB for the larger grid topologies.
