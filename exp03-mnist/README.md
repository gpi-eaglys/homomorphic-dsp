# MNIST - plaintext
* exp03 trains Multi-Layer Perceptron (MLP) models for MNIST classificaiton task

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

`grid_search_mlp.py` refuses to start if `torch.cuda.is_available()` is `False`, so a
missed reinstall fails fast instead of quietly running on CPU.

## Workflow: train plain -> export -> infer under CKKS

### 1. Run experiments
* run a grid search on MLP parameters of: `layers`/`activation`/`lr`/`dropout` (cf. `src/py/grid_search_mlp.py`).

```bash
./scripts/run-grid-search-mlp.sh --bg   # background, logs -> build/grid_search_mlp_<timestamp>.log
```

* each parameter combination is hashed
* model train out dir: `build/mdl/exp03/mlp-mnist_<param_hash>/`
* models are only dumped if the perform above `min_acc` above a threshold - on the test set
* upon completion, the best accuracy is written into `result.json` 


### 2. Pick a model — must be Quadratic activation
* use MLFlow's browser service 
* or run the following script to review the models

```bash
./scripts/trim-build-dir.sh 
```

* only `Quadratic` (`x^2`) models can be exported to CKKS 
  * `GELU`/`ReLU`/`Sigmoid` are not supported in exp03
* typcally single hidden models work the best, e.g.,  [784, 784, 10] (e.g. `afd773ed65fe`)


### 3.a Export model 
* export all CKKS-supported models using `export_features.py`
* models written into file `ckks_model.json` - next to the original model dir
* exportiing means encoding the data:
  * data layout changes
  * plaintext (no encryption)
  * Halevi-Shoup diagonal-packed weights are used: `mean`/`std`, `packed_dim` 


```
PYTHONPATH=src/py .venv/bin/python src/py/export_mdl.py
```

### 3.b Export features 
* features -> are just the input pictures 
* export means extracting .h5 data as plaintext 
* only test set feature from `mnist-test.h5` are exported


```bash
PYTHONPATH=src/py .venv/bin/python src/py/export_features.py
```

### 4. Run inference
* runs inference on MNIST test data 
* requires input: 
  * encoded model, i.e., the output of `export_model.py` 
  * encoded feauters, i.e., the output of `export_features.py`
* runs inference on encrypted data 
  * encrypts data before inference
  * decyprts results - for comparison


```bash
# id_file: one sample id per line, from build/fea/mnist.txt's first column
head -3 build/fea/mnist.txt | awk '{print $1}' > /tmp/ids.txt

./build/cmake/cmake-build-release/exp03-mnist/infer_mlp \
    build/mdl/exp03/mlp-mnist_<param_hash>/eNNNN/ckks_model.json \
    build/fea/mnist.txt \
    /tmp/ids.txt \
    /tmp/results.txt
```

Writes one line per sample to `output.txt`: `<id>\t<logit_0>\t<logit_1>\t...\t<logit_9>`
— the predicted class is `argmax` of the logits. `src/cpp/infer_mlp.cpp` discovers however
many `Wi_diag/bi` layer pairs the model actually has (not hardcoded to a fixed depth),
so it works for any exported MLP, not just single-hidden-layer ones. Rebuild after
changing it with `../scripts/build-cpp.sh` (from repo root).


