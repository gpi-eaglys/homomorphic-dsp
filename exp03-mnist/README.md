# MNIST - plaintext
* this experiment directory holds demonstrates and compares various approaches to 
  to MNIST digit classification
* some approaches aim to imitate CKKS constraints

## GPU setup

`lib/py/pyproject.toml` pins `torch==2.4.0+cpu` for the whole workspace (needed by
exp01/exp02's `kaldifeat+cpu.torch2.4.0` dependency), so `uv sync` will revert torch
to the CPU-only build. Training here is far too slow on CPU — after any `uv sync`,
reinstall CUDA torch:

```bash
uv pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124 --python .venv/bin/python
# or:
./scripts/install-torch+cu124.sh
```

`grid_search_mlp.py` refuses to start if `torch.cuda.is_available()` is `False`, so a
missed reinstall fails fast instead of quietly running on CPU.


