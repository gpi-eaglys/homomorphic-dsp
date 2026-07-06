#!/usr/bin/env bash
# Reinstall CUDA-enabled torch into the repo's .venv.
# `uv sync` reverts torch to the CPU build pinned by lib/py/pyproject.toml
# (needed by exp01/exp02's kaldifeat+cpu.torch2.4.0 dependency) — rerun this
# script after any `uv sync` if GPU training is needed again.
#
# Usage:
#   ./scripts/install-torch+cu124.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(realpath "${SCRIPT_DIR}/../..")"
PYTHON="${REPO_DIR}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Virtual environment not found: $PYTHON" >&2
    exit 1
fi

uv pip install "torch==2.6.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --python "$PYTHON"

"$PYTHON" -c "import torch; print('torch', torch.__version__, '  cuda available:', torch.cuda.is_available())"
