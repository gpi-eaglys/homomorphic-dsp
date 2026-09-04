#!/usr/bin/env bash
# Fast smoke test for exp05 — see src/py/check-train.py.
# Any extra args are passed through, e.g.:
#   ./scripts/run-check-train.sh --all-norms --activation Quadratic --layers 784 256 128 64 10
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${EXP_DIR}/.." && pwd)"

cd "${EXP_DIR}"
PYTHONPATH=src/py "${REPO_DIR}/.venv/bin/python" src/py/check-train.py "$@"
