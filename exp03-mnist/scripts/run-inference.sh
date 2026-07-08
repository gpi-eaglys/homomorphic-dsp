#!/usr/bin/env bash
# Thin wrapper around infer_mlp for CKKS inference on exp03 MNIST models.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(realpath "${SCRIPT_DIR}/../.." )"
BIN_INFER="${REPO_DIR}/build/cmake-build-release/exp03-mnist/infer_mlp"


if [[ $# -eq 0 ]]; then
    echo "Usage: $(basename "$0") <model.json> <features.txt> <id_file> <output.txt>" >&2
    echo "  model.json    exported by export_mdl.py" >&2
    echo "  features.txt  exported by export_features.py" >&2
    echo "  id_file       one sample id per line; only these samples are processed" >&2
    echo "  output.txt    results are written here" >&2
    exit 1
fi

if [[ $# -ne 4 ]]; then
    echo "Error: expected 4 arguments, got $#" >&2
    exit 1
fi

model="$1"
feat="$2"
id_file="$3"
output="$4"

if [[ ! -x "$BIN_INFER" ]]; then
    echo "Binary not found: $BIN_INFER" >&2
    exit 1
fi

exec "$BIN_INFER" "$model" "$feat" "$id_file" "$output"
