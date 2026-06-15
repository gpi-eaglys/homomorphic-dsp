#!/usr/bin/env bash
# Run plaintext MLP inference for all model / feature combinations.
# Expects export_ckks.py and export_features.py to have been run first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$EXP_DIR/build/cmake-build-release/infer_plaintext"
MDL_ROOT="$EXP_DIR/build/mdl"
FEAT_ROOT="$EXP_DIR/build/fea"

if [[ ! -x "$BIN" ]]; then
    echo "Binary not found: $BIN" >&2
    exit 1
fi

for mdl_dir in "$MDL_ROOT"/mlp-*/; do
    stem="${mdl_dir%/}"
    stem="${stem##*mlp-}"
    model="$mdl_dir/ckks_model.json"
    feat="$FEAT_ROOT/$stem.txt"

    if [[ ! -f "$model" ]]; then
        echo "Skipping $stem: missing $model" >&2
        continue
    fi
    if [[ ! -f "$feat" ]]; then
        echo "Skipping $stem: missing $feat" >&2
        continue
    fi

    echo "=== $stem ==="
    "$BIN" "$model" "$feat"
done
