#!/usr/bin/env bash
# Run plaintext inference for all models and save results to build/results/plaintext/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$EXP_DIR/build/cmake-build-release/infer_plaintext"
MDL_ROOT="$EXP_DIR/build/mdl"
FEAT_ROOT="$EXP_DIR/build/fea"
OUT_DIR="$EXP_DIR/build/results/plaintext"

if [[ ! -x "$BIN" ]]; then
    echo "Binary not found: $BIN" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

for mdl_dir in "$MDL_ROOT"/mlp-*/; do
    stem="${mdl_dir%/}"; stem="${stem##*mlp-}"
    model="$mdl_dir/ckks_model.json"
    feat="$FEAT_ROOT/$stem.txt"

    if [[ ! -f "$model" || ! -f "$feat" ]]; then
        echo "Skipping $stem: missing model or features" >&2
        continue
    fi

    "$BIN" "$model" "$feat" > "$OUT_DIR/$stem.txt"
    echo "Saved $stem -> build/results/plaintext/$stem.txt  ($(wc -l < "$OUT_DIR/$stem.txt") samples)"
done
