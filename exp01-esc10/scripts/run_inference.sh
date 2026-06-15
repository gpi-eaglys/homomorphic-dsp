#!/usr/bin/env bash
# Run plaintext and FHE inference for all models and save results.
#
# Plaintext: all samples, results -> build/results/plaintext/<stem>.txt (always overwritten)
# FHE:       pending samples only, results -> build/results/fhe/<stem>.txt (append only)
#            key files -> build/keys/<stem>_<timestamp>.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(realpath "${SCRIPT_DIR}/../.." )"
BLD_DIR="${REPO_DIR}/build"

BIN_PLAIN="${BLD_DIR}/cmake/cmake-build-release/infer_plaintext"
BIN_FHE="${BLD_DIR}/cmake/cmake-build-release/fhe_dps_test"
MDL_ROOT="${BLD_DIR}/mdl/exp01"
FEAT_ROOT="${BLD_DIR}/fea"
OUT_PLAIN="${BLD_DIR}/results/plaintext"
OUT_FHE="${BLD_DIR}/results/fhe"
KEYS_DIR="${BLD_DIR}/keys"


for bin in "$BIN_PLAIN" "$BIN_FHE"; do
    if [[ ! -x "$bin" ]]; then
        echo "Binary not found: $bin" >&2
        exit 1
    fi
done

mkdir -p "$OUT_PLAIN" "$OUT_FHE" "$KEYS_DIR"

TS=$(date +%Y%m%d_%H%M%S)

for mdl_dir in "$MDL_ROOT"/mlp-*/; do
    stem="${mdl_dir%/}"; stem="${stem##*mlp-}"
    model="$mdl_dir/ckks_model.json"
    feat="$FEAT_ROOT/$stem.txt"

    if [[ ! -f "$model" ]]; then
        echo "Skipping $stem: missing model at $model" >&2
        continue
    fi

    if [[ ! -f "$feat" ]]; then
        echo "Skipping $stem: missing features at $feat" >&2
        continue
    fi

    echo "Stem    : $stem" 
    echo "Model   : $model" 
    echo "Features: $feat" 

    echo "--- plaintext: $stem ---"
    "$BIN_PLAIN" "$model" "$feat" > "$OUT_PLAIN/$stem.txt"
    echo "Saved -> build/results/plaintext/$stem.txt"

    fhe_out="$OUT_FHE/$stem.txt"
    key_file="$KEYS_DIR/${stem}_${TS}.txt"

    if [[ -f "$fhe_out" ]]; then
        pending=$(comm -23 \
            <(awk '{print $1}' "$feat" | sort) \
            <(awk '{print $1}' "$fhe_out" | sort))
    else
        pending=$(awk '{print $1}' "$feat")
    fi

    if [[ -z "$pending" ]]; then
        echo "Skipping fhe $stem: all samples already computed" >&2
        continue
    fi

    echo "$pending" > "$key_file"
    n=$(wc -l < "$key_file")
    echo "--- fhe: $stem ($n pending) key file: build/keys/${stem}_${TS}.txt ---"
    "$BIN_FHE" "$model" "$feat" "$key_file" >> "$fhe_out"
    echo "Appended -> build/results/fhe/$stem.txt"
done
