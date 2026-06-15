#!/usr/bin/env bash
# Run FHE (CKKS) inference on the first N samples per model.
# All models run in parallel. Results saved to build/results/fhe/<stem>.txt
# Skips samples already present in the output file.

set -euo pipefail

N=${1:-20}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$EXP_DIR/build/cmake-build-release/fhe_dps_test"
MDL_ROOT="$EXP_DIR/build/mdl"
FEAT_ROOT="$EXP_DIR/build/fea"
OUT_DIR="$EXP_DIR/build/results/fhe"

if [[ ! -x "$BIN" ]]; then
    echo "Binary not found: $BIN" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

run_model() {
    local stem="$1"
    local model="$2"
    local feat="$3"
    local out="$OUT_DIR/$stem.txt"

    declare -A done_keys
    if [[ -f "$out" ]]; then
        while IFS= read -r line; do
            key=$(echo "$line" | awk '{print $1}')
            done_keys["$key"]=1
        done < "$out"
        echo "[$stem] Already done: ${#done_keys[@]}"
    fi

    keys=$(head -"$N" "$feat" | awk '{print $1}')
    i=1
    skipped=0
    while IFS= read -r key; do
        if [[ -n "${done_keys[$key]+x}" ]]; then
            skipped=$((skipped + 1))
        else
            echo "[$stem] [$i/$N] $key"
            "$BIN" "$model" "$feat" "$key" >> "$out"
        fi
        i=$((i + 1))
    done <<< "$keys"

    [[ $skipped -gt 0 ]] && echo "[$stem] Skipped $skipped sample(s)"
    echo "[$stem] Done -> build/results/fhe/$stem.txt"
}

pids=()
for mdl_dir in "$MDL_ROOT"/mlp-*/; do
    stem="${mdl_dir%/}"; stem="${stem##*mlp-}"
    model="$mdl_dir/ckks_model.json"
    feat="$FEAT_ROOT/$stem.txt"

    if [[ ! -f "$model" || ! -f "$feat" ]]; then
        echo "Skipping $stem: missing model or features" >&2
        continue
    fi

    run_model "$stem" "$model" "$feat" &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All models complete."
