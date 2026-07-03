#!/usr/bin/env bash
# Run FHE (CKKS) inference on the first N samples per model.
# All models run in parallel. Results saved to build/results/fhe/<stem>.txt
# Skips samples already present in the output file.

set -euo pipefail

N=${1:-20}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(realpath "${SCRIPT_DIR}/../.." )"
BIN="$REPO_DIR/build/cmake/cmake-build-release/fhe_dps_test"
BLD_DIR="${REPO_DIR}/build"
MDL_ROOT="$BLD_DIR/mdl/exp01"
FEAT_ROOT="$BLD_DIR/fea"
OUT_DIR="$BLD_DIR/results/fhe"


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

    local tmp_keys skipped=0 to_run=0
    tmp_keys=$(mktemp)
    while IFS= read -r key; do
        if [[ -v done_keys[$key] ]]; then
            skipped=$((skipped + 1))
        else
            echo "$key" >> "$tmp_keys"
            to_run=$((to_run + 1))
        fi
    done <<< "$(head -"$N" "$feat" | awk '{print $1}')"

    if [[ $to_run -gt 0 ]]; then
        echo "[$stem] Running $to_run sample(s)"
        "$BIN" "$model" "$feat" "$tmp_keys" >> "$out"
    fi
    rm -f "$tmp_keys"

    [[ $skipped -gt 0 ]] && echo "[$stem] Skipped $skipped sample(s)"
    echo "[$stem] Done -> build/results/fhe/$stem.txt"
}

pids=()
echo "[INFO]  Searching for models in ${MDL_ROOT}/"
for mdl_dir in `find ${MDL_ROOT} -type d -name "mlp-*"` ; do
    stem="${mdl_dir##*/mlp-}"  # strip path + "mlp-"  ->  esc50-mfb_e1010_acc=0.979
    stem="${stem%%_e*}"         # strip "_e..." suffix  ->  esc50-mfb
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

    run_model "$stem" "$model" "$feat" &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All models complete."
