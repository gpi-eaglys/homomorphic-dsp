#!/usr/bin/env bash
# Resumable wrapper around run-inference.sh for CKKS inference on exp03 MNIST models.
#
# Reads whichever sample ids are already present in <output.hyp>, computes the
# ids from <id_file> that are still missing, and runs inference only on those.
# Whatever the underlying infer_mlp process produces (even if interrupted
# partway through) is appended onto <output.hyp>, so re-running this script
# repeatedly picks up wherever the previous run left off.
#
# Usage: same 4 args as run-inference.sh, so it's a drop-in replacement.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_INFERENCE="${SCRIPT_DIR}/run-inference.sh"

if [[ $# -ne 4 ]]; then
    echo "Usage: $(basename "$0") <model.json> <features.txt> <id_file> <output.txt>" >&2
    echo "  model.json    exported by export_mdl.py" >&2
    echo "  features.txt  exported by export_features.py" >&2
    echo "  id_file       one sample id per line; only these samples are processed" >&2
    echo "  output.txt    results are accumulated here across resumed runs" >&2
    exit 1
fi

model="$1"
feat="$2"
id_file="$3"
output="$4"

if [[ ! -f "$id_file" ]]; then
    echo "id file not found: $id_file" >&2
    exit 1
fi

remaining_id_file="${id_file}.remaining"

# Pick the first unused "<output>.N" so each resumed run keeps its own
# scratch file instead of overwriting a previous one.
n=1
resume_output="${output}.${n}"
while [[ -e "$resume_output" ]]; do
    n=$((n + 1))
    resume_output="${output}.${n}"
done

# Ids already present in the accumulated output (first column of each line).
done_ids_file="$(mktemp)"
if [[ -f "$output" ]]; then
    cut -f1 "$output" > "$done_ids_file"
else
    : > "$done_ids_file"
fi

grep -vFf "$done_ids_file" "$id_file" > "$remaining_id_file" || true

total=$(wc -l < "$id_file")
done_count=$((total - $(wc -l < "$remaining_id_file")))
remaining_count=$(wc -l < "$remaining_id_file")

echo "Resume: ${done_count}/${total} sample(s) already in ${output}, ${remaining_count} remaining" >&2

if [[ "$remaining_count" -eq 0 ]]; then
    echo "Nothing to do." >&2
    exit 0
fi

append_progress() {
    if [[ -s "$resume_output" ]]; then
        cat "$resume_output" >> "$output"
    fi
}
trap append_progress EXIT

"$RUN_INFERENCE" "$model" "$feat" "$remaining_id_file" "$resume_output"
