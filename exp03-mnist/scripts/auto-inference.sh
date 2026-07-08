#!/usr/bin/env bash
# Wrapper around resume-inference.sh that derives the output path from the
# id_file name, and seeds it from any other .hyp files already sitting in
# the same output dir before running anything new.
#
# This is useful when sample sets overlap (e.g. a 100-sample id file that
# is a superset of an already-completed 20-sample run against the same
# model): the ids shared with an existing .hyp file are copied over instead
# of being recomputed, and only the genuinely new ids go through inference.
#
# Usage: auto-inference.sh <model.json> <features.txt> <id_file> <output_dir>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESUME_INFERENCE="${SCRIPT_DIR}/resume-inference.sh"

if [[ $# -ne 4 ]]; then
    echo "Usage: $(basename "$0") <model.json> <features.txt> <id_file> <output_dir>" >&2
    echo "  model.json    exported by export_mdl.py" >&2
    echo "  features.txt  exported by export_features.py" >&2
    echo "  id_file       one sample id per line; only these samples are processed" >&2
    echo "  output_dir    output is <output_dir>/<id_file basename>.hyp; other" >&2
    echo "                .hyp files already in this dir are reused for matching" >&2
    echo "                ids before running anything new" >&2
    exit 1
fi

model="$1"
feat="$2"
id_file="$3"
output_dir="$4"

if [[ ! -f "$id_file" ]]; then
    echo "id file not found: $id_file" >&2
    exit 1
fi
if [[ ! -d "$output_dir" ]]; then
    echo "output dir not found: $output_dir" >&2
    exit 1
fi

base="$(basename "$id_file")"
output="${output_dir}/${base%.ids}.hyp"
touch "$output"

# Ids already accounted for in the target output file.
have_ids_file="$(mktemp)"
cut -f1 "$output" > "$have_ids_file"

# Other .hyp files in the same output dir may already have results for some
# of the ids we need (e.g. a smaller sample set that's a subset of this one,
# run against the same model) -- reuse those instead of recomputing them.
other_hyps=()
for f in "$output_dir"/*.hyp; do
    [[ -e "$f" ]] || continue
    [[ "$(realpath "$f")" == "$(realpath "$output")" ]] && continue
    other_hyps+=("$f")
done

if [[ ${#other_hyps[@]} -gt 0 ]]; then
    before=$(wc -l < "$output")
    awk -v have_file="$have_ids_file" '
        FNR==NR { wanted[$1]=1; next }
        FILENAME==have_file { have[$1]=1; next }
        ($1 in wanted) && !($1 in have) && !($1 in seeded) { seeded[$1]=1; print }
    ' "$id_file" "$have_ids_file" "${other_hyps[@]}" >> "$output"
    after=$(wc -l < "$output")
    names="${other_hyps[*]##*/}"
    echo "Seeded $((after - before)) sample(s) from existing .hyp file(s): ${names}" >&2
fi

"$RESUME_INFERENCE" "$model" "$feat" "$id_file" "$output"
