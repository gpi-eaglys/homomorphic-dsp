#!/usr/bin/env bash
# Print (and optionally trim) exp03-mnist model dumps under build/mdl/exp03.
#
# Usage:
#   ./scripts/trim-build-dir.sh            # print only, no deletions
#   ./scripts/trim-build-dir.sh --trim     # also delete non-best checkpoints

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/.."
REPO_DIR="$(realpath "${SCRIPT_DIR}/../..")"
VENV="${REPO_DIR}/.venv/bin/python"

if [[ ! -x "$VENV" ]]; then
    echo "Virtual environment not found: $VENV" >&2
    exit 1
fi

PYTHONPATH="${EXP_DIR}/src/py" \
    exec "$VENV" "${EXP_DIR}/src/py/trim-build-dir.py" "$@"
