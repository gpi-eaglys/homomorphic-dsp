#!/usr/bin/env bash
# Run the MLflow grid search for exp03-mnist.
# Results are tracked in build/mlflow.db and viewable at http://<host>:10007
#
# Usage:
#   ./scripts/run_grid_search.sh            # foreground
#   ./scripts/run_grid_search.sh --bg       # background, logs -> build/grid_search.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(realpath "${SCRIPT_DIR}/../..")"
EXP_DIR="${SCRIPT_DIR}/.."
VENV="${REPO_DIR}/.venv/bin/python"
LOG_FILE="${REPO_DIR}/build/grid_search.log"

BACKGROUND=false
for arg in "$@"; do
    [[ "$arg" == "--bg" ]] && BACKGROUND=true
done

if [[ ! -x "$VENV" ]]; then
    echo "Virtual environment not found: $VENV" >&2
    exit 1
fi

mkdir -p "${REPO_DIR}/build"

if $BACKGROUND; then
    echo "Starting grid search in background -> $LOG_FILE"
    PYTHONPATH="${EXP_DIR}/src/py" \
        nohup "$VENV" "${EXP_DIR}/src/py/grid_search.py" \
        > "$LOG_FILE" 2>&1 &
    echo "PID=$!"
    echo "Follow: tail -f $LOG_FILE"
else
    PYTHONPATH="${EXP_DIR}/src/py" \
        exec "$VENV" "${EXP_DIR}/src/py/grid_search.py"
fi
