#!/usr/bin/env bash
# Run the MLflow CNN grid search for exp04-mnist.
# Results are tracked in build/mlflow.db and viewable at http://<host>:10007
#
# Usage:
#   ./scripts/run-grid-search-cnn.sh            # foreground, also tee'd to build/grid_search_cnn_<timestamp>.log
#   ./scripts/run-grid-search-cnn.sh --bg       # background, logs -> build/grid_search_cnn_<timestamp>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(realpath "${SCRIPT_DIR}/../..")"
EXP_DIR="${SCRIPT_DIR}/.."
VENV="${REPO_DIR}/.venv/bin/python"
LOG_FILE="${REPO_DIR}/build/grid_search_cnn_$(date +%Y%m%d_%H%M%S).log"

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
        nohup "$VENV" "${EXP_DIR}/src/py/grid_search_cnn.py" \
        > "$LOG_FILE" 2>&1 &
    echo "PID=$!"
    echo "Follow: tail -f $LOG_FILE"
else
    echo "Logging to $LOG_FILE"
    PYTHONPATH="${EXP_DIR}/src/py" \
        "$VENV" "${EXP_DIR}/src/py/grid_search_cnn.py" 2>&1 | tee "$LOG_FILE"
fi
