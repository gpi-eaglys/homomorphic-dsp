#!/usr/bin/env bash
# Start the MLflow UI for exp03-mnist.
# Opens at http://<host>:10007
#
# Usage:
#   ./scripts/run_mlflow_ui.sh            # foreground
#   ./scripts/run_mlflow_ui.sh --bg       # background

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(realpath "${SCRIPT_DIR}/../..")"
VENV="${REPO_DIR}/.venv/bin/mlflow"
DB="sqlite:///${REPO_DIR}/build/mlflow.db"
PORT=10007

BACKGROUND=false
for arg in "$@"; do
    [[ "$arg" == "--bg" ]] && BACKGROUND=true
done

COMMON_ARGS=(
    --backend-store-uri "$DB"
    --host 0.0.0.0
    --port "$PORT"
    --allowed-hosts "*"
    --cors-allowed-origins "*"
)

if $BACKGROUND; then
    nohup "$VENV" ui "${COMMON_ARGS[@]}" \
        > "${REPO_DIR}/build/mlflow_ui.log" 2>&1 &
    echo "PID=$!  ->  http://$(hostname -I | awk '{print $1}'):${PORT}"
else
    exec "$VENV" ui "${COMMON_ARGS[@]}"
fi
