#!/usr/bin/env bash
# MLflow tracking server + UI for LogiX-GIN (see utils/tracking.py).
#
#   scripts/mlflow_server.sh            # http://127.0.0.1:5055
#   PORT=5056 scripts/mlflow_server.sh
#
# Training processes log through it when MLFLOW_TRACKING_URI=http://127.0.0.1:$PORT
# (the run scripts set this). The server serialises writes to mlflow.db, which several
# processes writing sqlite directly would race on. From a laptop:
#   ssh -L 5055:127.0.0.1:5055 <this machine>   then open http://localhost:5055
#
# 5055 rather than MLflow's default 5000: this machine already runs another MLflow
# server on 5000, and a client pointed at the wrong server logs into it silently.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-5055}"
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT\$"; then
    echo "port $PORT is already in use - not starting (is it another MLflow server?)" >&2
    exit 1
fi
PY="${PY:-$HOME/miniconda3/envs/logix-gin/bin}"
export MLFLOW_DISABLE_AGENT_HINT=1
exec "$PY/mlflow" server \
    --backend-store-uri "sqlite:///$REPO/mlflow.db" \
    --default-artifact-root "file://$REPO/mlartifacts" \
    --host 127.0.0.1 --port "$PORT" --workers 2
