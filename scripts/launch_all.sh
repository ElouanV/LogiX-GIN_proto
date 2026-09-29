#!/usr/bin/env bash
# Start the MLflow server (port 5055) and, once it answers, the two experiment batches:
#   - AIDS + BBBP: teacher GIN then LogiX-GIN prototypes, 10 seeds each (scripts/run_proto_datasets.sh)
#   - Hoyer sparsification sweep on Mutagenicity seeds 0-2 (scripts/run_hoyer_sweep.sh)
# Everything is detached (survives logout); logs in logs/, UI at http://127.0.0.1:5055.
#
#   scripts/launch_all.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PORT="${PORT:-5055}"
mkdir -p logs

if ! curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; then
    setsid nohup scripts/mlflow_server.sh > logs/mlflow_server.log 2>&1 < /dev/null &
    for _ in $(seq 1 60); do
        curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && break
        sleep 2
    done
fi
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "MLflow server did not come up, see logs/mlflow_server.log" >&2; exit 1; }
# make sure the server on this port is the one serving this repo's store
pgrep -u "$(id -u)" -f "mlflow server --backend-store-uri sqlite:///$REPO/mlflow.db" >/dev/null \
    || { echo "port $PORT answers but is not this repo's MLflow server" >&2; exit 1; }
echo "MLflow server up: http://127.0.0.1:$PORT"

export MLFLOW_TRACKING_URI="http://127.0.0.1:$PORT"
DATASETS="${DATASETS:-AIDS BBBP}" JOBS="${JOBS:-3}" \
    setsid nohup scripts/run_proto_datasets.sh > logs/run_proto_datasets.log 2>&1 < /dev/null &
DATASET=Mutagenicity SEEDS="${SWEEP_SEEDS:-0 1 2}" JOBS="${SWEEP_JOBS:-2}" \
    setsid nohup scripts/run_hoyer_sweep.sh > logs/run_hoyer_sweep.log 2>&1 < /dev/null &
echo "launched: tail -f logs/run_proto_datasets.log logs/run_hoyer_sweep.log"
