#!/usr/bin/env bash
# Teacher GIN + LogiX-GIN prototype runs on new datasets, with the Mutagenicity configuration.
#
#   DATASETS="AIDS BBBP" JOBS=3 scripts/run_proto_datasets.sh
#
# Per dataset: 10 teacher seeds (parallel), eval pass, 10 prototype seeds (parallel),
# eval pass. Hyper-parameters are those of the Mutagenicity runs in results/ and
# results_proto/ (node-level prototypes), so the numbers are comparable.
# Logs: logs/<dataset>/*.log. Tracking: MLflow (scripts/mlflow_server.sh must be running).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
. "$REPO/scripts/env.sh"
DATASETS="${DATASETS:-AIDS BBBP}"
JOBS="${JOBS:-3}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
LEVEL="${LEVEL:-node}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5055}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1

BASE_ARGS="--epochs 500 --hidden_dim 64 --num_layers 3 --batch_size 128 --lr 0.001 --l2 1e-5 --dropout 0.15"
PROTO_ARGS="--epochs 3000 --warmup_epochs 1000 --batch_size 128 --lr 0.001 --l2 0.0 --conv_reg 0.001 \
--fc_reg 0.01 --proto_level $LEVEL --num_prototypes 16 --proto_div_reg 0.01 --proto_ent_reg 0.01 --push_every 0"
BASE_DIR_NAME="batch_size=128|dropout=0.15|epochs=500|hidden_dim=64|l2=1e-05|lr=0.001|nogumbel=False|num_layers=3"

for DS in $DATASETS; do
    mkdir -p "logs/$DS"
    echo "[$(date +%T)] $DS: teacher seeds ($SEEDS), $JOBS in parallel"
    printf '%s\n' $SEEDS | xargs -P "$JOBS" -I{} sh -c \
        "$PY train_baseline.py --dataset $DS $BASE_ARGS --seed {} > logs/$DS/baseline_seed{}.log 2>&1"
    $PY train_baseline.py --dataset "$DS" $BASE_ARGS --only_eval > "logs/$DS/baseline_eval.log" 2>&1
    BASE="results/$DS/$BASE_DIR_NAME"
    echo "[$(date +%T)] $DS: teacher done: $(cat "$BASE/results.json")"

    echo "[$(date +%T)] $DS: prototype seeds ($LEVEL), $JOBS in parallel"
    printf '%s\n' $SEEDS | xargs -P "$JOBS" -I{} sh -c \
        "$PY train_proto.py --dataset $DS --baseline_path '$BASE' $PROTO_ARGS --seed {} > logs/$DS/proto_${LEVEL}_seed{}.log 2>&1"
    $PY train_proto.py --dataset "$DS" --baseline_path "$BASE" $PROTO_ARGS --only_eval > "logs/$DS/proto_${LEVEL}_eval.log" 2>&1
    echo "[$(date +%T)] $DS: prototype done: $(cat results_proto/$DS/*proto_level=$LEVEL*/"$BASE_DIR_NAME"/results.json)"
done
