#!/usr/bin/env bash
# Base LogiX-GIN at a given hidden width, from scratch to extracted rules, one chain per seed:
#   teacher GIN (train_baseline.py) -> LogiX-GIN student (train_logic.py)
#   -> Hoyer sparsification (sparsify_proto.py) -> rules + formula size (unpack_rules.py)
# The student is distilled layer by layer, so the teacher must have the same width.
# Every stage is skipped when its output already exists, so the script also completes
# partially trained widths (e.g. HIDDEN=64, whose teachers and students exist).
#
#   HIDDEN=32 SEEDS="0 1 2" JOBS=3 scripts/run_hidden_dim.sh
#
# Same seed = same data split for every width (the split comes from the teacher's seed).
# Logs: logs/<dataset>/h<HIDDEN>_seed<k>.log. Tracking: MLflow (scripts/mlflow_server.sh).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
. "$REPO/scripts/env.sh"
DATASET="${DATASET:-Mutagenicity}"
HIDDEN="${HIDDEN:-32}"
SEEDS="${SEEDS:-0 1 2}"
JOBS="${JOBS:-3}"
HOYER="${HOYER:-3}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5055}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
mkdir -p "logs/$DATASET"

BASE_CFG="batch_size=128|dropout=0.15|epochs=500|hidden_dim=$HIDDEN|l2=1e-05|lr=0.001|nogumbel=False|num_layers=3"
LOGIC_CFG="batch_size=128|conv_reg=0.001|epochs=3000|fc_reg=0.01|l2=0.0|lr=0.001|warmup_epochs=1000"
SPARSE_CFG="epochs=300|hoyer_fc=$HOYER.0|hoyer_reg=$HOYER.0|prune_eps=0.01|recover_epochs=100"
export PY DATASET HIDDEN HOYER BASE_CFG LOGIC_CFG SPARSE_CFG

chain() {
    k=$1
    log="logs/$DATASET/h${HIDDEN}_seed$k.log"
    teacher="results/$DATASET/$BASE_CFG"
    student="results_logic/$DATASET/$LOGIC_CFG/$BASE_CFG/$k"
    sparse="$student/sparse/$SPARSE_CFG"
    {
        echo "[$(date +%T)] seed $k, hidden $HIDDEN"
        [ -f "$teacher/$k/best.pt" ] || $PY train_baseline.py --dataset "$DATASET" --epochs 500 --hidden_dim "$HIDDEN" \
            --num_layers 3 --batch_size 128 --lr 0.001 --l2 1e-5 --dropout 0.15 --seed "$k" || exit 1
        echo "[$(date +%T)] teacher ready"
        [ -f "$student/best.pt" ] || $PY train_logic.py --dataset "$DATASET" --baseline_path "$teacher" --epochs 3000 \
            --warmup_epochs 1000 --batch_size 128 --lr 0.001 --l2 0.0 --conv_reg 0.001 --fc_reg 0.01 --seed "$k" || exit 1
        echo "[$(date +%T)] student ready"
        [ -f "$sparse/best.pt" ] || $PY sparsify_proto.py --run_path "$student" --hoyer_reg "$HOYER" --hoyer_fc "$HOYER" \
            --epochs 300 --prune_eps 0.01 --recover_epochs 100 || exit 1
        echo "[$(date +%T)] sparse model ready"
        $PY unpack_rules.py --run_path "$sparse" || exit 1
        echo "[$(date +%T)] rules extracted: $sparse/rules"
    } > "$log" 2>&1
    echo "[$(date +%T)] seed $k done (h$HIDDEN): $(grep -E '^formula size' "$log")"
}
export -f chain

printf '%s\n' $SEEDS | xargs -P "$JOBS" -I{} bash -c 'chain {}'
