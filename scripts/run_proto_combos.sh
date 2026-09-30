#!/usr/bin/env bash
# Combinations of the prototype levers (push, binary care mask) at node and graph level,
# one train_proto.py run per (combo, seed). Same teacher and schedule as the dense
# node/graph runs, so they compare directly. Masked runs anneal T over the first
# MASK_ANNEAL_FRAC of the epochs and keep checkpoints only at T <= 1 (strict ANDs).
# Score them afterwards with interp_metrics.py.
#
#   COMBOS="node_push graph_mask" SEEDS="0 1 2" JOBS=5 scripts/run_proto_combos.sh
#
# Logs: logs/<dataset>/combo_<combo>_seed<k>.log
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PY:-$HOME/miniconda3/envs/logix-gin/bin/python}"
DATASET="${DATASET:-Mutagenicity}"
HIDDEN="${HIDDEN:-64}"
SEEDS="${SEEDS:-0 1 2}"
COMBOS="${COMBOS:-node_push graph_mask node_mask_push graph_mask_push node_mask}"
JOBS="${JOBS:-5}"
PUSH_EVERY="${PUSH_EVERY:-200}"
MASK_REG="${MASK_REG:-0.5}"
MASK_ANNEAL_FRAC="${MASK_ANNEAL_FRAC:-0.8}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5055}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "logs/$DATASET"
BASE_CFG="batch_size=128|dropout=0.15|epochs=500|hidden_dim=$HIDDEN|l2=1e-05|lr=0.001|nogumbel=False|num_layers=3"
export PY DATASET BASE_CFG PUSH_EVERY MASK_REG MASK_ANNEAL_FRAC

job() {
    combo=$1 k=$2
    level=${combo%%_*}
    flags=(--proto_level "$level")
    [[ $combo == *push* ]] && flags+=(--push_every "$PUSH_EVERY")
    [[ $combo == *mask* ]] && flags+=(--proto_mask --mask_reg "$MASK_REG" --mask_anneal_frac "$MASK_ANNEAL_FRAC"
                                      --mask_ckpt_temp 1.0)
    log="logs/$DATASET/combo_${combo}_seed$k.log"
    grep -q "^\[{'seed'" "$log" 2>/dev/null && { echo "[$(date +%T)] $combo seed $k: done already"; return; }
    $PY train_proto.py --dataset "$DATASET" --baseline_path "results/$DATASET/$BASE_CFG" --epochs 3000 \
        --warmup_epochs 1000 --batch_size 128 --lr 0.001 --conv_reg 0.001 --fc_reg 0.01 \
        --num_prototypes 16 "${flags[@]}" --seed "$k" > "$log" 2>&1
    echo "[$(date +%T)] $combo seed $k: exit $? $(grep -E "^\[\{'seed'" "$log")"
}
export -f job

for c in $COMBOS; do for k in $SEEDS; do echo "$c $k"; done; done | xargs -P "$JOBS" -L1 bash -c 'job $0 $1'
