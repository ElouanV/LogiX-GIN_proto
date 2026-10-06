#!/usr/bin/env bash
# Node-level LogiX-GIN prototypes with the binary care mask (train_proto.py --proto_mask),
# then rule extraction, one chain per (mask_reg, seed). Same teacher and schedule as the
# unmasked node runs, so they compare directly.
#
#   MASK_REGS="0.5 2.0" SEEDS="0 1 2" JOBS=3 scripts/run_proto_mask.sh
#
# Logs: logs/<dataset>/protomask_r<reg>_seed<k>.log
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
. "$REPO/scripts/env.sh"
DATASET="${DATASET:-Mutagenicity}"
HIDDEN="${HIDDEN:-64}"
SEEDS="${SEEDS:-0 1 2}"
MASK_REGS="${MASK_REGS:-0.5 2.0}"   # written as Python prints floats: they name the results dir
JOBS="${JOBS:-3}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5055}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
mkdir -p "logs/$DATASET"
BASE_CFG="batch_size=128|dropout=0.15|epochs=500|hidden_dim=$HIDDEN|l2=1e-05|lr=0.001|nogumbel=False|num_layers=3"
export PY DATASET BASE_CFG

job() {
    r=$1 k=$2
    teacher="results/$DATASET/$BASE_CFG"
    cfg="batch_size=128|conv_reg=0.001|epochs=3000|fc_reg=0.01|l2=0.0|lr=0.001|mask_reg=$r|mask_temp_end=1.0|mask_temp_start=None|num_prototypes=16|proto_div_reg=0.01|proto_ent_reg=0.01|proto_level=node|proto_mask=True|push_every=0|warmup_epochs=1000"
    run="results_proto/$DATASET/$cfg/$BASE_CFG/$k"
    log="logs/$DATASET/protomask_r${r}_seed$k.log"
    {
        [ -f "$run/best.pt" ] || $PY train_proto.py --dataset "$DATASET" --baseline_path "$teacher" --epochs 3000 \
            --warmup_epochs 1000 --batch_size 128 --lr 0.001 --conv_reg 0.001 --fc_reg 0.01 --proto_level node \
            --num_prototypes 16 --proto_mask --mask_reg "$r" --seed "$k" || exit 1
        [ -f "$run/rules/rules.json" ] || $PY unpack_rules.py --run_path "$run" || exit 1
    } > "$log" 2>&1
    echo "[$(date +%T)] mask_reg $r seed $k: $(grep -E '^formula size' "$log")"
}
export -f job

for r in $MASK_REGS; do for k in $SEEDS; do echo "$r $k"; done; done | xargs -P "$JOBS" -L1 bash -c 'job $0 $1'
