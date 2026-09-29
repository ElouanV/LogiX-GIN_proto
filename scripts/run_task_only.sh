#!/usr/bin/env bash
# Hoyer (+ unit penalty) sparsification of the dense base LogiX-GIN students with the
# class loss only (sparsify_proto.py --task_only, no layer-wise distillation), then
# rule extraction. Same epochs as the distilled Hoyer-3 runs of run_hidden_dim.sh.
#
#   WIDTHS="32 64" SEEDS="0 1 2" UNITS="0 1 3" JOBS=4 scripts/run_task_only.sh
#
# Logs: logs/<dataset>/taskonly_h<H>_u<unit>_seed<k>.log
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PY:-$HOME/miniconda3/envs/logix-gin/bin/python}"
DATASET="${DATASET:-Mutagenicity}"
WIDTHS="${WIDTHS:-32 64}"
SEEDS="${SEEDS:-0 1 2}"
UNITS="${UNITS:-0 1 3}"
HOYER="${HOYER:-3}"
JOBS="${JOBS:-4}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5055}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
mkdir -p "logs/$DATASET"
LOGIC_CFG="batch_size=128|conv_reg=0.001|epochs=3000|fc_reg=0.01|l2=0.0|lr=0.001|warmup_epochs=1000"
export PY DATASET HOYER LOGIC_CFG

job() {
    h=$1 u=$2 k=$3
    student="results_logic/$DATASET/$LOGIC_CFG/batch_size=128|dropout=0.15|epochs=500|hidden_dim=$h|l2=1e-05|lr=0.001|nogumbel=False|num_layers=3/$k"
    ucfg="" uarg=()
    [ "$u" != "0" ] && ucfg="|unit_hoyer=$u.0" && uarg=(--unit_hoyer "$u")
    out="$student/sparse/epochs=300|hoyer_fc=$HOYER.0|hoyer_reg=$HOYER.0$ucfg|task_only=True|prune_eps=0.01|recover_epochs=100"
    log="logs/$DATASET/taskonly_h${h}_u${u}_seed$k.log"
    {
        [ -f "$out/best.pt" ] || $PY sparsify_proto.py --run_path "$student" --hoyer_reg "$HOYER" --hoyer_fc "$HOYER" \
            --epochs 300 --prune_eps 0.01 --recover_epochs 100 --task_only "${uarg[@]}" || exit 1
        [ -f "$out/rules/rules.json" ] || $PY unpack_rules.py --run_path "$out" || exit 1
    } > "$log" 2>&1
    echo "[$(date +%T)] task-only h$h unit$u seed $k: $(grep -E '^formula size' "$log")"
}
export -f job

for h in $WIDTHS; do for u in $UNITS; do for k in $SEEDS; do echo "$h $u $k"; done; done; done \
    | xargs -P "$JOBS" -L1 bash -c 'job $0 $1 $2'
