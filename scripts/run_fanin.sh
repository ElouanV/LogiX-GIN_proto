#!/usr/bin/env bash
# Gradual fan-in capping of the Hoyer-sparsified base LogiX-GIN runs, then rule extraction.
# Starts from <student>/sparse/<Hoyer cfg>/ (made by scripts/run_hidden_dim.sh) with
# --epochs 0, so only the schedule + recovery are trained.
#
#   WIDTHS="32 64" SEEDS="0 1 2" JOBS=4 scripts/run_fanin.sh
#
# VARIANTS: "<schedule>:<head cap or ->[:<unit_hoyer>]" items; the head follows the conv
# schedule when "-"; unit_hoyer (default 0) leaves lower units unused (sparsify_proto.py).
# Logs: logs/<dataset>/fanin_h<H>_<schedule>_fc<cap>_seed<k>.log
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PY:-$HOME/miniconda3/envs/logix-gin/bin/python}"
DATASET="${DATASET:-Mutagenicity}"
WIDTHS="${WIDTHS:-32 64}"
SEEDS="${SEEDS:-0 1 2}"
JOBS="${JOBS:-4}"
VARIANTS="${VARIANTS:-12,8,6,4:- 12,8,6,4,3:- 12,8,6,4:8}"
STEP_EPOCHS="${STEP_EPOCHS:-50}"
RECOVER="${RECOVER:-100}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5055}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
mkdir -p "logs/$DATASET"
LOGIC_CFG="batch_size=128|conv_reg=0.001|epochs=3000|fc_reg=0.01|l2=0.0|lr=0.001|warmup_epochs=1000"
HOYER_CFG="epochs=300|hoyer_fc=3.0|hoyer_reg=3.0|prune_eps=0.01|recover_epochs=100"
export PY DATASET STEP_EPOCHS RECOVER LOGIC_CFG HOYER_CFG

job() {
    h=$1 k=$2 v=$3
    IFS=: read -r sched fc unit <<< "$v"; unit=${unit:-0}
    base="results_logic/$DATASET/$LOGIC_CFG/batch_size=128|dropout=0.15|epochs=500|hidden_dim=$h|l2=1e-05|lr=0.001|nogumbel=False|num_layers=3/$k/sparse/$HOYER_CFG"
    fcarg=() fccfg=""
    [ "$fc" != "-" ] && fcarg=(--fc_fanin "$fc") && fccfg="|fc_fanin=$fc"
    [ "$unit" != "0" ] && fcarg+=(--unit_hoyer "$unit") && fccfg+="|unit_hoyer=$unit.0"
    out="$base/sparse/epochs=0|hoyer_fc=1.0|hoyer_reg=1.0|fanin_schedule=${sched//,/-}|step_epochs=$STEP_EPOCHS$fccfg|prune_eps=0.01|recover_epochs=$RECOVER"
    log="logs/$DATASET/fanin_h${h}_${sched//,/-}_fc${fc}_u${unit}_seed$k.log"
    {
        [ -f "$out/best.pt" ] || $PY sparsify_proto.py --run_path "$base" --epochs 0 --fanin_schedule "$sched" \
            --step_epochs "$STEP_EPOCHS" --recover_epochs "$RECOVER" --prune_eps 0.01 "${fcarg[@]}" || exit 1
        [ -f "$out/rules/rules.json" ] || $PY unpack_rules.py --run_path "$out" || exit 1
    } > "$log" 2>&1
    echo "[$(date +%T)] h$h ${sched} fc$fc unit$unit seed $k: $(grep -E '^formula size' "$log")"
}
export -f job

for h in $WIDTHS; do for v in $VARIANTS; do for k in $SEEDS; do echo "$h $k $v"; done; done; done \
    | xargs -P "$JOBS" -L1 bash -c 'job $0 $1 $2'
