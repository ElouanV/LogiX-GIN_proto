#!/usr/bin/env bash
# Hoyer (+ unit penalty) sparsification with the class loss only (sparsify_proto.py
# --task_only, no layer-wise distillation), then rule extraction. Same epochs as the
# distilled Hoyer-3 runs of run_hidden_dim.sh. By default on the dense base LogiX-GIN
# students of each width; RUNS (a file with one seed directory per line, base or
# prototype) replaces them. EVAL=interp scores each result with interp_metrics.py
# instead of extracting every rule (unpack_rules.py), which can take >90 GB on
# prototype runs with dense layers.
#
#   WIDTHS="32 64" SEEDS="0 1 2" UNITS="0 1 3" JOBS=4 scripts/run_task_only.sh
#   RUNS=runs.txt UNITS="0 1" EVAL=interp JOBS=3 scripts/run_task_only.sh
#   RUNS=runs.txt UNITS=0 HOYER=0 HARD=1 EVAL=interp scripts/run_task_only.sh   # hard fine-tune
#
# HARD=1 fine-tunes in hard mode (sparsify_proto.py --hard, prototype runs only): the
# network then computes exactly its rules.
#
# Logs: logs/<dataset>/taskonly_<tag>_u<unit>_seed<k>.log, tag h<H> or, with RUNS, the
#       start of the run's config directory name
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
. "$REPO/scripts/env.sh"
DATASET="${DATASET:-Mutagenicity}"
WIDTHS="${WIDTHS:-32 64}"
SEEDS="${SEEDS:-0 1 2}"
UNITS="${UNITS:-0 1 3}"
HOYER="${HOYER:-3}"
JOBS="${JOBS:-4}"
RUNS="${RUNS:-}"
EVAL="${EVAL:-rules}"
HARD="${HARD:-0}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5055}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
mkdir -p "logs/$DATASET"
LOGIC_CFG="batch_size=128|conv_reg=0.001|epochs=3000|fc_reg=0.01|l2=0.0|lr=0.001|warmup_epochs=1000"
export PY DATASET HOYER LOGIC_CFG EVAL HARD

job() {
    student=$1 u=$2 tag=$3 k=$4
    ucfg="" uarg=()
    [ "$u" != "0" ] && ucfg="|unit_hoyer=$u.0" && uarg=(--unit_hoyer "$u")
    hcfg="" ltag=""
    [ "$HARD" = 1 ] && hcfg="|hard=True" && uarg+=(--hard) && ltag+="_hard"
    [ "$HOYER" != 3 ] && ltag+="_hoy$HOYER"
    out="$student/sparse/epochs=300|hoyer_fc=$HOYER.0|hoyer_reg=$HOYER.0$ucfg|task_only=True$hcfg|prune_eps=0.01|recover_epochs=100"
    log="logs/$DATASET/taskonly_${tag}${ltag}_u${u}_seed$k.log"
    {
        [ -f "$out/best.pt" ] || $PY sparsify_proto.py --run_path "$student" --hoyer_reg "$HOYER" --hoyer_fc "$HOYER" \
            --epochs 300 --prune_eps 0.01 --recover_epochs 100 --task_only "${uarg[@]}" || exit 1
        if [ "$EVAL" = interp ]; then
            $PY interp_metrics.py --run_path "$out" || exit 1
        else
            [ -f "$out/rules/rules.json" ] || $PY unpack_rules.py --run_path "$out" || exit 1
        fi
    } > "$log" 2>&1
    echo "[$(date +%T)] task-only $tag unit$u seed $k: $(grep -E '^formula size' "$log")$(grep -A1 -E '^ +acc' "$log" | tail -1)"
}
export -f job

students() {    # student path <TAB> log tag <TAB> seed
    if [ -n "$RUNS" ]; then
        while read -r p; do
            [ -n "$p" ] && printf '%s\t%s\t%s\n' "$p" "$(basename "$(dirname "$(dirname "$p")")" | cut -c1-18 | tr '|=' '__')" "$(basename "$p")"
        done < "$RUNS"
    else
        for h in $WIDTHS; do for k in $SEEDS; do
            printf '%s\t%s\t%s\n' "results_logic/$DATASET/$LOGIC_CFG/batch_size=128|dropout=0.15|epochs=500|hidden_dim=$h|l2=1e-05|lr=0.001|nogumbel=False|num_layers=3/$k" "h$h" "$k"
        done; done
    fi
}

students | while IFS=$'\t' read -r p tag k; do for u in $UNITS; do printf '%s\t%s\t%s\t%s\n' "$p" "$u" "$tag" "$k"; done; done \
    | xargs -P "$JOBS" -d '\n' -n1 bash -c 'IFS=$'"'"'\t'"'"' read -r p u tag k <<< "$0"; job "$p" "$u" "$tag" "$k"'
