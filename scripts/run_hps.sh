#!/usr/bin/env bash
# Optuna search of the six students of the sum-pooling ablation (optimize_optuna.py,
# utils/hps.py), one study per (dataset, model, fold), on the k-fold teachers.
#
#   DATASETS="MUTAG PROTEINS" FOLDS=0 N_TRIALS=25 JOBS=6 scripts/run_hps.sh
#
# One worker per study (WORKERS_PER_STUDY=1): the seeded TPE then proposes the same
# sequence of trials on a rerun. Studies are queued dataset by dataset, smallest first,
# and a free slot takes the next study, so results arrive incrementally. Studies resume
# from their journal, so the script can be stopped and restarted. When all are done the
# per-study files and results_hps/best_params.csv are exported.
# FOLDS=0 tunes on fold 0 only; FOLDS="0 1 ... 9" is per-fold (nested) selection.
# Logs: logs/hps/<dataset>_<model>_fold<k>[_w<i>].log
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
. "$REPO/scripts/env.sh"
DATASETS="${DATASETS:-MUTAG PROTEINS BaMultiShapes AIDS BBBP NCI1 Mutagenicity}"
MODELS="${MODELS:-classic classic_nosum node_mask_push node_mask_push_sum graph graph_sum}"
FOLDS="${FOLDS:-0}"
N_TRIALS="${N_TRIALS:-25}"
WORKERS_PER_STUDY="${WORKERS_PER_STUDY:-1}"
JOBS="${JOBS:-6}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5055}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PY N_TRIALS
mkdir -p logs/hps

for ds in $DATASETS; do for k in $FOLDS; do for m in $MODELS; do for w in $(seq 1 "$WORKERS_PER_STUDY"); do
    echo "$ds $m $k $w"
done; done; done; done | xargs -P "$JOBS" -L1 bash -c '
    log="logs/hps/$0_$1_fold$2_w$3.log"
    $PY optimize_optuna.py --dataset "$0" --model "$1" --fold "$2" --n_trials "$N_TRIALS" > "$log" 2>&1
    echo "[$(date +%T)] $0 $1 fold $2 worker $3: exit $? $(tail -1 "$log")"'
$PY optimize_optuna.py --export > logs/hps/export.log 2>&1
echo "[$(date +%T)] exported: results_hps/best_params.csv"
