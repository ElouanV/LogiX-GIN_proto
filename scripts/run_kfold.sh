#!/usr/bin/env bash
# 10-fold comparison across datasets (train_baseline.py --split kfold): teachers, then
# the students compared on them, all with the Mutagenicity configuration.
#
#   STAGE=teachers DATASETS="MUTAG PROTEINS" JOBS=4 scripts/run_kfold.sh
#   STAGE=students MODELS="classic node_mask_push" scripts/run_kfold.sh
#
# MODELS (students):
#   classic         LogiX-GIN (train_logic.py)
#   classic_nosum   LogiX-GIN with a mean ‖ max readout (train_logic.py --pool_ops mean,max)
#   node_mask_push  NMP: node prototypes + care mask + push (train_proto.py)
#   graph           dense graph-level prototypes (train_proto.py --proto_level graph)
#   *_sum           the same prototype model with a mean ‖ max ‖ sum readout
#                   (--pool_ops mean,max,sum): the sum-pooling ablation. Classic
#                   LogiX-GIN has sum upstream, so its ablation is classic_nosum.
# A fold whose log already holds its result line is skipped, so the script resumes.
# Folds are the seeds 0-9; small datasets first, so results come in early.
# Logs: logs/<dataset>/kfold_<model>_fold<k>.log.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PY:-$HOME/miniconda3/envs/logix-gin/bin/python}"
STAGE="${STAGE:-teachers}"
DATASETS="${DATASETS:-MUTAG BaMultiShapes PROTEINS AIDS BBBP NCI1 Mutagenicity}"
MODELS="${MODELS:-classic classic_nosum node_mask_push node_mask_push_sum graph graph_sum}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
JOBS="${JOBS:-6}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5055}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PY

BASE_ARGS="--epochs 500 --hidden_dim 64 --num_layers 3 --batch_size 128 --lr 0.001 --l2 1e-5 --dropout 0.15 --split kfold"
export TEACHER_CFG="batch_size=128|dropout=0.15|epochs=500|hidden_dim=64|l2=1e-05|lr=0.001|nogumbel=False|num_layers=3|split=kfold"
export STUDENT_ARGS="--epochs 3000 --warmup_epochs 1000 --batch_size 128 --lr 0.001 --conv_reg 0.001 --fc_reg 0.01"
export PROTO_ARGS="--num_prototypes 16"

student() {
    ds=$1 model=$2 k=$3
    teacher="results/$ds/$TEACHER_CFG"
    log="logs/$ds/kfold_${model}_fold$k.log"
    grep -q "^\[{'seed'" "$log" 2>/dev/null && { echo "[$(date +%T)] $ds $model fold $k: done already"; return; }
    [ -f "$teacher/$k/best.pt" ] || { echo "[$(date +%T)] $ds $model fold $k: no teacher"; return; }
    sum=()
    [[ $model == *_sum ]] && sum=(--pool_ops mean,max,sum)
    case ${model%_sum} in
        classic)        cmd=(train_logic.py) ;;
        classic_nosum)  cmd=(train_logic.py --pool_ops mean,max) ;;
        node_mask_push) cmd=(train_proto.py $PROTO_ARGS --proto_level node --push_every 200 --proto_mask
                             --mask_reg 0.5 --mask_anneal_frac 0.8 --mask_ckpt_temp 1.0 "${sum[@]}") ;;
        graph)          cmd=(train_proto.py $PROTO_ARGS --proto_level graph "${sum[@]}") ;;
        *) echo "unknown model $model"; return ;;
    esac
    $PY "${cmd[@]}" --dataset "$ds" --baseline_path "$teacher" $STUDENT_ARGS --seed "$k" > "$log" 2>&1
    echo "[$(date +%T)] $ds $model fold $k: exit $? $(grep -E "^\[\{'seed'" "$log")"
}
export -f student

for ds in $DATASETS; do mkdir -p "logs/$ds"; done
case $STAGE in
    teachers)
        for ds in $DATASETS; do for k in $FOLDS; do echo "$ds $k"; done; done | xargs -P "$JOBS" -L1 bash -c '
            log="logs/$0/kfold_teacher_fold$1.log"
            [ -f "results/$0/$TEACHER_CFG/$1/best.pt" ] && grep -q "^\[{.seed" "$log" 2>/dev/null && exit 0
            $PY train_baseline.py --dataset "$0" '"$BASE_ARGS"' --seed "$1" > "$log" 2>&1
            echo "[$(date +%T)] $0 teacher fold $1: exit $? $(grep -E "^\[\{.seed" "$log")"'
        for ds in $DATASETS; do      # summary over the 10 folds (results.json + MLflow summary run)
            $PY train_baseline.py --dataset "$ds" $BASE_ARGS --only_eval > "logs/$ds/kfold_teacher_eval.log" 2>&1
            echo "[$(date +%T)] $ds teachers: $(cat "results/$ds/$TEACHER_CFG/results.json")"
        done ;;
    students)
        for ds in $DATASETS; do for m in $MODELS; do for k in $FOLDS; do echo "$ds $m $k"; done; done; done \
            | xargs -P "$JOBS" -L1 bash -c 'student $0 $1 $2' ;;
    *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
