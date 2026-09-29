#!/usr/bin/env bash
# Hoyer sparsification sweep on trained prototype runs (sparsify_proto.py).
#
#   DATASET=Mutagenicity SEEDS="0 1 2" JOBS=2 scripts/run_hoyer_sweep.sh
#
# Grid: (hoyer_reg, epochs) in SETTINGS, same prune_eps / recovery for all. Each result
# lands in <run>/sparse/<config>/ and in MLflow experiment sparsify/<dataset>; compare
# the settings there with the after/* metrics (accuracy, rules per unit, shortest rule).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PY:-$HOME/miniconda3/envs/logix-gin/bin/python}"
DATASET="${DATASET:-Mutagenicity}"
LEVEL="${LEVEL:-node}"
SEEDS="${SEEDS:-0 1 2}"
JOBS="${JOBS:-2}"
SETTINGS="${SETTINGS:-1:600 3:300 3:600}"          # hoyer_reg:epochs
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5055}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
mkdir -p "logs/$DATASET"

jobs_list() {
    for SEED in $SEEDS; do
        RUN=$(ls -d results_proto/"$DATASET"/*proto_level="$LEVEL"\|push_every=0*/*/"$SEED" 2>/dev/null | head -1)
        [ -z "$RUN" ] && { echo "no $LEVEL run for seed $SEED" >&2; continue; }
        for S in $SETTINGS; do
            printf '%s\t%s\t%s\t%s\n' "$RUN" "${S%%:*}" "${S##*:}" "$SEED"
        done
    done
}

jobs_list | xargs -P "$JOBS" -d '\n' -I{} sh -c '
    IFS="$(printf "\t")"; set -- $1
    '"$PY"' sparsify_proto.py --run_path "$1" --hoyer_reg "$2" --hoyer_fc "$2" --epochs "$3" \
        --prune_eps 0.01 --recover_epochs 100 > "logs/'"$DATASET"'/sparsify_h$2_e$3_seed$4.log" 2>&1
    echo "[$(date +%T)] done hoyer=$2 epochs=$3 seed=$4"' _ {}
