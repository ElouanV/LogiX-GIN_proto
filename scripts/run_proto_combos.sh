#!/usr/bin/env bash
# Combinations of the prototype levers (push, binary care mask) at node and graph level,
# one train_proto.py run per (combo, seed). Same teacher and schedule as the dense
# node/graph runs, so they compare directly. Masked runs anneal T over the first
# MASK_ANNEAL_FRAC of the epochs and keep checkpoints only at T <= 1 (strict ANDs).
# Score them afterwards with interp_metrics.py.
#
# MASK_LAYER_COST (e.g. 0.25,1,1) weights the cared-bit penalty by conv layer
# (train_proto.py --mask_layer_cost) and VOCAB_REG adds the shared-vocabulary penalty
# (--vocab_reg); both are appended to the log name.
#
#   COMBOS="node_push graph_mask" SEEDS="0 1 2" JOBS=5 scripts/run_proto_combos.sh
#   COMBOS=node_mask_push MASK_LAYER_COST=0.25,1,1 scripts/run_proto_combos.sh
#   COMBOS=node_mask_push TRUNKS="64x2 32x3" scripts/run_proto_combos.sh
#
# TRUNKS lists teacher/student trunks as <hidden_dim>x<num_layers> (default 64x3). A
# missing teacher seed is trained first (train_baseline.py, as in run_hidden_dim.sh);
# the student is distilled layer by layer, so it always has its teacher's trunk.
#
# Logs: logs/<dataset>/combo_<combo>[_lc<cost>][_voc<reg>][_h<H>L<L>]_seed<k>.log (trunk tag only
#       when not 64x3); teachers: logs/<dataset>/teacher_h<H>L<L>_seed<k>.log
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PY:-$HOME/miniconda3/envs/logix-gin/bin/python}"
DATASET="${DATASET:-Mutagenicity}"
TRUNKS="${TRUNKS:-64x3}"
SEEDS="${SEEDS:-0 1 2}"
COMBOS="${COMBOS:-node_push graph_mask node_mask_push graph_mask_push node_mask}"
JOBS="${JOBS:-5}"
PUSH_EVERY="${PUSH_EVERY:-200}"
MASK_REG="${MASK_REG:-0.5}"
MASK_ANNEAL_FRAC="${MASK_ANNEAL_FRAC:-0.8}"
MASK_LAYER_COST="${MASK_LAYER_COST:-}"
VOCAB_REG="${VOCAB_REG:-}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5055}"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "logs/$DATASET"
export PY DATASET PUSH_EVERY MASK_REG MASK_ANNEAL_FRAC MASK_LAYER_COST VOCAB_REG

job() {
    combo=$1 trunk=$2 k=$3
    hidden=${trunk%x*} layers=${trunk#*x}
    base_cfg="batch_size=128|dropout=0.15|epochs=500|hidden_dim=$hidden|l2=1e-05|lr=0.001|nogumbel=False|num_layers=$layers"
    teacher="results/$DATASET/$base_cfg"
    if [ ! -f "$teacher/$k/best.pt" ]; then
        $PY train_baseline.py --dataset "$DATASET" --epochs 500 --hidden_dim "$hidden" --num_layers "$layers" \
            --batch_size 128 --lr 0.001 --l2 1e-5 --dropout 0.15 --seed "$k" \
            > "logs/$DATASET/teacher_h${hidden}L${layers}_seed$k.log" 2>&1 || { echo "teacher $trunk seed $k failed"; return; }
    fi
    level=${combo%%_*}
    flags=(--proto_level "$level")
    [[ $combo == *push* ]] && flags+=(--push_every "$PUSH_EVERY")
    [[ $combo == *mask* ]] && flags+=(--proto_mask --mask_reg "$MASK_REG" --mask_anneal_frac "$MASK_ANNEAL_FRAC"
                                      --mask_ckpt_temp 1.0)
    name=$combo
    if [[ $combo == *mask* && -n $MASK_LAYER_COST ]]; then
        flags+=(--mask_layer_cost "$MASK_LAYER_COST")
        name+="_lc${MASK_LAYER_COST//,/-}"
    fi
    if [[ $combo == *mask* && -n $VOCAB_REG ]]; then
        flags+=(--vocab_reg "$VOCAB_REG")
        name+="_voc$VOCAB_REG"
    fi
    [ "$trunk" != 64x3 ] && name+="_h${hidden}L$layers"
    log="logs/$DATASET/combo_${name}_seed$k.log"
    grep -q "^\[{'seed'" "$log" 2>/dev/null && { echo "[$(date +%T)] $name seed $k: done already"; return; }
    $PY train_proto.py --dataset "$DATASET" --baseline_path "$teacher" --epochs 3000 \
        --warmup_epochs 1000 --batch_size 128 --lr 0.001 --conv_reg 0.001 --fc_reg 0.01 \
        --num_prototypes 16 "${flags[@]}" --seed "$k" > "$log" 2>&1
    echo "[$(date +%T)] $name seed $k: exit $? $(grep -E "^\[\{'seed'" "$log")"
}
export -f job

for t in $TRUNKS; do for c in $COMBOS; do for k in $SEEDS; do echo "$c $t $k"; done; done; done \
    | xargs -P "$JOBS" -L1 bash -c 'job $0 $1 $2'
