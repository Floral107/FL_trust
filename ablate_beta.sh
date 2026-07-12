#!/bin/bash
# ablate_beta.sh — pick the shrinkage beta on celeba-K4 collapse cells.
#
# For each cell (mode,metric,seed) we train at each beta SEQUENTIALLY, reusing
# the canonical 420_<mode>_<metric>/... dir (eval acc on round_10 BEFORE the
# next beta overwrites it). Different cells run in PARALLEL. K=4 only — that is
# where celeba collapses (66-71% of cells revert to BL under the old guard).
#
# Output: ablation_beta.csv  (mode,metric,seed,beta,acc)  — compare acc vs the
# BL acc per (seed) to find the beta where collapse is gone but the weighting
# still differs from uniform. ST/DY share one transform, so the chosen beta
# transfers to the full sweep.
set -uo pipefail
cd /root/ulrich
source .venv/bin/activate 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export FLR_TF_INTRA=4 FLR_TF_INTER=2
mkdir -p logs
DS=celebanoniid; K=4
BETAS="0.2 0.3 0.5"
CSV=/root/ulrich/ablation_beta.csv
[ -f "$CSV" ] || echo "mode,metric,seed,beta,acc" > "$CSV"

cell() {  # mode metric seed
  local M=$1 WMET=$2 SEED=$3
  local WMET_LC; WMET_LC=$(printf '%s' "$WMET" | tr '[:upper:]' '[:lower:]')
  local DIR="420_${M}_${WMET_LC}/${DS}/${K}/${SEED}"
  for B in $BETAS; do
    rm -rf "$DIR" 2>/dev/null
    python3 set_num_cl.py --data="$DS" --clients="$K" >/dev/null 2>&1 || true
    ( cd "$DS" && seed=$SEED flwr run . --run-config \
        "num-server-rounds=10 weight-mode='$M' weight-metric='$WMET' weight-beta=$B et-mode='none' strategy='fedavg'" ) \
        > "logs/abl_${M}_${WMET_LC}_${SEED}_b${B}.log" 2>&1
    local ACC; ACC=$(python3 eval_acc_one.py "$DIR/global_model_round_10.keras" "$DS" 2>/dev/null | tail -1)
    echo "${M},${WMET_LC},${SEED},${B},${ACC}" >> "$CSV"
    echo "[$(date +%T)] done $M $WMET_LC s$SEED b$B acc=$ACC"
  done
  rm -rf "$DIR" 2>/dev/null
}

echo "=== BETA ABLATION START $(date) ==="
# 8 cells in parallel; each runs its 3 betas serially.
for SEED in 42 107; do
  for WMET in res fairDP acc; do
    cell dy "$WMET" "$SEED" &
  done
done
cell st res 42 &
cell st acc 42 &
wait
echo "=== BETA ABLATION DONE $(date) ==="
