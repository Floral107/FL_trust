#!/bin/bash
# res_check.sh — gate before the full run. Trains celeba-K4 res-weighted cells
# (st+dy) at beta=0.2 on seed 123 (avoids ablation's 42/107 → no dir collision),
# KEEPS the round_10 models (canonical dirs → the full run skips-if-valid, so
# zero redundant work), and evaluates the ACTUAL PGD res value + acc. Tells us
# whether res-weighting at beta=0.2 improves robustness over BL (res~0.52,
# acc~0.91) or merely costs accuracy.
set -uo pipefail
cd /root/ulrich
source .venv/bin/activate 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export FLR_TF_INTRA=4 FLR_TF_INTER=2
CSV=/root/ulrich/res_check.csv
echo "mode,metric,seed,beta,res,acc,bl_res,bl_acc" > "$CSV"
echo "=== RES CHECK START $(date) ==="

# purge the stale beta=1 seed-123 res models so we actually train at beta=0.2
rm -rf 420_st_res/celebanoniid/4/123 420_dy_res/celebanoniid/4/123 2>/dev/null

( BETA=0.2 bash st_dy_guarded_run.sh celebanoniid 4 st res 123 50 ) &
( BETA=0.2 bash st_dy_guarded_run.sh celebanoniid 4 dy res 123 51 ) &
wait

for M in st dy; do
  P="420_${M}_res/celebanoniid/4/123/global_model_round_10.keras"
  RA=$(python3 eval_resacc_one.py "$P" celebanoniid 2>/dev/null | tail -1)
  RES="${RA% *}"; ACC="${RA#* }"
  echo "${M},res,123,0.2,${RES},${ACC},0.52,0.91" >> "$CSV"
  echo "[rescheck] $M res s123 beta=0.2 -> res=${RES} acc=${ACC}  (BL res~0.52 acc~0.91)"
done
echo "=== RES CHECK DONE $(date) ==="
