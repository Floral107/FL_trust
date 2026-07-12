#!/bin/bash
# Group A: re-run the 20 celeba K20 fairDP/fairEO ST/DY cells that reached ROUND 10
# but lost their model to the transient round_10-save skip (same as imdb dy priv).
cd /root/ulrich
source .venv/bin/activate
export FLR_TF_INTRA=4 FLR_TF_INTER=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=-1
export BETA=0.2
MAX="${1:-4}"
SEEDS="42 107 123 2025 9928"
TASKS=()
for MODE in st dy; do
  for WM in fairDP fairEO; do
    for S in $SEEDS; do TASKS+=("$MODE $WM $S"); done
  done
done
echo "[groupA] ${#TASKS[@]} cells, MAX=$MAX $(date)"
i=0; running=0
for T in "${TASKS[@]}"; do
  set -- $T; MODE=$1; WM=$2; S=$3
  slot=$((81000+i))
  timeout 14400 bash st_dy_guarded_run.sh celebanoniid 20 "$MODE" "$WM" "$S" "$slot" >> full_rerun_b0.2.log 2>&1 &
  i=$((i+1)); running=$((running+1))
  if (( running >= MAX )); then wait -n; running=$((running-1)); fi
done
wait
echo "GROUPA_DONE $(date)"
