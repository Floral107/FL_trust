#!/bin/bash
# celeba_k20_dyfill.sh [MAX] — re-run the 10 celeba K20 DY rel/res cells with the
# COMPILED in-loop scoring (pgd_attack + fast_infer @tf.function; bit-identical
# numerics, ~2.7x faster on CPU). All 5 seeds each, run concurrently. wall-clock
# = slowest single cell (~26h with 2.7x), so a 36h per-cell timeout gives margin.
# st/dy_priv/fairDP/fairEO/acc/loss + all ST cells are already done; this fills
# only dy_rel + dy_res at K=20 (the influence/PGD-heavy DY column).
cd /root/ulrich
source .venv/bin/activate
export FLR_TF_INTRA=4 FLR_TF_INTER=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=-1
export BETA=0.2
MAX="${1:-10}"
SEEDS="42 107 123 2025 9928"
TIMEOUT=129600   # 36h per cell
TASKS=()
for WM in rel res; do
  for S in $SEEDS; do TASKS+=("dy $WM $S"); done
done
echo "[dyfill] ${#TASKS[@]} cells (compiled PGD), MAX=$MAX, timeout=${TIMEOUT}s $(date)"
i=0; running=0
for T in "${TASKS[@]}"; do
  set -- $T; MODE=$1; WM=$2; S=$3
  slot=$((84000+i))
  timeout $TIMEOUT bash st_dy_guarded_run.sh celebanoniid 20 "$MODE" "$WM" "$S" "$slot" >> full_rerun_b0.2.log 2>&1 &
  i=$((i+1)); running=$((running+1))
  if (( running >= MAX )); then wait -n; running=$((running-1)); fi
done
wait
echo "DYFILL_DONE $(date)"
