#!/bin/bash
# celeba_k20_fill.sh [MAX] — fill the remaining celeba K20 expensive-metric ST/DY
# cells (seeds 107,123,2025,9928; seed 42 runs separately in tmux k20diag).
# These use GTG-Shapley(30 perms) x heavy metric (res=PGD, rel=noise, priv=margin)
# and are CPU-bound-slow at K=20, NOT deadlocked. A 20h per-cell timeout is the
# feasibility arbiter: finishes -> data; times out (likely dy_res) -> documented gap.
# Ordered cheapest-first so the tractable data lands before the expensive res cells.
cd /root/ulrich
source .venv/bin/activate
export FLR_TF_INTRA=4 FLR_TF_INTER=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=-1
export BETA=0.2
MAX="${1:-8}"
SEEDS="107 123 2025 9928"
TIMEOUT=72000   # 20h per cell
# cheapest metric first, res (PGD) last
TASKS=()
for spec in "dy priv" "st res" "dy rel" "dy res"; do
  set -- $spec; MODE=$1; WM=$2
  for S in $SEEDS; do TASKS+=("$MODE $WM $S"); done
done
echo "[k20fill] ${#TASKS[@]} cells, MAX=$MAX, timeout=${TIMEOUT}s $(date)"
i=0; running=0
for T in "${TASKS[@]}"; do
  set -- $T; MODE=$1; WM=$2; S=$3
  slot=$((83000+i))
  timeout $TIMEOUT bash st_dy_guarded_run.sh celebanoniid 20 "$MODE" "$WM" "$S" "$slot" >> full_rerun_b0.2.log 2>&1 &
  i=$((i+1)); running=$((running+1))
  if (( running >= MAX )); then wait -n; running=$((running-1)); fi
done
wait
echo "K20FILL_DONE $(date)"
