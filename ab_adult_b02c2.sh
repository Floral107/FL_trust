#!/bin/bash
# ab_adult_b02c2.sh — transform ablation: beta=0.2 + cap=2.0 (shrink THEN cap)
# vs the canonical beta=1.0 + cap=2.0, on adult acc+loss cells (the healthy
# dataset, so differences reflect the TRANSFORM, not infra).
#
# Sandbox: OUTPRE=420b02c2 + FLR_SAVE_PREFIX (task.py) -> ckpts under
# 420b02c2_{st,dy}_{acc,loss}/adultnoniid/... — canonical 420_* untouched.
#
# WAITS until the canonical adult grid is complete (140 ckpts) so the driver's
# adult sweep cannot race this script's set_num_cl K-phasing on the shared
# adult pyproject.
set -uo pipefail
cd "$HOME/ulrich"
source .venv/bin/activate 2>/dev/null || true
MAX="${1:-3}"
export BETA=0.2 CAP=2.0 OUTPRE=420b02c2
export FLR_TF_INTRA=2 FLR_TF_INTER=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=-1
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
mkdir -p logs
LOG="$HOME/ulrich/ab_adult_b02c2.log"
SEEDS="42 107 123 2025 9928"
echo "=== A/B adult beta=0.2 cap=2.0 START $(date) ===" | tee -a "$LOG"

# wait for canonical adult to be complete (no pyproject race with the driver)
while true; do
  n=$(find 420_st_* 420_dy_* -path "*/adultnoniid/*" -name global_model_round_10.keras 2>/dev/null | wc -l)
  [ "$n" -ge 140 ] && break
  echo "[wait] canonical adult at $n/140 — sleeping 120s" | tee -a "$LOG"
  sleep 120
done

for K in 4 20; do
  echo "[phase K=$K]" | tee -a "$LOG"
  python3 set_num_cl.py --data=adultnoniid --clients=$K >>"$LOG" 2>&1
  python3 -c "import tomllib; tomllib.load(open('adultnoniid/pyproject.toml','rb'))" 2>>"$LOG" \
    || { echo "[ABORT] adult pyproject invalid" | tee -a "$LOG"; exit 1; }
  i=0; running=0
  for MODE in st dy; do for WMET in acc loss; do for SEED in $SEEDS; do
    SKIP_SETNC=1 BETA=0.2 CAP=2.0 OUTPRE=420b02c2 \
      bash st_dy_guarded_run.sh adultnoniid "$K" "$MODE" "$WMET" "$SEED" "$((500+i))" >>"$LOG" 2>&1 &
    i=$((i+1)); running=$((running+1))
    if (( running >= MAX )); then wait -n; running=$((running-1)); fi
  done; done; done
  wait
done
echo "=== A/B DONE $(date) ===" | tee -a "$LOG"
ok=0; for m in st dy; do for w in acc loss; do for K in 4 20; do for s in $SEEDS; do
  [ -f "420b02c2_${m}_${w}/adultnoniid/${K}/${s}/global_model_round_10.keras" ] && ok=$((ok+1)); done; done; done; done
echo "[A/B] ckpts present=$ok/40" | tee -a "$LOG"
