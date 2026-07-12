#!/bin/bash
# full_stdy_rerun.sh BETA [MAX] — the ONE coordinated CPU sweep that replaces the
# post-hoc do-no-harm guard with shrinkage(beta) across the whole Tab.4 ST/DY
# grid, AND retrains celeba et_adv at eps=0.01 (so the defense is trained and
# evaluated at the same epsilon). All CPU — runs in its own lane alongside the
# GPU celeba_adv and the CrySyS imdb_adv (which produce the res-contribution
# FIGURES, a separate output). Nothing here is recomputed elsewhere.
#
# Produces (canonical paths, OVERWRITING the old beta=1 collapsed models so the
# existing eval pipeline / eval_stdy_models.py works unchanged):
#   380 ST/DY : 420_<st|dy>_<metric>/<ds>/<K>/<seed>/global_model_round_10.keras
#    10 et_adv: 420_et_adv/celebanoniid/<K>/<seed>/...            (eps=0.01)
#
# Grid: {adult,celeba,imdb}noniid x K{4,20} x seeds{42,107,123,2025,9928}
#       x modes{st,dy} x weight_metrics(7 for adult/celeba, 5 for imdb).
set -uo pipefail
cd "$HOME/ulrich"   # /root/ulrich on A40, /home/crysys/ulrich on CrySyS
source .venv/bin/activate 2>/dev/null || true
BETA="${1:?usage: full_stdy_rerun.sh BETA [MAX] [CAP]}"
MAX="${2:-12}"
# Hard weight cap for the ST/DY transform (>=1, or 0 to disable). Default 0.0 =
# pure shrinkage(beta). The collapse-proof cap (cap=2.0) replaces beta=0.2 after
# the imdb-K20 A/B showed cap+DY beats BL with zero collapse.
CAP="${3:-0.0}"
export BETA CAP
export FLR_TF_INTRA=4 FLR_TF_INTER=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=-1
mkdir -p logs
LOG="$HOME/ulrich/full_rerun_b${BETA}.log"
echo "=== FULL ST/DY RE-RUN beta=$BETA cap=$CAP MAX=$MAX START $(date) ===" | tee -a "$LOG"

SEEDS="42 107 123 2025 9928"
# Which datasets this host handles (env override). Lets A40 do adult+celeba while
# CrySyS runs imdb in parallel. Default = all three (single-host behaviour).
DATASETS="${DATASETS:-adultnoniid celebanoniid imdbnoniid}"
echo "[scope] DATASETS=$DATASETS" | tee -a "$LOG"
declare -A WMETS
WMETS[adultnoniid]="acc loss fairDP fairEO rel res priv"
WMETS[celebanoniid]="acc loss fairDP fairEO rel res priv"
WMETS[imdbnoniid]="acc loss rel res priv"

# ── build task list ──────────────────────────────────────────────────────────
TASKS=()
# et_adv (celeba only, 10) first so its column re-evals early — only if celeba in
# scope AND explicitly requested. ET is unaffected by the ST/DY weight transform
# (no weighting), so a cap re-run skips it by default (ETADV=0). Set ETADV=1 to
# also retrain the celeba et_adv column.
ETADV="${ETADV:-0}"
if [ "$ETADV" = "1" ] && echo " $DATASETS " | grep -q " celebanoniid "; then
  for K in 4 20; do for SEED in $SEEDS; do TASKS+=("etadv $K $SEED"); done; done
fi
# then the ST/DY cells for the selected datasets.
for DS in $DATASETS; do
  for K in 4 20; do
    for MODE in st dy; do
      for WMET in ${WMETS[$DS]}; do
        for SEED in $SEEDS; do
          TASKS+=("stdy $DS $K $MODE $WMET $SEED")
        done
      done
    done
  done
done
echo "[plan] ${#TASKS[@]} tasks (datasets: $DATASETS), MAX=$MAX parallel" | tee -a "$LOG"

# ── purge OLD models ONCE so the per-cell skip-if-valid doesn't keep the stale
#    beta=1 checkpoints. A sentinel makes restarts resume-safe (purge only on the
#    first launch; subsequent restarts let skip-if-valid resume THIS run's work).
#    Results of the old beta=1 runs are preserved in data/{st,dy}; models are
#    fully regenerated here, so deleting the A40 checkpoints is non-destructive.
SENTINEL=".fullrerun_b${BETA}_c${CAP}.purged"
if [ ! -f "$SENTINEL" ]; then
  echo "[purge] removing old ST/DY checkpoints for datasets [$DATASETS] (beta=$BETA)" | tee -a "$LOG"
  for DS in $DATASETS; do rm -rf 420_st_*/"$DS" 420_dy_*/"$DS" 2>/dev/null; done
  if [ "$ETADV" = "1" ] && echo " $DATASETS " | grep -q " celebanoniid "; then
    for K in 4 20; do for S in $SEEDS; do rm -rf "420_et_adv/celebanoniid/${K}/${S}" 2>/dev/null; done; done
  fi
  touch "$SENTINEL"
else
  echo "[purge] sentinel present — resume mode (skip-if-valid keeps this run's done cells)" | tee -a "$LOG"
fi

run_task() {  # slot kind args...
  local SLOT=$1 KIND=$2; shift 2
  if [ "$KIND" = "stdy" ]; then
    BETA="$BETA" CAP="$CAP" bash st_dy_guarded_run.sh "$1" "$2" "$3" "$4" "$5" "$SLOT" >>"$LOG" 2>&1
  else  # etadv: celeba adv @ eps=0.01 (purge old eps=0.03 model so it retrains)
    local K=$1 SEED=$2
    rm -rf "420_et_adv/celebanoniid/${K}/${SEED}" 2>/dev/null
    ET_ADV_EPS=0.01 ET_ADV_ALPHA=0.0025 \
      bash et_guarded_run.sh celebanoniid "$K" adv "$SEED" "$SLOT" >>"$LOG" 2>&1
  fi
}

# ── sliding window; UNIQUE slot per task so per-slot flocks never contend
#    (concurrency is bounded solely by MAX via wait -n). ──────────────────────
i=0; running=0
for T in "${TASKS[@]}"; do
  run_task "$i" $T &
  i=$(( i+1 )); running=$(( running+1 ))
  if (( running >= MAX )); then wait -n; running=$(( running-1 )); fi
done
wait
echo "=== FULL ST/DY RE-RUN beta=$BETA DONE $(date) ===" | tee -a "$LOG"
