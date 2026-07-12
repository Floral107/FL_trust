#!/bin/bash
# cap_sweep.sh DS [CAP] [MAX] [KLIST] — race-free per-dataset ST/DY cap re-run.
#
# One dataset per invocation (each has its OWN pyproject, so 3 of these can run
# concurrently with no cross-dataset num-supernodes race). Within a dataset we
# K-PHASE: set num-supernodes ONCE per K (atomic), then run that K's cells with
# SKIP_SETNC=1 so concurrent cells never touch the shared pyproject — this is the
# fix for the set_num_cl write race that corrupted earlier sweeps. Finish a whole
# K-phase before changing K.
#
# Transform: BETA=1.0 + CAP (default 2.0) — the collapse-proof hard cap validated
# by the imdb-K20 A/B (DY+cap beats BL, zero collapse, res artifacts gone).
# Output: 420_<st|dy>_<wmet_lc>/<DS>/<K>/<seed>/global_model_round_10.keras
set -uo pipefail
cd "$HOME/ulrich"
source .venv/bin/activate 2>/dev/null || source /venv/main/bin/activate 2>/dev/null || true
DS="${1:?usage: cap_sweep.sh DS [CAP] [MAX] [KLIST]}"
CAP="${2:-2.0}"; MAX="${3:-8}"; KLIST="${4:-4 20}"
BETA=1.0
export BETA CAP
export FLR_TF_INTRA="${FLR_TF_INTRA:-2}" FLR_TF_INTER="${FLR_TF_INTER:-1}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
# NGPU>0 round-robins each cell's SERVER (which runs the GTG/PGD scoring) across
# NGPU GPUs via per-cell FLR_CUDA; clients stay on CPU (pyproject num-gpus=0).
# This accelerates the expensive in-loop scoring (celeba res/rel) ~10x. NGPU=0 =>
# pure CPU. MAX should be <= NGPU so each GPU gets at most one scoring cell.
NGPU="${NGPU:-0}"
export CUDA_VISIBLE_DEVICES="${FLR_CUDA:--1}"   # default; per-cell FLR_CUDA overrides below
# BOTH flags must be 0 — celeba (flwrlabs/celeba) trips OfflineModeIsEnabled if
# HF_DATASETS_OFFLINE is left unset/1 even when HF_HUB_OFFLINE=0.
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
mkdir -p logs
LOG="$HOME/ulrich/cap_sweep_${DS}.log"
SEEDS="42 107 123 2025 9928"
# WMETS env overrides the default per-dataset metric list (e.g. to split a cheap
# CPU lane from an expensive GPU lane). KLIST (4th arg) similarly subsets K.
if [ -n "${WMETS:-}" ]; then
  :  # use caller-provided WMETS
else
  case "$DS" in
    imdbnoniid) WMETS="acc loss rel res priv";;
    *)          WMETS="acc loss fairDP fairEO rel res priv";;
  esac
fi
echo "=== CAP SWEEP $DS beta=$BETA cap=$CAP MAX=$MAX KLIST='$KLIST' START $(date) ===" | tee -a "$LOG"

# Purge ONLY when explicitly asked (PURGE=1) — e.g. to clear old beta=0.2 models.
# Default no-purge so RE-RUNS RESUME via skip-if-valid (fills transient-fail gaps
# instead of wiping completed cells). On a fresh box there's nothing to purge.
if [ "${PURGE:-0}" = "1" ]; then
  for m in st dy; do for w in $WMETS; do wl=$(printf '%s' "$w" | tr '[:upper:]' '[:lower:]')
    rm -rf "420_${m}_${wl}/${DS}" 2>/dev/null; done; done
fi

for K in $KLIST; do
  echo "[phase $DS K=$K] set num-supernodes $(date +%T)" | tee -a "$LOG"
  python3 set_num_cl.py --data="$DS" --clients="$K" >>"$LOG" 2>&1
  python3 -c "import tomllib; tomllib.load(open('$DS/pyproject.toml','rb'))" 2>>"$LOG" \
    || { echo "[ABORT] $DS pyproject invalid K=$K" | tee -a "$LOG"; exit 1; }
  # Build the list of cells that ACTUALLY need work (round-10 ckpt missing), then
  # round-robin the GPU over THIS list. Assigning GPU by the full-enumeration index
  # collides mod NGPU on resume: with most cells already done (skipped instantly),
  # the few still-running cells get scattered indices that map to the same GPU ->
  # 2 cells share one 12GB card -> ResourceExhaustedError. Round-robining over only
  # the pending cells keeps the <=MAX(==NGPU) concurrent cells on distinct GPUs.
  PENDING=()
  for MODE in st dy; do for WMET in $WMETS; do
    wl=$(printf '%s' "$WMET" | tr '[:upper:]' '[:lower:]')
    for SEED in $SEEDS; do
      [ -f "420_${MODE}_${wl}/${DS}/${K}/${SEED}/global_model_round_10.keras" ] \
        || PENDING+=("$MODE $WMET $SEED")
    done
  done; done
  echo "[phase $DS K=$K] ${#PENDING[@]} cells pending" | tee -a "$LOG"
  if [ "$NGPU" -gt 0 ]; then
    # GPU-LANE POOL: NGPU parallel lanes, each PINNED to one GPU, running its
    # cells strictly sequentially. Guarantees exactly 1 cell per GPU at any time
    # (TF grabs ~10GB/cell, so 2 cells on a 12GB card OOM). A plain sliding window
    # can't do this: `wait -n` doesn't reveal which GPU freed, so a new cell may
    # land on a GPU still occupied. Lanes bind cell->GPU deterministically instead.
    for lane in $(seq 0 $(( NGPU - 1 ))); do
      (
        li=0
        for cell in "${PENDING[@]}"; do
          if [ $(( li % NGPU )) -eq "$lane" ]; then
            set -- $cell; M=$1; W=$2; S=$3
            SKIP_SETNC=1 BETA="$BETA" CAP="$CAP" FLR_CUDA="$lane" \
              bash st_dy_guarded_run.sh "$DS" "$K" "$M" "$W" "$S" "$(( 300 + li ))" >>"$LOG" 2>&1
          fi
          li=$(( li + 1 ))
        done
      ) &
    done
    wait
  else
    # CPU: sliding window of MAX concurrent cells.
    gi=0; running=0
    for cell in "${PENDING[@]}"; do
      set -- $cell; M=$1; W=$2; S=$3
      SKIP_SETNC=1 BETA="$BETA" CAP="$CAP" FLR_CUDA=-1 \
        bash st_dy_guarded_run.sh "$DS" "$K" "$M" "$W" "$S" "$(( 300 + gi ))" >>"$LOG" 2>&1 &
      gi=$(( gi + 1 )); running=$(( running + 1 ))
      if (( running >= MAX )); then wait -n; running=$(( running - 1 )); fi
    done
    wait
  fi
  echo "[phase $DS K=$K] DONE $(date +%T)" | tee -a "$LOG"
done

echo "=== CAP SWEEP $DS DONE $(date) ===" | tee -a "$LOG"
ok=0; miss=0
for m in st dy; do for w in $WMETS; do wl=$(printf '%s' "$w"|tr '[:upper:]' '[:lower:]'); for K in $KLIST; do for s in $SEEDS; do
  f="420_${m}_${wl}/${DS}/${K}/${s}/global_model_round_10.keras"
  if [ -f "$f" ]; then ok=$((ok+1)); else miss=$((miss+1)); fi
done; done; done; done
echo "[$DS] ckpts present=$ok missing=$miss" | tee -a "$LOG"
