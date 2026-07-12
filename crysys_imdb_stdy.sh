#!/bin/bash
# crysys_imdb_stdy.sh BETA [MAX] — race-free imdb ST/DY sweep for CrySyS.
#
# CrySyS ran imdb ST/DY fine in the original sweep; my concurrent orchestrator's
# per-cell set_num_cl writes corrupted the shared pyproject ("declared twice" /
# truncation). This launcher avoids that entirely:
#   - set num-supernodes ONCE per K-phase (serial, atomic write), verify it parses
#   - run that K's cells with SKIP_SETNC=1 so concurrent cells NEVER touch pyproject
#   - finish a whole K-phase before changing num-supernodes (no K4/K20 value race)
#
# imdb has 5 weight-metrics (no fairness): acc loss rel res priv.
# Grid: 5 wmet x {st,dy} x {K4,K20} x 5 seeds = 100 cells.
set -uo pipefail
cd "$HOME/ulrich"
source .venv/bin/activate 2>/dev/null || true
BETA="${1:-0.2}"; MAX="${2:-2}"; KLIST="${3:-4 20}"
# CrySyS has 23GB RAM — imdb sims (K client-actors each loading model+embedded
# text) OOM at MAX>2. And imdb-K20 (20 actors/sim) is too heavy here at any MAX.
# So run CrySyS on K=4 only (KLIST="4", MAX=2); A40 handles imdb-K20.
export BETA FLR_TF_INTRA=4 FLR_TF_INTER=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=-1
mkdir -p logs
LOG="$HOME/ulrich/imdb_stdy.log"
SEEDS="42 107 123 2025 9928"
WMETS="acc loss rel res priv"
echo "=== CrySyS IMDB ST/DY beta=$BETA MAX=$MAX START $(date) ===" | tee -a "$LOG"

# purge old beta=1 imdb ST/DY models once (results preserved in data/; regenerated here)
for m in st dy; do for w in $WMETS; do rm -rf "420_${m}_${w}/imdbnoniid" 2>/dev/null; done; done

for K in $KLIST; do
  echo "[phase K=$K] set num-supernodes once $(date +%T)" | tee -a "$LOG"
  python3 set_num_cl.py --data=imdbnoniid --clients=$K >>"$LOG" 2>&1
  if ! python3 -c "import tomllib; tomllib.load(open('imdbnoniid/pyproject.toml','rb'))" 2>>"$LOG"; then
    echo "[ABORT] imdb pyproject invalid for K=$K" | tee -a "$LOG"; exit 1
  fi
  i=0; running=0
  for MODE in st dy; do
    for WMET in $WMETS; do
      for SEED in $SEEDS; do
        SKIP_SETNC=1 BETA="$BETA" bash st_dy_guarded_run.sh imdbnoniid "$K" "$MODE" "$WMET" "$SEED" "$((100+i))" >>"$LOG" 2>&1 &
        i=$(( i+1 )); running=$(( running+1 ))
        if (( running >= MAX )); then wait -n; running=$(( running-1 )); fi
      done
    done
  done
  wait   # complete this K-phase before changing num-supernodes
  echo "[phase K=$K] DONE $(date +%T)" | tee -a "$LOG"
done
echo "=== CrySyS IMDB ST/DY DONE $(date) ===" | tee -a "$LOG"
