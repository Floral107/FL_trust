#!/bin/bash
# crysys_et_fair_sweep.sh — Point 3 only: adult+celeba ET-fair @ lambda=2.0
# (up from 0.5), 2 parallel lanes to saturate CrySyS's 18 cores.
set -uo pipefail
cd "$HOME/ulrich"
source .venv/bin/activate 2>/dev/null || true
grep -q 'ET_EXTRA_CONFIG' run_et_experiments.sh || \
  sed -i '/RUN_CONFIG="strategy/ s/"$/ ${ET_EXTRA_CONFIG:-}"/' run_et_experiments.sh

LOG=et_fair_sweep.log
: > "$LOG"; echo "=== FAIR SWEEP START $(date) lambda=2.0 (2 lanes) ===" >> "$LOG"
for DS in adultnoniid celebanoniid; do
  cp -rn "420_et_fair/$DS" "420_et_fair_l05_bak_$DS" 2>/dev/null || true
done
rm -rf 420_et_fair/adultnoniid 420_et_fair/celebanoniid

CELLS=()
for DS in adultnoniid celebanoniid; do for K in 4 20; do for S in 42 107 123 2025 9928; do
  CELLS+=("$DS $K fair $S"); done; done; done

run_lane() {  # $1 = slot id (0|1) = even/odd index
  local slot=$1 i=0 c
  for c in "${CELLS[@]}"; do
    if [ $((i % 2)) -eq "$slot" ]; then
      ET_EXTRA_CONFIG="et-lambda-fair=2.0" bash et_guarded_run.sh $c "$slot" >> "$LOG" 2>&1
      echo "[lane$slot DONE] $c $(date)" >> "$LOG"
    fi
    i=$((i+1))
  done
}
run_lane 0 & run_lane 1 & wait
echo "FAIR_SWEEP_DONE $(date)" >> "$LOG"
