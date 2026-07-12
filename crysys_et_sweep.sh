#!/bin/bash
# crysys_et_sweep.sh — points 2&3 ET retrain, 2 parallel lanes (lock slots 0+1)
# to saturate CrySyS's 18 cores (the A40-proven 2-slot pattern).
#   Point 2: imdb ET-adv  @ stronger PGD  (eps 0.0015->0.01, steps 7->10, alpha 0.005)
#   Point 3: adult+celeba ET-fair @ higher lambda (0.5 -> 2.0)
# Backs up the current (weak) models first; guard re-trains fresh + validates.
set -uo pipefail
cd "$HOME/ulrich"
source .venv/bin/activate 2>/dev/null || true
grep -q 'ET_EXTRA_CONFIG' run_et_experiments.sh || \
  sed -i '/RUN_CONFIG="strategy/ s/"$/ ${ET_EXTRA_CONFIG:-}"/' run_et_experiments.sh

LOG=et_sweep_crysys.log
: > "$LOG"; echo "=== SWEEP START $(date) (2 lanes) ===" >> "$LOG"

# backups (idempotent — never overwrite an existing backup)
cp -rn 420_et_adv/imdbnoniid 420_et_adv_eps0015_bak 2>/dev/null || true
for DS in adultnoniid celebanoniid; do
  cp -rn "420_et_fair/$DS" "420_et_fair_l05_bak_$DS" 2>/dev/null || true
done
# force fresh so the guard retrains at the new params
rm -rf 420_et_adv/imdbnoniid 420_et_fair/adultnoniid 420_et_fair/celebanoniid

# cell list "DS K MODE SEED|EXTRA_RUN_CONFIG"
CELLS=()
for K in 4 20; do for S in 42 107 123 2025 9928; do
  CELLS+=("imdbnoniid $K adv $S|et-adv-eps=0.01 et-adv-steps=10 et-adv-alpha=0.005"); done; done
for DS in adultnoniid celebanoniid; do for K in 4 20; do for S in 42 107 123 2025 9928; do
  CELLS+=("$DS $K fair $S|et-lambda-fair=2.0"); done; done; done

run_lane() {  # $1 = slot id (0|1) = even/odd cell index
  local slot=$1 i=0 cell args extra
  for cell in "${CELLS[@]}"; do
    if [ $((i % 2)) -eq "$slot" ]; then
      args="${cell%|*}"; extra="${cell#*|}"
      ET_EXTRA_CONFIG="$extra" bash et_guarded_run.sh $args "$slot" >> "$LOG" 2>&1
      echo "[lane$slot DONE] $args | $extra | $(date)" >> "$LOG"
    fi
    i=$((i+1))
  done
}
run_lane 0 & run_lane 1 & wait
echo "=== SWEEP ALL DONE $(date) ===" >> "$LOG"
echo "SWEEP_ALL_DONE" >> "$LOG"
