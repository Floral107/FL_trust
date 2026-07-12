#!/bin/bash
# Point 4: IMDB ET-adv with STRONGER PGD (eps 0.0015->0.03, steps 7->15, alpha 0.006),
# 2 parallel lanes. Then eval res at the table's eps=0.0015 to see if ET clears BL (0.82 K20).
set -uo pipefail
cd "$HOME/ulrich"
source .venv/bin/activate 2>/dev/null || true
grep -q 'ET_EXTRA_CONFIG' run_et_experiments.sh || \
  sed -i '/RUN_CONFIG="strategy/ s/"$/ ${ET_EXTRA_CONFIG:-}"/' run_et_experiments.sh

LOG=imdb_etadv_strong.log
: > "$LOG"; echo "=== imdb ET-adv STRONG (eps=0.03 alpha=0.006 steps=15) START $(date) ===" >> "$LOG"
rm -rf 420_et_adv/imdbnoniid   # force fresh (round_10 originals preserved in local a40_snapshot)

CELLS=(); for K in 4 20; do for s in 42 107 123 2025 9928; do CELLS+=("imdbnoniid $K adv $s"); done; done
run_lane() {
  local slot=$1 i=0 c
  for c in "${CELLS[@]}"; do
    if [ $((i % 2)) -eq "$slot" ]; then
      ET_EXTRA_CONFIG="et-adv-eps=0.03 et-adv-alpha=0.006 et-adv-steps=15" \
        bash et_guarded_run.sh $c "$slot" >> "$LOG" 2>&1
      echo "[lane$slot DONE] $c $(date)" >> "$LOG"
    fi
    i=$((i+1))
  done
}
run_lane 0 & run_lane 1 & wait
echo "=== training done; evaluating res @eps=0.0015 $(date) ===" >> "$LOG"
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 CUDA_VISIBLE_DEVICES= FLR_TF_INTRA=8
python eval_et_models.py --datasets imdbnoniid --modes adv >> "$LOG" 2>&1
echo "IMDB_ETADV_STRONG_DONE $(date)" >> "$LOG"
