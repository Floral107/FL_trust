#!/bin/bash
# Matched-threat proof: train imdb ET-adv at eps=0.005 (where BL is actually
# vulnerable: BL res@0.005 = 0.42 from the probe) and eval res @eps=0.005.
# If ET >> 0.42, adversarial training demonstrably beats BL at a meaningful threat.
set -uo pipefail
cd "$HOME/ulrich"
source .venv/bin/activate 2>/dev/null || true
grep -q 'ET_EXTRA_CONFIG' run_et_experiments.sh || \
  sed -i '/RUN_CONFIG="strategy/ s/"$/ ${ET_EXTRA_CONFIG:-}"/' run_et_experiments.sh

LOG=etadv_eps005_proof.log
: > "$LOG"; echo "=== ET-adv eps=0.005 matched-threat proof START $(date) ===" >> "$LOG"
rm -rf 420_et_adv/imdbnoniid

CELLS=(); for K in 4 20; do for s in 42 107 123 2025 9928; do CELLS+=("imdbnoniid $K adv $s"); done; done
run_lane() {
  local slot=$1 i=0 c
  for c in "${CELLS[@]}"; do
    if [ $((i % 2)) -eq "$slot" ]; then
      ET_EXTRA_CONFIG="et-adv-eps=0.005 et-adv-alpha=0.002 et-adv-steps=10" \
        bash et_guarded_run.sh $c "$slot" >> "$LOG" 2>&1
      echo "[lane$slot DONE] $c $(date)" >> "$LOG"
    fi
    i=$((i+1))
  done
}
run_lane 0 & run_lane 1 & wait
echo "=== training done; eval res @eps=0.005 (BL@0.005=0.42 from probe) $(date) ===" >> "$LOG"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
python - <<'PY' >> "$LOG" 2>&1
import sys, os; sys.path.insert(0, '/home/crysys/ulrich')
import numpy as np, tensorflow as tf
from reweight_eval import load_test_data
from attack_metric import calculate_pgd_score
x, y, *_ = load_test_data("imdbnoniid")
def res(p, eps):
    m = tf.keras.models.load_model(p, compile=False)
    return calculate_pgd_score(m, x, "imdbnoniid", y, epsilon=eps, alpha=eps, num_iter=10)
for K in (4, 20):
    et = []
    for s in (42, 107, 123, 2025, 9928):
        p = f"420_et_adv/imdbnoniid/{K}/{s}/global_model_round_10.keras"
        if os.path.exists(p):
            et.append(res(p, 0.005))
    print(f"PROOF imdb ET-adv K{K} (trained@0.005, eval@0.005): res={np.mean(et):.3f}+-{np.std(et):.3f} (n={len(et)}) | BL@0.005=0.42")
print("PROOF_DONE")
PY
echo "ETADV_EPS005_PROOF_DONE $(date)" >> "$LOG"
