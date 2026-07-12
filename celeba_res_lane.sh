#!/bin/bash
# celeba_res_lane.sh — dedicated lane for the 20 celeba res cells.
#
# res (PGD in-loop scoring) cannot share a 12GB GPU with GPU-resident clients
# (OOM even at num-gpus=0.5), so: WAIT until the driver finishes the celeba
# non-res grid (120 ckpts; driver EXP stops its celeba relaunches then), flip
# celeba clients to CPU (num-gpus=0.0), and run the res cells via cap_sweep's
# GPU-lane pool (server scoring on GPU, 1 cell/GPU, participation guard active).
set -uo pipefail
cd "$HOME/ulrich"
source .venv/bin/activate 2>/dev/null || true
LOG="$HOME/ulrich/celeba_res_lane.log"
echo "=== celeba res lane START $(date) ===" | tee -a "$LOG"
while true; do
  n=$(find 420_st_* 420_dy_* -path "*/celebanoniid/*" -name global_model_round_10.keras 2>/dev/null \
      | grep -vE "420_(st|dy)_res/" | wc -l)
  busy=$(pgrep -f "flower-simu[l]ation" -a 2>/dev/null | grep -c celebanoniid)
  if [ "$n" -ge 120 ] && [ "$busy" -eq 0 ]; then break; fi
  echo "[wait] celeba non-res $n/120, busy=$busy — sleep 180" | tee -a "$LOG"
  sleep 180
done
sed -i "s/^num-gpus = 0.25$/num-gpus = 0.0/" celebanoniid/pyproject.toml
grep -q "num-gpus = 0.0" celebanoniid/pyproject.toml || { echo "[ABORT] num-gpus flip failed" | tee -a "$LOG"; exit 1; }
# All other datasets are done, so use ALL 6 GPUs for res. res runs CPU clients
# (server PGD needs the whole GPU) and each CPU client holds ~2.7GB of images, so
# to fit 6 lanes we drop init num-cpus 4->2 (2 concurrent clients/cell = ~5.4GB;
# 6 lanes ~= 32-44GB, RAM-safe). Scoring (the res bottleneck) is unaffected.
sed -i "/init-args/{n;s/^num-cpus = 4$/num-cpus = 2/}" celebanoniid/pyproject.toml
echo "[flip] celeba clients -> CPU (num-gpus=0.0, init num-cpus=2); res on 4 GPUs" | tee -a "$LOG"
# SELF-HEALING loop: cap_sweep runs once and leaves any failed cell as a gap, so
# total never reaches 380. Loop it (resume skips done cells) until all 20 res
# ckpts exist. NGPU=4 (not 6): the heavy dy_res cells dropped clients under the
# 6-lane RAM pressure; 4 lanes x2 clients ~= 22GB is comfortably safe.
while true; do
  done=$(find 420_st_res 420_dy_res -path "*celebanoniid*" -name global_model_round_10.keras 2>/dev/null | wc -l)
  if [ "$done" -ge 20 ]; then echo "[res] all 20 res cells present" | tee -a "$LOG"; break; fi
  echo "[res] $done/20 done — running res sweep $(date +%T)" | tee -a "$LOG"
  # GPU servers (NGPU=4) with an AGGRESSIVE in-loop PGD cap (128 samples). CPU is
  # infeasible for K20 res (GTG over 20 clients x PGD/round ~= 6h/cell); GPU is fast
  # but OOM'd at 512 samples. 128-sample PGD tensors are tiny (~0.5GB) so they fit
  # the 12GB card even across the thousands of GTG coalition evals, and 128 samples
  # is ample for the weighting PROXY (reported res still comes from full post-hoc).
  NGPU=4 FLR_RES_MAX_ATTACK=128 WMETS="res" bash cap_sweep.sh celebanoniid 2.0 4 2>&1 | tee -a "$LOG"
  sleep 5
done
echo "=== celeba res lane DONE $(date) ===" | tee -a "$LOG"
