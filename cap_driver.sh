#!/bin/bash
# cap_driver.sh — self-healing supervisor for the cap re-run on the pids-capped
# box (cgroup pids.max=13568 here => ~20 Ray sims ceiling; we run ~13 for headroom).
# Keeps the 3 per-dataset sweeps alive; when one finishes (session gone) but is
# still incomplete, relaunches it (cap_sweep resumes via skip-if-valid, filling
# any transient/fork gaps). Exits when all 380 ST/DY ckpts exist.
cd "$HOME/ulrich"
# New box: pids.max=13568, 56 cores, 62 GB RAM, 6x RTX3060 (12GB ea).
# adult/imdb on CPU; celeba routed to the 6 GPUs at 1 cell/GPU (MAX=6). NOTE:
# 2 cells/GPU (MAX=12) GPU-OOMs on the 12GB 3060 for dy cells that re-score every
# round (ResourceExhaustedError -> rc=0 FAIL), so keep 1/GPU unless TF memory
# growth is enabled in celeba. Requires the celeba partition cache pre-warmed
# (prewarm_celeba.py 4 & 20) else cold preps thrash the dataset-wide lock.
# imdb MAX=2: the first sweep's imdb cells were poisoned by Ray OOM-killing the
# 1GB-USE client actors under RAM pressure (19/20-failure rounds silently
# aggregated) — cap concurrent imdb sims to keep RAM comfortable.
# LOW concurrency (~7 sims total): 12 sims + guard-relaunch churn + extra lanes
# drove RAM to 11GB-free and OOM-killed client actors on every dataset (death
# spiral). Keep total modest so cells actually complete and the guard rarely fires.
# celeba 2 (was 3): 3 K20 celeba lanes (33GB) + adult/imdb under the K20 tail load
# swap-DIED the 62GB box twice. 2 lanes = ~22GB, leaves headroom so it can't seize.
declare -A MAXOF=(  [adultnoniid]=2 [imdbnoniid]=2 [celebanoniid]=2 )
# celeba NGPU=3 (was 6): each celeba client holds ~2.7GB (up to ~19k 64x64x3
# float32 images in HOST RAM), so 6 lanes x4 clients x2.7GB ~= 65GB > 62GB box ->
# Ray OOM-kills clients -> guard rejects -> celeba stuck at 0. 3 lanes = ~32GB,
# RAM-safe. num-gpus=0.25 keeps client CNN training on-GPU (~10x vs CPU), 1
# cell/GPU via the lane pool. GPU flip does NOT fix RAM (images are host-side).
declare -A NGPUOF=( [adultnoniid]=0 [imdbnoniid]=0 [celebanoniid]=2 )
# EXP[celeba]=120 = the non-res grid this driver owns. Once reached, the driver
# STOPS relaunching the celeba sweep (its set_num_cl K-phasing would race the
# dedicated res lane, which flips celeba num-gpus to 0.0 and runs the 20 res
# cells with CPU clients + GPU servers).
declare -A EXP=(   [adultnoniid]=140 [imdbnoniid]=100 [celebanoniid]=120 )
declare -A SESS=(  [adultnoniid]=adult [imdbnoniid]=imdb [celebanoniid]=celeba )
# celeba runs with GPU clients (num-gpus=0.25) for the cheap metrics. res is
# EXCLUDED here: its PGD scoring is too GPU-memory-heavy to share the card with
# GPU-resident clients (OOMs). The lone remaining res cell (dy_res_K20) is run
# separately with CPU clients. EXP[celeba]=139 = 140 minus that 1 pending res
# cell, so the driver drives celeba's 14 cheap cells to completion then stops.
declare -A WMETSOF=( [celebanoniid]="acc loss fairDP fairEO rel priv" )
LOG="$HOME/ulrich/cap_driver.log"
echo "=== cap_driver START $(date) ===" | tee -a "$LOG"
while true; do
  total=0
  for ds in adultnoniid imdbnoniid celebanoniid; do
    n=$(find 420_st_* 420_dy_* -path "*/$ds/*" -name global_model_round_10.keras 2>/dev/null | wc -l)
    total=$(( total + n ))
    if ! tmux has-session -t "${SESS[$ds]}" 2>/dev/null; then
      if [ "$n" -lt "${EXP[$ds]}" ]; then
        echo "$(date +%H:%M) relaunch ${SESS[$ds]} ($ds $n/${EXP[$ds]})" | tee -a "$LOG"
        tmux new-session -d -s "${SESS[$ds]}" "cd $HOME/ulrich && exec env NGPU=${NGPUOF[$ds]} WMETS='${WMETSOF[$ds]:-}' bash cap_sweep.sh $ds 2.0 ${MAXOF[$ds]}"
        sleep 20   # stagger to avoid a fork burst
      fi
    fi
  done
  pids=$(cat /sys/fs/cgroup/pids/pids.current 2>/dev/null)
  pmax=$(cat /sys/fs/cgroup/pids/pids.max 2>/dev/null)
  echo "$(date +%H:%M) total=$total/380 (A=$(find 420_st_* 420_dy_* -path '*/adultnoniid/*' -name global_model_round_10.keras 2>/dev/null|wc -l) I=$(find 420_st_* 420_dy_* -path '*/imdbnoniid/*' -name global_model_round_10.keras 2>/dev/null|wc -l) C=$(find 420_st_* 420_dy_* -path '*/celebanoniid/*' -name global_model_round_10.keras 2>/dev/null|wc -l)) pids=$pids/$pmax" | tee -a "$LOG"
  if [ "$total" -ge 380 ]; then echo "=== ALL_DONE $(date) ===" | tee -a "$LOG"; break; fi
  sleep 120
done
