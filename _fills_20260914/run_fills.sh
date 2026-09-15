#!/usr/bin/env bash
# usage: run_fills.sh <jobfile>   ; job line: <dataset> <partition> <K> <seed> <method>
# robustness.py is resume-safe (skips done (seed,K,round)), so re-running a lane is free.
set -u
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=          # GPU eval throws XLA after PGD and writes 0 rows
export OMP_NUM_THREADS=4              # 3 lanes x 4 threads = 12 cores
export TF_CPP_MIN_LOG_LEVEL=3
while read -r ds part K sd m; do
  [ -z "${ds:-}" ] && continue
  case "$ds" in \#*) continue;; esac
  log="fill_${ds}_${K}_${sd}_${m}.log"
  echo "[$(date -u +%H:%M:%S)] START $ds K=$K seed=$sd $m" >> fills.log
  t0=$(date +%s)
  seed=$sd $PY ../Code/robustness.py --dataset "$ds" --partition "$part" --num_rounds 10 \
       --seed "$sd" --num_clients "$K" --method "$m" --root_dir ../420 > "$log" 2>&1
  rc=$?
  f="results_${ds}_${sd}_${m}_fedavg.csv"
  rows=$(( $(wc -l < "$f" 2>/dev/null || echo 1) - 1 ))
  echo "[$(date -u +%H:%M:%S)] DONE  $ds K=$K seed=$sd $m rc=$rc rows=$rows ($(( ($(date +%s)-t0)/60 ))m)" >> fills.log
done < "$1"
echo "[$(date -u +%H:%M:%S)] LANE $1 COMPLETE" >> fills.log
