#!/usr/bin/env bash
# usage: run_fills.sh <jobfile>   ; job line: <dataset> <partition> <K> <seed> <method> [sensitive_attr]
# Leave sensitive_attr out for the paper's attribute. Output lands next to this script,
# or in OUT_DIR (relative to it, e.g. OUT_DIR=../_direction_b) when set.
# robustness.py is resume-safe (skips done (seed,K,round)), so re-running a lane is free.
set -u
JOBS="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"   # resolve before changing directory
cd "$(dirname "$0")" && cd "${OUT_DIR:-.}" || exit 1
PY=../.venv/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=          # GPU eval throws XLA after PGD and writes 0 rows
export OMP_NUM_THREADS=4              # 3 lanes x 4 threads = 12 cores
export TF_CPP_MIN_LOG_LEVEL=3
while read -r ds part K sd m attr; do
  [ -z "${ds:-}" ] && continue
  case "$ds" in \#*) continue;; esac
  tag=${attr:+_s-${attr//[^A-Za-z0-9]/-}}   # the same tag robustness.py adds to the output name
  log="fill_${ds}_${K}_${sd}_${m}${tag}.log"
  echo "[$(date -u +%H:%M:%S)] START $ds K=$K seed=$sd $m${attr:+ attr=$attr}" >> fills.log
  t0=$(date +%s)
  seed=$sd $PY ../Code/robustness.py --dataset "$ds" --partition "$part" --num_rounds 10 \
       --seed "$sd" --num_clients "$K" --method "$m" --root_dir ../420 ${attr:+--sensitive_attr "$attr"} > "$log" 2>&1
  rc=$?
  f="results_${ds}_${sd}_${m}${tag}_fedavg.csv"
  rows=$(( $(wc -l < "$f" 2>/dev/null || echo 1) - 1 ))
  echo "[$(date -u +%H:%M:%S)] DONE  $ds K=$K seed=$sd $m${attr:+ attr=$attr} rc=$rc rows=$rows ($(( ($(date +%s)-t0)/60 ))m)" >> fills.log
done < "$JOBS"
echo "[$(date -u +%H:%M:%S)] LANE $1 COMPLETE" >> fills.log
