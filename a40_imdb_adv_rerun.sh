#!/bin/bash
# a40_imdb_adv_rerun.sh — re-run imdb adv (res) contributions with the FIXED
# attack_metric.py (batch_size bug). gtg_adv_pgd + l1o_adv_pgd x 5 seeds x K{4,20}
# on the BL models (420/imdbnoniid). CPU (GPUs left for celeba calibration).
cd /root/ulrich
source .venv/bin/activate
export PYTHONPATH=/root/ulrich seed=42 CUDA_VISIBLE_DEVICES=""
mkdir -p logs
LOG=/root/ulrich/imdb_adv_rerun.log
run(){
  echo "[start] s$1 $2 K$3 $(date +%T)" >> "$LOG"
  if python robustness.py --root_dir=/root/ulrich/420 --dataset=imdbnoniid \
       --partition=noniid --num_rounds=10 --seed="$1" --method="$2" \
       --strategy=fedavg --num_clients="$3" > "logs/imdbadv_${2}_${3}_${1}.log" 2>&1; then
    echo "[OK] s$1 $2 K$3 $(date +%T)" >> "$LOG"
  else
    echo "[FAIL] s$1 $2 K$3 $(date +%T)" >> "$LOG"
  fi
}
echo "=== IMDB ADV RERUN (fixed attack) $(date) ===" >> "$LOG"
python -c "import sys;sys.path.insert(0,'/root/ulrich');from imdb.imdb.task import process_text;process_text(['warm'])" >> "$LOG" 2>&1
echo "[USE warmed] $(date +%T)" >> "$LOG"
for K in 4 20; do
  for s in 42 107 123 2025 9928; do
    for m in gtg_adv_pgd l1o_adv_pgd; do
      run "$s" "$m" "$K" &
    done
  done
done
wait
echo "=== ALL DONE $(date) ===" >> "$LOG"
