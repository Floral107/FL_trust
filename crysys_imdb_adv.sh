#!/bin/bash
# crysys_imdb_adv.sh — re-run imdb (NIID) adv res CONTRIBUTIONS with the FIXED
# attack code. gtg_adv_pgd + l1o_adv_pgd, K{4,20}, 5 seeds = 20 jobs. Clears old
# (buggy/skip-prone) adv CSVs first. CPU, sliding window MAX=6 (RAM-safe on 24GB).
cd /home/crysys/ulrich
source .venv/bin/activate
export PYTHONPATH=/home/crysys/ulrich seed=42 CUDA_VISIBLE_DEVICES=""
LOG=/home/crysys/ulrich/imdb_adv_cry.log
mkdir -p logs
echo "=== IMDB ADV (NIID) fixed-attack START $(date) ===" >> "$LOG"
rm -f results_imdbnoniid_*_gtg_adv_pgd_fedavg.csv results_imdbnoniid_*_l1o_adv_pgd_fedavg.csv
python -c "import sys;sys.path.insert(0,'/home/crysys/ulrich');from imdb.imdb.task import process_text;process_text(['warm'])" >> "$LOG" 2>&1
echo "[USE warmed] $(date +%T)" >> "$LOG"
run(){
  echo "[start] s$1 $2 K$3 $(date +%T)" >> "$LOG"
  if python robustness.py --root_dir=/home/crysys/ulrich/420 --dataset=imdbnoniid \
       --partition=noniid --num_rounds=10 --seed=$1 --method=$2 --strategy=fedavg \
       --num_clients=$3 > "logs/imdbadv_${2}_${3}_${1}.log" 2>&1; then
    echo "[OK] s$1 $2 K$3 $(date +%T)" >> "$LOG"
  else
    echo "[FAIL] s$1 $2 K$3 $(date +%T)" >> "$LOG"
  fi
}
MAX=6; running=0
for K in 4 20; do
  for s in 42 107 123 2025 9928; do
    for m in gtg_adv_pgd l1o_adv_pgd; do
      run "$s" "$m" "$K" &
      running=$(( running + 1 ))
      if (( running >= MAX )); then wait -n; running=$(( running - 1 )); fi
    done
  done
done
wait
echo "=== ALL DONE $(date) ===" >> "$LOG"
