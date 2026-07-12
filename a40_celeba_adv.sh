#!/bin/bash
# a40_celeba_adv.sh — re-run celeba adv (res) CONTRIBUTIONS at the new eps=0.01
# (PGD_EVAL_PARAMS already updated). gtg_adv_pgd + l1o_adv_pgd, celebanoniid,
# K{4,20}, 5 seeds = 20 jobs, distributed across BOTH GPUs with a sliding
# concurrency window. Clears old eps=0.03 CSVs first (robustness.py is
# resume-aware and would otherwise skip them).
cd /root/ulrich
source .venv/bin/activate
export PYTHONPATH=/root/ulrich seed=42
LOG=/root/ulrich/celeba_adv.log
mkdir -p logs
echo "=== CELEBA ADV @ eps=0.01 START $(date) ===" >> "$LOG"
# clear stale adv outputs so robustness.py recomputes at eps=0.01
rm -f results_celebanoniid_*_gtg_adv_pgd_fedavg.csv results_celebanoniid_*_l1o_adv_pgd_fedavg.csv
run(){  # seed method K gpu
  echo "[start] s$1 $2 K$3 gpu$4 $(date +%T)" >> "$LOG"
  if CUDA_VISIBLE_DEVICES=$4 TF_FORCE_GPU_ALLOW_GROWTH=true python robustness.py \
       --root_dir=/root/ulrich/420 --dataset=celebanoniid --partition=noniid \
       --num_rounds=10 --seed=$1 --method=$2 --strategy=fedavg --num_clients=$3 \
       > "logs/celebaadv_${2}_${3}_${1}.log" 2>&1; then
    echo "[OK] s$1 $2 K$3 $(date +%T)" >> "$LOG"
  else
    echo "[FAIL] s$1 $2 K$3 $(date +%T)" >> "$LOG"
  fi
}
MAX=6   # ~3 jobs/GPU concurrent (celeba CNN is small; leaves headroom)
i=0; running=0
for K in 4 20; do
  for s in 42 107 123 2025 9928; do
    for m in gtg_adv_pgd l1o_adv_pgd; do
      gpu=$(( i % 2 )); run "$s" "$m" "$K" "$gpu" &
      i=$(( i + 1 )); running=$(( running + 1 ))
      if (( running >= MAX )); then wait -n; running=$(( running - 1 )); fi
    done
  done
done
wait
echo "=== ALL DONE $(date) ===" >> "$LOG"
