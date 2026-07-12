#!/bin/bash
# a40_celeba_iid_adv.sh — re-run celeba IID adv (res) contributions @ eps=0.01,
# mirroring the noniid celeba_adv but with dataset=celeba / partition=iid. Runs
# on GPU0 (free now the noniid celeba_adv is done) at LOW concurrency (MAX=2) so
# it doesn't blow the pid cap while the ST/DY sweep (CPU) + eval (GPU1) run.
# Clears only the IID CSVs (results_celeba_<digit>_*, NOT celebanoniid_*).
cd /root/ulrich
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/root/ulrich seed=42
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
LOG=/root/ulrich/celeba_iid_adv.log
mkdir -p logs
echo "=== CELEBA-IID ADV @eps=0.01 START $(date) ===" >> "$LOG"
rm -f results_celeba_[0-9]*_gtg_adv_pgd_fedavg.csv results_celeba_[0-9]*_l1o_adv_pgd_fedavg.csv
run(){  # seed method K
  echo "[start] s$1 $2 K$3 $(date +%T)" >> "$LOG"
  if CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python robustness.py \
       --root_dir=/root/ulrich/420 --dataset=celeba --partition=iid \
       --num_rounds=10 --seed=$1 --method=$2 --strategy=fedavg --num_clients=$3 \
       > "logs/celebaiidadv_${2}_${3}_${1}.log" 2>&1; then
    echo "[OK] s$1 $2 K$3 $(date +%T)" >> "$LOG"
  else
    echo "[FAIL] s$1 $2 K$3 $(date +%T)" >> "$LOG"
  fi
}
MAX=2; running=0
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
