#!/bin/bash
# celeba_iid_adv_2gpu.sh — finish celeba-IID adv K=20 across BOTH GPUs.
# robustness.py is resume-safe (skips already-written (seed,num_clients,round)
# rows), so the in-progress seeds (gtg 42@~100 rows, gtg 107@~80) just continue.
# Each robustness.py proc is light (~16 threads), so this adds ~nothing to the
# pid ceiling. MAX=1 per GPU = each cell gets a dedicated GPU (fastest per cell).
#
# Partition of the remaining K=20 cells (l1o seed42 already done):
#   GPU0: gtg {42(resume), 123, 9928}
#   GPU1: gtg {107(resume), 2025}  then fast l1o {107,123,2025,9928}
cd /root/ulrich
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/root/ulrich
PY=/root/ulrich/.venv/bin/python
run(){ # gpu seed method
  CUDA_VISIBLE_DEVICES=$1 TF_FORCE_GPU_ALLOW_GROWTH=true seed=$2 "$PY" robustness.py \
    --root_dir=/root/ulrich/420 --dataset=celeba --partition=iid \
    --num_rounds=10 --seed=$2 --method=$3 --strategy=fedavg --num_clients=20 \
    >> "celeba_adv_2gpu_gpu$1.out" 2>&1
  echo "[done] gpu$1 seed$2 $3 $(date)" >> celeba_adv_2gpu.log
}
echo "=== ADV 2GPU START $(date) ===" >> celeba_adv_2gpu.log
( run 0 42 gtg_adv_pgd; run 0 123 gtg_adv_pgd; run 0 9928 gtg_adv_pgd ) &
( run 1 107 gtg_adv_pgd; run 1 2025 gtg_adv_pgd; \
  for s in 107 123 2025 9928; do run 1 "$s" l1o_adv_pgd; done ) &
wait
echo "=== ADV 2GPU DONE $(date) ===" >> celeba_adv_2gpu.log
