#!/bin/bash
# Re-run the 3 imdb K4 dy priv cells that hit the transient round_10-save skip.
cd /root/ulrich
source .venv/bin/activate
export FLR_TF_INTRA=4 FLR_TF_INTER=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=-1
export BETA=0.2
for S in 123 2025 9928; do
  bash st_dy_guarded_run.sh imdbnoniid 4 dy priv "$S" "70$S" >> full_rerun_b0.2.log 2>&1 &
done
wait
echo "IMDBFILL_DONE $(date)"
