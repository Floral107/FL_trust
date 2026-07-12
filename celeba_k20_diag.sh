#!/bin/bash
# Diagnostic: one cell of each expensive type (seed 42) that stalls at ROUND 2.
# Long timeout (8h) + concurrent, so we can observe whether round 2 EVER completes.
# If they advance to ROUND 3+ within ~2h -> just slow, not deadlocked -> batch the rest.
# If still at ROUND 2 after ~2h with no log growth -> genuine deadlock -> different fix.
cd /root/ulrich
source .venv/bin/activate
export FLR_TF_INTRA=4 FLR_TF_INTER=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=-1
export BETA=0.2
echo "[k20diag] launching 4 expensive seed-42 cells $(date)"
timeout 28800 bash st_dy_guarded_run.sh celebanoniid 20 st res  42 82001 >> full_rerun_b0.2.log 2>&1 &
timeout 28800 bash st_dy_guarded_run.sh celebanoniid 20 dy res  42 82002 >> full_rerun_b0.2.log 2>&1 &
timeout 28800 bash st_dy_guarded_run.sh celebanoniid 20 dy rel  42 82003 >> full_rerun_b0.2.log 2>&1 &
timeout 28800 bash st_dy_guarded_run.sh celebanoniid 20 dy priv 42 82004 >> full_rerun_b0.2.log 2>&1 &
wait
echo "K20DIAG_DONE $(date)"
