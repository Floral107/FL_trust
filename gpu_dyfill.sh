#!/bin/bash
# gpu_dyfill.sh — finish the remaining celeba K20 dy cells on the two free GPUs.
# Server-side GTG/PGD scoring runs on GPU (~10x faster than CPU); client actors
# stay on CPU (pyproject client num-gpus=0). dy_res_2025 already runs in tmux
# gputest on GPU0; this fills the other 4 across both GPUs, 1 cell per GPU.
#   GPU1 lane: dy_rel {42,107,9928}  (sequential)
#   GPU0 lane: wait for dy_res_2025 to finish, then dy_res_9928
cd /root/ulrich
source .venv/bin/activate
export FLR_TF_INTRA=4 FLR_TF_INTER=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export BETA=0.2
runcell(){ # gpu mode wm seed slot
  FLR_CUDA=$1 timeout 43200 bash st_dy_guarded_run.sh celebanoniid 20 "$2" "$3" "$4" "$5" >> full_rerun_b0.2.log 2>&1
  echo "[gpu_dyfill] done gpu$1 $2/$3/$4 $(date)" >> gpu_dyfill.log
}
echo "=== gpu_dyfill START $(date) ===" >> gpu_dyfill.log
# GPU1 lane: the three dy_rel cells
( for s in 42 107 9928; do runcell 1 dy rel "$s" "8810$s"; done ) &
# GPU0 lane: after dy_res_2025 (gputest) lands, run dy_res_9928 on GPU0
( while [ ! -f 420_dy_res/celebanoniid/20/2025/global_model_round_10.keras ]; do sleep 300; done
  runcell 0 dy res 9928 880928 ) &
wait
echo "=== gpu_dyfill DONE $(date) ===" >> gpu_dyfill.log
