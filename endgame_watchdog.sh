#!/bin/bash
# endgame_watchdog.sh — self-healing watchdog that drives the celeba K20 endgame
# to completion WITHOUT supervision. Every 10-min tick it:
#   * relaunches any required ST/DY cell that is missing AND not currently running
#     (thread-guarded so it never duplicates a running cell or breaches the pid cap)
#   * keeps the celeba eval alive (idempotent: skips already-scored cells)
#   * restarts the dual-GPU adv lane if it died before finishing
# When every training cell is on disk -> runs the FINAL all-dataset CPU eval once
# (CUDA_VISIBLE_DEVICES='' to dodge the K20-res XLA error). Exits when training +
# final eval + adv are all complete. All actions are logged to watchdog.log.
set -uo pipefail
cd /root/ulrich
source .venv/bin/activate 2>/dev/null || true
export FLR_TF_INTRA=4 FLR_TF_INTER=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=-1 BETA=0.2
LOG=/root/ulrich/watchdog.log
SEEDS="42 107 123 2025 9928"
TARGETS="st:res dy:rel dy:res"      # the only cells still in flight for Tab.4
THREAD_GUARD=28000

log(){ echo "[$(date +'%F %T')] $*" >> "$LOG"; }
done_cell(){ [ -f "420_$1_$2/celebanoniid/20/$3/global_model_round_10.keras" ]; }   # mode wm seed
run_cell(){   # mode wm seed -> 0 if a live flwr proc has that seed env
  local p s
  for p in $(pgrep -f "weight-mode='$1' weight-metric='$2'" 2>/dev/null); do
    s=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^seed=//p')
    [ "$s" = "$3" ] && return 0
  done
  return 1
}

FINAL_EVAL_DONE=0
log "=== endgame watchdog START (targets: $TARGETS) ==="
while true; do
  THREADS=$(cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0)
  pending=0; i=0
  for t in $TARGETS; do
    mode=${t%:*}; wm=${t#*:}
    for s in $SEEDS; do
      slot=$((86000+i)); i=$((i+1))
      done_cell "$mode" "$wm" "$s" && continue
      pending=$((pending+1))
      run_cell "$mode" "$wm" "$s" && continue
      if [ "$THREADS" -gt "$THREAD_GUARD" ]; then
        log "threads=$THREADS > $THREAD_GUARD : defer relaunch $mode/$wm/$s"; continue
      fi
      log "RELAUNCH $mode/$wm/$s (missing, not running) slot=$slot"
      timeout 86400 bash st_dy_guarded_run.sh celebanoniid 20 "$mode" "$wm" "$s" "$slot" >> full_rerun_b0.2.log 2>&1 &
      THREADS=$((THREADS+2500))
    done
  done

  # keep celeba eval scoring finished cells while training continues
  if [ "$pending" -gt 0 ] && ! tmux has-session -t evalstdy 2>/dev/null; then
    log "celeba eval not running -> relaunch"
    tmux new-session -d -s evalstdy "cd /root/ulrich && source .venv/bin/activate; OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python eval_stdy_models.py --datasets celebanoniid >> eval_stdy.out 2>&1"
  fi

  # restart adv lane if it died before completing
  if ! tmux has-session -t adv2gpu 2>/dev/null && ! grep -q 'ADV 2GPU DONE' celeba_adv_2gpu.log 2>/dev/null; then
    log "adv2gpu died before done -> restart"
    tmux new-session -d -s adv2gpu 'cd /root/ulrich && bash celeba_iid_adv_2gpu.sh'
  fi

  log "tick: training_pending=$pending threads=$(cat /sys/fs/cgroup/pids.current 2>/dev/null) final_eval_done=$FINAL_EVAL_DONE"

  # once all training cells exist, run the final all-dataset eval exactly once
  if [ "$pending" -eq 0 ] && [ "$FINAL_EVAL_DONE" -eq 0 ]; then
    log "ALL TRAINING CELLS DONE -> running FINAL all-dataset CPU eval"
    tmux kill-session -t evalstdy 2>/dev/null
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python eval_stdy_models.py --datasets adultnoniid celebanoniid imdbnoniid >> eval_stdy.out 2>&1
    FINAL_EVAL_DONE=1
    log "FINAL eval complete"
  fi

  # exit when training + final eval + adv are all done
  if [ "$pending" -eq 0 ] && [ "$FINAL_EVAL_DONE" -eq 1 ] && grep -q 'ADV 2GPU DONE' celeba_adv_2gpu.log 2>/dev/null; then
    log "=== endgame watchdog DONE: training + eval + adv all complete ==="
    break
  fi

  sleep 600
done
