#!/bin/bash
# watchdog.sh — productivity check, called every ~30 min by the remote monitor.
# Verdict lines are machine-greppable:
#   PRODUCING  job's per-job log written recently (real progress)
#   STARTING   process younger than 15 min (grace period)
#   WEDGED     process >15 min old AND its log stale >30 min  -> kill it; the
#              lane logs FAIL and moves on (jobs are resume-aware)
#   NOGROWTH   total result rows flat since last check while jobs are running
#   FAILDELTA  new FAILs since last check
# Also reports: lane tmux sessions, totals, disk, GPU, pids.
cd "$HOME/ulrich" || exit 1
STATE="/tmp/watchdog_state"; NEW="/tmp/watchdog_state.new"; : > "$NEW"
now=$(date +%s)
echo "=== WATCHDOG $(hostname) $(date +'%F %T') ==="

ps -eo pid=,etimes=,pcpu=,args= | grep '[r]obustness.py' > /tmp/wd_procs || true
njobs=0
while read -r pid et cpu args; do
  njobs=$((njobs+1))
  seed=$(printf '%s' "$args" | sed -n 's/.*--seed=\([^ ]*\).*/\1/p')
  meth=$(printf '%s' "$args" | sed -n 's/.*--method=\([^ ]*\).*/\1/p')
  nc=$(printf '%s' "$args" | sed -n 's/.*--num_clients=\([^ ]*\).*/\1/p')
  log=$(ls -t logs/*_"${meth}"_"${nc}"_"${seed}".log 2>/dev/null | head -1)
  age=-1
  [ -n "$log" ] && age=$(( now - $(stat -c %Y "$log") ))
  if [ "$age" -ge 0 ] && [ "$age" -lt 1800 ]; then
    echo "PRODUCING pid=$pid $meth nc=$nc seed=$seed cpu=${cpu}%% log_age=${age}s"
  elif [ "$et" -lt 900 ]; then
    echo "STARTING pid=$pid $meth nc=$nc seed=$seed et=${et}s"
  else
    echo "WEDGED pid=$pid $meth nc=$nc seed=$seed et=${et}s cpu=${cpu}%% log_age=${age}s"
  fi
done < /tmp/wd_procs
echo "running_jobs=$njobs"

tmux ls 2>/dev/null | cut -d: -f1 | grep -E 'lane|cry_' | sed 's/^/session_alive=/'

tot=0
for f in results_*_fedavg.csv; do
  [ -f "$f" ] || continue
  tot=$(( tot + $(wc -l < "$f") ))
done
old=$(grep '^tot_rows=' "$STATE" 2>/dev/null | cut -d= -f2)
echo "tot_rows=$tot (prev=${old:-n/a})"
echo "tot_rows=$tot" >> "$NEW"
if [ -n "$old" ] && [ "$tot" -le "$old" ] && [ "$njobs" -gt 0 ]; then
  echo "NOGROWTH result rows flat since last check"
fi

okc=$(grep -h ' OK ' *lane_*.log 2>/dev/null | wc -l)
flc=$(grep -h ' FAIL ' *lane_*.log 2>/dev/null | wc -l)
oldf=$(grep '^fails=' "$STATE" 2>/dev/null | cut -d= -f2)
echo "jobs_ok=$okc fails=$flc (prev_fails=${oldf:-0})"
echo "fails=$flc" >> "$NEW"
[ -n "$oldf" ] && [ "$flc" -gt "$oldf" ] && echo "FAILDELTA $((flc-oldf)) new"

df -h "$HOME" 2>/dev/null | tail -1 | awk '{print "disk_free="$4}'
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
[ -f /sys/fs/cgroup/pids.current ] && echo "pids=$(cat /sys/fs/cgroup/pids.current)"
mv "$NEW" "$STATE"
