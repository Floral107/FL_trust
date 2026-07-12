#!/usr/bin/env bash
# FLRobustness server health check. Emits PASS/WARN/FAIL signals per section.
cd /root/ulrich 2>/dev/null || { echo "FAIL: cannot cd /root/ulrich"; exit 1; }
now=$(date "+%Y-%m-%d %H:%M:%S")
echo "===== FLRobustness health @ $now ====="

# ---------- 1. CPU / GPU ----------
echo "## CPU_GPU"
read -r one five fifteen rest < /proc/loadavg
cores=$(nproc)
echo "  load: ${one} (1m) / ${five} (5m) over ${cores} cores"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
             --format=csv,noheader,nounits |
  while IFS=, read -r idx util mused mtot temp pw; do
    util=$(echo "$util" | tr -d ' '); mused=$(echo "$mused" | tr -d ' '); mtot=$(echo "$mtot" | tr -d ' ')
    flag="ok"
    if [ "$mused" -gt 1000 ] && [ "$util" -lt 5 ]; then flag="WARN_busy_mem_idle_compute"; fi
    echo "  GPU${idx}: util=${util}% mem=${mused}/${mtot}MiB temp=${temp}C pw=${pw}W [$flag]"
  done
else
  echo "  WARN: no nvidia-smi"
fi
ntrain=$(pgrep -fc "flwr|flower|robustness.py|run_experiments|cont_evals" 2>/dev/null)
[ -z "$ntrain" ] && ntrain=0
echo "  training_procs: ${ntrain}"
busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
if [ "${ntrain:-0}" -gt 0 ] && [ "${busy:-0}" -lt 5 ]; then
  echo "  WARN: training process alive but all GPUs ~0% util (possible stall)"
fi

# ---------- 2. GENERAL HEALTH ----------
echo "## HEALTH"
df -h /root/ulrich | tail -1 | awk '{print "  disk: "$4" free ("$5" used)"}'
free -g | awk 'NR==2{print "  ram: "$7" GB avail of "$2" GB"}'
# Exclude this script's own output log so the health check never flags itself.
LOGS=$(ls -t *.log 2>/dev/null | grep -v '^health_history.log$')
LOG=$(echo "$LOGS" | head -1)
if [ -n "$LOG" ]; then
  echo "  newest_log: $LOG"
  tail -4 "$LOG" | sed 's/^/    /'
fi
echo "  logs_with_errors:"
hits=$(grep -ilE "FAIL|error|exception|traceback|OOM|out of memory" $LOGS 2>/dev/null | head -8)
if [ -n "$hits" ]; then
  for f in $hits; do
    n=$(grep -icE "FAIL|error|exception|traceback|OOM|out of memory" "$f")
    echo "    $f ($n hits)"
  done
else
  echo "    none"
fi

# ---------- 3. CORRECT DATA ----------
echo "## DATA"
newest_csv=$(ls -t data/*/*.csv 2>/dev/null | head -1)
if [ -n "$newest_csv" ]; then
  age_h=$(( ( $(date +%s) - $(stat -c %Y "$newest_csv") ) / 3600 ))
  rows=$(( $(wc -l < "$newest_csv") - 1 ))
  echo "  newest_csv: $newest_csv (${age_h}h old, ${rows} rows)"
  echo "  header: $(head -1 "$newest_csv")"
  bad=$(grep -icE "nan|inf" "$newest_csv")
  echo "  nan/inf_rows: ${bad}"
else
  echo "  WARN: no CSVs found under data/"
fi
tiny=$(find 420 -name "*.keras" -size -100k 2>/dev/null | wc -l)
newest_ck=$(ls -t 420/*/*/*/*.keras 2>/dev/null | head -1)
echo "  newest_ckpt: ${newest_ck}"
echo "  truncated_ckpts(<100KB): ${tiny}"
echo "===== end ====="
