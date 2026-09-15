"""Hold the system awake while fill jobs run. Process-scoped: releases on exit,
changes NO power settings. Self-terminating so it cannot outlive the work.

Exits when no robustness.py process has been seen for 3 consecutive checks,
or after 24h, whichever comes first.
"""
import ctypes, subprocess, sys, time

PATTERN = sys.argv[1] if len(sys.argv) > 1 else "robustness.py"  # command-line substring of the process(es) to stay awake for
ES_CONTINUOUS        = 0x80000000
ES_SYSTEM_REQUIRED   = 0x00000001
ES_DISPLAY_REQUIRED  = 0x00000002  # Modern Standby laptops enter standby when the display idles off; SYSTEM_REQUIRED alone did not stop it (2026-09-15 01:24)
kernel32 = ctypes.windll.kernel32

def jobs_running() -> int:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" |"
             " Where-Object { $_.CommandLine -like '*" + PATTERN + "*' }).Count"],
            capture_output=True, text=True, timeout=60)
        return int((out.stdout or "0").strip() or 0)
    except Exception:
        return -1          # unknown -> treat as "still running", never sleep early

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

if kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) == 0:
    log("SetThreadExecutionState FAILED - system may still sleep"); sys.exit(1)
log("system sleep AND display idle-off BLOCKED (Modern Standby needs both; power plan untouched); waiting on *" + PATTERN + "*")

deadline = time.time() + 24 * 3600
idle_checks = 0
try:
    while time.time() < deadline:
        time.sleep(60)
        n = jobs_running()
        idle_checks = idle_checks + 1 if n == 0 else 0
        if idle_checks == 1 or n > 0 and idle_checks == 0:
            pass
        if idle_checks >= 3:
            log("no fill jobs for 3 checks - releasing"); break
    else:
        log("24h cap reached - releasing")
finally:
    kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    log("sleep block RELEASED")
