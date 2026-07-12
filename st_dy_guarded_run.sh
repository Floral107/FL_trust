#!/bin/bash
# st_dy_guarded_run.sh — SAFE single ST/DY training with the same guards as
# et_guarded_run.sh (mutex + validation + orphan reaper). Isolation via the
# mutex prevents the client-dropout collapse that corrupted the chaotic ET runs.
#
# Usage: st_dy_guarded_run.sh <dataset> <K> <wmode> <wmetric> <seed> [slot]
#   wmode   : st | dy
#   wmetric : acc loss fairDP fairEO rel res priv
# Output: 420_<wmode>_<wmetric>/<dataset>/<K>/<seed>/global_model_round_10.keras
set -uo pipefail
# Ray actors are socket-heavy; the default 1024 fd limit starved them under
# concurrency (mass "0 results/N failures" actor deaths). Raise the soft limit.
ulimit -n 8192 2>/dev/null || ulimit -n 4096 2>/dev/null || true
cd "$HOME/ulrich"
# Self-source the venv so `flwr` is on PATH regardless of how we're launched
# (a bare `bash sweep.sh` without prior activation otherwise gives rc=127).
source .venv/bin/activate 2>/dev/null || true
export FLR_TF_INTRA="${FLR_TF_INTRA:-4}" FLR_TF_INTER="${FLR_TF_INTER:-2}"
# Cap thread-pool libs. On high-core hosts (e.g. the 255-core A40) OpenMP/BLAS/
# NumExpr each spawn ~1 thread per core PER PROCESS (ignoring the TF cap), so a
# single sim balloons to ~9k threads (= cgroup pids). Capping these keeps each
# sim to ~2-3k pids, letting many sims run under the pid ceiling. Harmless on
# small hosts (CrySyS) too. Ray workers inherit these from the driver env.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
# FL training stays on CPU. It's orchestration/Shapley-bound (GPU was only ~15-20%
# on the A100), and ~20 client actors/sim would thrash a shared GPU. Eval (separate)
# uses the GPU. Harmless on CPU-only hosts (CrySyS).
export CUDA_VISIBLE_DEVICES="${FLR_CUDA:--1}"

DS=$1 K=$2 WMODE=$3 WMET=$4 SEED=$5
SLOT=${6:-0}
# Shrinkage strength for the ST/DY weight transform. Injected by the sweep
# orchestrator (full_stdy_rerun.sh). Default 1.0 = legacy raw transform.
BETA="${BETA:-1.0}"
# Hard weight cap for the ST/DY transform (>=1, or 0 to disable). Injected by an
# A/B orchestrator. Default 0.0 = disabled (legacy behaviour unchanged).
CAP="${CAP:-0.0}"
LOCK=/tmp/flwr_sim_slot${SLOT}.lock
# server_app lowercases weight_metric before saving (e.g. fairDP -> fairdp), so
# the output dir is ALWAYS lowercase. Build DIR with the lowercased metric or the
# guard checks the wrong path for mixed-case metrics (fairDP/fairEO) and reports a
# false FAIL on every run — never skipping done work, never seeing the real ckpt.
WMET_LC=$(printf '%s' "$WMET" | tr '[:upper:]' '[:lower:]')
# OUTPRE redirects the checkpoint tree (default 420 = canonical). Used by
# transform A/Bs (e.g. beta0.2+cap2.0 sandbox) so they never overwrite the
# canonical models.
OUTPRE="${OUTPRE:-420}"
# task.py's _resolve_save_dir honors FLR_SAVE_PREFIX — keep it in lockstep with
# OUTPRE so the flwr-side saves land where this script validates/prunes.
export FLR_SAVE_PREFIX="$OUTPRE"
DIR="${OUTPRE}_${WMODE}_${WMET_LC}/${DS}/${K}/${SEED}"
OUT="$DIR/global_model_round_10.keras"
R0="$DIR/global_model_round_0.keras"

log() { echo "[$(date +'%F %H:%M:%S')] [stdy $DS/$K/$WMODE/$WMET/$SEED b$BETA] $*"; }

reap_orphans() {
    for pid in $(ps -ef | grep 'ray::ClientAppActor' | grep -v grep | awk '{print $2}'); do
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ -z "$ppid" ] || ! ps -p "$ppid" >/dev/null 2>&1; then
            kill -9 "$pid" 2>/dev/null && log "reaped orphan actor $pid"
        fi
    done
}

is_valid() {
    # Validate the GLOBAL round-10 model only (that's what Tab.4 evaluates).
    # Requiring all K client ckpts was too strict: K=20 runs that don't save
    # every client (sampling, transient failures) got falsely purged even when
    # the global is fine. Init-fake detection = round_10 differs from round_0.
    # Dropout-collapse (severely degraded global) is caught later at EVAL via
    # accuracy, not here.
    [ -f "$OUT" ] || return 1
    [ -f "$R0" ] || return 1
    local s10 s0
    s10=$(stat -c %s "$OUT" 2>/dev/null); s0=$(stat -c %s "$R0" 2>/dev/null)
    [ "$s10" = "$s0" ] && return 1
    return 0
}

reap_orphans
if is_valid; then log "SKIP — valid ckpt present"; exit 0; fi

exec 9>"$LOCK"
if ! flock -w 21600 9; then log "FAIL — lock timeout"; exit 2; fi
log "lock(slot $SLOT) acquired, training"

rm -rf "$DIR" 2>/dev/null
# Ensure num-supernodes matches K. SKIP_SETNC=1 skips this — used when the
# caller has already set num-supernodes once for a whole K-phase, so concurrent
# cells never touch the shared pyproject (avoids the set_num_cl write race that
# broke CrySyS's imdb run: "pyproject.toml does not exist" mid-write).
if [ -z "${SKIP_SETNC:-}" ]; then
  python3 set_num_cl.py --data="$DS" --clients="$K" >> logs/stdy_setnc.log 2>&1 || true
fi

cd "$DS"
export seed="$SEED"
RUNLOG="logs/stdy_${DS}_K${K}_${WMODE}_${WMET}_${SEED}_b${BETA}.log"
flwr run . --run-config "num-server-rounds=10 weight-mode='$WMODE' weight-metric='$WMET' weight-beta=$BETA weight-cap=$CAP et-mode='none' strategy='fedavg'" \
    > "../$RUNLOG" 2>&1
rc=$?
cd ..
flock -u 9

reap_orphans
# PARTICIPATION GUARD: flwr's FedAvg default accept_failures=True silently
# aggregates partial results — a round with "1 results and 19 failures" yields a
# single-client global model that LOOKS trained (passes is_valid) but is garbage
# (this poisoned the acc/loss cells of the first cap sweep: Ray OOM-killed client
# actors, cells passed, table collapsed to base rate). But rejecting ANY single
# failure is too strict — a transient one-actor hiccup at GPU startup is benign
# (FedAvg over 3/4 or 19/20 clients is fine). Reject only CATASTROPHIC dropout:
# a round that aggregated 0 clients, or fewer than HALF of that round's clients.
bad=$(grep -hoE "received [0-9]+ results and [0-9]+ failures" "$RUNLOG" \
      | awk '{r=$2; f=$5; if (r==0 || r*2 < (r+f)) c++} END{print c+0}')
if [ "${bad:-0}" -gt 0 ]; then
    worst=$(grep -hoE "received [0-9]+ results and [0-9]+ failures" "$RUNLOG" \
            | awk '{r=$2; f=$5; if (r==0 || r*2 < (r+f)) print $0}' | head -1)
    log "FAIL — catastrophic dropout in $bad round(s) [$worst], purging"
    rm -rf "$DIR" 2>/dev/null
    exit 1
fi
if is_valid; then
    # Auto-prune intermediate rounds 1-9 (Tab.4 ST/DY eval uses round_10 only;
    # round_0 kept for the init-fake guard check). Caps disk growth at ~2
    # rounds/training instead of 11 — prevents the disk-full crash that killed
    # CrySyS. Keeps global+client round_0 and round_10.
    rm -f "$DIR"/global_model_round_[1-9].keras "$DIR"/client_*_round_[1-9].keras 2>/dev/null
    log "OK — valid ckpt (rc=$rc); pruned intermediate rounds 1-9"
    exit 0
else
    log "FAIL — no valid ckpt (rc=$rc), purging"
    rm -rf "$DIR" 2>/dev/null
    exit 1
fi
