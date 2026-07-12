#!/bin/bash
# imdb_rerun.sh — the ONE IMDB rerun (advisor comments 1, 2, 4-IMDB, 5).
# Self-sequencing, resume-safe (sentinels in imdb_rerun_state/), relation-gated:
# a failed gate writes a FLAG file and stops downstream phases instead of
# training garbage.
#
#  0  archive old-LR imdb models (mv, not rm) + prewarm embedding caches
#  1  pilot: BL imdbnoniid K20 seed42 @LR1e-3 -> GATE1 acc>0.70 & CE<0.65
#  2  BL fleet: imdb + imdbnoniid x K{4,20} x 5 seeds (resume via round-10 ckpt)
#     + eps_calibrate.py on the pilot model (comment 5) -> imdb_eps.txt
#  3  WAIT for EPS_APPLIED sentinel (human checkpoint: review curve, update
#     attack_metric.py + score_metrics.py + pyproject et-adv params, push, touch)
#  4  ST/DY: cap_sweep.sh imdbnoniid (beta=1.0 cap=2.0) — non-res immediately,
#     res cells after EPS_APPLIED
#  5  ET: dp immediately, adv after EPS_APPLIED (ET_ADV_EPS from imdb_eps.txt)
#  6  post-hoc scoring: robustness.py, CPU, all methods x both partitions
#  7  evals -> data/{bl,st,dy,et}; merges/tables happen locally per PIPELINE.md
set -uo pipefail
cd "$HOME/ulrich"
source .venv/bin/activate 2>/dev/null || true
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
export FLR_TF_INTRA=2 FLR_TF_INTER=1 OMP_NUM_THREADS=2
ST=imdb_rerun_state; mkdir -p "$ST" logs
LOG="$HOME/ulrich/imdb_rerun.log"
SEEDS="42 107 123 2025 9928"
log(){ echo "[$(date +'%F %T')] $*" | tee -a "$LOG"; }
flag(){ log "FLAG: $*"; touch "$ST/FLAG_$1"; }

bl_done(){ [ -f "420/$1/$2/$3/global_model_round_10.keras" ]; }  # ds K seed

run_bl(){  # ds K seed — one BL cell (et/weight modes default to none in pyproject)
    local DS=$1 K=$2 S=$3
    bl_done "$DS" "$K" "$S" && return 0
    export seed=$S GLOBAL_SEED=$S PYTHONHASHSEED=$S
    ( cd "$DS" && flwr run . --run-config "strategy='fedavg'" ) \
        >> "logs/bl_${DS}_K${K}_s${S}.log" 2>&1
    bl_done "$DS" "$K" "$S"
}

bl_fleet_for(){  # ds — K-phased (set_num_cl once per K), seeds sequential
    local DS=$1
    for K in 20 4; do
        python3 set_num_cl.py --data="$DS" --clients="$K" >>"$LOG" 2>&1
        for S in $SEEDS; do
            bl_done "$DS" "$K" "$S" && continue
            log "[BL] $DS K=$K seed=$S"
            run_bl "$DS" "$K" "$S" || flag "BL_${DS}_K${K}_s${S}" "BL cell failed"
        done
    done
}

# ---- phase 0: archive old-LR artifacts + prewarm ----------------------------
if [ ! -f "$ST/p0" ]; then
    log "phase 0: archive old-LR imdb models + prewarm"
    mkdir -p _old_lr_imdb
    for d in 420/imdb 420/imdbnoniid 420_st_*/imdbnoniid 420_dy_*/imdbnoniid \
             420_et_adv/imdbnoniid 420_et_dp/imdbnoniid; do
        [ -d "$d" ] && mkdir -p "_old_lr_imdb/$(dirname "$d")" && mv "$d" "_old_lr_imdb/$d" \
            && log "  archived $d"
    done
    for K in 1 4 20; do python3 prewarm_imdb.py $K >>"$LOG" 2>&1 || true; done
    # iid partitions embed lazily on first cell; USE model + HF dataset now cached
    touch "$ST/p0"
fi

# ---- phase 1: pilot + gate ---------------------------------------------------
if [ ! -f "$ST/p1" ]; then
    log "phase 1: pilot BL imdbnoniid K20 seed42 @1e-3"
    python3 set_num_cl.py --data=imdbnoniid --clients=20 >>"$LOG" 2>&1
    run_bl imdbnoniid 20 42 || { flag PILOT "pilot training failed"; exit 1; }
    GATE=$(CUDA_VISIBLE_DEVICES= seed=42 python3 - <<'PY'
import os, numpy as np, tensorflow as tf
from imdbnoniid.imdbnoniid.task import load_data, process_text
_,_,xt,yt = load_data(0,1)
m = tf.keras.models.load_model("420/imdbnoniid/20/42/global_model_round_10.keras")
m.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
loss, acc = m.evaluate(process_text(xt), np.asarray(yt).astype("int32"), verbose=0)
print(f"GATE1 acc={acc:.4f} ce={loss:.4f}")
print("PASS" if (acc > 0.70 and loss < 0.65) else "FAIL")
PY
)
    log "$GATE"
    echo "$GATE" | grep -q PASS || { flag GATE1 "pilot did not converge — rethink LR before fleet"; exit 1; }
    touch "$ST/p1"
fi

# ---- phase 2: BL fleet + eps calibration (parallel lanes) --------------------
if [ ! -f "$ST/p2" ]; then
    log "phase 2: BL fleet (2 app-dir lanes) + eps calibration"
    bl_fleet_for imdbnoniid &  P_N=$!
    bl_fleet_for imdb       &  P_I=$!
    if [ ! -f imdb_eps.txt ]; then
        CUDA_VISIBLE_DEVICES= seed=42 python3 eps_calibrate.py >>"$LOG" 2>&1 \
            || flag EPS "no eps in target band — see imdb_eps_curve.txt"
    fi
    wait $P_N $P_I
    ls "$ST"/FLAG_BL_* >/dev/null 2>&1 && { log "BL fleet has failures — stopping"; exit 1; }
    touch "$ST/p2"
fi

# ---- phase 4a: ST/DY non-res (no eps dependency) ------------------------------
if [ ! -f "$ST/p4a" ]; then
    log "phase 4a: ST/DY imdbnoniid non-res (cap_sweep beta=1.0 cap=2.0)"
    WMETS="acc loss rel priv" bash cap_sweep.sh imdbnoniid 2.0 2 >>"$LOG" 2>&1
    n=$(find 420_st_acc 420_st_loss 420_st_rel 420_st_priv 420_dy_acc 420_dy_loss 420_dy_rel 420_dy_priv \
        -path "*/imdbnoniid/*" -name global_model_round_10.keras 2>/dev/null | wc -l)
    [ "$n" -ge 80 ] && touch "$ST/p4a" || flag P4A "ST/DY non-res incomplete ($n/80)"
fi

# ---- phase 5a: ET-dp (no eps dependency) --------------------------------------
if [ ! -f "$ST/p5a" ]; then
    log "phase 5a: ET-dp imdbnoniid"
    bash run_et_experiments.sh imdbnoniid all dp all >>"$LOG" 2>&1
    n=$(find 420_et_dp -path "*/imdbnoniid/*" -name global_model_round_10.keras 2>/dev/null | wc -l)
    [ "$n" -ge 10 ] && touch "$ST/p5a" || flag P5A "ET-dp incomplete ($n/10)"
fi

# ---- phase 3 barrier: wait for eps review -------------------------------------
while [ ! -f "$ST/EPS_APPLIED" ]; do
    log "waiting for EPS_APPLIED (review imdb_eps_curve.txt, update attack_metric.py + score_metrics.py + push, then: touch $ST/EPS_APPLIED)"
    sleep 600
done
EPS=$(cat imdb_eps.txt)
log "eps applied: $EPS"

# ---- phase 4b: ST/DY res cells ------------------------------------------------
if [ ! -f "$ST/p4b" ]; then
    log "phase 4b: ST/DY imdbnoniid res"
    WMETS="res" bash cap_sweep.sh imdbnoniid 2.0 2 >>"$LOG" 2>&1
    n=$(find 420_st_res 420_dy_res -path "*/imdbnoniid/*" -name global_model_round_10.keras 2>/dev/null | wc -l)
    [ "$n" -ge 20 ] && touch "$ST/p4b" || flag P4B "ST/DY res incomplete ($n/20)"
fi

# ---- phase 5b: ET-adv at calibrated eps ---------------------------------------
if [ ! -f "$ST/p5b" ]; then
    log "phase 5b: ET-adv imdbnoniid at eps=$EPS (ratio in pyproject, steps per push)"
    ET_ADV_EPS=$EPS ET_ADV_ALPHA=$(python3 -c "print($EPS/4)") \
        bash run_et_experiments.sh imdbnoniid all adv all >>"$LOG" 2>&1
    n=$(find 420_et_adv -path "*/imdbnoniid/*" -name global_model_round_10.keras 2>/dev/null | wc -l)
    [ "$n" -ge 10 ] && touch "$ST/p5b" || flag P5B "ET-adv incomplete ($n/10)"
fi

# ---- phase 6: post-hoc scoring (CPU) -------------------------------------------
if [ ! -f "$ST/p6" ]; then
    log "phase 6: post-hoc LOO+GTG scoring"
    for DS_PART in "imdb iid" "imdbnoniid noniid"; do
        set -- $DS_PART; DS=$1; PART=$2
        for M in gtg_acc gtg_loss gtg_rob gtg_adv_pgd gtg_priv \
                 l1o_acc l1o_loss l1o_rob l1o_adv_pgd l1o_priv; do
            for S in $SEEDS; do
                while [ "$(jobs -rp | wc -l)" -ge 6 ]; do sleep 30; done
                CUDA_VISIBLE_DEVICES= seed=$S python3 robustness.py \
                    --dataset=$DS --partition=$PART --num_rounds=10 \
                    --seed=$S --method=$M --strategy=fedavg \
                    >> "logs/score_${DS}_${M}_${S}.log" 2>&1 &
            done
        done
    done
    wait
    touch "$ST/p6"
fi

# ---- phase 7: evals --------------------------------------------------------------
if [ ! -f "$ST/p7" ]; then
    log "phase 7: model evals (CPU)"
    export CUDA_VISIBLE_DEVICES=
    python3 eval_bl_models.py   --datasets imdbnoniid                >>"$LOG" 2>&1
    python3 eval_stdy_models.py --datasets imdbnoniid                >>"$LOG" 2>&1
    python3 eval_et_models.py   --datasets imdbnoniid --modes adv dp >>"$LOG" 2>&1
    for K in 4 20; do
        python3 reweight_eval.py --dataset imdbnoniid --num_clients $K >>"$LOG" 2>&1
    done
    touch "$ST/p7"
fi

log "=== IMDB RERUN COMPLETE — pull CSVs, run sanity_gate.py, rebuild tables/figs locally ==="
