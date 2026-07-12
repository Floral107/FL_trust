#!/bin/bash
# crysys_cap_ab.sh — weight-CAP A/B prototype on the IMDB K=20 acc cells.
#
# Tests whether the hard weight cap (weight-cap=2.0, beta=1.0) eliminates the
# base-rate collapse seen under beta=0.2 AND lifts accuracy on the one genuine
# headroom case (imdb K20, BL acc ~0.63). Trains ST+DY x acc x 5 seeds and saves
# to the canonical 420_{st,dy}_acc/imdbnoniid/20/<seed>/ paths (no beta=0.2 K20
# ckpts exist on CrySyS, so nothing is overwritten). Eval is separate
# (eval_cap_ab.py -> _cap_ab_imdb_k20.csv) so canonical data/ stays untouched.
set -uo pipefail
cd "$HOME/ulrich"
source .venv/bin/activate 2>/dev/null || true
CAP="${1:-2.0}"; BETA="${2:-1.0}"; MAX="${3:-2}"
DS=imdbnoniid; K=20; WMET=acc
SEEDS="42 107 123 2025 9928"
export FLR_TF_INTRA=4 FLR_TF_INTER=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=-1
mkdir -p logs
LOG="$HOME/ulrich/cap_ab.log"
echo "=== CAP A/B imdb K20 acc  cap=$CAP beta=$BETA MAX=$MAX  START $(date) ===" | tee -a "$LOG"

# Fresh: remove any prior cap-run ckpts for these cells (idempotent re-launch)
for m in st dy; do rm -rf "420_${m}_acc/${DS}/${K}" 2>/dev/null; done

# Set num-supernodes once (serial, atomic), verify parse
python3 set_num_cl.py --data="$DS" --clients="$K" >>"$LOG" 2>&1
python3 -c "import tomllib; tomllib.load(open('$DS/pyproject.toml','rb'))" 2>>"$LOG" \
  || { echo "[ABORT] $DS pyproject invalid" | tee -a "$LOG"; exit 1; }

i=0; running=0
for MODE in st dy; do
  for SEED in $SEEDS; do
    SKIP_SETNC=1 BETA="$BETA" CAP="$CAP" \
      bash st_dy_guarded_run.sh "$DS" "$K" "$MODE" "$WMET" "$SEED" "$((200+i))" >>"$LOG" 2>&1 &
    i=$(( i+1 )); running=$(( running+1 ))
    if (( running >= MAX )); then wait -n; running=$(( running-1 )); fi
  done
done
wait
echo "=== CAP A/B DONE $(date) ===" | tee -a "$LOG"
echo "ckpts:"; for m in st dy; do for s in $SEEDS; do
  f="420_${m}_acc/${DS}/${K}/${s}/global_model_round_10.keras"
  [ -f "$f" ] && echo "  OK $f" || echo "  MISSING $f"
done; done | tee -a "$LOG"
