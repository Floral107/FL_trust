#!/bin/bash
# run_et_experiments.sh
# Usage: ./run_et_experiments.sh [dataset] [clients] [et_mode] [seed]
# All args optional. Omit to loop over all values.
#
# Examples:
#   ./run_et_experiments.sh                                # all combos
#   ./run_et_experiments.sh celebanoniid 4 adv 42          # one combo
#   ./run_et_experiments.sh adultnoniid all all all        # all et_modes, all seeds for adult-4 and adult-20
#
# Output: 420_et_{fair,adv,dp}/<dataset>/<nc>/<seed>/global_model_round_*.keras
#
# Notes:
#  - Only noniid datasets (the paper reports ET on non-IID only).
#  - imdbnoniid + et_mode=fair is skipped (no sensitive attribute).
#  - ET runs use FedAvg aggregation; FedProx is mutually exclusive.

set -e

DATASETS_ALL=(adultnoniid celebanoniid imdbnoniid)
CLIENTS_ALL=(4 20)
ET_MODES_ALL=(fair adv dp)
SEEDS_ALL=(42 107 123 2025 9928)

# ── arg parsing ────────────────────────────────────────────────────────────────
DATASET_ARG=${1:-all}
CLIENTS_ARG=${2:-all}
ET_MODE_ARG=${3:-all}
SEED_ARG=${4:-all}

[[ "$DATASET_ARG" == "all" ]] && DATASETS=("${DATASETS_ALL[@]}") || DATASETS=("$DATASET_ARG")
[[ "$CLIENTS_ARG" == "all" ]] && CLIENT_LIST=("${CLIENTS_ALL[@]}") || CLIENT_LIST=("$CLIENTS_ARG")
[[ "$ET_MODE_ARG" == "all" ]] && ET_MODES=("${ET_MODES_ALL[@]}") || ET_MODES=("$ET_MODE_ARG")
[[ "$SEED_ARG"    == "all" ]] && SEEDS=("${SEEDS_ALL[@]}")       || SEEDS=("$SEED_ARG")

# Activate venv if it exists.
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

for DATA in "${DATASETS[@]}"; do
    for NC in "${CLIENT_LIST[@]}"; do
        for ET in "${ET_MODES[@]}"; do
            # IMDB has no sensitive attribute → fair would be a BL no-op; skip.
            if [[ "$DATA" == "imdbnoniid" && "$ET" == "fair" ]]; then
                echo "[SKIP] imdbnoniid + fair (no sensitive attribute)"
                continue
            fi

            for SEED in "${SEEDS[@]}"; do
                echo "================================================================"
                echo "[ET] dataset=$DATA  clients=$NC  et_mode=$ET  seed=$SEED"
                echo "================================================================"

                export seed=$SEED
                export GLOBAL_SEED=$SEED
                export PYTHONHASHSEED=$GLOBAL_SEED
                export TF_ENABLE_ONEDNN_OPTS=0
                export TF_DETERMINISTIC_OPS=1
                # export CUDA_VISIBLE_DEVICES=""  # disabled on iA — A100 GPU available

                if ! python3 set_num_cl.py --data="$DATA" --clients="$NC" >/dev/null; then
                    echo "[ERROR] set_num_cl.py failed for $DATA, seed $SEED, $NC clients"
                    continue
                fi

                pushd "$DATA" >/dev/null || { echo "[ERROR] cd $DATA failed"; continue; }

                # ET takes precedence over strategy; we always pass fedavg + et-mode.
                RUN_CONFIG="strategy='fedavg' et-mode='$ET'"
                # Optional adv-PGD overrides (env): used by the celeba et_adv
                # retrain at eps=0.01 so the defense is trained and evaluated at
                # the same epsilon (drops the train/eval decoupling footnote).
                [ -n "${ET_ADV_EPS:-}" ]   && RUN_CONFIG="$RUN_CONFIG et-adv-eps=$ET_ADV_EPS"
                [ -n "${ET_ADV_ALPHA:-}" ] && RUN_CONFIG="$RUN_CONFIG et-adv-alpha=$ET_ADV_ALPHA"

                if flwr run . --run-config "$RUN_CONFIG"; then
                    echo "[OK] $DATA / nc=$NC / et=$ET / seed=$SEED"
                else
                    echo "[ERROR] flwr run failed: $DATA / nc=$NC / et=$ET / seed=$SEED"
                fi

                popd >/dev/null
            done
        done
    done
done

echo "================================================================"
echo "ET experiments completed."
echo "Run: python reweight_eval.py --et-mode fair --dataset adultnoniid --num_clients 4"
echo "     etc. to evaluate round-10 checkpoints."
