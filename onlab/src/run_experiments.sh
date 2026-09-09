#!/bin/bash

set -e

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null

DATASETS=("imdb" "imdbnoniid")
SEEDS=(42 107 123 2025 9928)
CLIENT_COUNTS=(4 20)

for DATA in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do   
        for CLIENTS in "${CLIENT_COUNTS[@]}"; do
            echo "Running: Dataset=$DATA, Seed=$SEED, Clients=$CLIENTS"

            export seed=$SEED
            export GLOBAL_SEED=$SEED
            export PYTHONHASHSEED=$GLOBAL_SEED
            export TF_ENABLE_ONEDNN_OPTS=0
            export TF_DETERMINISTIC_OPS=1

            if ! python3 /home/ulrich/set_num_cl.py --data=$DATA --clients=$CLIENTS >/dev/null; then
                echo "[ERROR] set_num_cl.py failed for $DATA, seed $SEED, $CLIENTS clients"
                continue
            fi

            if ! cd /home/ulrich/$DATA; then
                echo "[ERROR] Could not cd to $DATA"
                continue
            fi

            if flwr run .; then
                echo "Flower run for $DATA, seed $SEED, $CLIENTS clients"
            else
                echo "[ERROR] Flower run failed for $DATA, seed $SEED, $CLIENTS clients"
            fi

            cd ..
        done
    done
done

echo "Experiments completed."
echo "Experiments are finished" | mail -s "Seed X" ulrichcs1234@gmail.com
