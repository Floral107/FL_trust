#!/bin/bash

# Name: eval.sh

# Exit on any error
set -e

# Check for required arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset> <partition> <rounds>"
    echo "Example: $0 cifar iid"
    exit 1
fi

DATASET=$1
PARTITION=$2
ROUNDS=$3

# Create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (silent unless error)
pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null


SEEDS=(42 107 123 2025 9928)
for SEED in "${SEEDS[@]}"; do
    export seed=$SEED
# Run robustness analysis (only log errors)
    python3 /home/ulrich/robustness.py --dataset="$DATASET" --partition="$PARTITION" --num_rounds="$ROUNDS" --seed="$seed"
done

echo "Experiment completed."
