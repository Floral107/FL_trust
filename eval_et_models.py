#!/usr/bin/env python3
"""eval_et_models.py — score ET-trained models on all 7 metrics for Tab. 4.

Reads existing ET ckpts at 420_et_<mode>/<dataset>/<K>/<seed>/global_model_round_10.keras
and evaluates each on (loss, acc, fairDP, fairEO, rel, res, priv).

Output: results_ET_eval.csv with one row per (dataset, K, mode, seed).
Idempotent — skips combos already in the output CSV.

Run from project root after at least one ET training completes.
"""
import os
import sys
import argparse
import pandas as pd
import tensorflow as tf

# Re-use reweight_eval's helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reweight_eval import load_test_data, eval_all_metrics, EVAL_ROUND, load_global_model

DATASETS = ["adultnoniid", "imdbnoniid", "celebanoniid"]
MODES = ["fair", "adv", "dp"]
KS = [4, 20]
SEEDS = [42, 107, 123, 2025, 9928]

OUT_DIR = "data/et"


def out_csv_for(dataset):
    """Per-dataset CSV path mirroring data/combo schema."""
    return os.path.join(OUT_DIR, f"results_{dataset}_et.csv")


def load_existing(dataset):
    """Read existing per-dataset ET CSV if present."""
    p = out_csv_for(dataset)
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


def already_done(df, k, mode, seed):
    if df is None or df.empty:
        return False
    return ((df["num_clients"] == k) & (df["mode"] == mode) & (df["seed"] == seed)).any()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    cache_test = {}     # dataset → (x_test, y_test, s_test)
    existing_by_ds = {} # dataset → DataFrame|None
    new_rows = 0

    for dataset in args.datasets:
        existing = existing_by_ds.setdefault(dataset, load_existing(dataset))
        for mode in args.modes:
            # imdbnoniid + fair: no sensitive attribute — script skips, no ckpt produced
            if dataset == "imdbnoniid" and mode == "fair":
                continue
            for k in args.ks:
                for seed in args.seeds:
                    if already_done(existing, k, mode, seed):
                        continue
                    ckpt = f"420_et_{mode}/{dataset}/{k}/{seed}/global_model_round_10.keras"
                    if not os.path.exists(ckpt):
                        # silent skip — too verbose for the full sweep
                        continue
                    print(f"\n=== {dataset} K={k} mode={mode} seed={seed} ===", flush=True)
                    model_dir = os.path.dirname(ckpt)
                    try:
                        model = load_global_model(model_dir, EVAL_ROUND, dataset)
                    except Exception as e:
                        print(f"    FAIL load: {e}")
                        continue

                    if dataset not in cache_test:
                        print(f"    loading test data for {dataset}")
                        cache_test[dataset] = load_test_data(dataset)
                    x_test, y_test, s_test = cache_test[dataset]

                    try:
                        scores = eval_all_metrics(model, x_test, y_test, s_test, dataset)
                    except Exception as e:
                        print(f"    FAIL eval: {e}")
                        continue

                    # Match data/combo schema: dataset, partition, num_clients, seed, ...
                    # ET-specific extras: mode, round (10 — final ET model only)
                    partition = "noniid"  # Tab. 4 is NIID only
                    row = dict(
                        dataset=dataset, partition=partition,
                        num_clients=k, seed=seed, mode=mode, round=EVAL_ROUND,
                        **scores,
                    )
                    out_csv = out_csv_for(dataset)
                    pd.DataFrame([row]).to_csv(
                        out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                    existing_by_ds[dataset] = load_existing(dataset)
                    new_rows += 1
                    print(f"    scores: {scores}")

    print(f"\n=== DONE — {new_rows} new rows written to {OUT_DIR}/ ===")


if __name__ == "__main__":
    main()
