#!/usr/bin/env python3
"""eval_bl_models.py — score the BASELINE (plain FedAvg) models on all 7
metrics for the Tab.4 BL rows.

Reads:  420/<dataset>/<K>/<seed>/global_model_round_10.keras
Writes: data/bl/results_<dataset>_bl.csv  (one row per ds,K,seed)
Columns: dataset,partition,num_clients,seed,round,acc,loss,rel,res,fairDP,fairEO,priv
         -> mirrors data/st,dy schema (no weight_metric: BL has none). Uses the
         exact same eval_all_metrics as ST/DY/ET so all four Tab.4 strategies
         are scored identically (incl. per-dataset PGD params).

Idempotent: skips (K, seed) rows already in the output CSV.
Per-server: use --datasets to run only the models a given host has.
"""
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reweight_eval import load_test_data, eval_all_metrics, EVAL_ROUND, load_global_model

DATASETS = ["adultnoniid", "imdbnoniid", "celebanoniid"]
KS = [4, 20]
SEEDS = [42, 107, 123, 2025, 9928]


def out_csv_for(dataset):
    return os.path.join("data", "bl", f"results_{dataset}_bl.csv")


def load_existing(dataset):
    p = out_csv_for(dataset)
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def already_done(df, k, seed):
    if df is None or df.empty:
        return False
    return ((df["num_clients"] == k) & (df["seed"] == seed)).any()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    os.makedirs(os.path.join("data", "bl"), exist_ok=True)
    cache_test = {}
    new_rows = 0

    for dataset in args.datasets:
        existing = load_existing(dataset)
        for k in args.ks:
            for seed in args.seeds:
                if already_done(existing, k, seed):
                    continue
                ckpt = f"420/{dataset}/{k}/{seed}/global_model_round_10.keras"
                if not os.path.exists(ckpt):
                    print(f"missing: {ckpt}")
                    continue
                print(f"\n=== BL {dataset} K={k} seed={seed} ===", flush=True)
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
                row = dict(dataset=dataset, partition="noniid", num_clients=k,
                           seed=seed, round=EVAL_ROUND, **scores)
                out_csv = out_csv_for(dataset)
                pd.DataFrame([row]).to_csv(
                    out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                existing = load_existing(dataset)
                new_rows += 1
                print(f"    scores: {scores}")

    print(f"\n=== DONE — {new_rows} new BL rows written ===")


if __name__ == "__main__":
    main()
