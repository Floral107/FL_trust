#!/usr/bin/env python3
"""eval_stdy_models.py — score ST/DY (static/dynamic weighted) models on all 7
metrics for the Tab.4 data collection.

Reads:  420_<strat>_<weight_metric>/<dataset>/<K>/<seed>/global_model_round_10.keras
        strat in {st, dy}; weight_metric in {acc,loss,fairdp,faireo,rel,res,priv}
        (imdb has no sensitive attr -> no fairdp/faireo weightings, fair eval = NaN)

Writes: data/<strat>/results_<dataset>_<strat>.csv  (one row per ds,K,seed,weight_metric)
Columns: dataset,partition,num_clients,seed,weight_metric,round,acc,loss,rel,res,fairDP,fairEO,priv
         -> mirrors data/et schema (mode -> weight_metric). The full metric matrix:
            the diagonal (eval metric == weight_metric) is the Tab.4 cell; off-diagonal
            lets you see cross-metric effects of each weighting.

Idempotent: skips (K, weight_metric, seed) rows already in the output CSV.
Per-server: use --datasets to run only the models/data a given host has.

Run from project root after at least one ST/DY training completes.
"""
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reweight_eval import load_test_data, eval_all_metrics, EVAL_ROUND, load_global_model

DATASETS = ["adultnoniid", "imdbnoniid", "celebanoniid"]
STRATS = ["st", "dy"]
KS = [4, 20]
SEEDS = [42, 107, 123, 2025, 9928]
# weight-metric dirs are lowercased by server_app at save time
WMS_FULL = ["acc", "loss", "fairdp", "faireo", "rel", "res", "priv"]
WMS_NOFAIR = ["acc", "loss", "rel", "res", "priv"]   # imdb: no sensitive attr


def wms_for(dataset):
    return WMS_NOFAIR if "imdb" in dataset else WMS_FULL


def out_csv_for(strat, dataset):
    return os.path.join("data", strat, f"results_{dataset}_{strat}.csv")


def load_existing(strat, dataset):
    p = out_csv_for(strat, dataset)
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def already_done(df, k, wm, seed):
    if df is None or df.empty:
        return False
    return ((df["num_clients"] == k) & (df["weight_metric"] == wm) & (df["seed"] == seed)).any()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--strats", nargs="+", default=STRATS)
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--wms", nargs="+", default=None, help="override weight-metrics")
    args = ap.parse_args()

    for s in args.strats:
        os.makedirs(os.path.join("data", s), exist_ok=True)

    cache_test = {}      # dataset -> (x_test, y_test, s_test)
    new_rows = 0

    for dataset in args.datasets:
        wms = args.wms or wms_for(dataset)
        for strat in args.strats:
            existing = load_existing(strat, dataset)
            for wm in wms:
                for k in args.ks:
                    for seed in args.seeds:
                        if already_done(existing, k, wm, seed):
                            continue
                        ckpt = f"420_{strat}_{wm}/{dataset}/{k}/{seed}/global_model_round_10.keras"
                        if not os.path.exists(ckpt):
                            continue
                        print(f"\n=== {dataset} {strat} wm={wm} K={k} seed={seed} ===", flush=True)
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
                        row = dict(
                            dataset=dataset, partition="noniid", num_clients=k,
                            seed=seed, weight_metric=wm, round=EVAL_ROUND, **scores,
                        )
                        out_csv = out_csv_for(strat, dataset)
                        pd.DataFrame([row]).to_csv(
                            out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                        existing = load_existing(strat, dataset)
                        new_rows += 1
                        print(f"    scores: {scores}")

    print(f"\n=== DONE — {new_rows} new ST/DY rows written ===")


if __name__ == "__main__":
    main()
