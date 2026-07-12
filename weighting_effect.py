#!/usr/bin/env python3
"""weighting_effect.py — did ST/DY weighting actually change the model vs BL?

For each (dataset, strat, K, seed, weight_metric) it loads the ST/DY round-10
global model and the matching BL (420/) round-10 global model and computes the
relative weight-space difference:  ||W_stdy - W_bl|| / ||W_bl||.

If reldiff ~ 0 the weighting produced uniform (all-1.0) weights every round and
the run is effectively identical to BL — i.e. the metric had NO signal to weight
by (e.g. imdb 'res': resilience ~0 so every client looks the same). reldiff > 0
means the weighting genuinely steered aggregation.

Output: data/weighting_effect.csv  (idempotent-ish; appends, dedup at merge).
Columns: dataset,strat,num_clients,seed,weight_metric,reldiff_vs_bl,equiv_bl

Only needs TensorFlow (load weights) — no dataset/test-data, runs on any host
that has both 420/ and 420_{st,dy}_* for the requested datasets.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import tensorflow as tf

DATASETS = ["adultnoniid", "imdbnoniid", "celebanoniid"]
STRATS = ["st", "dy"]
KS = [4, 20]
SEEDS = [42, 107, 123, 2025, 9928]
WMS_FULL = ["acc", "loss", "fairdp", "faireo", "rel", "res", "priv"]
WMS_NOFAIR = ["acc", "loss", "rel", "res", "priv"]
EQUIV_THRESH = 1e-4   # reldiff below this => effectively identical to BL
OUT = "data/weighting_effect.csv"


def load_w(path):
    m = tf.keras.models.load_model(path, compile=False)
    w = m.get_weights()
    del m
    tf.keras.backend.clear_session()
    return w


def reldiff(wa, wb):
    num = den = 0.0
    for a, b in zip(wa, wb):
        a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        num += float(np.sum((a - b) ** 2)); den += float(np.sum(b ** 2))
    return float((num / den) ** 0.5) if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        try:
            d = pd.read_csv(OUT)
            done = set(zip(d.dataset, d.strat, d.num_clients, d.seed, d.weight_metric))
        except Exception:
            pass
    n = 0
    for ds in args.datasets:
        wms = WMS_NOFAIR if "imdb" in ds else WMS_FULL
        for k in args.ks:
            for seed in args.seeds:
                bl = f"420/{ds}/{k}/{seed}/global_model_round_10.keras"
                if not os.path.exists(bl):
                    continue
                blw = None
                for strat in STRATS:
                    for wm in wms:
                        if (ds, strat, k, seed, wm) in done:
                            continue
                        p = f"420_{strat}_{wm}/{ds}/{k}/{seed}/global_model_round_10.keras"
                        if not os.path.exists(p):
                            continue
                        if blw is None:
                            blw = load_w(bl)
                        rd = reldiff(load_w(p), blw)
                        row = dict(dataset=ds, strat=strat, num_clients=k, seed=seed,
                                   weight_metric=wm, reldiff_vs_bl=rd,
                                   equiv_bl=bool(rd < EQUIV_THRESH))
                        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
                        n += 1
                        print(f"{ds} {strat} K{k} s{seed} {wm}: reldiff={rd:.6f} equiv_bl={rd<EQUIV_THRESH}", flush=True)
    print(f"=== {n} comparisons written to {OUT} ===")


if __name__ == "__main__":
    main()
