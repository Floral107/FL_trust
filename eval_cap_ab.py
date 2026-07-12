#!/usr/bin/env python3
"""eval_cap_ab.py — evaluate the weight-CAP A/B ckpts (imdb K20 acc, ST+DY) and
write to a SEPARATE csv so the canonical data/{st,dy} CSVs stay untouched.

Reuses reweight_eval.eval_all_metrics (same scoring as eval_stdy_models.py).
Run on CPU:  CUDA_VISIBLE_DEVICES= python3 eval_cap_ab.py
Output: _cap_ab_imdb_k20.csv  (cols: mode,seed,acc,loss,rel,res,fairDP,fairEO,priv)
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reweight_eval import load_test_data, eval_all_metrics, EVAL_ROUND, load_global_model

DS = "imdbnoniid"; K = 20; WM = "acc"
SEEDS = [42, 107, 123, 2025, 9928]
OUT = "_cap_ab_imdb_k20.csv"

def main():
    x_test, y_test, s_test = load_test_data(DS)
    rows = []
    for mode in ("st", "dy"):
        for seed in SEEDS:
            ckpt = f"420_{mode}_{WM}/{DS}/{K}/{seed}/global_model_round_10.keras"
            if not os.path.exists(ckpt):
                print(f"MISSING {ckpt}"); continue
            try:
                model = load_global_model(os.path.dirname(ckpt), EVAL_ROUND, DS)
                scores = eval_all_metrics(model, x_test, y_test, s_test, DS)
            except Exception as e:
                print(f"FAIL {mode} seed={seed}: {e}"); continue
            row = dict(mode=mode, seed=seed, **scores)
            rows.append(row)
            print(f"{mode} seed={seed}: acc={scores.get('acc'):.3f} loss={scores.get('loss'):.3f} res={scores.get('res'):.3f}")
    if rows:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"\nwrote {OUT} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
