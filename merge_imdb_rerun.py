"""merge_imdb_rerun.py — fold the 2026-07-10 rerun data into data/combo.

1. IMDB (both partitions): FULL REPLACE of results_imdb{,noniid}_{gtg,l1o}.csv —
   the rerun retrained every BL trajectory (LR 1e-3), so old combo rows are a
   different (non-converged) model generation and must not be mixed.
   Input: data/_imdbpull_20260710/scores/results_<ds>_<seed>_<method>_fedavg.csv
   (method = {gtg,l1o}_{acc,loss,rob,adv_pgd,priv}); adv_pgd -> adv in combo.
2. CelebA gtg_loss: column-replace gtg_loss_contribution/global_gtg_loss with
   the loss_flag-correct recompute for every covered (K, seed) block
   (celeba: all; celebanoniid: K4 all + K20 seeds 123/2025/9928 — 42/107 ckpts
   no longer exist anywhere, those keep the sign-fixed values).
Sign convention: inputs are post-2026-07-08 cont_evals output = canonical
(higher=better incl. loss). NO sign logic here — see PIPELINE.md.
Backup: data/combo_bak_premergererun_20260710/. Run: python merge_imdb_rerun.py
"""
import glob
import os
import re
import shutil

import pandas as pd

KEY = ["dataset", "partition", "num_clients", "seed", "round", "client_id"]
PULL = "data/_imdbpull_20260710"
BAK = "data/combo_bak_premergererun_20260710"

if not os.path.exists(BAK):
    os.makedirs(BAK)
    for ds in ["imdb", "imdbnoniid", "celeba", "celebanoniid"]:
        for m in ["gtg", "l1o"]:
            shutil.copy(f"data/combo/results_{ds}_{m}.csv", BAK)
    print(f"backup -> {BAK}")

# ---- 1. IMDB full replace ----------------------------------------------------
for ds in ["imdb", "imdbnoniid"]:
    for meth in ["gtg", "l1o"]:
        # bucket per metric (concat seeds), THEN merge across metrics on KEY
        per_metric = {}
        for fp in sorted(glob.glob(f"{PULL}/scores/results_{ds}_*_{meth}_*_fedavg.csv")):
            m = re.match(rf"results_{ds}_(\d+)_{meth}_([a-z_]+)_fedavg\.csv", os.path.basename(fp))
            if not m:
                continue
            metric = m.group(2)
            out_metric = "adv" if metric == "adv_pgd" else metric
            d = pd.read_csv(fp)
            gcol_in, ccol_in = f"global_{meth}_{metric}", f"{meth}_{metric}_contribution"
            gcol, ccol = f"global_{meth}_{out_metric}", f"{meth}_{out_metric}_contribution"
            d = d[KEY + [gcol_in, ccol_in]].rename(columns={gcol_in: gcol, ccol_in: ccol})
            per_metric.setdefault(out_metric, []).append(d)
        merged = None
        for metric, parts in per_metric.items():
            block = pd.concat(parts, ignore_index=True).drop_duplicates(subset=KEY, keep="last")
            merged = block if merged is None else merged.merge(block, on=KEY, how="outer")
        # combo alias expected by figs/tables
        merged["acc_contribution"] = merged[f"{meth}_acc_contribution"]
        out = f"data/combo/results_{ds}_{meth}.csv"
        merged = merged.sort_values(KEY).reset_index(drop=True)
        merged.to_csv(out, index=False)
        print(f"wrote {out}: {len(merged)} rows, cols={len(merged.columns)}")

# ---- 2. CelebA gtg_loss column replace ----------------------------------------
for ds in ["celeba", "celebanoniid"]:
    combo = pd.read_csv(f"data/combo/results_{ds}_gtg.csv")
    n_repl = 0
    for fp in sorted(glob.glob(f"results_{ds}_*_gtg_loss_fedavg.csv")
                     + glob.glob(f"{PULL}/results_{ds}_*_gtg_loss_fedavg.csv")):
        d = pd.read_csv(fp)
        d = d[pd.to_numeric(d["round"], errors="coerce").notna()].copy()
        for c in ["num_clients", "seed", "round", "client_id"]:
            d[c] = pd.to_numeric(d[c]).astype(int)
        d = d.drop_duplicates(subset=["num_clients", "seed", "round", "client_id"], keep="last")
        d = d.set_index(["num_clients", "seed", "round", "client_id"])
        ci = combo.set_index(["num_clients", "seed", "round", "client_id"]).index
        mask = ci.isin(d.index)
        for col in ["gtg_loss_contribution", "global_gtg_loss"]:
            vals = d[col].reindex(ci[mask]).to_numpy()
            combo.loc[mask, col] = pd.to_numeric(vals)
        n_repl += int(mask.sum())
    combo.to_csv(f"data/combo/results_{ds}_gtg.csv", index=False)
    print(f"{ds}: replaced gtg_loss on {n_repl} row-instances")

print("merge done — run sanity_gate.py next")
