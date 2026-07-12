"""Merge per-metric per-seed CelebA result CSVs into the combo format that
short_figs.py / article_figs.ipynb expect.

Input  : data/celeba_to_distribute/results_celeba_<seed>_<method>_<metric>_fedavg.csv
         data/celeba_to_distribute/results_celebanoniid_<seed>_<method>_<metric>_fedavg.csv
Output : data/combo/results_celeba_<method>.csv
         data/combo/results_celebanoniid_<method>.csv

Metric -> column mapping (matches existing adult combo schema):
  acc     -> global_<method>_acc      , <method>_acc_contribution     , acc_contribution (alias of <method>_acc_contribution)
  fair    -> global_<method>_fair     , <method>_fair_contribution
  fair_eo -> global_<method>_fair_eo  , <method>_fair_eo_contribution
  rob     -> global_<method>_rob      , <method>_rob_contribution
  adv     -> global_<method>_adv      , <method>_adv_contribution     (input filename uses 'adv_pgd' -> normalised)
  priv    -> global_<method>_priv     , <method>_priv_contribution
  loss    -> global_<method>_loss     , <method>_loss_contribution

Rows are joined on (dataset, partition, num_clients, seed, round, client_id).
Missing metrics for a (method, dataset) pair are simply not in the output —
short_figs.py handles missing columns gracefully.

SIGN CONVENTION (do NOT add sign logic here): contribution columns are
higher=better for every metric, including loss. Since 2026-07-08 cont_evals.py
negates loss contributions at the source, so raw files merge verbatim.
Raw celeba files produced by PRE-fix code carry the inverse (raw-loss) sign —
if you ever re-merge from an old archive, run sanity_gate.py afterwards; it
catches the inversion. See PIPELINE.md.
"""

import os
import re
import glob
import pandas as pd

SRC_DIR = os.path.join("data", "celeba_to_distribute")
OUT_DIR = os.path.join("data", "combo")
os.makedirs(OUT_DIR, exist_ok=True)

KEY = ["dataset", "partition", "num_clients", "seed", "round", "client_id"]


def _parse_filename(name):
    """Returns (dataset, seed, method, metric) or None if filename doesn't fit."""
    m = re.match(
        r"results_(?P<ds>celeba(?:noniid)?)_(?P<seed>\d+)_(?P<method>gtg|l1o)_(?P<metric>[a-z_]+?)_fedavg\.csv",
        name,
    )
    if not m:
        return None
    metric = m.group("metric")
    # 'adv_pgd' in filename -> store as 'adv' in combo (matches adult schema)
    if metric == "adv_pgd":
        metric = "adv"
    return m.group("ds"), int(m.group("seed")), m.group("method"), metric


def _per_metric_columns(df, method, metric):
    """Pick the (global_*, *_contribution) columns for this metric and rename
    them to the combo-schema names. Returns a DataFrame keyed on KEY."""
    global_col = f"global_{method}_{metric}"
    contrib_col = f"{method}_{metric}_contribution"

    if metric == "adv":
        # filenames write the global col with 'adv_pgd' but contribution col with 'adv_pgd' too
        in_global = f"global_{method}_adv_pgd"
        in_contrib = f"{method}_adv_pgd_contribution"
    else:
        in_global = global_col
        in_contrib = contrib_col

    cols = list(KEY)
    if in_global in df.columns:
        cols.append(in_global)
    if in_contrib in df.columns:
        cols.append(in_contrib)
    sub = df[cols].copy()
    sub = sub.rename(columns={in_global: global_col, in_contrib: contrib_col})
    return sub


def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.csv")))
    if not files:
        print(f"No files in {SRC_DIR}")
        return

    # bucket[(dataset, method, metric)] -> list of per-seed DFs (same schema, just stack)
    per_metric = {}
    for fp in files:
        parsed = _parse_filename(os.path.basename(fp))
        if not parsed:
            print(f"  skip (unparseable): {os.path.basename(fp)}")
            continue
        ds, seed, method, metric = parsed
        d = pd.read_csv(fp)
        sub = _per_metric_columns(d, method, metric)
        per_metric.setdefault((ds, method, metric), []).append(sub)

    # Concat seeds for each metric, then outer-merge across metrics on KEY.
    grouped = {}  # (ds, method) -> {metric: concat_df}
    for (ds, method, metric), parts in per_metric.items():
        concat_df = pd.concat(parts, ignore_index=True)
        # If duplicate KEY rows appear across seeds (shouldn't, seeds differ),
        # average them defensively so the merge has unique keys.
        concat_df = concat_df.groupby(KEY, as_index=False).mean(numeric_only=True)
        grouped.setdefault((ds, method), {})[metric] = concat_df

    for (ds, method), per_m in grouped.items():
        merged = None
        for metric, sub in per_m.items():
            merged = sub if merged is None else merged.merge(sub, on=KEY, how="outer")

        # short_figs.py expects 'acc_contribution' too (alias of <method>_acc_contribution)
        acc_contrib = f"{method}_acc_contribution"
        if acc_contrib in merged.columns and "acc_contribution" not in merged.columns:
            merged["acc_contribution"] = merged[acc_contrib]

        out = os.path.join(OUT_DIR, f"results_{ds}_{method}.csv")
        merged.to_csv(out, index=False)
        seeds = sorted({s for parts_list in per_metric.values() for sub in parts_list for s in sub.get("seed", pd.Series([])).unique() if sub is not None})
        print(f"  wrote {out} ({len(merged)} rows, metrics: {sorted(per_m.keys())})")


if __name__ == "__main__":
    main()
