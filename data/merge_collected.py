"""merge_collected.py — merge result CSVs for et/st/dy from all sources
(data/_collect/<host>/ staging dirs + whatever is already in data/<fam>/) into
the canonical data/<fam>/results_<ds>_<fam>.csv, deduping by experiment key.

Lets partial CSVs from different compute hosts (A40 has celeba/K20; CrySyS has
adult/imdb-K4) combine into one complete file per (family, dataset).
Run from repo root: python data/merge_collected.py
"""
import pandas as pd, glob, os, re

KEYS = {
    "et": ["dataset", "num_clients", "seed", "mode"],
    "st": ["dataset", "num_clients", "seed", "weight_metric"],
    "dy": ["dataset", "num_clients", "seed", "weight_metric"],
}
PAT = re.compile(r"results_(.+?)_(et|st|dy)\.csv$")

for fam in ["et", "st", "dy"]:
    os.makedirs(f"data/{fam}", exist_ok=True)
    files = glob.glob(f"data/_collect/*/results_*_{fam}.csv") + \
            glob.glob(f"data/{fam}/results_*_{fam}.csv")
    by_ds = {}
    for fp in files:
        m = PAT.search(os.path.basename(fp))
        if not m or m.group(2) != fam:
            continue
        ds = m.group(1)
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if len(df):
            by_ds.setdefault(ds, []).append(df)
    for ds, dfs in sorted(by_ds.items()):
        alldf = pd.concat(dfs, ignore_index=True)
        k = [c for c in KEYS[fam] if c in alldf.columns]
        alldf = alldf.drop_duplicates(subset=k, keep="last").sort_values(k)
        out = f"data/{fam}/results_{ds}_{fam}.csv"
        alldf.to_csv(out, index=False)
        print(f"{out}: {len(alldf)} rows  (keys={k})")

# weighting_effect.csv — the ST/DY-vs-BL "did weighting do anything" flag.
we_files = glob.glob("data/_collect/*/weighting_effect.csv") + (
    ["data/weighting_effect.csv"] if os.path.exists("data/weighting_effect.csv") else [])
we = [pd.read_csv(f) for f in we_files if os.path.getsize(f) > 0]
if we:
    wedf = pd.concat(we, ignore_index=True).drop_duplicates(
        subset=["dataset", "strat", "num_clients", "seed", "weight_metric"], keep="last")
    # Recompute the flag from raw reldiff: no-weighting cluster ~2e-4, real
    # weighting 0.2-0.8, so 1e-3 cleanly separates them (the per-run 1e-4 was too tight).
    wedf["equiv_bl"] = wedf["reldiff_vs_bl"] < 1e-3
    wedf = wedf.sort_values(["dataset", "strat", "num_clients", "seed", "weight_metric"])
    wedf.to_csv("data/weighting_effect.csv", index=False)
    nbl = int(wedf["equiv_bl"].sum()) if "equiv_bl" in wedf else 0
    print(f"data/weighting_effect.csv: {len(wedf)} rows, {nbl} flagged equiv_bl (no weighting signal)")
print("merge done")
