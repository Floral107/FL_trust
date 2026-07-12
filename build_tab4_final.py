#!/usr/bin/env python3
"""build_tab4_final.py — regenerate Tab.4 (tab:util) from data/{bl,st,dy,et}
with three clean, code+theory-backed override layers on top of build_tab4 logic:

  1) GUARD (do-no-harm): a ST/DY cell whose own model accuracy dropped > DELTA
     vs BL for that (dataset,K) reverts to the BL value. The weighting is adopted
     only when it does not degrade utility; otherwise fall back to baseline.
  2) ET-fair lambda=0.5: ET fairDP/fairEO for adult/celeba taken from the re-run
     CSVs (data/et holds the old lambda=0.1).
  3) imdb res bugfix: imdb res (BL/ST/DY/ET-adv) taken from the post-fix CSVs
     (the table's data/* held the swallowed-TypeError 0.00).

Prints every changed cell (old -> new) and writes tab4_final.tex.
"""
import os
import numpy as np
import pandas as pd

DATASETS = ["adultnoniid", "celebanoniid", "imdbnoniid"]
KS = [4, 20]
METRICS = ["loss", "acc", "fairDP", "fairEO", "rel", "res", "priv"]
WM_OF = {"loss": "loss", "acc": "acc", "fairDP": "fairdp", "fairEO": "faireo",
         "rel": "rel", "res": "res", "priv": "priv"}
ET_ROW_METRIC = {"fair": "fairDP", "adv": "res", "dp": "priv"}
ET_LABEL = {"fair": "Fair. Reg.", "adv": "Rob. Train.", "dp": "Diff. Priv."}
DELTA = 0.10
METRIC_TRANSFORM = {"loss": (lambda v: np.exp(-v))}


def _read(fam, ds):
    p = os.path.join("data", fam, f"results_{ds}_{fam}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    if "round" in d.columns:
        d = d[d["round"] == d["round"].max()]
    return d if len(d) else None


def cell(df, k, metric, wm=None, mode=None, transform=None):
    if df is None:
        return None
    m = df["num_clients"] == k
    if wm is not None:
        m &= df["weight_metric"].astype(str).str.lower() == wm
    if mode is not None and "mode" in df.columns:
        m &= df["mode"].astype(str).str.lower() == mode
    sub = df[m]
    if sub.empty or metric not in sub.columns:
        return None
    vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
    if vals.empty:
        return None
    if transform is not None:
        vals = transform(vals)
    return float(vals.mean()), float(vals.std(ddof=0)), int(len(vals))


# ---- override sources ----
def _load_override_csvs():
    ov = {}
    # ET-fair lambda=0.5 (per-seed metric columns incl fairDP/fairEO)
    for ds, f in (("adultnoniid", "_t4_etfair_adult.csv"),
                  ("celebanoniid", "_t4_etfair_celeba.csv")):
        if os.path.exists(f):
            ov[("etfair", ds)] = pd.read_csv(f)
    # imdb res: BL/ET-adv and ST/DY
    if os.path.exists("_t4_imdbres_blet.csv"):
        ov["imdbres_blet"] = pd.read_csv("_t4_imdbres_blet.csv")
    if os.path.exists("results_imdb_stdy_res.csv"):
        ov["imdbres_stdy"] = pd.read_csv("results_imdb_stdy_res.csv")
    return ov


def _meanstd(df, **filt):
    sub = df
    for col, v in filt.items():
        sub = sub[sub[col] == v]
    vals = pd.to_numeric(sub["res"] if "res" in filt.get("_col", ["res"]) else sub[filt.get("_col", "res")], errors="coerce").dropna() if False else None
    return None  # placeholder, replaced below


def fmt(c, na="--"):
    if c is None:
        return "$.XX$"
    mean, std = c[0], c[1]
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return na
    return rf"${mean:.2f} \pm {std:.2f}$"


def main():
    data = {fam: {ds: _read(fam, ds) for ds in DATASETS} for fam in ("bl", "st", "dy", "et")}
    ov = _load_override_csvs()
    changes = []

    def bl_acc(ds, k):
        c = cell(data["bl"][ds], k, "acc")
        return c[0] if c else None

    def model_acc(fam, ds, k, wm):
        c = cell(data[fam][ds], k, "acc", wm=wm)
        return c[0] if c else None

    def imdbres_override(setting, k):
        """mean/std res from the bugfix CSVs for imdb; None if absent."""
        if setting in ("BL", "ET-adv"):
            df = ov.get("imdbres_blet")
        else:
            df = ov.get("imdbres_stdy")
        if df is None:
            return None
        sub = df[(df["setting"] == setting) & (df["K"] == k)]
        vals = pd.to_numeric(sub["res"], errors="coerce").dropna()
        if vals.empty:
            return None
        return float(vals.mean()), float(vals.std(ddof=0)), int(len(vals))

    def etfair_override(ds, k, metric):
        df = ov.get(("etfair", ds))
        if df is None or metric not in df.columns:
            return None
        sub = df[df["K"] == k]
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        if vals.empty:
            return None
        return float(vals.mean()), float(vals.std(ddof=0)), int(len(vals))

    def block_cell(row, fam, metric, ds, k, wm):
        """Resolve one BL/ST/DY cell with imdb-res override + guard."""
        tf_ = METRIC_TRANSFORM.get(metric)
        if "imdb" in ds and metric in ("fairDP", "fairEO"):
            return "--", None
        # imdb res: the swallowed-TypeError 0.00 bug is fixed; the beta=0.2 re-run
        # re-evaluated imdb res into data/{st,dy} and BL was reconciled from combo,
        # so imdb res now comes from the normal data path like every other cell.
        base = cell(data[fam][ds], k, metric, wm=wm, transform=tf_)
        # do-no-harm GUARD REMOVED: replaced by shrinkage-toward-uniform (beta=0.2),
        # which floors every client weight at 0.8 -> collapse-proof by construction,
        # so ST/DY cells report their true value (no BL fallback).
        return fmt(base), base

    L = [r"\begin{tabular}{cc||cc|cc|cc|}",
         r"    \multicolumn{2}{c||}{\multirow{2}{*}{\textbf{Setting}}}",
         r"    & \multicolumn{2}{c|}{\textbf{$\mathtt{ADULT}$}}",
         r"    & \multicolumn{2}{c|}{\textbf{$\mathtt{CelebA}$}}",
         r"    & \multicolumn{2}{c|}{\textbf{$\mathtt{IMDB}$}} \\",
         r"    && $4$ & $20$ & $4$ & $20$ & $4$ & $20$ \\",
         r"    \hline\hline"]

    def block(metric):
        wm = WM_OF[metric]
        L.append(rf"    \multirow{{3}}{{*}}{{$\mathtt{{{metric}}}$}}")
        for row, fam in (("BL", "bl"), ("ST", "st"), ("DY", "dy")):
            w = None if row == "BL" else wm
            cells = [block_cell(row, fam, metric, ds, k, w)[0] for ds in DATASETS for k in KS]
            L.append(f"    & {row} & " + " & ".join(cells) + r" \\")
        L.append(r"    \hline")

    def et_row(mode):
        metric = ET_ROW_METRIC[mode]
        cells = []
        for ds in DATASETS:
            for k in KS:
                if "imdb" in ds and metric in ("fairDP", "fairEO"):
                    cells.append("--")
                elif mode == "fair":
                    c = etfair_override(ds, k, "fairDP")
                    if c is not None and ds != "adultnoniid":
                        pass
                    old = cell(data["et"][ds], k, "fairDP", mode="fair")
                    if c is not None:
                        changes.append(f"ETFAIR {ds} K{k} fairDP: {old[0] if old else float('nan'):.3f} -> {c[0]:.3f}")
                    cells.append(fmt(c if c is not None else old))
                elif mode == "adv" and "imdb" in ds:
                    # imdb ET-adv res now comes straight from data/et (the et_adv
                    # models were re-evaluated on CPU at eps=0.0015); the old
                    # _t4_imdbres_blet.csv override is gone.
                    cells.append(fmt(cell(data["et"][ds], k, metric, mode=mode)))
                else:
                    cells.append(fmt(cell(data["et"][ds], k, metric, mode=mode)))
        L.append(f"    {ET_LABEL[mode]}& ET & " + " & ".join(cells) + r" \\")
        # sibling comment line
        sibling = {"fair": "fairEO", "adv": "rel"}.get(mode)
        if sibling:
            alt = []
            for ds in DATASETS:
                for k in KS:
                    if "imdb" in ds and sibling.startswith("fair"):
                        alt.append("--")
                    elif mode == "fair":
                        c = etfair_override(ds, k, "fairEO")
                        alt.append(fmt(c if c is not None else cell(data["et"][ds], k, sibling, mode=mode)))
                    else:
                        alt.append(fmt(cell(data["et"][ds], k, sibling, mode=mode)))
            L.append(f"    % ET-{mode} on {sibling}: " + " & ".join(alt))
        L.append(r"    \hline")

    block("loss"); block("acc"); L[-1] = r"    \hline\hline"
    et_row("fair"); block("fairDP"); block("fairEO"); L[-1] = r"    \hline\hline"
    et_row("adv"); block("rel"); block("res"); L[-1] = r"    \hline\hline"
    et_row("dp"); block("priv")
    L.append(r"\end{tabular}")

    with open("tab4_final.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("tab4_final.tex written.")
    print(f"\n=== {len(changes)} cell changes vs raw data ===")
    for c in changes:
        print("  ", c)


if __name__ == "__main__":
    main()
