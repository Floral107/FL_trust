#!/usr/bin/env python3
"""build_tab4.py — assemble MAIN.tex Tab.4 (tab:util) from data/{bl,st,dy,et}.

Cell semantics (per the table in MAIN.tex):
  - metric block X, row BL : the unweighted baseline model evaluated on X
  - metric block X, row ST : model weighted BY X (round-2 frozen), evaluated ON X
  - metric block X, row DY : model weighted BY X (fresh each round), evaluated ON X
  - ET rows (one per intervention): Fair.Reg -> fairDP, Rob.Train -> res,
    Diff.Priv -> priv (variants on the other pillar metric are emitted as
    comments so the author can swap).
All cells: mean over the 5 seeds (std in the summary CSV / comments).
imdb fairness cells are '--' (no sensitive attribute).

Outputs:
  data/tab4_summary.csv  - long form: dataset,K,row,metric,mean,std,n
  tab4_filled.tex        - the tabular block, missing cells left as .XX

Usage: python build_tab4.py   (from repo root; reads local data/)
"""
import os
import pandas as pd
import numpy as np

DATASETS = ["adultnoniid", "celebanoniid", "imdbnoniid"]   # table column order
KS = [4, 20]
METRICS = ["loss", "acc", "fairDP", "fairEO", "rel", "res", "priv"]
WM_OF = {"loss": "loss", "acc": "acc", "fairDP": "fairdp", "fairEO": "faireo",
         "rel": "rel", "res": "res", "priv": "priv"}
ET_ROW_METRIC = {"fair": "fairDP", "adv": "res", "dp": "priv"}
ET_LABEL = {"fair": "Fair. Reg.", "adv": "Rob. Train.", "dp": "Diff. Priv."}
SEEDS_EXPECTED = 5


def _read(fam, ds):
    p = os.path.join("data", fam, f"results_{ds}_{fam}.csv")
    if not os.path.exists(p):
        return None
    try:
        d = pd.read_csv(p)
        return d if len(d) else None
    except Exception:
        return None


def cell(df, k, metric, wm=None, mode=None, transform=None):
    """mean/std/n of `metric` column over seeds for one table cell.

    `transform` (callable on the per-seed Series) is applied BEFORE mean/std so
    loss can be reported as the Eq.(2) utility e^{-loss} (higher=better), making
    the whole table direction-consistent. Transform per-seed, then aggregate.
    """
    if df is None:
        return None
    m = df["num_clients"] == k
    if wm is not None:
        m &= df["weight_metric"].astype(str).str.lower() == wm
    if mode is not None:
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


# loss reported as e^{-loss} (Eq.2 utility) so every table column is higher=better.
METRIC_TRANSFORM = {"loss": (lambda v: np.exp(-v))}


def fmt(c, na="--"):
    if c is None:
        return "$.XX$"
    mean, std = c[0], c[1]
    if np.isnan(mean):
        return na
    return rf"${mean:.2f} \pm {std:.2f}$"


def main():
    data = {fam: {ds: _read(fam, ds) for ds in DATASETS} for fam in
            ("bl", "st", "dy", "et")}

    rows_long, missing = [], []

    def collect(ds, k, row, metric, c):
        if c is None:
            missing.append(f"{ds} K{k} {row} {metric}")
            rows_long.append(dict(dataset=ds, K=k, row=row, metric=metric,
                                  mean=np.nan, std=np.nan, n=0))
        else:
            rows_long.append(dict(dataset=ds, K=k, row=row, metric=metric,
                                  mean=c[0], std=c[1], n=c[2]))

    def six(rowname, metric, fam, wm=None, mode=None):
        """One table line: cells for (ds,K) in table order. Returns latex cells."""
        out = []
        tf_ = METRIC_TRANSFORM.get(metric)
        for ds in DATASETS:
            for k in KS:
                if "imdb" in ds and metric in ("fairDP", "fairEO"):
                    collect(ds, k, rowname, metric, (float("nan"), 0.0, 0))
                    out.append("--")
                    continue
                c = cell(data[fam][ds], k, metric, wm=wm, mode=mode, transform=tf_)
                collect(ds, k, rowname, metric, c)
                out.append(fmt(c))
        return out

    L = []
    L.append(r"\begin{tabular}{cc||cc|cc|cc|}")
    L.append(r"    \multicolumn{2}{c||}{\multirow{2}{*}{\textbf{Setting}}}")
    L.append(r"    & \multicolumn{2}{c|}{\textbf{$\mathtt{ADULT}$}}")
    L.append(r"    & \multicolumn{2}{c|}{\textbf{$\mathtt{CelebA}$}}")
    L.append(r"    & \multicolumn{2}{c|}{\textbf{$\mathtt{IMDB}$}} \\")
    L.append(r"    && $4$ & $20$ & $4$ & $20$ & $4$ & $20$ \\")
    L.append(r"    \hline\hline")

    def block(metric):
        wm = WM_OF[metric]
        L.append(rf"    \multirow{{3}}{{*}}{{$\mathtt{{{metric}}}$}}")
        for row, fam, w in (("BL", "bl", None), ("ST", "st", wm), ("DY", "dy", wm)):
            cells = six(f"{row}", metric, fam, wm=w)
            L.append(f"    & {row} & " + " & ".join(cells) + r" \\")
        L.append(r"    \hline")

    def et_row(mode):
        metric = ET_ROW_METRIC[mode]
        cells = six(f"ET-{mode}", metric, "et", mode=mode)
        L.append(f"    {ET_LABEL[mode]}& ET & " + " & ".join(cells) + r" \\")
        # comment variants on the sibling pillar metric for the author
        sibling = {"fair": "fairEO", "adv": "rel"}.get(mode)
        if sibling:
            alt = []
            for ds in DATASETS:
                for k in KS:
                    if "imdb" in ds and sibling.startswith("fair"):
                        alt.append("--")
                    else:
                        alt.append(fmt(cell(data["et"][ds], k, sibling, mode=mode)))
            L.append(f"    % ET-{mode} on {sibling}: " + " & ".join(alt))
        L.append(r"    \hline")

    block("loss")
    block("acc")
    L[-1] = r"    \hline\hline"
    et_row("fair")
    block("fairDP")
    block("fairEO")
    L[-1] = r"    \hline\hline"
    et_row("adv")
    block("rel")
    block("res")
    L[-1] = r"    \hline\hline"
    et_row("dp")
    block("priv")
    L.append(r"\end{tabular}")

    pd.DataFrame(rows_long).to_csv("data/tab4_summary.csv", index=False)
    with open("tab4_filled.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    done = sum(1 for r in rows_long if r["n"] > 0 or np.isnan(r["mean"]))
    print(f"tab4_filled.tex written; cells filled: {done}/{len(rows_long)}")
    if missing:
        print(f"missing ({len(missing)}):")
        for m in missing[:30]:
            print("  ", m)
    # sanity: seeds per filled cell
    bad = [r for r in rows_long if 0 < r["n"] < SEEDS_EXPECTED]
    if bad:
        print(f"WARNING: {len(bad)} cells have <{SEEDS_EXPECTED} seeds:")
        for r in bad[:10]:
            print("  ", r["dataset"], r["K"], r["row"], r["metric"], "n=", r["n"])


if __name__ == "__main__":
    main()
