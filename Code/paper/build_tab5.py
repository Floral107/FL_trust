"""build_tab5.py — regenerate the tab:util body (MAIN.tex Tab.5) from
data/{bl,st,dy,et}. Replaces the box-era generate_thesis_table.py (lost with
the box) with the minimal equivalent.

Cell = mean +/- std over seeds of the round-10 model. ST/DY rows use the
DIAGONAL (weight_metric == eval metric). ET rows: fair->fairDP, adv->res,
dp->priv. Display: loss = exp(-CE) (higher better, matches existing table);
all other metrics raw. imdb fair columns are '--'.

Output: tab5_new.tex (body only). Also prints a diff vs the values currently
in MAIN.tex so intended changes are auditable. Run after sanity_gate.py.
"""
import numpy as np
import pandas as pd

DS = ["adultnoniid", "celebanoniid", "imdbnoniid"]
KS = [4, 20]
SEEDS = {42, 107, 123, 2025, 9928}

def load(fam, ds):
    d = pd.read_csv(f"data/{fam}/results_{ds}_{fam}.csv")
    return d[d.seed.isin(SEEDS)]

def cell(vals, transform=None):
    v = pd.to_numeric(vals, errors="coerce").dropna()
    if len(v) == 0:
        return "--"
    if transform:
        v = transform(v)
    return f"${v.mean():.2f} \\pm {v.std(ddof=0):.2f}$"

def stdy_cell(fam, ds, K, metric, wm):
    d = load(fam, ds)
    r = d[(d.num_clients == K) & (d.weight_metric.str.lower() == wm.lower())]
    if metric in ("fairDP", "fairEO") and "imdb" in ds:
        return "--"
    return cell(r[metric], np.exp(-pd.to_numeric(r[metric])) if False else None) \
        if metric != "loss" else cell(r[metric], lambda v: np.exp(-v))

def bl_cell(ds, K, metric):
    d = load("bl", ds)
    r = d[d.num_clients == K]
    if metric in ("fairDP", "fairEO") and "imdb" in ds:
        return "--"
    return cell(r[metric], lambda v: np.exp(-v)) if metric == "loss" else cell(r[metric])

def et_cell(ds, K, mode, metric):
    if "imdb" in ds and mode == "fair":
        return "--"
    d = load("et", ds)
    r = d[(d.num_clients == K) & (d["mode"] == mode)]
    return cell(r[metric])

def row(fn, *args):
    return " & ".join(fn(ds, K, *args) for ds in DS for K in KS)

L = []
def block(metric, wm):
    L.append(f"    \\multirow{{3}}{{*}}{{$\\mathtt{{{metric}}}$}}")
    L.append(f"    & BL & {row(bl_cell, metric)} \\\\")
    L.append(f"    & ST & {row(lambda d,k: stdy_cell('st', d, k, metric, wm))} \\\\")
    L.append(f"    & DY & {row(lambda d,k: stdy_cell('dy', d, k, metric, wm))} \\\\")

block("loss", "loss"); L.append("    \\hline")
block("acc", "acc"); L.append("    \\hline\\hline")
L.append(f"    Fair. Reg.& ET & {row(et_cell, 'fair', 'fairDP')} \\\\")
L.append(f"    % ET-fair on fairEO: {row(et_cell, 'fair', 'fairEO')}")
L.append("    \\hline")
block("fairDP", "fairDP"); L.append("    \\hline")
block("fairEO", "fairEO"); L.append("    \\hline\\hline")
L.append(f"    Rob. Train.& ET & {row(et_cell, 'adv', 'res')} \\\\")
L.append(f"    % ET-adv on rel: {row(et_cell, 'adv', 'rel')}")
L.append("    \\hline")
block("rel", "rel"); L.append("    \\hline")
block("res", "res"); L.append("    \\hline\\hline")
L.append(f"    Diff. Priv.& ET & {row(et_cell, 'dp', 'priv')} \\\\")
L.append("    \\hline")
block("priv", "priv"); L.append("    \\hline")

open("tab5_new.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"wrote tab5_new.tex ({len(L)} lines)")
