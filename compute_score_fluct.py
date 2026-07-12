"""compute_score_fluct.py — regenerate tab:score_fluct FRESH from data/combo (GTG).
Round-wise fluctuation = std of each client's contribution across SCORING rounds
(2-10; round 1 is a pre-stabilization artifact), averaged over clients -> per-seed
value; cell = mean +/- std over seeds. imdb fairDP/fairEO -> --. Emits the table
body to tab_scorefluct_new.tex (text left unchanged per user)."""
import pandas as pd, numpy as np, os
METHOD = "gtg"
RMIN = 2
METRICS = [("acc", "acc"), ("loss", "loss"), ("fairDP", "fair"), ("fairEO", "fair_eo"),
           ("rel", "rob"), ("res", "adv"), ("priv", "priv")]
DS = [("ADULT", "adult"), ("CelebA", "celeba"), ("IMDB", "imdb")]
COMBO = "data/combo"

def pick(df, *c):
    for x in c:
        if x in df.columns and df[x].notna().any():
            return x
    return None

def cell(ds_base, K, part, lbl, suf):
    fn = f"{COMBO}/results_{ds_base}{'noniid' if part=='noniid' else ''}_{METHOD}.csv"
    if not os.path.exists(fn): return None
    df = pd.read_csv(fn)
    df = df[(df["num_clients"] == K) & (df["partition"] == part) & (df["round"] >= RMIN)]
    if df.empty: return None
    if "imdb" in ds_base and lbl in ("fairDP", "fairEO"): return None
    if lbl == "acc":    col = pick(df, "acc_contribution", f"{METHOD}_acc_contribution")
    elif lbl == "loss": col = pick(df, f"{METHOD}_loss_contribution", "loss_contribution")
    else:               col = pick(df, f"{METHOD}_{suf}_contribution", f"{suf}_contribution")
    if col is None: return None
    per_seed = []
    for s, g in df.groupby("seed"):
        stds = [np.std(pd.to_numeric(gc.sort_values("round")[col], errors="coerce").dropna())
                for cid, gc in g.groupby("client_id")
                if pd.to_numeric(gc[col], errors="coerce").notna().sum() >= 2]
        if stds: per_seed.append(np.mean(stds))
    if not per_seed: return None
    return float(np.mean(per_seed)), float(np.std(per_seed))

def sci(v):
    if v is None or v < 1e-7: return "0.0"
    mant, exp = f"{v:.1e}".split("e")
    return f"{mant}$e${int(exp)}"

def fmt(c):
    return "--" if c is None else f"{sci(c[0])}$\\pm${sci(c[1])}"

def row(ds_base, K, part):
    return " & ".join(fmt(cell(ds_base, K, part, lbl, suf)) for lbl, suf in METRICS)

L = []
for di, (dname, dbase) in enumerate(DS):
    L.append(rf"        \multirow{{4}}{{*}}{{\rotatebox{{90}}{{{dname}}}}} & \multirow{{2}}{{*}}{{4}} & I & {row(dbase,4,'iid')} \\")
    L.append(rf"        && N & {row(dbase,4,'noniid')} \\")
    L.append(r"        \cline{2-10}")
    L.append(rf"        & \multirow{{2}}{{*}}{{20}} & I & {row(dbase,20,'iid')} \\")
    L.append(rf"        && N & {row(dbase,20,'noniid')} \\")
    if di < len(DS) - 1:
        L.append(r"        \hline")
open("tab_scorefluct_new.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote tab_scorefluct_new.tex")
print("sanity ADULT K4: I:", row("adult",4,"iid")[:90])
print("sanity CelebA K4 I:", row("celeba",4,"iid")[:90])
print("sanity IMDB K20 N:", row("imdb",20,"noniid")[:90])
