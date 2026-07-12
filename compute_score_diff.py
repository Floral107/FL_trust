"""compute_score_diff.py — regenerate tab:score_diff_merged FRESH from data/combo
(round-10 contributions), per seed -> mean +/- std. Phi = Spearman(metric_contrib,
loss_contrib) over clients; L2 = RMSE(metric_contrib, loss_contrib) over clients.
Phi cell = mean+/-std over seeds with a defined Spearman, else \textemdash.
Outputs the LaTeX body (dataset blocks) to tab_scorediff_new.tex.
"""
import pandas as pd, numpy as np, os
from scipy.stats import spearmanr

COMBO = "data/combo"
METRICS = [("acc", "acc"), ("fairDP", "fair"), ("fairEO", "fair_eo"),
           ("rel", "rob"), ("res", "adv"), ("priv", "priv")]
DS = [("ADULT", "adult"), ("CelebA", "celeba"), ("IMDB", "imdb")]

def stats(ds_base, K, part, method):
    fn = f"{COMBO}/results_{ds_base}{'noniid' if part=='noniid' else ''}_{method}.csv"
    if not os.path.exists(fn):
        return {}
    df = pd.read_csv(fn)
    df = df[(df["num_clients"] == K) & (df["partition"] == part)]
    if df.empty:
        return {}
    df = df[df["round"] == df["round"].max()]
    cols = set(df.columns)
    def pick(*cands):
        for c in cands:
            if c in cols:
                return c
        return None
    # combo naming differs by dataset: adult/imdb use loss_contribution/acc_contribution
    # (no prefix); celeba (merge_celeba) uses gtg_loss_contribution etc. Try both.
    loss_c = pick(f"{method}_loss_contribution", "loss_contribution")
    out = {}
    for lbl, suf in METRICS:
        if "imdb" in ds_base and lbl in ("fairDP", "fairEO"):
            out[lbl] = None; continue
        if lbl == "acc":
            met_c = pick("acc_contribution", f"{method}_acc_contribution")
        else:
            met_c = pick(f"{method}_{suf}_contribution", f"{suf}_contribution")
        if met_c is None or loss_c is None:
            out[lbl] = None; continue
        # Caption scheme: PRIMARY Phi/L2 computed on SEED-AVERAGED contributions
        # (matches the visual heatmaps); the +- std comes from per-seed values.
        cid = pick("client_id", "client", "cid", "partition_id")
        phis, l2s, Ms, Ls = [], [], [], []
        for s, g in df.groupby("seed"):
            if cid: g = g.sort_values(cid)
            m = pd.to_numeric(g[met_c], errors="coerce").to_numpy()
            l = pd.to_numeric(g[loss_c], errors="coerce").to_numpy()
            # loss in performance (-L) convention applied at DATA level (combo), uniform gtg+l1o
            ok = ~(np.isnan(m) | np.isnan(l)); m, l = m[ok], l[ok]
            if len(m) < 2: continue
            Ms.append(m); Ls.append(l)
            l2s.append(float(np.sqrt(np.mean((m - l) ** 2))))
            if np.std(m) > 1e-12 and np.std(l) > 1e-12:
                phi = spearmanr(m, l).correlation
                if not np.isnan(phi): phis.append(phi)
        phiA, l2A = float("nan"), float("nan")
        if Ms and len({len(x) for x in Ms}) == 1:
            mA = np.mean(Ms, axis=0); lA = np.mean(Ls, axis=0)
            l2A = float(np.sqrt(np.mean((mA - lA) ** 2)))
            if np.std(mA) > 1e-12 and np.std(lA) > 1e-12:
                p = spearmanr(mA, lA).correlation
                if not np.isnan(p): phiA = float(p)
        out[lbl] = {"phiA": phiA, "l2A": l2A, "phi": phis, "l2": l2s}
    return out

def cell_phi(d):
    if d is None: return "\\textemdash"
    if np.isnan(d["phiA"]): return "\\textemdash"
    sd = np.std(d["phi"]) if d["phi"] else 0.0
    return f"${d['phiA']:.2f} \\pm {sd:.2f}$"

def cell_l2(d):
    if d is None: return "\\textemdash"
    if np.isnan(d["l2A"]): return "\\textemdash"
    sd = np.std(d["l2"]) if d["l2"] else 0.0
    return f"${d['l2A']:.2f} \\pm {sd:.2f}$"

def row_cells(ds_base, K, part, method):
    st = stats(ds_base, K, part, method)
    cells = []
    for lbl, _ in METRICS:
        d = st.get(lbl)
        cells.append(cell_phi(d)); cells.append(cell_l2(d))
    return " & ".join(cells)

L = []
for di, (dname, dbase) in enumerate(DS):
    L.append(rf"        \multirow{{8}}{{*}}{{\rotatebox{{90}}{{{dname}}}}} & \multirow{{4}}{{*}}{{4}} & \multirow{{2}}{{*}}{{\rotatebox{{90}}{{IID}}}} &")
    L.append(rf"        LOO & {row_cells(dbase,4,'iid','l1o')} \\")
    L.append(rf"        &&& GTG & {row_cells(dbase,4,'iid','gtg')} \\")
    L.append(r"        \cline{3-16}")
    L.append(r"        && \multirow{2}{*}{\rotatebox{90}{NIID}} &")
    L.append(rf"        LOO & {row_cells(dbase,4,'noniid','l1o')} \\")
    L.append(rf"        &&& GTG & {row_cells(dbase,4,'noniid','gtg')} \\")
    L.append(r"        \cline{2-16}")
    L.append(r"        & \multirow{4}{*}{20} & \multirow{2}{*}{\rotatebox{90}{IID}} &")
    L.append(rf"        LOO & {row_cells(dbase,20,'iid','l1o')} \\")
    L.append(rf"        &&& GTG & {row_cells(dbase,20,'iid','gtg')} \\")
    L.append(r"        \cline{3-16}")
    L.append(r"        && \multirow{2}{*}{\rotatebox{90}{NIID}} &")
    L.append(rf"        LOO & {row_cells(dbase,20,'noniid','l1o')} \\")
    L.append(rf"        &&& GTG & {row_cells(dbase,20,'noniid','gtg')} \\")
    L.append(r"        \hline" if di < len(DS) - 1 else "")
open("tab_scorediff_new.tex", "w", encoding="utf-8").write("\n".join(l for l in L if l != "") + "\n")
print("wrote tab_scorediff_new.tex")
print("=== sanity: adult NIID K4 ===")
for m in ["l1o", "gtg"]:
    print(m, row_cells("adult", 4, "noniid", m)[:160])
