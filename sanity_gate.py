"""sanity_gate.py — run BEFORE regenerating any figure or table.

Advisor's baseline-inversion check: for every dataset, the acc~loss Spearman
correlation of client contributions must be POSITIVE under both LOO and GTG.
Two metrics that both capture predictive quality must agree; a negative value
means a loss-sign inversion somewhere upstream (the bug class that hit LOO in
June and celeba-GTG in July 2026).

Convention enforced (see cont_evals.py evaluate_round): every *_contribution
column in data/combo is higher=better, INCLUDING loss. Sign logic lives in
cont_evals.py only — never flip signs in merge/collect/plot scripts.

Pass rule: per (dataset-family, method), the mean of acc~loss phi across
settings (partition x K, seed-averaged, final round) must be >= 0. Individual
cells may be negative (K=4 Spearman over 4 rank points is noisy); a negative
MEAN is systematic and fails the gate. Degenerate (constant) score vectors are
skipped. Datasets not in the paper (cifar) are reported but not gating.

Exit 0 = all pass; exit 1 = inversion detected (regeneration must not proceed).
Run: python sanity_gate.py
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PAPER = ["adult", "celeba", "imdb"]          # gating families
EXTRA = ["cifar"]                            # reported only
METHODS = ["l1o", "gtg"]

def accloss_phi(path, method):
    """Seed-averaged acc~loss Spearman per K in the final round; None if degenerate."""
    df = pd.read_csv(path)
    lc = f"{method}_loss_contribution" if f"{method}_loss_contribution" in df.columns else "loss_contribution"
    ac = "acc_contribution" if "acc_contribution" in df.columns else f"{method}_acc_contribution"
    d = df[df["round"] == df["round"].max()]
    out = {}
    for K, g0 in d.groupby("num_clients"):
        Ms, Ls = [], []
        for _, g in g0.groupby("seed"):
            g = g.sort_values("client_id")
            m = pd.to_numeric(g[ac], errors="coerce").to_numpy()
            l = pd.to_numeric(g[lc], errors="coerce").to_numpy()
            ok = ~(np.isnan(m) | np.isnan(l))
            m, l = m[ok], l[ok]
            if len(m) >= 2:
                Ms.append(m); Ls.append(l)
        if not Ms or len({len(x) for x in Ms}) != 1:
            out[K] = None; continue
        mA, lA = np.mean(Ms, axis=0), np.mean(Ls, axis=0)
        if np.std(mA) <= 1e-12 or np.std(lA) <= 1e-12:
            out[K] = None
        else:
            out[K] = float(spearmanr(mA, lA).correlation)
    return out

def main():
    failed = []
    for fam in PAPER + EXTRA:
        for method in METHODS:
            cells = []
            for ds in [fam, fam + "noniid"]:
                try:
                    phis = accloss_phi(f"data/combo/results_{ds}_{method}.csv", method)
                except FileNotFoundError:
                    continue
                for K, phi in sorted(phis.items()):
                    cells.append((ds, K, phi))
            vals = [p for _, _, p in cells if p is not None]
            mean = np.mean(vals) if vals else float("nan")
            gating = fam in PAPER
            ok = (not gating) or (not vals) or mean >= 0
            status = "PASS" if ok else "FAIL"
            if not gating:
                status = "info"
            detail = "  ".join(
                f"{ds}/K{K}={'degen' if p is None else f'{p:+.2f}'}" for ds, K, p in cells)
            print(f"[{status}] {fam:6s} {method}: mean acc~loss phi = {mean:+.2f}   ({detail})")
            if not ok:
                failed.append((fam, method))
    if failed:
        print("\nGATE FAILED — loss-sign inversion suspected in:",
              ", ".join(f"{f}/{m}" for f, m in failed))
        print("Do NOT regenerate figures/tables. Check the offending combo CSVs' "
              "loss_contribution sign against cont_evals.py's convention.")
        return 1
    print("\nGATE PASSED — safe to regenerate figures and tables.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
