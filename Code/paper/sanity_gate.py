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
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEEDS = [42, 107, 123, 2025, 9928]

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


def coverage(strict: bool):
    """Report per-(dataset, metric, K) seed coverage at the final round.

    Every table/figure builder aggregates with a plain pandas mean, which drops
    NaN silently -- so a cell backed by 2 seeds renders identically to one backed
    by 5. Nothing else in the pipeline counts them. This does.

    Returns the list of under-covered cells. Warn-only unless --strict.
    """
    print("\n" + "=" * 72)
    print("COVERAGE - seeds contributing to each final-round cell (expected 5)")
    print("=" * 72)
    thin = []
    for path in sorted(glob.glob("data/combo/*.csv")):
        base = os.path.basename(path)
        method = "gtg" if "_gtg" in base else "l1o"
        df = pd.read_csv(path)
        df = df[df["round"] == df["round"].max()]
        bad = []
        for col in df.columns:
            m = re.fullmatch(rf"{method}_(.+)_contribution", col)
            if not m or m.group(1).endswith("_abs"):
                continue
            metric = m.group(1)
            for K, g in df.groupby("num_clients"):
                v = pd.to_numeric(g[col], errors="coerce")
                n = int(g.loc[v.notna(), "seed"].nunique())
                # A present-but-constant column (GTG truncation returns exact 0.0
                # for every client when coalition utilities barely move) counts as
                # covered, yet compute_score_diff drops it: Spearman needs std>0.
                # So a 5/5 cell can still build Phi from 3 seeds. Count both.
                nvar = 0
                for _s, gs in g.groupby("seed"):
                    vs = pd.to_numeric(gs[col], errors="coerce").to_numpy()
                    if not np.isnan(vs).all() and np.nanstd(vs) > 1e-12:
                        nvar += 1
                if n < len(SEEDS) or nvar < n:
                    bad.append((metric, int(K), n, nvar))
                if n < len(SEEDS):  # only real holes gate --strict; zero-variance (A32) is unfixable, report-only
                    thin.append((base, metric, int(K), n))
        if bad:
            print(f"  {base}")
            for metric, K, n, nvar in bad:
                note = "   <-- EMPTY" if n == 0 else ""
                if nvar < n:
                    note += f"   [{n - nvar} seed(s) zero-variance -> only {nvar} enter Phi]"
                print(f"      {metric:10s} K={K:<3d} {n}/{len(SEEDS)} seeds" + note)
    if not thin:
        print("  all cells backed by all 5 seeds")
    else:
        print(f"\n  {len(thin)} under-covered cell(s). Means over these are NOT 5-seed means.")
        print("  " + ("STRICT: refusing to regenerate." if strict
                      else "Warning only; pass --strict to make this fail the gate."))
    return thin


FAMILY_METRICS = ["acc", "loss", "fair", "fair_eo", "rel", "res", "priv"]


def coverage_families(strict: bool):
    """Same coverage question for the Tab. 5 families (data/{bl,st,dy,et}).

    coverage() above only reads data/combo, which backs Tab. 3/4. Tab. 5 is built
    from a different layer that nothing counted -- so an ST/DY cell missing a seed
    rendered as a 5-seed mean exactly like a combo one did.
    """
    print()
    print("=" * 72)
    print("COVERAGE - Tab.5 families data/{bl,st,dy,et} (expected 5 seeds)")
    print("=" * 72)
    thin = []
    for path in sorted(glob.glob("data/*/results_*.csv")):
        fam = os.path.basename(os.path.dirname(path))
        if fam not in ("bl", "st", "dy", "et"):
            continue
        base = fam + "/" + os.path.basename(path)
        df = pd.read_csv(path)
        if "seed" not in df.columns:
            continue
        # rows are keyed by K plus whichever variant column the family carries
        keys = ["num_clients"] + [c for c in ("weight_metric", "mode") if c in df.columns]
        bad = []
        for metric in [m for m in FAMILY_METRICS if m in df.columns]:
            for key, g in df.groupby(keys):
                v = pd.to_numeric(g[metric], errors="coerce")
                n = int(g.loc[v.notna(), "seed"].nunique())
                if n < len(SEEDS):
                    label = key if isinstance(key, tuple) else (key,)
                    bad.append((metric, label, n))
                    thin.append((base, metric, label, n))
        if bad:
            print("  " + base)
            for metric, label, n in bad:
                loc = " ".join(str(x) for x in label)
                tag = "   <-- EMPTY" if n == 0 else ""
                print(f"      {metric:8s} {loc:20s} {n}/{len(SEEDS)} seeds" + tag)
    if not thin:
        print("  all cells backed by all 5 seeds")
    else:
        print()
        print(f"  {len(thin)} under-covered cell(s) in the Tab.5 layer.")
        print("  " + ("STRICT: refusing to regenerate." if strict
                      else "Warning only; pass --strict to make this fail the gate."))
    return thin


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="also fail the gate when a cell is not backed by all 5 seeds")
    args = ap.parse_args()

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
    thin = coverage(args.strict)
    thin = thin + coverage_families(args.strict)

    if failed:
        print("\nGATE FAILED — loss-sign inversion suspected in:",
              ", ".join(f"{f}/{m}" for f, m in failed))
        print("Do NOT regenerate figures/tables. Check the offending combo CSVs' "
              "loss_contribution sign against cont_evals.py's convention.")
        return 1
    if thin and args.strict:
        print("\nGATE FAILED — incomplete coverage (--strict).")
        return 1
    print("\nGATE PASSED — safe to regenerate figures and tables.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
