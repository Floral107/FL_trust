"""fold_in.py - fold fill CSVs into data/combo, writing ONLY where the combo cell is NaN.

Replaces the dead archive/oneoffs/merge_celeba.py path (its data/celeba_to_distribute
staging dir is a stale multi-host map). Never overwrites an existing value, so it
cannot move a published number - it can only turn a hole into a value. Idempotent.

  python fold_in.py            # dry run: report what would be filled
  python fold_in.py --apply    # back up combo/, then write

Paths are cwd-relative: run it from a run folder at the repo root, e.g.
  cd _direction_b && FLR_COMBO=../Code/data/combo_v2 python ../_fills_20260914/fold_in.py --apply

Two targets, never mixed:
  data/combo        (default) the shipped numbers: plain-mean coalitions, best-subset GTG.
                    Output of the fixed scorer (it has a coalition_avg column) is skipped.
  FLR_COMBO=<dir>   e.g. ../Code/data/combo_v2, for the fixed scorer only: files, rows and
                    columns are created as runs arrive; old-scorer output is skipped.
A kept cell whose incoming value differs is counted and reported, never overwritten.
"""
import pandas as pd, numpy as np, glob, os, re, sys, shutil, datetime

SHIPPED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Code", "data", "combo")
COMBO = os.environ.get("FLR_COMBO") or "../Code/data/combo"
# combo_v2 mode only on an explicit FLR_COMBO that isn't the shipped folder (by absolute path)
V2 = bool(os.environ.get("FLR_COMBO")) and \
    os.path.normcase(os.path.abspath(COMBO)) != os.path.normcase(os.path.abspath(SHIPPED))
KEY = ["num_clients", "seed", "round", "client_id"]
PAT = re.compile(r"^results_(adult|celeba|imdb)(noniid)?_(\d+)_((?:gtg|l1o)_\w+)_fedavg\.csv$")
APPLY = "--apply" in sys.argv
PER_METHOD = ("shapley", "shapley_m_a", "shapley_m_b", "contribution_m_a", "contribution_m_b")
GROUP = ("group_a", "n_group_a", "n_group_b")  # differ between dp and eop, so stored per method


def differing(old, new):
    """How many kept cells the incoming values disagree with."""
    try:
        return int((np.abs(old.to_numpy(dtype=float) - np.asarray(new, dtype=float)) > 1e-9).sum())
    except (TypeError, ValueError):
        return int((old.astype(str).to_numpy() != np.asarray(new).astype(str)).sum())


fills = {}
for fp in sorted(glob.glob("results_*_fedavg.csv")):
    m = PAT.match(os.path.basename(fp))
    if not m:
        print(f"  [skip] unrecognised name: {fp}"); continue
    base, noniid, seed, method = m.group(1), m.group(2) or "", m.group(3), m.group(4)
    fam = "gtg" if method.startswith("gtg") else "l1o"
    fills.setdefault(f"{base}{noniid}_{fam}", []).append((fp, method))

if APPLY:
    os.makedirs(COMBO, exist_ok=True)
    bak = f"{COMBO}_bak_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    shutil.copytree(COMBO, bak); print(f"backup -> {os.path.abspath(bak)}\n")

total = conflicts = 0
for target, items in sorted(fills.items()):
    tgt = f"{COMBO}/results_{target}.csv"
    srcs = {}
    for fp, method in items:
        src = pd.read_csv(fp)
        if ("coalition_avg" in src.columns) != V2:
            where = "FLR_COMBO=../Code/data/combo_v2" if V2 is False else "data/combo"
            print(f"  [skip] {fp}: written by the {'fixed' if not V2 else 'old'} scorer; it belongs in {where}")
            continue
        srcs[fp] = (method, src)
    if not srcs:
        continue
    if V2:
        keys = pd.concat([s[["dataset", "partition"] + KEY] for _, s in srcs.values()]).drop_duplicates(KEY)
        if os.path.exists(tgt):
            combo = pd.read_csv(tgt)
            new = keys[~keys.set_index(KEY).index.isin(combo.set_index(KEY).index)]
            combo = pd.concat([combo, new], ignore_index=True)
        else:
            combo = keys.reset_index(drop=True)
    elif os.path.exists(tgt):
        combo = pd.read_csv(tgt)
    else:
        print(f"[MISS] no combo file {tgt}"); continue
    filled_here = 0
    for fp, (method, src) in sorted(srcs.items()):
        # combo and fill share column names: global_<method>, <method>_contribution.
        # 'acc' additionally has a bare acc_contribution that the table builders
        # prefer (compute_score_diff/fluct pick it first), so fill that too.
        pairs = [(f"global_{method}", f"global_{method}"),
                 (f"{method}_contribution", f"{method}_contribution")]
        if method.endswith("_acc"):
            pairs.append((f"{method}_contribution", "acc_contribution"))
        if V2:
            pairs += [(c, c) for c in [f"{method}_{e}" for e in PER_METHOD]
                      + [f"global_{method}_m_a", f"global_{method}_m_b", "coalition_avg"] if c in src.columns]
            pairs += [(c, f"{method}_{c}") for c in GROUP if c in src.columns]
        idx = combo.set_index(KEY).index
        s = src.set_index(KEY)
        for src_col, dst_col in pairs:
            if src_col not in src.columns:
                print(f"  [warn] {fp}: no column {src_col}"); continue
            if dst_col not in combo.columns:
                if not V2:
                    print(f"  [warn] {tgt}: no column {dst_col}"); continue
                combo[dst_col] = float("nan")  # combo_v2 grows columns as methods arrive
            vals = s[src_col].reindex(idx)
            # measure BEFORE writing, else every cell we just filled reads back
            # as "already present" and the report is meaningless
            hole = combo[dst_col].isna() & vals.notna().to_numpy()
            both = combo[dst_col].notna() & vals.notna().to_numpy()
            overlap = int(both.sum())
            differ = differing(combo.loc[both, dst_col], vals.to_numpy()[both.to_numpy()]) if overlap else 0
            conflicts += differ
            n = int(hole.sum())
            if n:
                if combo[dst_col].dtype != vals.dtype:
                    combo[dst_col] = combo[dst_col].astype(object)
                combo.loc[hole, dst_col] = vals.to_numpy()[hole.to_numpy()]
                filled_here += n
            print(f"  {os.path.basename(fp):52s} -> {dst_col:32s} fill {n:4d} | kept {overlap:4d}"
                  + (f" | DIFFERENT {differ}" if differ else ""))
    if filled_here:
        total += filled_here
        if APPLY:
            combo.to_csv(tgt, index=False)
            print(f"[WROTE] {tgt}: {filled_here} cells\n")
        else:
            print(f"[DRY]   {tgt}: would fill {filled_here} cells\n")

print(f"{'FILLED' if APPLY else 'WOULD FILL'} {total} cells total")
if conflicts:
    print(f"[WARN] {conflicts} kept cells differ from the incoming values; the old values were kept")
if not APPLY:
    print("(dry run - pass --apply to write)")
