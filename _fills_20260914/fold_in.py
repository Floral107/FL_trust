"""fold_in.py - fold fill CSVs into data/combo, writing ONLY where the combo cell is NaN.

Replaces the dead archive/oneoffs/merge_celeba.py path (its data/celeba_to_distribute
staging dir is a stale multi-host map). Never overwrites an existing value, so it
cannot move a published number - it can only turn a hole into a value. Idempotent.

  python fold_in.py            # dry run: report what would be filled
  python fold_in.py --apply    # back up combo/, then write
"""
import pandas as pd, glob, os, re, sys, shutil, datetime

COMBO = "../Code/data/combo"
KEY = ["num_clients", "seed", "round", "client_id"]
PAT = re.compile(r"^results_(adult|celeba|imdb)(noniid)?_(\d+)_((?:gtg|l1o)_\w+)_fedavg\.csv$")
APPLY = "--apply" in sys.argv

fills = {}
for fp in sorted(glob.glob("results_*_fedavg.csv")):
    m = PAT.match(os.path.basename(fp))
    if not m:
        print(f"  [skip] unrecognised name: {fp}"); continue
    base, noniid, seed, method = m.group(1), m.group(2) or "", m.group(3), m.group(4)
    fam = "gtg" if method.startswith("gtg") else "l1o"
    fills.setdefault(f"{base}{noniid}_{fam}", []).append((fp, method))

if APPLY:
    bak = f"{COMBO}_bak_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    shutil.copytree(COMBO, bak); print(f"backup -> {os.path.abspath(bak)}\n")

total = 0
for target, items in sorted(fills.items()):
    tgt = f"{COMBO}/results_{target}.csv"
    if not os.path.exists(tgt):
        print(f"[MISS] no combo file {tgt}"); continue
    combo = pd.read_csv(tgt)
    filled_here = 0
    for fp, method in sorted(items):
        src = pd.read_csv(fp)
        # combo and fill share column names: global_<method>, <method>_contribution.
        # 'acc' additionally has a bare acc_contribution that the table builders
        # prefer (compute_score_diff/fluct pick it first), so fill that too.
        pairs = [(f"global_{method}", f"global_{method}"),
                 (f"{method}_contribution", f"{method}_contribution")]
        if method.endswith("_acc"):
            pairs.append((f"{method}_contribution", "acc_contribution"))
        idx = combo.set_index(KEY).index
        s = src.set_index(KEY)
        for src_col, dst_col in pairs:
            if src_col not in src.columns:
                print(f"  [warn] {fp}: no column {src_col}"); continue
            if dst_col not in combo.columns:
                print(f"  [warn] {tgt}: no column {dst_col}"); continue
            vals = s[src_col].reindex(idx)
            # measure BEFORE writing, else every cell we just filled reads back
            # as "already present" and the report is meaningless
            hole = combo[dst_col].isna() & vals.notna().to_numpy()
            overlap = int((combo[dst_col].notna() & vals.notna().to_numpy()).sum())
            n = int(hole.sum())
            if n:
                combo.loc[hole, dst_col] = vals.to_numpy()[hole.to_numpy()]
                filled_here += n
            print(f"  {os.path.basename(fp):52s} -> {dst_col:28s} fill {n:4d} | kept {overlap:4d}")
    if filled_here:
        total += filled_here
        if APPLY:
            combo.to_csv(tgt, index=False)
            print(f"[WROTE] {tgt}: {filled_here} cells\n")
        else:
            print(f"[DRY]   {tgt}: would fill {filled_here} cells\n")

print(f"{'FILLED' if APPLY else 'WOULD FILL'} {total} cells total")
if not APPLY:
    print("(dry run - pass --apply to write)")
