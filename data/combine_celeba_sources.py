"""Combine celeba per-(seed,method,metric) result files from all sources
(local celeba_to_distribute + A40 csv_done + A40 csv_root), union+dedupe by
(num_clients, round, client_id), write the most-complete version back into
celeba_to_distribute/, and report exactly which combos are still incomplete.

Target per file: 240 data rows (K4: 4 clients x10 rounds=40 + K20: 20x10=200).
"""
import pandas as pd, glob, os, re
from collections import defaultdict

SRCS = ["celeba_to_distribute", "_a40_csv_done", "_a40_csv_root"]
PAT = re.compile(r"results_(celeba(?:noniid)?)_(\d+)_(gtg|l1o)_([a-z_]+?)_fedavg\.csv$")
DEDUPE_KEY = ["num_clients", "seed", "round", "client_id"]
METRICS = ["acc", "loss", "fair", "fair_eo", "rob", "adv_pgd", "priv"]
SEEDS = [42, 107, 123, 2025, 9928]
TARGET = 240

groups = defaultdict(list)
for d in SRCS:
    for fp in glob.glob(os.path.join(d, "*.csv")):
        m = PAT.match(os.path.basename(fp))
        if not m:
            continue
        ds, seed, method, metric = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if len(df):
            groups[(ds, seed, method, metric)].append(df)

os.makedirs("celeba_to_distribute", exist_ok=True)
rows = {}
for (ds, seed, method, metric), dfs in groups.items():
    alldf = pd.concat(dfs, ignore_index=True)
    keys = [k for k in DEDUPE_KEY if k in alldf.columns]
    alldf = alldf.drop_duplicates(subset=keys)
    out = f"celeba_to_distribute/results_{ds}_{seed}_{method}_{metric}_fedavg.csv"
    alldf.to_csv(out, index=False)
    rows[(ds, seed, method, metric)] = len(alldf)

# Gap report
print("=== STILL-INCOMPLETE combos (rows < 240) ===")
need = defaultdict(list)
for ds in ["celeba", "celebanoniid"]:
    for method in ["gtg", "l1o"]:
        for seed in SEEDS:
            for metric in METRICS:
                n = rows.get((ds, seed, method, metric), 0)
                if n < TARGET:
                    need[(ds, method)].append((seed, metric, n))
for (ds, method), items in sorted(need.items()):
    print(f"\n{ds} {method}: {len(items)} incomplete")
    # group by seed for readability
    byseed = defaultdict(list)
    for seed, metric, n in items:
        byseed[seed].append(f"{metric}={n}")
    for seed in SEEDS:
        if seed in byseed:
            print(f"  seed {seed}: {', '.join(byseed[seed])}")
if not need:
    print("  NONE — all complete!")
print(f"\n=== total per-file combos held: {len(rows)} ===")
