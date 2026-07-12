"""_run_figs.py — regenerate the article figures locally from data/combo/ using
article_figs.ipynb's plotting code (already advisor-spec-compliant). Execs all
code cells in order (skipping the Colab download cell + bare functions() call),
then runs both gtg + l1o. Output -> data/combo/plots/.
"""
import matplotlib
matplotlib.use("Agg")
import json, os, traceback

REPO = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(REPO, "article_figs.ipynb"), encoding="utf-8"))
code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

os.chdir(os.path.join(REPO, "data", "combo"))
os.makedirs("plots", exist_ok=True)

g = {}
for src in code_cells:
    if "google.colab" in src or "!zip" in src:      # skip Colab download cell
        continue
    src = "\n".join(l for l in src.split("\n") if l.strip() != "functions()")
    try:
        exec(src, g)
    except Exception as e:
        print(f"[cell exec ERR] {e}")

# --- column-resolution fix --------------------------------------------------
# The notebook concatenates ALL datasets into one df, so the union of columns
# contains both the bare 'acc_contribution'/'loss_contribution' (cifar) and the
# method-prefixed 'gtg_acc_contribution'/'gtg_loss_contribution' (adult/imdb/
# celeba). The original contrib_columns picked the FIRST *present* candidate,
# which for adult/imdb is the bare column that is all-NaN there -> acc/loss were
# silently skipped. Re-bind it to pick the first candidate that is present AND
# actually has data, so every metric (incl. acc & loss) renders for every dataset.
if "_contrib_candidates" in g:
    _cand = g["_contrib_candidates"]
    def contrib_columns(df, method, include_fair=True):
        out = []
        for key, cands in _cand(method):
            if not include_fair and key in ("fairDP", "fairEO"):
                continue
            col = next((c for c in cands if c in df.columns and df[c].notna().any()), None)
            if col is not None:
                out.append((key, col))
        return out
    g["contrib_columns"] = contrib_columns

FUNCS = ["global_metric_evolutions", "imdb_boxplots_3metrics_4cli",
         "adult_celeba_boxplots_4metrics_4cli", "imdb_boxplots_4metrics_20cli",
         "adult_celeba_boxplots_4metrics_20cli", "generate_heatmaps",
         "distribution_scores"]
ok, err = 0, 0
for m in ["gtg", "l1o"]:
    g["METHOD"] = m
    for fn in FUNCS:
        try:
            g[fn](method=m); ok += 1
        except Exception as e:
            err += 1; print(f"ERR {fn} {m}: {e}")
print(f"\nDONE: {ok} fn-runs ok, {err} errors. PNGs: {len([f for f in os.listdir('plots') if f.endswith('.png')])}")
