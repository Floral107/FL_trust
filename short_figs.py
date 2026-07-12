"""Generate the short-paper figures for FedKDD.tex into figs/short/.

Produces only the subset of figures the conference paper needs:
  - Client Scores  : 4 clients, non-IID    -> S_{A,C}_4_N_{l1o,gtg}.png
  - Score Dist     : 20 clients, non-IID   -> SD_20_{A,C}_N_{l1o,gtg}.png
  - Heat Map       : 4 clients, non-IID    -> H_4_{A,C}_N_{l1o,gtg}.png
  - Legend                                  -> S_legend.png

Metrics shown: perf, fair, rel, res (privacy and fair_eo dropped, IMDB dropped).

Expects merged CSVs `results_<dataset>_<method>.csv` in the working
directory (same format article_figs.ipynb uses). Copy them next to this
script before running, or edit CSV_DIR below.
"""

import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


CSV_DIR = "."
OUT_DIR = os.path.join("figs", "short")
os.makedirs(OUT_DIR, exist_ok=True)


DOMAIN_COLORS = {
    "perf": "#1f77b4",
    "fair": "#1b7837",
    "rel":  "#b2182b",
    "res":  "#ef8a62",
}

DATASETS = ["adult", "celeba"]
DATASET_CODE = {"adult": "A", "celeba": "C"}


def _read(method):
    """Load merged CSVs for `method` ('gtg' or 'l1o') across adult and celeba.
    Tolerates missing files: warns and skips."""
    frames = []
    for base in DATASETS:
        for suffix in ("", "noniid"):
            path = os.path.join(CSV_DIR, f"results_{base}{suffix}_{method}.csv")
            if not os.path.exists(path):
                print(f"[warn] missing {path}")
                continue
            d = pd.read_csv(path)
            d["dataset"] = d["dataset"].astype(str).str.strip().str.lower()
            d["partition"] = d["partition"].astype(str).str.strip().str.lower()
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"No results_*_{method}.csv found in {CSV_DIR}")
    return pd.concat(frames, ignore_index=True)


def _short_cols(method):
    return {
        "perf": "acc_contribution",
        "fair": f"{method}_fair_contribution",
        "rel":  f"{method}_rob_contribution",
        "res":  f"{method}_adv_contribution",
    }


def _palette(method):
    cols = _short_cols(method)
    return {cols[k]: DOMAIN_COLORS[k] for k in cols}


def _color_box_borders(ax):
    """Color box edges + whisker/cap/median lines so shrunk boxes (20 clients)
    stay visible."""
    boxes = [p for p in ax.patches if hasattr(p, "get_facecolor")]
    if not boxes:
        return
    n_lines = len(ax.lines)
    per_box = n_lines // len(boxes) if boxes else 0
    for i, patch in enumerate(boxes):
        fc = patch.get_facecolor()
        patch.set_edgecolor(fc)
        patch.set_linewidth(1.2)
        for j in range(per_box):
            idx = i * per_box + j
            if idx < n_lines:
                ax.lines[idx].set_color(fc)
                ax.lines[idx].set_markeredgecolor(fc)
                ax.lines[idx].set_markerfacecolor(fc)


def client_scores_4(method, partition="noniid"):
    df = _read(method)
    cols = _short_cols(method)
    value_vars = list(cols.values())
    palette = _palette(method)
    pcode = "N" if partition == "noniid" else "I"

    for ds in DATASETS:
        sub = df[(df["dataset"] == ds) & (df["partition"] == partition) & (df["num_clients"] == 4)]
        if sub.empty:
            print(f"[warn] no rows for {ds} 4 {partition} {method} — skipping S_{DATASET_CODE[ds]}_4_{pcode}_{method}.png")
            continue

        agg = sub.groupby(["client_id", "round"])[value_vars].mean().reset_index()
        melted = agg.melt(id_vars=["client_id"], value_vars=value_vars, var_name="type", value_name="value")

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(
            data=melted, x="client_id", y="value", hue="type",
            hue_order=value_vars, ax=ax, palette=palette,
            showfliers=False, legend=False,
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Client ID", fontsize=16, labelpad=10)
        ax.set_ylabel("Contribution", fontsize=16, labelpad=10)
        ax.tick_params(axis="both", labelsize=14, width=2.0, length=6)

        for i in range(len(sorted(melted["client_id"].unique())) - 1):
            ax.axvline(i + 0.5, color="gray", linestyle="-", linewidth=1, zorder=0)

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"S_{DATASET_CODE[ds]}_4_{pcode}_{method}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


def score_dist_20(method, partition="noniid"):
    df = _read(method)
    cols = _short_cols(method)
    pcode = "N" if partition == "noniid" else "I"

    ZOOM_ABS_Q = 0.98
    ZOOM_PAD = 0.05
    MIN_LIM = 1e-6

    for ds in DATASETS:
        sub = df[(df["dataset"] == ds) & (df["partition"] == partition) & (df["num_clients"] == 20)]
        if sub.empty:
            print(f"[warn] no rows for {ds} 20 {partition} {method} — skipping SD_20_{DATASET_CODE[ds]}_{pcode}_{method}.png")
            continue

        series = {}
        for key, col in cols.items():
            if col not in sub.columns:
                continue
            v = sub[col].dropna()
            if not v.empty:
                series[key] = v
        if not series:
            continue

        fig, ax = plt.subplots(figsize=(5, 4))
        for key, v in series.items():
            sns.kdeplot(data=v, ax=ax, linewidth=1.6, fill=True, alpha=0.3, color=DOMAIN_COLORS[key])

        ax.axvline(x=0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Contribution", fontsize=16)
        ax.set_ylabel("Density", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        all_vals = np.concatenate([v.to_numpy() for v in series.values()])
        lim = np.quantile(np.abs(all_vals), ZOOM_ABS_Q)
        lim = max(lim, MIN_LIM) * (1.0 + ZOOM_PAD)
        ax.set_xlim(-lim, lim)

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"SD_20_{DATASET_CODE[ds]}_{pcode}_{method}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


def heatmap_4(method, partition="noniid"):
    df = _read(method)
    cols = _short_cols(method)
    keys = list(cols.keys())
    val_cols = [cols[k] for k in keys]
    pcode = "N" if partition == "noniid" else "I"

    for ds in DATASETS:
        sub = df[(df["dataset"] == ds) & (df["partition"] == partition) & (df["num_clients"] == 4)].copy()
        if sub.empty:
            print(f"[warn] no rows for {ds} 4 {partition} {method} — skipping H_4_{DATASET_CODE[ds]}_{pcode}_{method}.png")
            continue

        # Round-averaged Spearman: for each round, compute the per-pair
        # correlation matrix on per-client means (NaN where a column has
        # zero variance), then average element-wise across rounds with
        # nanmean. More robust than picking a single round when individual
        # rounds have sparse / zero-variance columns.
        per_round = []
        rounds_used = []
        for r in sorted(sub["round"].unique()):
            w = sub[sub["round"] == r].groupby("client_id", as_index=False)[val_cols].mean()
            w = w.dropna(subset=val_cols)
            if w.shape[0] < 3:
                continue
            # Spearman returns NaN for zero-variance columns; that's fine —
            # nanmean across rounds will skip those NaNs cell-by-cell.
            m = w[val_cols].corr(method="spearman").to_numpy()
            per_round.append(m)
            rounds_used.append(r)

        if not per_round:
            print(f"[warn] no usable rounds for {ds} 4 {partition} {method} — skipping")
            continue

        avg = np.nanmean(np.stack(per_round, axis=0), axis=0)
        mat = pd.DataFrame(avg, index=keys, columns=keys)
        print(f"  [info] {ds} 4 {partition} {method}: averaged over rounds {rounds_used}")

        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        sns.heatmap(
            mat, ax=ax, annot=True, fmt=".1f", annot_kws={"size": 16},
            vmin=-1, vmax=1, center=0, cmap="RdBu_r", square=True,
            linewidths=0.8, cbar=True, cbar_kws={"ticks": [-1, -0.5, 0, 0.5, 1]},
        )
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, pad=6)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14, width=2.0, length=6)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"H_4_{DATASET_CODE[ds]}_{pcode}_{method}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


def legend():
    """Single shared legend (perf / fair / rel / res) used by all S_* figures."""
    entries = [
        ("Performance",  DOMAIN_COLORS["perf"]),
        ("Fairness",     DOMAIN_COLORS["fair"]),
        ("Reliability",  DOMAIN_COLORS["rel"]),
        ("Resilience",   DOMAIN_COLORS["res"]),
    ]
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=18,
               markerfacecolor=c, markeredgecolor=c, label=lab)
        for lab, c in entries
    ]
    fig, ax = plt.subplots(figsize=(5, 1.0))
    ax.legend(handles=handles, ncol=4, fontsize=12, frameon=False, loc="center")
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "S_legend.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main(also_iid=False):
    """If also_iid=True, also produce IID variants (useful while non-IID runs
    are still in progress — gives visual stand-ins for the CelebA figures)."""
    print("Generating short-paper figures into", OUT_DIR)
    legend()
    partitions = ("noniid",) if not also_iid else ("noniid", "iid")
    for method in ("l1o", "gtg"):
        print(f"--- method={method} ---")
        for part in partitions:
            client_scores_4(method, partition=part)
            score_dist_20(method, partition=part)
            heatmap_4(method, partition=part)
    print("Done.")


if __name__ == "__main__":
    main()
