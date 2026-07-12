"""Load and normalize the precomputed FLRobustness contribution CSVs.

The research pipeline writes one CSV per (dataset, method-family) with wide columns like
``l1o_rob_contribution`` / ``gtg_fair_contribution``. The dashboard wants one tidy frame:

    dataset | partition | num_clients | seed | round | client_id | method | dimension | global | contribution

Dimensions surfaced (matching the thesis): rob (certified robustness), adv (PGD adversarial),
fair (demographic parity), fair_eo (equalized odds), priv (MIA privacy), acc, loss.
"""

from __future__ import annotations

import os
import re
from glob import glob

import pandas as pd

# Human-facing names for score dimensions, in display order.
DIMENSIONS = {
    "rob": "Certified robustness",
    "adv": "Adversarial (PGD)",
    "fair": "Fairness (DP)",
    "fair_eo": "Fairness (EO)",
    "priv": "Privacy (MIA)",
    "acc": "Accuracy",
    "loss": "Loss",
}

METHODS = {"l1o": "Leave-One-Out", "gtg": "GTG-Shapley"}

# The four headline trustworthiness axes for the trade-off views.
TRADEOFF_DIMS = ["rob", "adv", "fair", "priv"]

_FILE_RE = re.compile(r"results_(?P<dataset>[a-z]+?)(?P<noniid>noniid)?_(?P<method>l1o|gtg)\.csv$")


def _melt_one(path: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_cols = ["dataset", "partition", "num_clients", "seed", "round", "client_id"]
    frames = []
    for dim in DIMENSIONS:
        # Column layouts drifted across pipeline versions: acc/loss contributions may be
        # "{method}_{dim}_contribution" or bare "{dim}_contribution"; globals may be
        # "global_{method}_{dim}" or the raw metric column ("accuracy"/"loss").
        contrib_candidates = [f"{method}_{dim}_contribution", f"{dim}_contribution"]
        global_candidates = [f"global_{method}_{dim}", "accuracy" if dim == "acc" else "loss"]

        contrib_col = next((c for c in contrib_candidates if c in df.columns), None)
        if contrib_col is None:
            continue
        global_col = next((c for c in global_candidates if c in df.columns), None)

        sub = df[id_cols].copy()
        sub["method"] = method
        sub["dimension"] = dim
        sub["global"] = pd.to_numeric(df[global_col], errors="coerce") if global_col else float("nan")
        sub["contribution"] = pd.to_numeric(df[contrib_col], errors="coerce")
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_results(data_dir: str) -> pd.DataFrame:
    """Load every results_*_{l1o,gtg}.csv under ``data_dir`` into one tidy frame."""
    frames = []
    for path in sorted(glob(os.path.join(data_dir, "results_*.csv"))):
        m = _FILE_RE.search(os.path.basename(path))
        if not m:
            continue
        tidy = _melt_one(path, m.group("method"))
        if not tidy.empty:
            frames.append(tidy)
    if not frames:
        return pd.DataFrame(
            columns=["dataset", "partition", "num_clients", "seed", "round",
                     "client_id", "method", "dimension", "global", "contribution"]
        )
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["contribution"])
    return out


def tradeoff_matrix(df: pd.DataFrame, dims: list[str] | None = None) -> pd.DataFrame:
    """Pivot the tidy frame to one row per client with one column per dimension.

    Contributions are averaged over whatever seeds/rounds remain after the caller's
    filtering — the paper reports seed-averaged scores the same way.
    """
    dims = dims or TRADEOFF_DIMS
    sub = df[df["dimension"].isin(dims)]
    wide = (
        sub.groupby(["client_id", "dimension"])["contribution"]
        .mean()
        .unstack("dimension")
        .reindex(columns=dims)
    )
    wide.index.name = "client_id"
    return wide.reset_index()
