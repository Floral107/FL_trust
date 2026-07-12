"""Live Leave-One-Out contribution scoring for uploaded tabular datasets.

This mirrors the thesis pipeline's L1O definition —

    contribution_i = Metric(model trained on ALL clients) - Metric(model trained on ALL \\ {i})

— but swaps the federated Keras training for a fast sklearn model so scoring K clients
takes seconds on CPU instead of GPU-days. The *scoring semantics* (metric definitions,
L1O differencing, sign conventions) match the research code:

  * accuracy      — plain test accuracy (higher = better)
  * loss          — negative log-loss, so positive contribution = client reduces loss
  * fairness (DP) — 1 - |P(yhat=1 | s=0) - P(yhat=1 | s=1)|  (demographic parity)
  * fairness (EO) — 1 - |TPR(s=0) - TPR(s=1)|                (equalized odds)
  * noise robustness — accuracy under Gaussian feature noise, a cheap stand-in for the
    paper's certified robustness (same "consistency under perturbation" idea, no
    randomized-smoothing certificates)

The expensive axes from the paper (PGD adversarial, MIA privacy) are intentionally NOT
computed live; the Explore tab shows those from precomputed research results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

NOISE_SIGMA = 0.1   # matches the research pipeline's certified-robustness sigma
NOISE_REPS = 5      # noise draws averaged per robustness eval


@dataclass
class LiveScoreResult:
    """Per-client L1O contributions plus the all-clients global metrics."""
    global_metrics: dict[str, float]
    contributions: pd.DataFrame  # index: client id, columns: metric names
    client_sizes: dict[str, int]
    n_test: int
    warnings: list[str] = field(default_factory=list)


def _make_model(x: pd.DataFrame, seed: int):
    """Logistic regression with standard preprocessing — fast, deterministic, no GPU."""
    num_cols = x.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in x.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    return make_pipeline(pre, LogisticRegression(max_iter=1000, random_state=seed))


def _metrics(model, x_test, y_test, s_test, numeric_cols, rng) -> dict[str, float]:
    yhat = model.predict(x_test)
    proba = model.predict_proba(x_test)
    out = {
        "accuracy": float(accuracy_score(y_test, yhat)),
        "neg_log_loss": float(-log_loss(y_test, proba, labels=model.classes_)),
    }

    # Noise robustness: accuracy with N(0, sigma) added to standardized numeric features.
    if numeric_cols:
        accs = []
        scale = x_test[numeric_cols].std(ddof=0).replace(0, 1.0)
        for _ in range(NOISE_REPS):
            x_noisy = x_test.copy()
            noise = rng.normal(0.0, NOISE_SIGMA, size=x_test[numeric_cols].shape)
            x_noisy[numeric_cols] = x_test[numeric_cols] + noise * scale.values
            accs.append(accuracy_score(y_test, model.predict(x_noisy)))
        out["noise_robustness"] = float(np.mean(accs))

    # Fairness needs a binary-ish sensitive attribute and binary target.
    if s_test is not None:
        groups = pd.Series(s_test).astype(str).values
        uniq = np.unique(groups)
        classes = model.classes_
        if len(uniq) == 2 and len(classes) == 2:
            pos = classes[1]
            g0, g1 = (groups == uniq[0]), (groups == uniq[1])
            yhat_pos = (yhat == pos)
            # Demographic parity: gap in positive prediction rate.
            dp_gap = abs(yhat_pos[g0].mean() - yhat_pos[g1].mean())
            out["fairness_dp"] = float(1.0 - dp_gap)
            # Equalized odds (TPR flavour, as in the thesis): gap in TPR.
            y_pos = (np.asarray(y_test) == pos)
            tprs = []
            for g in (g0, g1):
                mask = g & y_pos
                tprs.append(yhat_pos[mask].mean() if mask.sum() > 0 else np.nan)
            if not any(np.isnan(t) for t in tprs):
                out["fairness_eo"] = float(1.0 - abs(tprs[0] - tprs[1]))
    return out


def score_clients(
    df: pd.DataFrame,
    client_col: str,
    target_col: str,
    sensitive_col: str | None = None,
    test_size: float = 0.3,
    seed: int = 42,
) -> LiveScoreResult:
    """Compute L1O contribution scores for every client in an uploaded dataset.

    Trains one model on all clients' train data, then one per left-out client, and
    reports contribution_i = metric(all) - metric(all minus i) for each metric.
    """
    warnings: list[str] = []
    df = df.dropna(subset=[client_col, target_col]).copy()
    if df[target_col].nunique() < 2:
        raise ValueError("Target column needs at least 2 classes.")

    feature_cols = [c for c in df.columns if c not in {client_col, target_col}]
    if sensitive_col and sensitive_col in feature_cols:
        pass  # sensitive attr stays a feature, as in the Adult setup
    if not feature_cols:
        raise ValueError("No feature columns left after removing client/target columns.")

    clients = sorted(df[client_col].astype(str).unique())
    if len(clients) < 2:
        raise ValueError("Need at least 2 clients to compute contributions.")
    if len(clients) > 50:
        raise ValueError(f"{len(clients)} clients is too many for the live demo (max 50).")

    df["_client"] = df[client_col].astype(str)

    # Global stratified train/test split — the test set is shared by every coalition,
    # like the held-out test set in the research pipeline.
    strat = df[target_col] if df[target_col].value_counts().min() >= 2 else None
    if strat is None:
        warnings.append("Target too imbalanced to stratify the test split.")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)

    x_test = test_df[feature_cols]
    y_test = test_df[target_col].values
    s_test = test_df[sensitive_col].values if sensitive_col else None
    numeric_cols = x_test.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        warnings.append("No numeric features — noise-robustness score skipped.")

    def fit_and_eval(train_subset: pd.DataFrame) -> dict[str, float] | None:
        if train_subset[target_col].nunique() < 2:
            return None
        model = _make_model(train_subset[feature_cols], seed)
        model.fit(train_subset[feature_cols], train_subset[target_col].values)
        # Fresh generator per eval so leave-one-out order can't change the noise draws.
        return _metrics(model, x_test, y_test, s_test, numeric_cols, np.random.default_rng(seed))

    global_metrics = fit_and_eval(train_df)
    if global_metrics is None:
        raise ValueError("Could not train the all-clients model.")

    rows = {}
    for cid in clients:
        rest = train_df[train_df["_client"] != cid]
        m = fit_and_eval(rest)
        if m is None:
            warnings.append(f"Client {cid}: leaving it out removes a whole class — scores set to NaN.")
            rows[cid] = {k: np.nan for k in global_metrics}
            continue
        rows[cid] = {k: global_metrics[k] - m.get(k, np.nan) for k in global_metrics}

    contributions = pd.DataFrame.from_dict(rows, orient="index")
    contributions.index.name = "client_id"
    sizes = train_df.groupby("_client").size().to_dict()
    return LiveScoreResult(
        global_metrics=global_metrics,
        contributions=contributions,
        client_sizes={str(k): int(v) for k, v in sizes.items()},
        n_test=len(test_df),
        warnings=warnings,
    )
