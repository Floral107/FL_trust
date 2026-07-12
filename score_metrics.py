"""score_metrics.py — adapters that produce a model→float metric_fn for each
of the 7 trustworthiness metrics used in Tab. 4.

Each builder takes the dataset's held-out evaluation tensors (x_test, y_test,
optionally s_test) and returns a callable `metric_fn(model) -> float` that
matches the GTGShapley utility-function shape. These are designed for IN-LOOP
score computation during ST/DY FL training — i.e. they must be fast (no long
PGD iterations, no MIA training) since they'll fire every coalition × every
permutation × every round.

For the heavy metrics (res, priv) we use cheaper proxies during in-loop
scoring than the post-hoc evaluation in cont_evals.py. The paper's "GTG-based"
score for these metrics is what drives ST/DY weighting; the actual cell value
in Tab. 4 still comes from the proper post-hoc eval.

Convention: higher = better for every metric except loss (loss_flag).
"""
from __future__ import annotations

import os
from typing import Callable, Optional
import numpy as np

# Compiled forward-pass helper: routes the many in-loop model(x) calls through a
# cached @tf.function so coalition scoring on CPU is not bottlenecked by per-op
# eager dispatch. Identical math; only the execution backend changes.
from fast_infer import get_infer

# Peak-memory bound for scoring forward passes. A single full-batch forward over
# the whole eval set (e.g. 5000 CelebA images) allocates huge conv activations
# (~[5000,64,64,32] ≈ 2.6 GB/layer), which forces the scoring server to own a GPU
# by itself. Chunking the forward into FLR_SCORE_CHUNK-row batches bounds that
# peak so client training can share the GPU — without changing the result:
# inference is per-sample independent (BatchNorm uses moving stats, dropout off),
# so concatenating chunked outputs equals the full-batch output.
_SCORE_CHUNK = int(os.environ.get("FLR_SCORE_CHUNK", "512"))


def _predict_np(model, x, chunk: Optional[int] = None) -> np.ndarray:
    """model(x, training=False) done in chunks -> concatenated numpy array.

    Numerically identical to the full-batch forward (per-sample independent at
    inference), but with bounded peak activation memory.
    """
    import tensorflow as tf
    c = int(chunk or _SCORE_CHUNK)
    xt = x if tf.is_tensor(x) else tf.convert_to_tensor(np.asarray(x))
    n = int(xt.shape[0])
    if n <= c:
        p = model(xt, training=False)
        return p.numpy() if hasattr(p, "numpy") else np.asarray(p)
    outs = []
    for i in range(0, n, c):
        p = model(xt[i:i + c], training=False)
        outs.append(p.numpy() if hasattr(p, "numpy") else np.asarray(p))
    return np.concatenate(outs, axis=0)


def _infer_np(infer, x, chunk: Optional[int] = None) -> np.ndarray:
    """Apply a compiled get_infer() fn over x in chunks -> concatenated numpy.
    Same bound/identity rationale as _predict_np; fast_infer sets
    reduce_retracing so the smaller final chunk doesn't re-trace expensively.
    """
    import tensorflow as tf
    c = int(chunk or _SCORE_CHUNK)
    xt = x if tf.is_tensor(x) else tf.convert_to_tensor(np.asarray(x))
    n = int(xt.shape[0])
    if n <= c:
        return infer(xt).numpy()
    outs = []
    for i in range(0, n, c):
        outs.append(infer(xt[i:i + c]).numpy())
    return np.concatenate(outs, axis=0)


# ── basic metrics (cheap; valid for both in-loop and post-hoc) ──────────────

def build_acc_metric(x_test, y_test) -> Callable:
    """Accuracy metric: correct-predictions / total. Higher is better."""
    def metric(model) -> float:
        preds = _predict_np(model, x_test)
        if preds.ndim == 2 and preds.shape[-1] == 1:
            y_hat = (preds.reshape(-1) >= 0.5).astype(int)
        else:
            y_hat = preds.argmax(axis=-1)
        y_true = np.asarray(y_test).reshape(-1).astype(int)
        return float((y_hat == y_true).mean())
    return metric


def build_loss_metric(x_test, y_test) -> Callable:
    """Cross-entropy / BCE loss. LOWER is better (set loss_flag=True in GTG).
    Returns negative loss so the GTG utility increases with quality — keeps
    the rest of the pipeline metric-direction-agnostic. (NOTE: if we want to
    rank by raw loss with loss_flag, that's a separate choice — see caller.)
    """
    # Build a loss-fn that handles both sparse-categorical and binary outputs
    def metric(model) -> float:
        preds = np.asarray(_predict_np(model, x_test), dtype=np.float32)
        eps = 1e-7
        preds = np.clip(preds, eps, 1.0 - eps)
        y = np.asarray(y_test).reshape(-1).astype(int)
        if preds.ndim == 2 and preds.shape[-1] == 1:
            # binary: BCE
            p = preds.reshape(-1)
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        else:
            # categorical: sparse CE
            loss = -np.mean(np.log(preds[np.arange(len(y)), y]))
        return -float(loss)  # negative so larger = better
    return metric


# ── fairness ───────────────────────────────────────────────────────────────

def build_fairDP_metric(x_test, y_test, s_test) -> Callable:
    """Demographic Parity: 1 - |P(ŷ=pos|S=0) - P(ŷ=pos|S=1)|. Higher is better.

    Skips if either group is missing (returns 1.0 — no disparity measurable).
    """
    s = np.asarray(s_test).reshape(-1).astype(int)
    def metric(model) -> float:
        preds = _predict_np(model, x_test)
        if preds.ndim == 2 and preds.shape[-1] == 1:
            y_hat = (preds.reshape(-1) >= 0.5).astype(int)
        else:
            y_hat = preds.argmax(axis=-1)
        mask0 = (s == 0); mask1 = (s == 1)
        if not mask0.any() or not mask1.any():
            return 1.0
        p0 = float(y_hat[mask0].mean())
        p1 = float(y_hat[mask1].mean())
        return 1.0 - abs(p0 - p1)
    return metric


def build_fairEO_metric(x_test, y_test, s_test) -> Callable:
    """Equalised Odds (positive-rate parity, conditional on y=1).

    Score: 1 - |TPR(S=0) - TPR(S=1)|. Higher is better.
    """
    s = np.asarray(s_test).reshape(-1).astype(int)
    y_true = np.asarray(y_test).reshape(-1).astype(int)
    def metric(model) -> float:
        preds = _predict_np(model, x_test)
        if preds.ndim == 2 and preds.shape[-1] == 1:
            y_hat = (preds.reshape(-1) >= 0.5).astype(int)
        else:
            y_hat = preds.argmax(axis=-1)
        positives = (y_true == 1)
        if not positives.any():
            return 1.0
        mask0 = (s == 0) & positives
        mask1 = (s == 1) & positives
        if not mask0.any() or not mask1.any():
            return 1.0
        tpr0 = float(y_hat[mask0].mean())
        tpr1 = float(y_hat[mask1].mean())
        return 1.0 - abs(tpr0 - tpr1)
    return metric


# ── robustness (CHEAP proxies for in-loop scoring) ─────────────────────────

def build_rel_metric(x_test, y_test, sigma: float = 0.1, num_samples: int = 8) -> Callable:
    """Reliability proxy: fraction of samples whose prediction is UNCHANGED
    under Gaussian noise σ, averaged over `num_samples` noise draws.

    The paper uses L=100 noise samples for post-hoc evaluation. For in-loop
    scoring we use a smaller L (default 8) to keep coalition evaluation
    affordable; this is a noisier estimator but preserves ranking signal.
    """
    import tensorflow as tf
    x_t = tf.convert_to_tensor(np.asarray(x_test, dtype=np.float32))
    def metric(model) -> float:
        infer = get_infer(model)
        clean = _infer_np(infer, x_t)
        clean_hat = clean.argmax(axis=-1) if clean.ndim == 2 and clean.shape[-1] > 1 \
                    else (clean.reshape(-1) >= 0.5).astype(int)
        agree = 0
        total = 0
        for _ in range(num_samples):
            noise = tf.random.normal(shape=tf.shape(x_t), mean=0.0, stddev=sigma)
            noisy = tf.clip_by_value(x_t + noise, 0.0, 1.0)
            np_ = _infer_np(infer, noisy)
            np_hat = np_.argmax(axis=-1) if np_.ndim == 2 and np_.shape[-1] > 1 \
                     else (np_.reshape(-1) >= 0.5).astype(int)
            agree += int((np_hat == clean_hat).sum())
            total += len(clean_hat)
        return float(agree / max(total, 1))
    return metric


# In-loop res proxy params per dataset: same eps/alpha as the post-hoc eval
# (attack_metric.PGD_EVAL_PARAMS / MAIN.tex) but fewer PGD iterations so
# coalition evaluation stays affordable. Lower num_iter = noisier estimate
# but preserves ranking.
RES_PROXY_PARAMS = {
    "adult": (0.3, 0.007, 7), "adultnoniid": (0.3, 0.007, 7),
    "celeba": (0.03, 0.007, 7), "celebanoniid": (0.03, 0.007, 7),
    "cifar": (0.03, 0.007, 7), "cifarnoniid": (0.03, 0.007, 7),
    "imdb": (0.005, 0.00125, 5), "imdbnoniid": (0.005, 0.00125, 5),  # eps recalibrated 2026-07-08 (see attack_metric.py); 5 iters kept for in-loop affordability
}


def build_res_metric(x_test, y_test, epsilon=None, alpha=None, num_iter=None,
                     dataset=None) -> Callable:
    """Resilience proxy: 1 - PGD attack success rate.

    epsilon/alpha/num_iter resolve from RES_PROXY_PARAMS by `dataset` when not
    given explicitly (fallback: 0.03/0.007/7, the legacy uniform values).
    """
    d_eps, d_alpha, d_iter = RES_PROXY_PARAMS.get(
        (dataset or "").lower(), (0.03, 0.007, 7))
    epsilon = d_eps if epsilon is None else epsilon
    alpha = d_alpha if alpha is None else alpha
    num_iter = d_iter if num_iter is None else num_iter
    import tensorflow as tf
    from pgd_attack import pgd_attack
    x_np = np.asarray(x_test, dtype=np.float32)
    x_t = tf.convert_to_tensor(x_np)
    y_t = tf.convert_to_tensor(np.asarray(y_test).reshape(-1).astype(np.int32))
    # Clip bounds from the data's own range so PGD doesn't distort inputs:
    # images→~[0,1], adult (StandardScaled)→~[-3,3], imdb embeddings→their range.
    _lo, _hi = float(x_np.min()), float(x_np.max())
    def metric(model) -> float:
        # Pick loss matching the model head: adult is binary sigmoid (1 output)
        # → binary_crossentropy; celeba/imdb are 2-class softmax → sparse cat.
        # Using sparse-cat on a 1-output model crashes (label 1 out of range).
        is_sigmoid = (len(model.output_shape) >= 2 and model.output_shape[-1] == 1)
        loss_name = "binary_crossentropy" if is_sigmoid else "sparse_categorical_crossentropy"
        # Filter to clean-correct first
        infer = get_infer(model)
        preds = _infer_np(infer, x_t)
        if preds.ndim == 2 and preds.shape[-1] > 1:
            clean_hat = preds.argmax(axis=-1)
        else:
            clean_hat = (preds.reshape(-1) >= 0.5).astype(int)
        correct_mask = (clean_hat == y_t.numpy())
        if not correct_mask.any():
            return 1.0  # nothing to attack
        idx = np.where(correct_mask)[0]
        # Cap the in-loop PGD attack set. PGD holds a full-batch gradient tape
        # through the CNN, so ~4000 celeba images OOM a 12GB GPU — fatal for DY
        # (scores every round). This is the in-loop res PROXY that only drives
        # client weights; the res value reported in Tab.5 comes from the full
        # post-hoc eval (attack_metric), so bounding the attack set here is a
        # memory fix, not a metric change. Deterministic subsample (seeded).
        _max_atk = int(os.environ.get("FLR_RES_MAX_ATTACK", "512"))
        if len(idx) > _max_atk:
            idx = np.sort(np.random.default_rng(0).choice(idx, _max_atk, replace=False))
        x_c = tf.gather(x_t, idx)
        y_c = tf.gather(y_t, idx)
        x_adv = pgd_attack(
            model, x_c, y_c,
            epsilon=epsilon, alpha=alpha, num_iter=num_iter,
            clip_min=_lo, clip_max=_hi, from_logits=False,
            loss_name=loss_name,
            random_start=True,
        )
        adv_preds = _infer_np(infer, x_adv)
        if adv_preds.ndim == 2 and adv_preds.shape[-1] > 1:
            adv_hat = adv_preds.argmax(axis=-1)
        else:
            adv_hat = (adv_preds.reshape(-1) >= 0.5).astype(int)
        success_rate = float((adv_hat != y_c.numpy()).mean())
        return 1.0 - success_rate
    return metric


# ── privacy (CHEAP proxy: confidence-margin-based, not full MIA) ────────────

def build_priv_metric(x_test, y_test) -> Callable:
    """Privacy proxy: 1 - avg max-prob margin on correct samples.

    The full MIA-based priv (in cont_evals.py) requires fine-tuning a shadow
    model on D'_in — way too expensive for in-loop coalition scoring. As a
    cheaper proxy that correlates with memorisation: high confidence on
    members = lower privacy. We measure mean(max prob) on samples the model
    classifies correctly. Lower margin = higher privacy → returns 1 - margin.
    """
    def metric(model) -> float:
        preds = _predict_np(model, x_test)
        if preds.ndim == 2 and preds.shape[-1] > 1:
            y_hat = preds.argmax(axis=-1)
            max_p = preds.max(axis=-1)
        else:
            p = preds.reshape(-1)
            y_hat = (p >= 0.5).astype(int)
            max_p = np.maximum(p, 1 - p)
        y_true = np.asarray(y_test).reshape(-1).astype(int)
        correct = (y_hat == y_true)
        if not correct.any():
            return 1.0
        avg_margin = float(max_p[correct].mean())
        return 1.0 - avg_margin
    return metric


# ── dispatch ────────────────────────────────────────────────────────────────

def build_metric(name: str, x_test, y_test, s_test=None, dataset=None) -> Callable:
    """Dispatch by metric name. Returns metric_fn(model) -> float.

    For metrics that need s_test (fairDP, fairEO), s_test must be provided
    or a ValueError is raised. `dataset` selects per-dataset res-proxy PGD
    parameters (RES_PROXY_PARAMS); omitted -> legacy uniform defaults.
    """
    n = name.lower()
    if n == "acc":
        return build_acc_metric(x_test, y_test)
    if n == "loss":
        return build_loss_metric(x_test, y_test)
    if n in ("fairdp", "fair_dp", "fair"):
        if s_test is None:
            raise ValueError("fairDP metric needs s_test")
        return build_fairDP_metric(x_test, y_test, s_test)
    if n in ("faireo", "fair_eo"):
        if s_test is None:
            raise ValueError("fairEO metric needs s_test")
        return build_fairEO_metric(x_test, y_test, s_test)
    if n == "rel":
        return build_rel_metric(x_test, y_test)
    if n == "res":
        return build_res_metric(x_test, y_test, dataset=dataset)
    if n == "priv":
        return build_priv_metric(x_test, y_test)
    raise ValueError(f"unknown metric name: {name!r}")


# Whether to set GTGShapley(loss_flag=True) for this metric.
# Currently all metrics are framed as "higher = better" (loss is negated).
def is_loss_metric(name: str) -> bool:
    return name.lower() == "loss" and False  # always returns False — we negate loss
