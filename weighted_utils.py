"""weighted_utils.py — pure-math helpers for ST/DY aggregation.

NO flwr or TF dependency — importable on any machine with numpy.

The transforms and weighted aggregation are factored out here so they can be
unit-tested in isolation. WeightedFedAvg (in weighted_strategy.py) reuses
these helpers.
"""
from __future__ import annotations
from typing import List
import numpy as np


def shift_and_normalize(
    scores: List[float], beta: float = 1.0, cap: float = 0.0
) -> List[float]:
    """Transform per paper eq., with optional shrinkage toward uniform and/or a
    hard ratio cap:
        s'  = s + |min(0, min_k s_k)|     # shift to non-negative
        w   = s' / mean(s')               # normalize → weights hover at 1
        w   = (1 - beta) + beta * w       # shrink toward the uniform value 1
        w   = clip(w, 1/cap, cap)         # bound each client's deviation from uniform

    The raw transform (beta=1, cap off) is unbounded: under GTG truncation the
    contribution scores go near-one-hot, w spikes (e.g. one client at K·max,
    the rest ~0), and the size-modulated aggregation collapses onto a single
    client — discarding most of the federation's data. This is the failure the
    old post-hoc do-no-harm guard was papering over.

    Two collapse-proofing knobs, usable together or independently:

    * ``beta`` — shrinkage toward uniform. Floors each weight at (1 - beta) and
      keeps the mean at ~1.
        beta = 1 → legacy raw transform (kept for backward-compat / ablation)
        beta = 0 → exact uniform  ==  standard size-weighted FedAvg (BL)
        0<beta<1 → damped contribution signal.
      Limitation: beta only floors the *low* end; the top weight can still reach
      (1-beta)+beta·K, which at K=20, beta=0.2 is 4.8 — enough to dominate.

    * ``cap`` — hard multiplicative cap (cap >= 1, or 0/None to disable). After
      shrinkage, weights are clipped to [1/cap, cap]. Since pre-clip weights have
      mean 1, this bounds each client's deviation from uniform: the top client
      gets at most cap x the uniform share and the bottom at least 1/cap x, so a
      single client can never capture the aggregation no matter how one-hot the
      GTG scores degenerate (independent of K). cap=2 -> top client capped at 2x
      uniform. No renormalization is applied: the aggregation coefficients
      (n_k w_k / sum_j n_j w_j) are scale-invariant, so re-centering the mean is
      unnecessary and would re-inflate the clipped top past the cap.

    Degenerate case: if the shifted mean is ~0 (all-zero shifted), fall back to
    uniform 1.0 (so the weighted aggregation reduces to plain FedAvg).
    """
    s = np.asarray(scores, dtype=float)
    shift = -min(0.0, float(np.min(s)))
    shifted = s + shift
    mean = float(np.mean(shifted))
    if mean < 1e-12:
        return [1.0] * len(scores)
    w = shifted / mean
    if beta != 1.0:
        w = (1.0 - beta) + beta * w
    if cap and cap > 0.0:
        if cap < 1.0:
            raise ValueError(f"cap must be >= 1 (or 0 to disable), got {cap!r}")
        # Clip but do NOT renormalize: aggregation is scale-invariant, and
        # re-centering would push the clipped top back above the cap.
        w = np.clip(w, 1.0 / cap, cap)
    return w.tolist()


def weighted_aggregate(
    client_params,
    client_num_examples: List[int],
    client_weights: List[float],
):
    """Option (a) — size × weight modulation:

        θ_X = Σ_k  (n_k · w_k / Σ_j n_j w_j)  ·  θ_k^X

    When all w_k = 1, this reduces exactly to standard size-weighted FedAvg.
    When the score-derived weights vary, high-scoring clients are amplified
    relative to their natural size-based share.

    Falls back to uniform coefficients if Σ (n_k · w_k) ≈ 0 (defensive).
    """
    n = len(client_params)
    if n == 0:
        raise ValueError("weighted_aggregate called with no client params")
    if len(client_num_examples) != n or len(client_weights) != n:
        raise ValueError("length mismatch among params/num_examples/weights")

    raw = np.asarray(client_num_examples, dtype=float) * np.asarray(
        client_weights, dtype=float
    )
    total = float(raw.sum())
    if total < 1e-12:
        coeffs = np.full(n, 1.0 / n)
    else:
        coeffs = raw / total

    # Layer-by-layer weighted sum (works for any list-of-ndarrays "params").
    out = []
    for layer_idx in range(len(client_params[0])):
        layer_acc = sum(
            float(coeffs[k]) * client_params[k][layer_idx] for k in range(n)
        )
        out.append(layer_acc)
    return out


def fedavg_aggregate(client_params, client_num_examples: List[int]):
    """Standard size-weighted FedAvg — used as the 'no scoring' baseline.

    Equivalent to weighted_aggregate with all weights = 1.0, but explicit so
    callers can short-circuit when weighting is disabled.
    """
    total = float(sum(client_num_examples))
    if total < 1e-12:
        coeffs = [1.0 / len(client_params)] * len(client_params)
    else:
        coeffs = [float(n) / total for n in client_num_examples]
    out = []
    for layer_idx in range(len(client_params[0])):
        layer_acc = sum(
            coeffs[k] * client_params[k][layer_idx] for k in range(len(client_params))
        )
        out.append(layer_acc)
    return out
