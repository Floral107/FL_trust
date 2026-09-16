"""gap_metrics.py - direction B: group-gap utilities for LOO / GTG client scoring.

    u = 1 - |M(A) - M(B)|

M is one of the paper's base metrics, computed separately on the two sensitive
groups (A: s == 1, B: s == 0). The filter decides which test samples count:
    dp  - every sample of the group
    eop - only samples whose TRUE label is positive (y == 1), as in Hardt et al.'s
          equal opportunity

Method names are {l1o,gtg}_gap_{metric}_{filter}. The filter comes last on
purpose: cont_evals negates contributions and switches GTG to minimisation for
any name ending in "_loss", and gap utilities are higher-is-better.

Pure numpy (TensorFlow is only touched lazily to slice tensors), so the logic is
testable without TF: see test_gap_metrics.py.
"""
import re

import numpy as np

METRICS = ("acc", "loss", "rel", "res")  # ponytail: priv parked until reviewer feedback; add "priv" + a per-group MIA AUC then
FILTERS = ("dp", "eop")
_NAME = re.compile(r"^(?:l1o|gtg)_gap_(%s)_(%s)$" % ("|".join(METRICS), "|".join(FILTERS)))


def gap_methods():
    return [f"{sc}_gap_{m}_{f}" for sc in ("l1o", "gtg") for m in METRICS for f in FILTERS]


def parse(method):
    """(metric, filter) for a gap method name, None for every other method."""
    m = _NAME.match(method)
    return (m.group(1), m.group(2)) if m else None


def group_indices(s, y, filt):
    """Row indices of group A (s == 1) and group B (s == 0) that the filter keeps."""
    s = np.asarray(s).ravel().astype(int)
    keep = np.ones(s.shape, dtype=bool) if filt == "dp" else np.asarray(y).ravel() >= 0.5
    return np.flatnonzero(keep & (s == 1)), np.flatnonzero(keep & (s == 0))


def _take(x, idx):
    if isinstance(x, np.ndarray):
        return x[idx]
    import tensorflow as tf  # CelebA hands over tf tensors
    return tf.gather(x, idx)


def make_gap_value(metric_fn, x, y, s, filt):
    """model -> 1 - |M(A) - M(B)|, with metric_fn(model, x_group, y_group) -> float.

    The returned function also has .parts(model) -> (M(A), M(B)): a gap alone can't say
    whether groups got equal because one improved or because the other got worse.
    Groups are sliced once here, not once per coalition. An empty group gives NaN,
    never a number: a silent 0 or 1 would read as a real measurement in LOO/GTG.
    """
    ia, ib = group_indices(s, y, filt)
    if len(ia) == 0 or len(ib) == 0:
        parts = lambda model: (float("nan"), float("nan"))
    else:
        xa, ya, xb, yb = _take(x, ia), _take(y, ia), _take(x, ib), _take(y, ib)
        parts = lambda model: (float(metric_fn(model, xa, ya)), float(metric_fn(model, xb, yb)))

    def gap_value(model):
        m_a, m_b = parts(model)
        return 1.0 - abs(m_a - m_b)

    gap_value.parts = parts
    return gap_value
