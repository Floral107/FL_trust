"""
privacy_metric.py

Clean, working MIA-privacy metric for coalition models.

What it does per coalition model:
1) Clone the model
2) Fine-tune on a deterministic "member" subset of the TEST set for a few epochs
3) Run ART MembershipInferenceBlackBox
   - We use LOSS as features in a robust way: compute per-sample loss ourselves and pass it via `pred=...`
     so we avoid the ART-internal "loss + one-hot" shape mismatch for sigmoid binary models.

Outputs:
- privacy score in [0,1], higher = better privacy
- raw MIA accuracy / AUC

Dependencies:
- tensorflow
- adversarial-robustness-toolbox[tensorflow]
- scikit-learn
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import tensorflow as tf

_CPU = "/CPU:0"
# Pinning forward/backward MIA ops to CPU is a stability guard from when we
# saw flaky behaviour on certain TF/cuDNN versions for ft_model.fit(). Now
# that the GPU stack is current (TF 2.19 + cu12 wheels) we want to run those
# ops on GPU — a 10-15× speed-up on MIA which dominates gtg_priv wall-time.
# Override via FLR_MIA_CPU=1 to restore the legacy CPU-only behaviour.
import os as _os
if _os.environ.get("FLR_MIA_CPU", "0") == "1":
    _MIA_DEVICE = _CPU
else:
    _MIA_DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else _CPU

# -----------------------------
# Config
# -----------------------------
@dataclass
class MIAConfig:
    # Fine-tuning on the member set
    finetune_epochs: int = 2
    finetune_lr: float = 1e-3
    batch_size: int = 16  # smaller batches → more gradient steps → stronger per-epoch memorisation signal

    # Attack model in ART: "lr", "rf", "gb", or "nn"
    attack_model_type: str = "gb"  # gradient boosting > RF for low-dim tabular MIA features
    nn_model_epochs: int = 50
    nn_model_batch_size: int = 256
    nn_model_learning_rate: float = 1e-4

    # Features for the attack model:
    #   "loss"           -> per-sample loss after fine-tuning (1D)
    #   "combined"       -> loss + max_confidence + entropy + correctness (4D)
    #   "combined_delta" -> delta-loss + combined (5D, strongest signal without extra epochs)
    #   "prediction"     -> model predicted probabilities/logits
    feature_type: str = "combined_delta"

    # ART estimator settings
    clip_values: Optional[Tuple[float, float]] = None
    use_logits: bool = False

    # Privacy mapping
    use_auc_for_privacy: bool = True  # privacy derived from AUC advantage if available


# -----------------------------
# Deterministic splits (reused across coalitions in the same round)
# -----------------------------
@dataclass(frozen=True)
class MIASplits:
    idx_mem_tr: np.ndarray
    idx_mem_ev: np.ndarray
    idx_non_tr: np.ndarray
    idx_non_ev: np.ndarray


def make_mia_splits(
    y_test: Any,
    *,
    seed: int,
    round_num: int,
    member_ratio: float = 0.5,
    attack_train_ratio: float = 0.5,
    balance: bool = True,
) -> MIASplits:
    """
    Deterministic indices for:
      - member vs non-member selection (from test)
      - attack train vs eval splits on each side
    """
    y = _to_label_index(y_test)
    n = len(y)
    rng = np.random.default_rng(int(seed) + 1000 * int(round_num))

    idx_all = np.arange(n)
    rs = int(rng.integers(0, 2**31 - 1))

    strat = y if len(np.unique(y)) > 1 else None
    try:
        idx_mem, idx_rest = train_test_split(idx_all, train_size=member_ratio, random_state=rs, stratify=strat)
    except Exception:
        idx_mem, idx_rest = train_test_split(idx_all, train_size=member_ratio, random_state=rs, shuffle=True)

    if balance:
        m = len(idx_mem)
        if len(idx_rest) >= m:
            idx_non = rng.choice(idx_rest, size=m, replace=False)
        else:
            idx_non = idx_rest
    else:
        idx_non = idx_rest

    def _side_split(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rs2 = int(rng.integers(0, 2**31 - 1))
        y_side = y[indices]
        strat2 = y_side if len(np.unique(y_side)) > 1 else None
        try:
            tr, ev = train_test_split(indices, train_size=attack_train_ratio, random_state=rs2, stratify=strat2)
        except Exception:
            tr, ev = train_test_split(indices, train_size=attack_train_ratio, random_state=rs2, shuffle=True)
        return tr, ev

    idx_mem_tr, idx_mem_ev = _side_split(idx_mem)
    idx_non_tr, idx_non_ev = _side_split(idx_non)

    return MIASplits(
        idx_mem_tr=np.asarray(idx_mem_tr, dtype=int),
        idx_mem_ev=np.asarray(idx_mem_ev, dtype=int),
        idx_non_tr=np.asarray(idx_non_tr, dtype=int),
        idx_non_ev=np.asarray(idx_non_ev, dtype=int),
    )


# -----------------------------
# Main scoring
# -----------------------------
def _reset_optimizer(model: tf.keras.Model, lr: float) -> None:
    """Reset Adam optimizer state to fresh zeros without recompiling.

    Avoids TF graph retracing that compile() triggers.
    """
    opt = model.optimizer
    opt_vars = opt.variables if isinstance(opt.variables, list) else opt.variables()
    for var in opt_vars:
        var.assign(tf.zeros_like(var))
    # Reset learning rate in case config differs (shouldn't, but safe)
    opt.learning_rate.assign(lr)


def _finetune_on_cpu(
    coalition_model: tf.keras.Model,
    config: MIAConfig,
    x_mem_tr: Any,
    y_mem_tr_idx: np.ndarray,
    ft_template: Optional[tf.keras.Model] = None,
) -> Tuple[tf.keras.Model, int]:
    """Fine-tune on member training split.

    If ft_template is provided (GTG reuse), it is already compiled.
    We reset its weights + optimizer state without calling compile() again.
    """
    with tf.device(_MIA_DEVICE):
        if ft_template is not None:
            ft_model = ft_template
            ft_model.set_weights(coalition_model.get_weights())
            _reset_optimizer(ft_model, config.finetune_lr)
        else:
            ft_model = tf.keras.models.clone_model(coalition_model)
            ft_model.set_weights(coalition_model.get_weights())

    units = _output_units(ft_model)
    if units == 1:
        y_fit = y_mem_tr_idx.astype(np.float32).reshape(-1, 1)
    else:
        y_fit = y_mem_tr_idx.astype(np.int64).reshape(-1)

    with tf.device(_MIA_DEVICE):
        if ft_template is None:
            if units == 1:
                ft_loss = tf.keras.losses.BinaryCrossentropy(from_logits=config.use_logits)
            else:
                ft_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=config.use_logits)
            ft_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config.finetune_lr),
                loss=ft_loss,
                metrics=["accuracy"],
            )
        if config.finetune_epochs > 0:
            ft_model.fit(x_mem_tr, y_fit, epochs=int(config.finetune_epochs),
                         batch_size=int(config.batch_size), verbose=0)
    return ft_model, units


def _delta_features_from_precomputed(
    ft_model: tf.keras.Model,
    x: Any,
    y_idx: np.ndarray,
    units: int,
    use_logits: bool,
    batch_size: int,
    orig_loss: np.ndarray,
) -> np.ndarray:
    """5-dimensional features: [loss_delta, loss_ft, max_confidence, entropy, correctness].

    loss_delta = loss_before_finetune - loss_after_finetune.
    Members (data the model was fine-tuned on) have a large positive delta;
    non-members have near-zero delta. This is the strongest signal per epoch.

    orig_loss is pre-computed in one batch to avoid redundant forward passes.
    """
    combined_ft = _combined_features(ft_model, x, y_idx, units, use_logits, batch_size)
    loss_ft = combined_ft[:, 0]
    delta = (orig_loss.reshape(-1) - loss_ft).reshape(-1)
    return np.stack([delta, combined_ft[:, 0], combined_ft[:, 1], combined_ft[:, 2], combined_ft[:, 3]], axis=1).astype(np.float32)


def _attack_combined(ft_model, config, units, seed, x_splits, y_splits, orig_model=None):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    x_mem_tr, x_non_tr, x_mem_ev, x_non_ev = x_splits
    y_mem_tr_idx, y_non_tr_idx, y_mem_ev_idx, y_non_ev_idx = y_splits
    use_delta = (config.feature_type == "combined_delta") and (orig_model is not None)

    mem_tr_feat = non_tr_feat = mem_ev_feat = non_ev_feat = None
    x_atk_tr = y_atk_tr = x_atk_ev = None
    clf = None
    all_orig_loss = None

    try:
        with tf.device(_MIA_DEVICE):
            # Pre-compute ALL orig-model losses in one forward pass (4→1)
            # This is the main GTG memory/compute saving for combined_delta
            orig_losses = [None, None, None, None]
            if use_delta:
                all_x = np.concatenate([x_mem_tr, x_non_tr, x_mem_ev, x_non_ev], axis=0)
                all_y = np.concatenate([y_mem_tr_idx, y_non_tr_idx, y_mem_ev_idx, y_non_ev_idx])
                all_orig_loss = _per_sample_loss(orig_model, all_x, all_y, units, config.use_logits, config.batch_size)
                ns = [len(y_mem_tr_idx), len(y_non_tr_idx), len(y_mem_ev_idx), len(y_non_ev_idx)]
                orig_losses = np.split(all_orig_loss, np.cumsum(ns[:-1]))
                del all_x, all_y, all_orig_loss

            if use_delta:
                mem_tr_feat = _delta_features_from_precomputed(ft_model, x_mem_tr, y_mem_tr_idx, units, config.use_logits, config.batch_size, orig_losses[0])
                non_tr_feat = _delta_features_from_precomputed(ft_model, x_non_tr, y_non_tr_idx, units, config.use_logits, config.batch_size, orig_losses[1])
                mem_ev_feat = _delta_features_from_precomputed(ft_model, x_mem_ev, y_mem_ev_idx, units, config.use_logits, config.batch_size, orig_losses[2])
                non_ev_feat = _delta_features_from_precomputed(ft_model, x_non_ev, y_non_ev_idx, units, config.use_logits, config.batch_size, orig_losses[3])
            else:
                mem_tr_feat = _combined_features(ft_model, x_mem_tr, y_mem_tr_idx, units, config.use_logits, config.batch_size)
                non_tr_feat = _combined_features(ft_model, x_non_tr, y_non_tr_idx, units, config.use_logits, config.batch_size)
                mem_ev_feat = _combined_features(ft_model, x_mem_ev, y_mem_ev_idx, units, config.use_logits, config.batch_size)
                non_ev_feat = _combined_features(ft_model, x_non_ev, y_non_ev_idx, units, config.use_logits, config.batch_size)
            del orig_losses

        x_atk_tr = np.concatenate([mem_tr_feat, non_tr_feat], axis=0)
        y_atk_tr = np.concatenate([np.ones(len(mem_tr_feat)), np.zeros(len(non_tr_feat))])

        if config.attack_model_type == "gb":
            clf = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=int(seed),
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=100,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=int(seed),
            )
        clf.fit(x_atk_tr, y_atk_tr)

        x_atk_ev = np.concatenate([mem_ev_feat, non_ev_feat], axis=0)
        y_score_all = clf.predict_proba(x_atk_ev)[:, 1]
        y_pred_cls = (y_score_all >= 0.5).astype(int)
        n = len(mem_ev_feat)

        return y_score_all[:n], y_pred_cls[:n], y_score_all[n:], y_pred_cls[n:], mem_ev_feat, non_ev_feat

    finally:
        del x_atk_tr, y_atk_tr, x_atk_ev, clf
        gc.collect()


def _make_art_feature_fn(feature_type, ft_model, units, use_logits, batch_size):
    """Returns a (x, y) -> features callable for the ART attack."""
    if feature_type == "prediction":
        return lambda x, _: _predictions(ft_model, x, batch_size)
    if feature_type == "loss":
        return lambda x, y: _per_sample_loss(ft_model, x, y, units, use_logits, batch_size)
    raise ValueError("config.feature_type must be 'loss', 'combined', or 'prediction'.")


def _attack_art(ft_model, config, units, x_splits, y_splits):
    """ART MembershipInferenceBlackBox attack using loss or prediction features."""
    try:
        from art.estimators.classification import KerasClassifier
        from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
    except Exception as e:
        raise ImportError(
            "Missing ART. Install with: pip install 'adversarial-robustness-toolbox[tensorflow]'"
        ) from e

    x_mem_tr, x_non_tr, x_mem_ev, x_non_ev = x_splits
    y_mem_tr_idx, y_non_tr_idx, y_mem_ev_idx, y_non_ev_idx = y_splits

    feats = _make_art_feature_fn(config.feature_type, ft_model, units, config.use_logits, config.batch_size)

    art_est = KerasClassifier(model=ft_model, clip_values=config.clip_values, use_logits=bool(config.use_logits))
    mia = MembershipInferenceBlackBox(
        estimator=art_est, input_type="prediction",
        attack_model_type=str(config.attack_model_type),
        nn_model_epochs=int(config.nn_model_epochs),
        nn_model_batch_size=int(config.nn_model_batch_size),
        nn_model_learning_rate=float(config.nn_model_learning_rate),
    )

    with tf.device(_MIA_DEVICE):
        mem_tr_feat = feats(x_mem_tr, y_mem_tr_idx)
        non_tr_feat = feats(x_non_tr, y_non_tr_idx)
        mem_ev_feat = feats(x_mem_ev, y_mem_ev_idx)
        non_ev_feat = feats(x_non_ev, y_non_ev_idx)

    mia.fit(x=x_mem_tr, y=y_mem_tr_idx, test_x=x_non_tr, test_y=y_non_tr_idx,
            pred=mem_tr_feat, test_pred=non_tr_feat)
    s_mem, c_mem = _membership_scores(mia.infer(x=x_mem_ev, y=y_mem_ev_idx, pred=mem_ev_feat, probabilities=True))
    s_non, c_non = _membership_scores(mia.infer(x=x_non_ev, y=y_non_ev_idx, pred=non_ev_feat, probabilities=True))
    return s_mem, c_mem, s_non, c_non, mem_ev_feat, non_ev_feat


def mia_privacy_score_for_coalition(
    coalition_model: tf.keras.Model,
    x_test: Any,
    y_test: Any,
    *,
    seed: int,
    round_num: int,
    splits: Optional[MIASplits] = None,
    config: Optional[MIAConfig] = None,
    prepared: Optional[Dict[str, Any]] = None,
    ft_template: Optional[tf.keras.Model] = None,
) -> Dict[str, Any]:
    if config is None:
        config = MIAConfig()
    if splits is None:
        splits = make_mia_splits(y_test, seed=seed, round_num=round_num)

    if prepared is None:
        prepared = prepare_mia_inputs(x_test, y_test, splits)

    x_mem_tr = prepared["x_mem_tr"]
    x_mem_ev = prepared["x_mem_ev"]
    x_non_tr = prepared["x_non_tr"]
    x_non_ev = prepared["x_non_ev"]
    x_mem_all = prepared["x_mem_all"]

    y_mem_tr_idx = prepared["y_mem_tr_idx"]
    y_mem_ev_idx = prepared["y_mem_ev_idx"]
    y_non_tr_idx = prepared["y_non_tr_idx"]
    y_non_ev_idx = prepared["y_non_ev_idx"]
    y_mem_all_idx = prepared["y_mem_all_idx"]

    x_splits = (x_mem_tr, x_non_tr, x_mem_ev, x_non_ev)
    y_splits = (y_mem_tr_idx, y_non_tr_idx, y_mem_ev_idx, y_non_ev_idx)

    ft_model = None
    mem_ev_feat = non_ev_feat = None
    s_mem = c_mem = s_non = c_non = None

    try:
        ft_model, units = _finetune_on_cpu(coalition_model, config, x_mem_all, y_mem_all_idx, ft_template=ft_template)

        if config.feature_type in ("combined", "combined_delta"):
            s_mem, c_mem, s_non, c_non, mem_ev_feat, non_ev_feat = _attack_combined(
                ft_model, config, units, seed, x_splits, y_splits,
                orig_model=coalition_model if config.feature_type == "combined_delta" else None)
        else:
            s_mem, c_mem, s_non, c_non, mem_ev_feat, non_ev_feat = _attack_art(
                ft_model, config, units, x_splits, y_splits)

        y_true = np.concatenate([np.ones_like(c_mem), np.zeros_like(c_non)])
        y_pred = np.concatenate([c_mem, c_non])
        y_score = np.concatenate([s_mem, s_non])

        mia_acc = float(accuracy_score(y_true, y_pred))
        try:
            mia_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            mia_auc = float("nan")

        if config.feature_type in ("combined", "combined_delta"):
            try:
                # Yeom threshold: score = -loss_ft (lower loss → more likely member)
                # For combined_delta col0=delta, col1=loss_ft; for combined col0=loss
                loss_col = 1 if config.feature_type == "combined_delta" else 0
                yeom_scores = np.concatenate([-mem_ev_feat[:, loss_col], -non_ev_feat[:, loss_col]])
                yeom_auc = float(roc_auc_score(y_true, yeom_scores))
                if not np.isnan(yeom_auc) and (np.isnan(mia_auc) or yeom_auc > mia_auc):
                    mia_auc = yeom_auc
                # Also try delta directly (larger delta → more likely member)
                if config.feature_type == "combined_delta":
                    delta_scores = np.concatenate([mem_ev_feat[:, 0], non_ev_feat[:, 0]])
                    delta_auc = float(roc_auc_score(y_true, delta_scores))
                    if not np.isnan(delta_auc) and delta_auc > mia_auc:
                        mia_auc = delta_auc
            except Exception:
                pass

        score = mia_auc if config.use_auc_for_privacy and not np.isnan(mia_auc) else mia_acc
        advantage = 2.0 * abs(score - 0.5)
        privacy = float(np.clip(1.0 - advantage, 0.0, 1.0))

        return {
            "privacy": privacy,
            "mia_acc": mia_acc,
            "mia_auc": mia_auc,
            "member_eval_size": int(len(c_mem)),
            "nonmember_eval_size": int(len(c_non)),
        }

    finally:
        if ft_template is None:
            del ft_model
        del mem_ev_feat, non_ev_feat, s_mem, c_mem, s_non, c_non
        gc.collect()


def prepare_mia_inputs(x_test: Any, y_test: Any, splits: MIASplits) -> Dict[str, Any]:
    idx_mem_all = np.concatenate([splits.idx_mem_tr, splits.idx_mem_ev])

    prepared = {
        "x_mem_tr": _take(x_test, splits.idx_mem_tr),
        "x_mem_ev": _take(x_test, splits.idx_mem_ev),
        "x_non_tr": _take(x_test, splits.idx_non_tr),
        "x_non_ev": _take(x_test, splits.idx_non_ev),
        "x_mem_all": _take(x_test, idx_mem_all),

        "y_mem_tr_idx": _to_label_index(_take(y_test, splits.idx_mem_tr)),
        "y_mem_ev_idx": _to_label_index(_take(y_test, splits.idx_mem_ev)),
        "y_non_tr_idx": _to_label_index(_take(y_test, splits.idx_non_tr)),
        "y_non_ev_idx": _to_label_index(_take(y_test, splits.idx_non_ev)),
        "y_mem_all_idx": _to_label_index(_take(y_test, idx_mem_all)),
    }
    return prepared


# -----------------------------
# Minimal internal utilities (kept compact)
# -----------------------------
def _take(x: Any, idx: np.ndarray) -> Any:
    if isinstance(x, np.ndarray):
        return x[idx]
    if isinstance(x, tf.Tensor):
        return tf.gather(x, idx)
    if isinstance(x, (list, tuple)):
        return type(x)(_take(xi, idx) for xi in x)
    if isinstance(x, dict):
        return {k: _take(v, idx) for k, v in x.items()}
    return x[idx]


def _to_label_index(y: Any) -> np.ndarray:
    y_np = np.asarray(y)
    if y_np.ndim == 2 and y_np.shape[1] >= 2:
        return np.argmax(y_np, axis=1).astype(np.int64)
    return y_np.reshape(-1).astype(np.int64)


def _output_units(model: tf.keras.Model) -> int:
    out = model.output_shape
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        # (None, units)
        return int(out[-1])
    return 1


def _predictions(model: tf.keras.Model, x: Any, batch_size: int) -> np.ndarray:
    p = model.predict(x, batch_size=batch_size, verbose=0)
    p = np.asarray(p)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    return p.astype(np.float32)


def _per_sample_loss(
    model: tf.keras.Model,
    x: Any,
    y_idx: np.ndarray,
    units: int,
    from_logits: bool,
    batch_size: int,
) -> np.ndarray:
    # compute predictions
    y_pred = model.predict(x, batch_size=batch_size, verbose=0)
    y_pred = tf.convert_to_tensor(y_pred)

    if units == 1:
        y_true = tf.convert_to_tensor(y_idx.astype(np.float32).reshape(-1, 1))
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits, reduction="none")
        l = loss_fn(y_true, y_pred)  # (N,)
    else:
        y_true = tf.convert_to_tensor(y_idx.astype(np.int64).reshape(-1))
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction="none")
        l = loss_fn(y_true, y_pred)  # (N,)

    l = tf.reshape(l, (-1, 1))
    return l.numpy().astype(np.float32)


def _combined_features(
    model: tf.keras.Model,
    x: Any,
    y_idx: np.ndarray,
    units: int,
    from_logits: bool,
    batch_size: int,
) -> np.ndarray:
    """4-dimensional features: [loss, max_confidence, entropy, correctness].
    Richer signal than loss alone — members tend to have lower loss,
    higher confidence, lower entropy, and higher correctness after fine-tuning.
    """
    y_pred = model.predict(x, batch_size=batch_size, verbose=0)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    # --- loss ---
    y_pred_t = tf.convert_to_tensor(y_pred)
    if units == 1:
        y_true_t = tf.convert_to_tensor(y_idx.astype(np.float32).reshape(-1, 1))
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits, reduction="none")
        loss = loss_fn(y_true_t, y_pred_t).numpy().reshape(-1)
        # for binary: expand to 2-class probs for entropy/confidence
        p1 = y_pred.reshape(-1)
        probs = np.stack([1.0 - p1, p1], axis=1)
    else:
        y_true_t = tf.convert_to_tensor(y_idx.astype(np.int64).reshape(-1))
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction="none")
        loss = loss_fn(y_true_t, y_pred_t).numpy().reshape(-1)
        probs = y_pred  # (N, C)

    # --- max confidence ---
    max_conf = probs.max(axis=1)

    # --- entropy (normalised to [0,1]) ---
    eps = 1e-12
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    n_classes = probs.shape[1]
    ent_norm = ent / np.log(n_classes + eps)

    # --- correctness ---
    pred_class = probs.argmax(axis=1)
    correct = (pred_class == y_idx).astype(np.float32)

    feats = np.stack([loss, max_conf, ent_norm, correct], axis=1)
    return feats.astype(np.float32)


def _membership_scores(p: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    ART infer(probabilities=True) kimenet lehet:
      - (N,2)  -> [P(non-member), P(member)]
      - (N,1)  -> P(member)
      - (N,)   -> P(member) vagy 0/1 label
    Mi mindig visszaadjuk:
      score = P(member)  (float32)
      cls   = score >= 0.5 (int)
    """
    arr = np.asarray(p)

    if arr.ndim == 0:
        score = np.asarray([float(arr)], dtype=np.float32)
        cls = (score >= 0.5).astype(int)
        return score, cls

    if arr.ndim == 1:
        score = arr.astype(np.float32)
        cls = (score >= 0.5).astype(int)
        return score, cls

    if arr.ndim == 2:
        if arr.shape[1] == 1:
            score = arr[:, 0].astype(np.float32)
        else:
            # ha (N,2) akkor a member tipikusan az utolsó oszlop
            score = arr[:, -1].astype(np.float32)
        cls = (score >= 0.5).astype(int)
        return score, cls

    # fallback: lapítsuk
    flat = arr.reshape(-1).astype(np.float32)
    cls = (flat >= 0.5).astype(int)
    return flat, cls