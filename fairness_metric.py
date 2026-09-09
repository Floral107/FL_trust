# fairness_metric.py

import numpy as np


def _get_predicted_labels(model, X, batch_size=256, threshold=0.5):
    """
    Returns hard class labels from a Keras-like model.
    - Binary (sigmoid): y_pred >= threshold -> class 1, else 0
    - Multiclass (softmax): argmax over last axis
    """
    y_pred = model.predict(X, batch_size=batch_size, verbose=0)
    y_pred = np.asarray(y_pred)

    # Multiclass: shape (N, C), C > 1
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        return np.argmax(y_pred, axis=1)

    # Binary: shape (N,) or (N,1) with probabilities
    probs = y_pred.ravel()
    return (probs >= threshold).astype(int)


def _to_true_labels(y_test, threshold=0.5):
    """
    Converts y_test into hard labels:
    - one-hot / multiclass probs -> argmax
    - binary probs -> threshold
    - already int labels -> unchanged (after ravel)
    """
    y = np.asarray(y_test)

    # one-hot or multiclass probs
    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1).astype(int)

    # binary vector (could be {0,1} or probabilities)
    y = y.ravel()
    if np.issubdtype(y.dtype, np.floating):
        return (y >= threshold).astype(int)

    return y.astype(int)


def _get_cifar_sensitive_from_images(X, atol: float = 0.0) -> np.ndarray:
    """
    CIFAR spec sensitive attrib:
    1, ha a kep grayscale (R == G == B minden pixelen), kulonben 0.

    Feltetelezi: X.shape = (N, H, W, 3)
    """
    x_np = np.asarray(X)
    if x_np.ndim != 4 or x_np.shape[-1] != 3:
        raise ValueError(
            f"CIFAR sensitive attr only defined for (N,H,W,3) images, got shape {x_np.shape}."
        )

    ch0 = x_np[..., 0]
    ch1 = x_np[..., 1]
    ch2 = x_np[..., 2]

    if atol == 0.0:
        eq01 = np.all(ch0 == ch1, axis=(1, 2))
        eq12 = np.all(ch1 == ch2, axis=(1, 2))
    else:
        eq01 = np.all(np.abs(ch0 - ch1) <= atol, axis=(1, 2))
        eq12 = np.all(np.abs(ch1 - ch2) <= atol, axis=(1, 2))

    is_gray = np.logical_and(eq01, eq12)
    return is_gray.astype(int)


def fairness_score_demographic_parity(
    model,
    X_test,
    s_test,
    hat_y=1,
    group_value=1,
    batch_size=256,
    threshold=0.5,
):
    y_hat = _get_predicted_labels(model, X_test, batch_size=batch_size, threshold=threshold)
    s = np.asarray(s_test).ravel()

    if y_hat.shape[0] != s.shape[0]:
        raise ValueError("X_test and s_test must have the same number of samples.")

    in_S = (s == group_value)
    in_not_S = ~in_S

    if in_S.sum() == 0 or in_not_S.sum() == 0:
        raise ValueError("Both S and D'/S must contain at least one sample.")

    p_S     = np.mean(y_hat[in_S] == hat_y)
    p_not_S = np.mean(y_hat[in_not_S] == hat_y)

    return 1 - abs(p_S - p_not_S)


def equalized_odds_score(
    model,
    X_test,
    y_test,
    s_test,
    group_value=1,
    batch_size=256,
    threshold=0.5,
):
    """
    Equalized Odds score in [0,1], higher is better.

    For each class c (binary or multiclass, via one-vs-rest):
      TPR_c(g) = P(ŷ=c | y=c, s=g)
      FPR_c(g) = P(ŷ=c | y!=c, s=g)

    EO gap per class:
      gap_c = 0.5*(|TPR_c(S)-TPR_c(~S)| + |FPR_c(S)-FPR_c(~S)|)

    score_c = 1 - gap_c
    Final score = mean(score_c over computable classes)

    If a class lacks positives/negatives in a group (so TPR/FPR undefined),
    that class is skipped. If nothing is computable, returns 0.0.
    """
    y_hat = _get_predicted_labels(model, X_test, batch_size=batch_size, threshold=threshold)
    y_true = _to_true_labels(y_test, threshold=threshold)
    s = np.asarray(s_test).ravel()

    if y_hat.shape[0] != s.shape[0] or y_true.shape[0] != s.shape[0]:
        raise ValueError("X_test, y_test, and s_test must have the same number of samples.")

    in_S = (s == group_value)
    in_not_S = ~in_S

    if in_S.sum() == 0 or in_not_S.sum() == 0:
        # cannot compare groups
        return 0.0

    classes = np.unique(y_true)
    scores = []

    for c in classes:
        pos_S = in_S & (y_true == c)
        neg_S = in_S & (y_true != c)

        pos_N = in_not_S & (y_true == c)
        neg_N = in_not_S & (y_true != c)

        # Need both denominators to define TPR/FPR in both groups
        if pos_S.sum() == 0 or pos_N.sum() == 0 or neg_S.sum() == 0 or neg_N.sum() == 0:
            continue

        tpr_S = np.mean(y_hat[pos_S] == c)
        tpr_N = np.mean(y_hat[pos_N] == c)

        fpr_S = np.mean(y_hat[neg_S] == c)
        fpr_N = np.mean(y_hat[neg_N] == c)

        gap = 0.5 * (abs(tpr_S - tpr_N) + abs(fpr_S - fpr_N))
        score_c = 1.0 - gap
        # numerical safety
        score_c = float(np.clip(score_c, 0.0, 1.0))
        scores.append(score_c)

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))


def calculate_fairness_score(
    model,
    X_test,
    s_test,
    dataset: str,
    hat_y: int = 1,
    group_value: int = 1,
    batch_size: int = 256,
    threshold: float = 0.5,
    # NEW:
    y_test=None,
    fairness_metric: str = "dp",   # "dp" or "eo"
) -> float:
    ds = dataset.lower()
    fm = fairness_metric.lower().strip()

    # Determine sensitive attribute (Adult/CelebA use provided s_test; CIFAR derives from images)
    if "cifar" in ds:
        s_used = _get_cifar_sensitive_from_images(X_test)
    else:
        s_used = s_test

    if "adult" in ds or "cifar" in ds or "celeba" in ds:
        if fm in ("dp", "demographic_parity"):
            return fairness_score_demographic_parity(
                model=model,
                X_test=X_test,
                s_test=s_used,
                hat_y=hat_y,
                group_value=group_value,
                batch_size=batch_size,
                threshold=threshold,
            )

        if fm in ("eo", "equalized_odds"):
            if y_test is None:
                raise ValueError("Equalized Odds requires y_test (ground truth labels).")
            return equalized_odds_score(
                model=model,
                X_test=X_test,
                y_test=y_test,
                s_test=s_used,
                group_value=group_value,
                batch_size=batch_size,
                threshold=threshold,
            )

        raise ValueError(f"Unknown fairness_metric='{fairness_metric}'. Use 'dp' or 'eo'.")

    # IMDB (or any dataset where you didn't define a sensitive attr) stays as before
    return 0.0
