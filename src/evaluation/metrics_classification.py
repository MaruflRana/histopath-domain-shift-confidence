"""Binary classification metrics for the tumor / non-tumor patch classifier.

Milestone 4B. Positive class is **tumor = 1**. Threshold-based metrics use a decision
threshold on ``prob_1`` (default 0.5); ranking metrics (AUROC / AUPRC) use the raw
``prob_1`` scores.

Edge cases are handled explicitly rather than allowed to raise:
  * empty input -> a clear ``ValueError`` (there is nothing to score),
  * a single class present in ``y_true`` -> AUROC and AUPRC are **undefined**; they are
    returned as ``None`` with a recorded warning, while the threshold metrics (which are
    still well-defined) are computed normally,
  * degenerate confusion-matrix rows/columns (e.g. no predicted positives) -> the
    dependent ratio is returned as ``None`` (0/0 is not silently reported as 0).

``sklearn`` is used for AUROC / AUPRC / balanced accuracy; the confusion-matrix counts
and simple ratios are computed directly so their edge behaviour is explicit.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Return numerator/denominator, or ``None`` when the denominator is 0."""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def compute_binary_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    prob_1: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute binary classification metrics with tumor (1) as the positive class.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels in {0, 1} (0 = non-tumor, 1 = tumor).
    prob_1:
        Predicted probability of the positive class (tumor). Used directly for AUROC /
        AUPRC and thresholded at ``threshold`` for the point metrics.
    threshold:
        Decision threshold on ``prob_1`` for the hard prediction (``prob_1 >= threshold``
        -> predict 1). Default 0.5.

    Returns
    -------
    dict
        ``{n, accuracy, balanced_accuracy, auroc, auprc, sensitivity, specificity,
        precision, f1, tn, fp, fn, tp, threshold, positive_class, warnings}``.
        ``auroc`` / ``auprc`` are ``None`` when undefined (single class in ``y_true``);
        ratio metrics with a zero denominator are likewise ``None``. ``warnings`` is a
        list of human-readable strings describing any such fallbacks.

    Raises
    ------
    ValueError
        If the inputs are empty or of mismatched length, or if ``y_true`` contains a
        value outside {0, 1}.
    """
    y_true = np.asarray(y_true).ravel()
    prob_1 = np.asarray(prob_1, dtype=np.float64).ravel()

    if y_true.shape[0] != prob_1.shape[0]:
        raise ValueError(
            f"y_true and prob_1 length mismatch: {y_true.shape[0]} vs {prob_1.shape[0]}."
        )
    n = int(y_true.shape[0])
    if n == 0:
        raise ValueError("Cannot compute metrics on empty input (n == 0).")

    y_true = y_true.astype(np.int64)
    unique = set(np.unique(y_true).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"y_true must contain only 0/1 labels; got values {sorted(unique)}.")

    warnings: list[str] = []

    # --- hard predictions + confusion-matrix counts (tumor = positive) --------- #
    y_pred = (prob_1 >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / n

    sensitivity = _safe_ratio(tp, tp + fn)   # recall / TPR (positives = tumor)
    specificity = _safe_ratio(tn, tn + fp)   # TNR
    precision = _safe_ratio(tp, tp + fp)     # PPV

    if sensitivity is None:
        warnings.append("sensitivity undefined (no actual positives / tumor examples).")
    if specificity is None:
        warnings.append("specificity undefined (no actual negatives / non-tumor examples).")
    if precision is None:
        warnings.append("precision undefined (no predicted positives).")

    # balanced accuracy = mean(sensitivity, specificity) when both are defined.
    if sensitivity is not None and specificity is not None:
        balanced_accuracy: Optional[float] = 0.5 * (sensitivity + specificity)
    else:
        balanced_accuracy = None
        warnings.append("balanced_accuracy undefined (sensitivity or specificity undefined).")

    # F1 = 2*precision*recall / (precision+recall), guarding a zero denominator.
    if precision is not None and sensitivity is not None and (precision + sensitivity) > 0:
        f1: Optional[float] = 2 * precision * sensitivity / (precision + sensitivity)
    else:
        f1 = None
        warnings.append("f1 undefined (precision/recall undefined or both zero).")

    # --- ranking metrics: require both classes present in y_true --------------- #
    auroc: Optional[float] = None
    auprc: Optional[float] = None
    if len(unique) < 2:
        only = next(iter(unique))
        warnings.append(
            f"AUROC/AUPRC undefined: y_true is single-class (all == {only}); "
            "returning None for both."
        )
    else:
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc = float(roc_auc_score(y_true, prob_1))
        auprc = float(average_precision_score(y_true, prob_1))

    return {
        "n": n,
        "accuracy": float(accuracy),
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": float(threshold),
        "positive_class": 1,
        "warnings": warnings,
    }
