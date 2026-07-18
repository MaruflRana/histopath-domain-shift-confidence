"""Operating-point (threshold) analysis for the tumor / non-tumor classifier.

Milestone 7B. This module is **threshold analysis only** — it consumes already-saved
prediction scores (``prob_1``) and ground-truth labels; it does **no** model inference,
**no** training, **no** calibration. Positive class is **tumor = 1**.

The purpose is to compare two ID-trained models (plain ERM vs stain-augmentation ERM)
*fairly*: instead of judging both at the arbitrary default threshold 0.5, we pick an
operating threshold on ``id_val`` (in-distribution, where model selection is allowed) and
then apply that *same* threshold to ``ood_val`` for a development-stage OOD comparison.
This separates two very different failure modes:

  * the ranking (score ordering) genuinely worsened under shift  -> AUROC / AUPRC drop,
  * only the *threshold* became mis-placed (too conservative)    -> AUROC holds but the
    0.5 operating point collapses sensitivity.

Definitions used throughout (tumor = positive = 1):
  * ``pred = 1 if prob_1 >= threshold else 0``
  * ``specificity = TN / (TN + FP)``
  * ``sensitivity = TP / (TP + FN)``
  * false negative = ``label == 1 and pred == 0``
  * false positive = ``label == 0 and pred == 1``

Edge cases are handled explicitly (returned as ``None`` with the reason recorded) rather
than allowed to raise: single-class inputs, zero-denominator ratios, and unachievable
targets.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Metric field order shared by the point-metric dicts (keeps CSV columns stable).
METRIC_FIELDS = [
    "threshold",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
    "false_negative_rate_over_tumors",
    "predicted_positive_rate",
]


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Return numerator/denominator, or ``None`` when the denominator is 0."""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _as_arrays(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to 1-D arrays and validate shape / label domain (raises on bad input)."""
    y_true = np.asarray(y_true).ravel().astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"y_true and y_score length mismatch: {y_true.shape[0]} vs {y_score.shape[0]}."
        )
    if y_true.shape[0] == 0:
        raise ValueError("Cannot compute operating-point metrics on empty input (n == 0).")
    unique = set(np.unique(y_true).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"y_true must contain only 0/1 labels; got values {sorted(unique)}.")
    return y_true, y_score


def compute_metrics_at_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict:
    """Confusion-matrix + summary metrics for ``prob_1 >= threshold`` (tumor = 1).

    Returns a dict with the keys in :data:`METRIC_FIELDS` plus ``n``, ``positive_class``,
    and ``warnings``. Ratio metrics with a zero denominator are returned as ``None`` (0/0
    is never silently reported as 0). ``false_negative_rate_over_tumors`` is
    ``FN / (FN + TP) = 1 - sensitivity``; ``predicted_positive_rate`` is
    ``(TP + FP) / n``.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    n = int(y_true.shape[0])
    threshold = float(threshold)

    y_pred = (y_score >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    warnings: list[str] = []

    accuracy = (tp + tn) / n
    sensitivity = _safe_ratio(tp, tp + fn)   # TPR / recall over tumors
    specificity = _safe_ratio(tn, tn + fp)   # TNR
    precision = _safe_ratio(tp, tp + fp)     # PPV

    if sensitivity is None:
        warnings.append("sensitivity undefined (no actual tumor examples).")
    if specificity is None:
        warnings.append("specificity undefined (no actual non-tumor examples).")
    if precision is None:
        warnings.append("precision undefined (no predicted positives).")

    if sensitivity is not None and specificity is not None:
        balanced_accuracy: Optional[float] = 0.5 * (sensitivity + specificity)
    else:
        balanced_accuracy = None
        warnings.append("balanced_accuracy undefined (sensitivity or specificity undefined).")

    if precision is not None and sensitivity is not None and (precision + sensitivity) > 0:
        f1: Optional[float] = 2 * precision * sensitivity / (precision + sensitivity)
    else:
        f1 = None
        warnings.append("f1 undefined (precision/recall undefined or both zero).")

    # FN rate over the tumor population = 1 - sensitivity (None if no tumors).
    fnr_over_tumors = _safe_ratio(fn, fn + tp)
    predicted_positive_rate = (tp + fp) / n

    return {
        "n": n,
        "threshold": threshold,
        "accuracy": float(accuracy),
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_negative_rate_over_tumors": fnr_over_tumors,
        "predicted_positive_rate": float(predicted_positive_rate),
        "positive_class": 1,
        "warnings": warnings,
    }


def _candidate_thresholds(y_score: np.ndarray) -> np.ndarray:
    """Distinct decision thresholds, ascending, including an 'accept nothing' cut.

    With ``pred = (score >= t)``, using each observed score value as ``t`` enumerates every
    distinct partition (the smallest observed score reproduces 'predict all positive'). We
    append one threshold strictly above the maximum so that 'predict all negative'
    (specificity 1.0) is always reachable.
    """
    uniq = np.unique(y_score)
    top = float(uniq[-1])
    # a cut above every score -> all predictions negative
    above_all = np.nextafter(top, np.inf)
    if above_all <= top:  # numerical guard for extreme values
        above_all = top + 1.0
    return np.concatenate([uniq, np.asarray([above_all], dtype=np.float64)])


def threshold_for_target_specificity(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    target_specificity: float,
) -> Optional[float]:
    """Smallest threshold whose ``id``-set specificity is >= ``target_specificity``.

    Specificity is non-decreasing in the threshold, so the *smallest* qualifying threshold
    keeps sensitivity as high as possible. Returns ``None`` if specificity is undefined
    (no negatives present) — with negatives present the 'predict all negative' cut always
    attains specificity 1.0, so a target in [0, 1] is always reachable.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    n_neg = int(np.sum(y_true == 0))
    if n_neg == 0:
        return None
    target = float(target_specificity)
    for t in _candidate_thresholds(y_score):  # ascending
        y_pred = (y_score >= t).astype(np.int64)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        spec = tn / (tn + fp)  # denominator == n_neg > 0
        if spec >= target:
            return float(t)
    return None  # unreachable when n_neg > 0, but explicit


def threshold_for_target_sensitivity(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    target_sensitivity: float,
) -> Optional[float]:
    """Largest threshold whose ``id``-set sensitivity is >= ``target_sensitivity``.

    Sensitivity is non-increasing in the threshold, so the *largest* qualifying threshold
    keeps specificity as high as possible. Returns ``None`` if sensitivity is undefined
    (no positives present) — with positives present the smallest observed score attains
    sensitivity 1.0, so a target in [0, 1] is always reachable.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return None
    target = float(target_sensitivity)
    best: Optional[float] = None
    for t in _candidate_thresholds(y_score):  # ascending; keep the largest that qualifies
        y_pred = (y_score >= t).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        sens = tp / (tp + fn)  # denominator == n_pos > 0
        if sens >= target:
            best = float(t)
    return best


def make_threshold_sweep(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] | np.ndarray | None = None,
) -> list[dict]:
    """Metrics at a grid of thresholds (default 0.00..1.00 step 0.01).

    Returns a list of metric dicts (one per threshold), each with the :data:`METRIC_FIELDS`
    keys plus ``n``. Useful for ROC / sensitivity-specificity trade-off plots and for
    tabulating how each model behaves across the whole threshold range.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    if thresholds is None:
        thresholds = np.round(np.linspace(0.0, 1.0, 101), 4)
    rows: list[dict] = []
    for t in thresholds:
        m = compute_metrics_at_threshold(y_true, y_score, float(t))
        rows.append(m)
    return rows


def apply_operating_points(
    id_val_df,
    ood_val_df,
    target_specificities: Sequence[float],
    target_sensitivities: Sequence[float],
) -> list[dict]:
    """Select thresholds on ``id_val``, then evaluate the *same* threshold on ``ood_val``.

    ``id_val_df`` / ``ood_val_df`` are pandas DataFrames with ``label`` and ``prob_1``
    columns. For each target specificity and each target sensitivity, a threshold is chosen
    on ``id_val`` only; both splits are then measured at that threshold. Returns a list of
    **long-format** rows (one per criterion x target x split)::

        {criterion, target, threshold, achieved_on_id_val, split, <METRIC_FIELDS...>}

    ``threshold`` is the id_val-selected value (identical across the two splits). If a
    target is unachievable / undefined on ``id_val``, ``threshold`` is ``None`` and the
    metric fields are ``None`` (``achieved_on_id_val`` is False).
    """
    id_y = id_val_df["label"].to_numpy()
    id_s = id_val_df["prob_1"].to_numpy(dtype=np.float64)
    ood_y = ood_val_df["label"].to_numpy()
    ood_s = ood_val_df["prob_1"].to_numpy(dtype=np.float64)

    rows: list[dict] = []

    def _emit(criterion: str, target: float, threshold: Optional[float]) -> None:
        achieved = threshold is not None
        for split, y, s in (("id_val", id_y, id_s), ("ood_val", ood_y, ood_s)):
            row: dict = {
                "criterion": criterion,
                "target": float(target),
                "threshold": (float(threshold) if achieved else None),
                "achieved_on_id_val": achieved,
                "split": split,
            }
            if achieved:
                m = compute_metrics_at_threshold(y, s, float(threshold))
                for k in METRIC_FIELDS:
                    row[k] = m[k]
            else:
                for k in METRIC_FIELDS:
                    row[k] = None
                row["threshold"] = None
            rows.append(row)

    for spec in target_specificities:
        t = threshold_for_target_specificity(id_y, id_s, float(spec))
        _emit("specificity", float(spec), t)

    for sens in target_sensitivities:
        t = threshold_for_target_sensitivity(id_y, id_s, float(sens))
        _emit("sensitivity", float(sens), t)

    return rows
