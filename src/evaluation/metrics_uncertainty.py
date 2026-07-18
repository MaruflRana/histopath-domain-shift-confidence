"""Deterministic uncertainty / error-detection metrics (Milestone 5B).

This module computes uncertainty scores and error-detection metrics **purely from
saved prediction tables** (per-example ``prob_0``, ``prob_1``, ``logit_0``, ``logit_1``,
``label``, ``pred_label``, ``correct``). It performs **no** model inference, no training,
no MC-dropout, no ensembling, and no selective-prediction thresholding. Everything here
is a closed-form function of already-exported predictions, so results are exactly
reproducible from the CSVs.

Conventions (shared with the rest of the pipeline):
  * positive class = **tumor = 1**; ``prob_1`` is P(tumor).
  * ``confidence = max(prob_0, prob_1)`` (probability of the predicted class).
  * an **error** is ``correct == False`` (equivalently ``pred_label != label``).
  * a **false negative** is ``label == 1 and pred_label == 0`` — a missed tumor.
  * every uncertainty score is oriented so **higher = more uncertain**.

Score functions (vectorised, numpy):
  * ``max_confidence``              — max(prob_0, prob_1)              (higher = certain)
  * ``uncertainty_from_confidence`` — 1 - max(prob_0, prob_1)
  * ``predictive_entropy``          — -Σ p log p  (natural log; max = ln 2 at p=0.5)
  * ``probability_margin``          — |prob_1 - prob_0|               (higher = certain)
  * ``uncertainty_from_margin``     — 1 - |prob_1 - prob_0|
  * ``energy_score_from_logits``    — -logsumexp(logits)  (higher = more uncertain/OOD)

Evaluation functions (operate on a prediction DataFrame):
  * ``compute_error_detection_metrics``
  * ``compute_high_confidence_error_summary``
  * ``compute_high_confidence_false_negative_summary``
  * ``compute_risk_coverage_curve``

Edge cases are handled explicitly: a metric that is undefined (e.g. AUROC when only one
error class is present, or a rate with a zero denominator) is returned as ``None`` with a
recorded warning, rather than raising or silently reporting a misleading number.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Tolerance for the [0, 1] range check and the prob_0 + prob_1 == 1 check.
_PROB_TOL = 1e-6
_SUM_TOL = 1e-5


# --------------------------------------------------------------------------- #
# Input coercion / validation
# --------------------------------------------------------------------------- #
def _as_prob_pair(
    prob_0: Sequence[float] | np.ndarray,
    prob_1: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``prob_0`` / ``prob_1`` to float64 arrays and validate them.

    Checks: equal length, non-empty, finite, within ``[0, 1]`` (up to a tiny tolerance),
    and ``prob_0 + prob_1 ~= 1``. Tiny numerical overshoot is clipped back into range.
    """
    p0 = np.asarray(prob_0, dtype=np.float64).ravel()
    p1 = np.asarray(prob_1, dtype=np.float64).ravel()
    if p0.shape[0] != p1.shape[0]:
        raise ValueError(f"prob_0 and prob_1 length mismatch: {p0.shape[0]} vs {p1.shape[0]}.")
    if p0.shape[0] == 0:
        raise ValueError("Cannot compute uncertainty scores on empty input (n == 0).")
    if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
        raise ValueError("prob_0/prob_1 contain non-finite values (NaN/Inf).")
    for name, p in (("prob_0", p0), ("prob_1", p1)):
        if np.any(p < -_PROB_TOL) or np.any(p > 1.0 + _PROB_TOL):
            raise ValueError(f"{name} out of [0,1] (min={float(p.min())}, max={float(p.max())}).")
    s = p0 + p1
    if np.any(np.abs(s - 1.0) > _SUM_TOL):
        worst = float(np.max(np.abs(s - 1.0)))
        raise ValueError(f"prob_0 + prob_1 != 1 (max deviation {worst} > {_SUM_TOL}).")
    return np.clip(p0, 0.0, 1.0), np.clip(p1, 0.0, 1.0)


def _as_logit_pair(
    logit_0: Sequence[float] | np.ndarray,
    logit_1: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``logit_0`` / ``logit_1`` to float64 arrays and finiteness-check them."""
    l0 = np.asarray(logit_0, dtype=np.float64).ravel()
    l1 = np.asarray(logit_1, dtype=np.float64).ravel()
    if l0.shape[0] != l1.shape[0]:
        raise ValueError(f"logit_0 and logit_1 length mismatch: {l0.shape[0]} vs {l1.shape[0]}.")
    if l0.shape[0] == 0:
        raise ValueError("Cannot compute energy on empty input (n == 0).")
    if not (np.all(np.isfinite(l0)) and np.all(np.isfinite(l1))):
        raise ValueError("logit_0/logit_1 contain non-finite values (NaN/Inf).")
    return l0, l1


def _error_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean array: ``True`` where the example is an **error** (``correct == False``).

    ``correct`` is stored as 0/1 in the saved CSVs; this also tolerates real booleans and
    the strings ``"True"``/``"False"``. Falls back to ``pred_label != label`` only if a
    ``correct`` column is absent.
    """
    if "correct" in df.columns:
        c = df["correct"]
        if c.dtype == bool:
            correct = c.to_numpy()
        elif c.dtype == object:
            correct = c.astype(str).str.strip().str.lower().map(
                {"1": True, "0": False, "true": True, "false": False}
            ).to_numpy()
            if pd.isna(correct).any():
                raise ValueError("Unparseable value(s) in 'correct' column.")
            correct = correct.astype(bool)
        else:
            correct = c.to_numpy().astype(float) != 0.0
        return ~correct
    return df["pred_label"].to_numpy() != df["label"].to_numpy()


def _false_negative_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean array: ``True`` where ``label == 1 and pred_label == 0`` (missed tumor)."""
    label = df["label"].to_numpy()
    pred = df["pred_label"].to_numpy()
    return (label == 1) & (pred == 0)


# --------------------------------------------------------------------------- #
# Uncertainty score functions (higher = more uncertain, unless noted)
# --------------------------------------------------------------------------- #
def max_confidence(
    prob_0: Sequence[float] | np.ndarray,
    prob_1: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Confidence = probability of the predicted class = ``max(prob_0, prob_1)``.

    NOTE: this is a *confidence* (higher = more certain), not an uncertainty score.
    """
    p0, p1 = _as_prob_pair(prob_0, prob_1)
    return np.maximum(p0, p1)


def uncertainty_from_confidence(
    prob_0: Sequence[float] | np.ndarray,
    prob_1: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """``1 - max(prob_0, prob_1)``. Higher = more uncertain."""
    return 1.0 - max_confidence(prob_0, prob_1)


def predictive_entropy(
    prob_0: Sequence[float] | np.ndarray,
    prob_1: Sequence[float] | np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Shannon entropy of the 2-class distribution (natural log). Higher = more uncertain.

    Ranges from 0 (a one-hot prediction) to ``ln 2 ~= 0.6931`` at ``p = 0.5``.
    Probabilities are clipped to ``[eps, 1]`` only to keep ``log`` finite.
    """
    p0, p1 = _as_prob_pair(prob_0, prob_1)
    p0c = np.clip(p0, eps, 1.0)
    p1c = np.clip(p1, eps, 1.0)
    return -(p0 * np.log(p0c) + p1 * np.log(p1c))


def probability_margin(
    prob_0: Sequence[float] | np.ndarray,
    prob_1: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """``|prob_1 - prob_0|``. Higher = more certain (this is a margin, not uncertainty)."""
    p0, p1 = _as_prob_pair(prob_0, prob_1)
    return np.abs(p1 - p0)


def uncertainty_from_margin(
    prob_0: Sequence[float] | np.ndarray,
    prob_1: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """``1 - |prob_1 - prob_0|``. Higher = more uncertain."""
    return 1.0 - probability_margin(prob_0, prob_1)


def energy_score_from_logits(
    logit_0: Sequence[float] | np.ndarray,
    logit_1: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Free-energy score ``E = -logsumexp(logit_0, logit_1)`` (temperature 1).

    Following Liu et al. (2020) "Energy-based OOD Detection": in-distribution / confident
    inputs have **lower** energy, OOD / uncertain inputs have **higher** energy. So higher
    energy = more uncertain, consistent with the other scores here. Computed with a
    numerically stable log-sum-exp.
    """
    l0, l1 = _as_logit_pair(logit_0, logit_1)
    m = np.maximum(l0, l1)
    lse = m + np.log(np.exp(l0 - m) + np.exp(l1 - m))
    return -lse


# --------------------------------------------------------------------------- #
# Error-detection metrics
# --------------------------------------------------------------------------- #
def compute_error_detection_metrics(
    df: pd.DataFrame,
    score_col: str,
    higher_means_more_uncertain: bool = True,
) -> dict:
    """How well ``score_col`` separates errors from correct predictions.

    Treats **error (``correct == False``) as the positive class** and asks whether the
    uncertainty score ranks errors above correct predictions (error-detection AUROC). Also
    reports the mean score on the correct vs error subsets.

    Parameters
    ----------
    df:
        Prediction table containing ``score_col`` and a ``correct`` (or ``label`` +
        ``pred_label``) column.
    score_col:
        Column holding the per-example uncertainty (or confidence) score.
    higher_means_more_uncertain:
        If ``True`` the raw score is used as the error predictor; if ``False`` the score is
        negated first so that "more uncertain" always points toward the error class.

    Returns
    -------
    dict
        ``{score_col, higher_means_more_uncertain, n, n_error, n_correct, error_rate,
        error_detection_auroc, mean_uncertainty_correct, mean_uncertainty_error,
        warnings}``. ``error_detection_auroc`` is ``None`` (with a warning) when only one
        error class is present.
    """
    if score_col not in df.columns:
        raise ValueError(f"score_col {score_col!r} not found in DataFrame columns.")

    score = np.asarray(df[score_col].to_numpy(), dtype=np.float64)
    if score.shape[0] == 0:
        raise ValueError("Cannot compute error-detection metrics on empty input (n == 0).")
    if not np.all(np.isfinite(score)):
        raise ValueError(f"score_col {score_col!r} contains non-finite values.")

    is_error = _error_mask(df)
    n = int(score.shape[0])
    n_error = int(np.sum(is_error))
    n_correct = n - n_error

    warnings: list[str] = []

    mean_unc_correct: Optional[float] = (
        float(np.mean(score[~is_error])) if n_correct > 0 else None
    )
    mean_unc_error: Optional[float] = (
        float(np.mean(score[is_error])) if n_error > 0 else None
    )
    if n_correct == 0:
        warnings.append("mean_uncertainty_correct undefined (no correct predictions).")
    if n_error == 0:
        warnings.append("mean_uncertainty_error undefined (no errors).")

    # Orient the score so higher => more likely to be an error, then AUROC vs the error
    # label. AUROC needs both classes (at least one error AND one correct) present.
    auroc: Optional[float] = None
    if n_error == 0 or n_correct == 0:
        only = "all-correct" if n_error == 0 else "all-error"
        warnings.append(
            f"error_detection_auroc undefined: only one error class present ({only}); "
            "returning None."
        )
    else:
        oriented = score if higher_means_more_uncertain else -score
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(is_error.astype(np.int64), oriented))

    return {
        "score_col": score_col,
        "higher_means_more_uncertain": bool(higher_means_more_uncertain),
        "n": n,
        "n_error": n_error,
        "n_correct": n_correct,
        "error_rate": (n_error / n) if n > 0 else None,
        "error_detection_auroc": auroc,
        "mean_uncertainty_correct": mean_unc_correct,
        "mean_uncertainty_error": mean_unc_error,
        "warnings": warnings,
    }


def compute_high_confidence_error_summary(
    df: pd.DataFrame,
    confidence_col: str,
    thresholds: Sequence[float] = (0.90, 0.95, 0.99),
) -> list[dict]:
    """Count / rate of **errors among high-confidence predictions**.

    For each threshold ``t`` the "high-confidence" subset is ``confidence_col >= t``.
    Returns, per threshold: the subset size, how many of those are errors, and the error
    rate within the subset (``None`` when the subset is empty). High-confidence errors are
    the dangerous kind — the model is both wrong and sure.
    """
    if confidence_col not in df.columns:
        raise ValueError(f"confidence_col {confidence_col!r} not found in DataFrame columns.")
    conf = np.asarray(df[confidence_col].to_numpy(), dtype=np.float64)
    if not np.all(np.isfinite(conf)):
        raise ValueError(f"confidence_col {confidence_col!r} contains non-finite values.")

    is_error = _error_mask(df)
    n_total = int(conf.shape[0])
    n_error_total = int(np.sum(is_error))

    rows: list[dict] = []
    for t in thresholds:
        mask = conf >= float(t)
        n_high = int(np.sum(mask))
        n_high_err = int(np.sum(mask & is_error))
        rate = (n_high_err / n_high) if n_high > 0 else None
        rows.append({
            "confidence_col": confidence_col,
            "threshold": float(t),
            "n_total": n_total,
            "n_high_conf": n_high,
            "high_conf_coverage": (n_high / n_total) if n_total > 0 else None,
            "n_high_conf_error": n_high_err,
            "high_conf_error_rate": rate,
            "n_error_total": n_error_total,
        })
    return rows


def compute_high_confidence_false_negative_summary(
    df: pd.DataFrame,
    confidence_col: str,
    thresholds: Sequence[float] = (0.90, 0.95, 0.99),
) -> list[dict]:
    """Count / rate of **false negatives among high-confidence predictions**.

    A false negative (``label == 1 and pred_label == 0``) is a missed tumor. Among
    predictions the model is confident about (``confidence_col >= t``), how many are these
    confidently-missed tumors? Returns, per threshold: the high-confidence subset size, the
    number of high-confidence FNs, the FN rate within the subset, the total FN count, and
    the fraction of all FNs that fall in the high-confidence subset (``None`` on a zero
    denominator).
    """
    if confidence_col not in df.columns:
        raise ValueError(f"confidence_col {confidence_col!r} not found in DataFrame columns.")
    conf = np.asarray(df[confidence_col].to_numpy(), dtype=np.float64)
    if not np.all(np.isfinite(conf)):
        raise ValueError(f"confidence_col {confidence_col!r} contains non-finite values.")

    is_fn = _false_negative_mask(df)
    n_total = int(conf.shape[0])
    n_fn_total = int(np.sum(is_fn))

    rows: list[dict] = []
    for t in thresholds:
        mask = conf >= float(t)
        n_high = int(np.sum(mask))
        n_high_fn = int(np.sum(mask & is_fn))
        rate = (n_high_fn / n_high) if n_high > 0 else None
        captured = (n_high_fn / n_fn_total) if n_fn_total > 0 else None
        rows.append({
            "confidence_col": confidence_col,
            "threshold": float(t),
            "n_total": n_total,
            "n_high_conf": n_high,
            "n_high_conf_false_negative": n_high_fn,
            "high_conf_false_negative_rate": rate,
            "n_false_negative_total": n_fn_total,
            "fraction_of_fn_in_high_conf": captured,
        })
    return rows


# --------------------------------------------------------------------------- #
# Risk–coverage (reporting only; NOT selective-prediction threshold selection)
# --------------------------------------------------------------------------- #
def compute_aurc(df: pd.DataFrame, uncertainty_col: str) -> Optional[float]:
    """Area Under the Risk–Coverage curve, from the full ranking by ``uncertainty_col``.

    Examples are ranked from most-certain (lowest uncertainty) to least. Sweeping coverage
    from ``1/n`` to ``1`` and computing the cumulative error rate (risk) of the retained,
    most-confident prefix gives a risk–coverage curve; AURC is its trapezoidal area over
    coverage. Lower AURC = uncertainty ranks errors better. This is a descriptive summary
    of already-saved predictions — it selects **no** operating threshold.

    Ties in the score are broken by the stable sort order (dataset order); with ~2-decimal
    ties this only perturbs AURC negligibly. Returns ``None`` (undefined) if ``n == 0``.
    """
    unc = np.asarray(df[uncertainty_col].to_numpy(), dtype=np.float64)
    is_error = _error_mask(df).astype(np.float64)
    n = int(unc.shape[0])
    if n == 0:
        return None
    order = np.argsort(unc, kind="stable")          # ascending uncertainty = descending confidence
    err_sorted = is_error[order]
    cum_err = np.cumsum(err_sorted)
    k = np.arange(1, n + 1)
    risk = cum_err / k                              # risk of the top-k most-confident prefix
    coverage = k / n
    # np.trapz was removed in NumPy 2.x -> np.trapezoid; fall back for NumPy 1.x.
    trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapezoid(risk, coverage))


def compute_risk_coverage_curve(
    df: pd.DataFrame,
    uncertainty_col: str,
    coverage_levels: Sequence[float] = (1.0, 0.95, 0.90, 0.80, 0.70, 0.50),
) -> dict:
    """Risk (error rate) of the most-confident retained fraction, at fixed coverages.

    Examples are ranked by ``uncertainty_col`` ascending (most-confident first). For each
    requested coverage ``c`` we keep the top ``round(c * n)`` most-confident examples and
    report the error rate (risk) within that retained set. Coverage ``1.0`` reproduces the
    overall error rate. This characterises how error rate would fall if the least-confident
    predictions were withheld — but it **does not** choose or commit to any threshold
    (selective prediction is a later, separately authorized milestone).

    Returns ``{uncertainty_col, n, aurc, points}`` where ``points`` is a list of
    ``{coverage, n_selected, n_error_selected, risk}`` (``risk`` is ``None`` if the
    retained set is empty).
    """
    if uncertainty_col not in df.columns:
        raise ValueError(f"uncertainty_col {uncertainty_col!r} not found in DataFrame columns.")
    unc = np.asarray(df[uncertainty_col].to_numpy(), dtype=np.float64)
    if not np.all(np.isfinite(unc)):
        raise ValueError(f"uncertainty_col {uncertainty_col!r} contains non-finite values.")

    is_error = _error_mask(df).astype(np.int64)
    n = int(unc.shape[0])
    order = np.argsort(unc, kind="stable")          # most-confident (lowest uncertainty) first
    err_sorted = is_error[order]

    points: list[dict] = []
    for c in coverage_levels:
        c = float(c)
        k = int(round(c * n))
        k = max(0, min(n, k))
        if k == 0:
            points.append({
                "coverage": c, "n_selected": 0, "n_error_selected": 0, "risk": None,
            })
            continue
        n_err = int(np.sum(err_sorted[:k]))
        points.append({
            "coverage": c,
            "n_selected": k,
            "n_error_selected": n_err,
            "risk": n_err / k,
        })

    return {
        "uncertainty_col": uncertainty_col,
        "n": n,
        "aurc": compute_aurc(df, uncertainty_col),
        "points": points,
    }
