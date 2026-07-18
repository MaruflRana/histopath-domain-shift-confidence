"""Selective-prediction / abstention analysis from saved uncertainty scores (Milestone 5C).

This module converts the per-example uncertainty scores exported in Milestone 5B into
**selective-prediction / abstention policy tables** — coverage, risk, retained accuracy,
and (crucially for a tumor detector) how false negatives redistribute between the
*retained* and *abstained* sets as the model is allowed to abstain on its least-confident
predictions.

It reads **only** already-saved CSVs: no model inference, no training, no MC-dropout, no
ensembling. Every number is a closed-form function of the saved predictions, so results
are exactly reproducible.

Conventions (shared with the pipeline):
  * uncertainty score: **higher = more uncertain** (e.g. ``entropy``, ``energy``,
    ``uncertainty_confidence``, ``uncertainty_margin`` from Milestone 5B).
  * **retained** = the lowest-uncertainty (most-confident) examples we still predict on;
    **abstained** = the highest-uncertainty examples we decline to predict.
  * ``coverage = retained_n / total_n``  (fraction we still act on).
  * ``risk = retained_errors / retained_n`` (error rate among retained).
  * ``error`` = ``correct == False``; ``false negative`` = ``label == 1 and
    pred_label == 0`` (a missed tumor); tumor class = ``label == 1``.

Retention is defined by **rank** (a stable ascending sort of the uncertainty score, top-k
retained) so the actual coverage equals ``round(target * n) / n`` exactly and ties are
broken deterministically by dataset order. A reported ``threshold`` is the largest
uncertainty value among the retained set (retain iff ``score <= threshold``).

This is **development-stage** selective-prediction analysis on capped balanced subsets
(`id_val`, `ood_val`), **not** final clinical performance, and it commits to **no**
operating threshold for deployment. ``ood_test`` is never read.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.metrics_uncertainty import _error_mask, _false_negative_mask

DEFAULT_COVERAGE_LEVELS = (1.00, 0.95, 0.90, 0.80, 0.70, 0.50)


def _k_for_coverage(n: int, target_coverage: float) -> int:
    """Number retained for a target coverage: ``round(target * n)`` clamped to ``[0, n]``."""
    if n <= 0:
        return 0
    k = int(round(float(target_coverage) * n))
    return max(0, min(n, k))


def threshold_for_target_coverage(
    df: pd.DataFrame,
    uncertainty_col: str,
    target_coverage: float,
) -> Optional[float]:
    """Uncertainty threshold that retains ~``target_coverage`` of the most-confident rows.

    Rows are retained iff ``uncertainty_col <= threshold``. The threshold is the k-th
    smallest uncertainty score, where ``k = round(target_coverage * n)`` (rank-based, so
    coverage is exact up to rounding). Returns ``None`` when nothing is retained
    (``k == 0``) — there is no finite "retain iff <= t" threshold in that case.
    """
    if uncertainty_col not in df.columns:
        raise ValueError(f"uncertainty_col {uncertainty_col!r} not found in DataFrame columns.")
    unc = np.asarray(df[uncertainty_col].to_numpy(), dtype=np.float64)
    n = int(unc.shape[0])
    if not np.all(np.isfinite(unc)):
        raise ValueError(f"uncertainty_col {uncertainty_col!r} contains non-finite values.")
    k = _k_for_coverage(n, target_coverage)
    if k == 0:
        return None
    sorted_unc = np.sort(unc, kind="stable")
    return float(sorted_unc[k - 1])


def retained_subset_by_coverage(
    df: pd.DataFrame,
    uncertainty_col: str,
    target_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split ``df`` into (retained, abstained) at a target coverage.

    Retains the ``k = round(target_coverage * n)`` lowest-uncertainty rows (stable sort,
    so ties break by dataset order). Returns ``(retained_df, abstained_df, info)`` where
    ``info`` = ``{uncertainty_col, target_coverage, actual_coverage, threshold, retained_n,
    abstained_n}``. ``threshold`` is ``None`` when ``k == 0``.
    """
    if uncertainty_col not in df.columns:
        raise ValueError(f"uncertainty_col {uncertainty_col!r} not found in DataFrame columns.")
    unc = np.asarray(df[uncertainty_col].to_numpy(), dtype=np.float64)
    n = int(unc.shape[0])
    if not np.all(np.isfinite(unc)):
        raise ValueError(f"uncertainty_col {uncertainty_col!r} contains non-finite values.")

    order = np.argsort(unc, kind="stable")          # ascending uncertainty (most-confident first)
    k = _k_for_coverage(n, target_coverage)
    retained_pos = order[:k]
    abstained_pos = order[k:]

    retained_df = df.iloc[retained_pos]
    abstained_df = df.iloc[abstained_pos]
    threshold = float(np.sort(unc, kind="stable")[k - 1]) if k > 0 else None

    info = {
        "uncertainty_col": uncertainty_col,
        "target_coverage": float(target_coverage),
        "actual_coverage": (k / n) if n > 0 else None,
        "threshold": threshold,
        "retained_n": int(k),
        "abstained_n": int(n - k),
    }
    return retained_df, abstained_df, info


def _selective_row(
    df: pd.DataFrame,
    uncertainty_col: str,
    target_coverage: float,
    variant: Optional[str] = None,
) -> dict:
    """Compute one selective-prediction row (all fields) at a single coverage level."""
    unc = np.asarray(df[uncertainty_col].to_numpy(), dtype=np.float64)
    n = int(unc.shape[0])

    is_error = _error_mask(df)
    is_fn = _false_negative_mask(df)
    is_tumor = df["label"].to_numpy() == 1

    total_errors = int(np.sum(is_error))
    total_fn = int(np.sum(is_fn))
    total_tumors = int(np.sum(is_tumor))

    order = np.argsort(unc, kind="stable")
    k = _k_for_coverage(n, target_coverage)
    ret_pos = order[:k]
    abs_pos = order[k:]

    ret_err = int(np.sum(is_error[ret_pos]))
    ret_fn = int(np.sum(is_fn[ret_pos]))
    abs_err = total_errors - ret_err
    abs_fn = total_fn - ret_fn

    threshold = float(np.sort(unc, kind="stable")[k - 1]) if k > 0 else None
    retained_risk = (ret_err / k) if k > 0 else None
    retained_accuracy = (1.0 - retained_risk) if retained_risk is not None else None
    ret_fn_rate_ret = (ret_fn / k) if k > 0 else None
    ret_fn_rate_tumors = (ret_fn / total_tumors) if total_tumors > 0 else None

    row = {
        "variant": variant,
        "uncertainty_col": uncertainty_col,
        "target_coverage": float(target_coverage),
        "actual_coverage": (k / n) if n > 0 else None,
        "threshold": threshold,
        "retained_n": int(k),
        "abstained_n": int(n - k),
        "retained_errors": ret_err,
        "retained_risk": retained_risk,
        "retained_accuracy": retained_accuracy,
        "retained_false_negatives": ret_fn,
        "retained_false_negative_rate_over_retained": ret_fn_rate_ret,
        "retained_false_negative_rate_over_all_tumors": ret_fn_rate_tumors,
        "abstained_errors": abs_err,
        "abstained_false_negatives": abs_fn,
        "total_errors": total_errors,
        "total_false_negatives": total_fn,
        "total_tumors": total_tumors,
    }
    return row


def selective_metrics_at_coverages(
    df: pd.DataFrame,
    uncertainty_col: str,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
    variant: Optional[str] = None,
) -> list[dict]:
    """Selective-prediction metrics for ``uncertainty_col`` at each coverage level.

    Returns one dict per coverage level with the full field set (coverage, threshold,
    retained/abstained counts, risk, retained accuracy, false-negative breakdowns, and the
    dataset-level totals). ``variant`` is copied into each row if given.
    """
    if uncertainty_col not in df.columns:
        raise ValueError(f"uncertainty_col {uncertainty_col!r} not found in DataFrame columns.")
    unc = np.asarray(df[uncertainty_col].to_numpy(), dtype=np.float64)
    if unc.shape[0] == 0:
        raise ValueError("Cannot compute selective metrics on empty input (n == 0).")
    if not np.all(np.isfinite(unc)):
        raise ValueError(f"uncertainty_col {uncertainty_col!r} contains non-finite values.")
    return [_selective_row(df, uncertainty_col, c, variant=variant) for c in coverage_levels]


def summarize_selective_prediction(
    df: pd.DataFrame,
    uncertainty_col: str,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
    variant: Optional[str] = None,
) -> dict:
    """Bundle the per-coverage selective-prediction rows plus dataset-level context.

    Returns ``{variant, uncertainty_col, n, total_errors, total_false_negatives,
    total_tumors, points}`` where ``points`` is the list from
    :func:`selective_metrics_at_coverages`.
    """
    points = selective_metrics_at_coverages(df, uncertainty_col, coverage_levels, variant=variant)
    n = int(len(df))
    is_error = _error_mask(df)
    is_fn = _false_negative_mask(df)
    is_tumor = df["label"].to_numpy() == 1
    return {
        "variant": variant,
        "uncertainty_col": uncertainty_col,
        "n": n,
        "total_errors": int(np.sum(is_error)),
        "total_false_negatives": int(np.sum(is_fn)),
        "total_tumors": int(np.sum(is_tumor)),
        "points": points,
    }


def summarize_false_negatives_under_abstention(
    df: pd.DataFrame,
    uncertainty_col: str,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
    variant: Optional[str] = None,
) -> list[dict]:
    """How false negatives (missed tumors) split between retained and abstained sets.

    For a tumor detector, the key selective-prediction question is not just overall risk
    but where the *missed tumors* end up: ideally abstention catches them (moves them out
    of the retained/acted-on set). For each coverage level this returns the retained vs
    abstained false-negative counts, the fraction of all FNs that fall in the abstained
    set, and the retained FN rate (over retained, and over all tumors). Rates with a zero
    denominator are ``None``.
    """
    if uncertainty_col not in df.columns:
        raise ValueError(f"uncertainty_col {uncertainty_col!r} not found in DataFrame columns.")
    unc = np.asarray(df[uncertainty_col].to_numpy(), dtype=np.float64)
    if not np.all(np.isfinite(unc)):
        raise ValueError(f"uncertainty_col {uncertainty_col!r} contains non-finite values.")

    is_fn = _false_negative_mask(df)
    is_tumor = df["label"].to_numpy() == 1
    total_fn = int(np.sum(is_fn))
    total_tumors = int(np.sum(is_tumor))
    n = int(unc.shape[0])
    order = np.argsort(unc, kind="stable")

    rows: list[dict] = []
    for c in coverage_levels:
        k = _k_for_coverage(n, c)
        ret_pos = order[:k]
        ret_fn = int(np.sum(is_fn[ret_pos]))
        abs_fn = total_fn - ret_fn
        rows.append({
            "variant": variant,
            "uncertainty_col": uncertainty_col,
            "target_coverage": float(c),
            "actual_coverage": (k / n) if n > 0 else None,
            "retained_n": int(k),
            "abstained_n": int(n - k),
            "total_false_negatives": total_fn,
            "retained_false_negatives": ret_fn,
            "abstained_false_negatives": abs_fn,
            "fraction_of_fn_abstained": (abs_fn / total_fn) if total_fn > 0 else None,
            "retained_false_negative_rate_over_retained": (ret_fn / k) if k > 0 else None,
            "retained_false_negative_rate_over_all_tumors": (
                ret_fn / total_tumors if total_tumors > 0 else None
            ),
            "total_tumors": total_tumors,
        })
    return rows
