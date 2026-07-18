"""Calibration metrics for the tumor / non-tumor probability outputs (Milestone 5A).

Positive class is **tumor = 1**; ``prob_1`` is the predicted probability of tumor. All
metrics here measure how well the predicted probabilities match observed frequencies —
they say nothing new about ranking/accuracy (see ``metrics_classification``).

Implemented:
  * ``compute_ece``           — Expected Calibration Error (equal-width confidence bins),
  * ``compute_brier_score``   — Brier score of ``prob_1`` against the 0/1 label,
  * ``compute_nll_from_probs``— mean negative log-likelihood (probabilities clipped ONLY
                                 for numerical stability of the log),
  * ``compute_calibration_metrics`` — bundles ``{n, ece, brier, nll, n_bins}``.

Numpy only (no sklearn dependency). Invalid probabilities (non-finite, or outside
[0, 1] beyond a tiny tolerance) raise a clear ``ValueError`` rather than silently
producing a meaningless number.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Tolerance for the [0, 1] range check (softmax outputs can be a hair over 1.0).
_PROB_TOL = 1e-6


def _validate(y_true: np.ndarray, prob_1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce, length-check, finiteness-check, and range-check the inputs."""
    y_true = np.asarray(y_true).ravel()
    prob_1 = np.asarray(prob_1, dtype=np.float64).ravel()

    if y_true.shape[0] != prob_1.shape[0]:
        raise ValueError(
            f"y_true and prob_1 length mismatch: {y_true.shape[0]} vs {prob_1.shape[0]}."
        )
    if y_true.shape[0] == 0:
        raise ValueError("Cannot compute calibration metrics on empty input (n == 0).")

    if not np.all(np.isfinite(prob_1)):
        raise ValueError("prob_1 contains non-finite values (NaN/Inf).")
    if np.any(prob_1 < -_PROB_TOL) or np.any(prob_1 > 1.0 + _PROB_TOL):
        lo, hi = float(prob_1.min()), float(prob_1.max())
        raise ValueError(f"prob_1 out of [0,1] (min={lo}, max={hi}).")

    y_true = y_true.astype(np.int64)
    uniq = set(np.unique(y_true).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"y_true must contain only 0/1 labels; got {sorted(uniq)}.")

    # Clip tiny numerical overshoot back into range (does not distort real values).
    prob_1 = np.clip(prob_1, 0.0, 1.0)
    return y_true, prob_1


def compute_ece(
    y_true: Sequence[int] | np.ndarray,
    prob_1: Sequence[float] | np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error using equal-width confidence bins.

    Confidence is the probability of the *predicted* class:
    ``conf = max(prob_1, 1 - prob_1)`` (2-class case). Accuracy within a bin is the
    fraction of examples whose hard prediction (``prob_1 >= 0.5`` -> tumor) is correct.
    ECE is the sample-weighted average of ``|accuracy - confidence|`` across bins.
    """
    y_true, prob_1 = _validate(y_true, prob_1)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")

    n = y_true.shape[0]
    pred = (prob_1 >= 0.5).astype(np.int64)
    correct = (pred == y_true).astype(np.float64)
    confidence = np.maximum(prob_1, 1.0 - prob_1)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # Last bin is closed on the right so confidence == 1.0 is included.
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(confidence[mask]))
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier_score(
    y_true: Sequence[int] | np.ndarray,
    prob_1: Sequence[float] | np.ndarray,
) -> float:
    """Brier score = mean squared error between ``prob_1`` and the 0/1 tumor label."""
    y_true, prob_1 = _validate(y_true, prob_1)
    return float(np.mean((prob_1 - y_true.astype(np.float64)) ** 2))


def compute_nll_from_probs(
    y_true: Sequence[int] | np.ndarray,
    prob_1: Sequence[float] | np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Mean negative log-likelihood (log loss) of ``prob_1`` vs the tumor label.

    Probabilities are clipped to ``[eps, 1 - eps]`` **only** to keep ``log`` finite;
    this does not otherwise alter the values.
    """
    y_true, prob_1 = _validate(y_true, prob_1)
    p = np.clip(prob_1, eps, 1.0 - eps)
    y = y_true.astype(np.float64)
    nll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(nll))


def compute_calibration_metrics(
    y_true: Sequence[int] | np.ndarray,
    prob_1: Sequence[float] | np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Bundle the calibration metrics into a single dict.

    Returns ``{n, ece, brier, nll, n_bins}`` (positive class = tumor = 1).
    """
    y_true_arr, prob_1_arr = _validate(y_true, prob_1)
    return {
        "n": int(y_true_arr.shape[0]),
        "ece": compute_ece(y_true_arr, prob_1_arr, n_bins=n_bins),
        "brier": compute_brier_score(y_true_arr, prob_1_arr),
        "nll": compute_nll_from_probs(y_true_arr, prob_1_arr),
        "n_bins": int(n_bins),
    }
