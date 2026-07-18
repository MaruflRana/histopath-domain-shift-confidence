"""Evaluation utilities for the Camelyon17 confidence-awareness project.

Milestone 4B: a reusable evaluation scaffold — run inference on cached balanced
subsets, export per-example predictions, and compute classification metrics. This
package does NOT perform calibration, uncertainty, or selective prediction, and it does
NOT read the OOD splits (``ood_val`` / ``ood_test``). Those are later, separately
authorized milestones.

Public API:
    predict_to_dataframe(model, dataloader, device, split_name)
    compute_binary_classification_metrics(y_true, prob_1, threshold=0.5)
"""

from __future__ import annotations

from evaluation.metrics_classification import compute_binary_classification_metrics
from evaluation.predict import predict_to_dataframe

__all__ = [
    "compute_binary_classification_metrics",
    "predict_to_dataframe",
]
