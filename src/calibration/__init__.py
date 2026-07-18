"""Post-hoc calibration for the Camelyon17 classifier (Milestone 5A).

Currently: temperature scaling — a single scalar ``T`` that divides the logits before
softmax, fit by minimizing NLL on a held-out OOD set (``ood_val``). Temperature scaling
does NOT change the model's weights or its hard predictions (argmax is invariant to a
positive scalar divide); it only rescales confidence.

Public API:
    TemperatureScaler(nn.Module)
    fit_temperature(logits, labels, ...) -> dict
    apply_temperature_to_logits(logits, temperature) -> Tensor
"""

from __future__ import annotations

from calibration.temperature_scaling import (
    TemperatureScaler,
    apply_temperature_to_logits,
    fit_temperature,
)

__all__ = [
    "TemperatureScaler",
    "apply_temperature_to_logits",
    "fit_temperature",
]
