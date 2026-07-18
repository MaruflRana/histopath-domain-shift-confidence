"""Temperature scaling (Guo et al., 2017) for post-hoc confidence calibration.

A single positive scalar ``T`` rescales the logits: ``z -> z / T``. Fitting ``T`` on a
held-out set by minimizing NLL leaves the model weights and the argmax prediction
untouched (dividing all logits by a positive scalar preserves their order) while making
the softmax confidences better match observed accuracy.

Stable parameterization: the learnable parameter is ``log_temperature`` so that
``T = exp(log_temperature)`` is always strictly positive; the reported/applied ``T`` is
additionally clamped to ``[min_temperature, max_temperature]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """One learnable scalar temperature applied as ``logits / T``.

    Parameters
    ----------
    temperature_init:
        Initial temperature (must be > 0). Stored internally as ``log(T)``.
    min_temperature, max_temperature:
        Bounds enforced whenever the temperature is read (``get_temperature`` / the
        forward pass), so the effective ``T`` never leaves ``[min, max]``.
    """

    def __init__(
        self,
        temperature_init: float = 1.0,
        min_temperature: float = 0.05,
        max_temperature: float = 10.0,
    ) -> None:
        super().__init__()
        if temperature_init <= 0:
            raise ValueError(f"temperature_init must be > 0, got {temperature_init}.")
        if not (0 < min_temperature <= max_temperature):
            raise ValueError(
                f"require 0 < min_temperature <= max_temperature; got "
                f"{min_temperature}, {max_temperature}."
            )
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(float(temperature_init)))
        )

    def get_temperature(self) -> float:
        """Return the current temperature as a Python float, clamped to bounds."""
        t = torch.exp(self.log_temperature)
        t = torch.clamp(t, self.min_temperature, self.max_temperature)
        return float(t.item())

    def _temperature_tensor(self) -> torch.Tensor:
        """Bounded temperature as a tensor (keeps autograd through exp/clamp)."""
        t = torch.exp(self.log_temperature)
        return torch.clamp(t, self.min_temperature, self.max_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits ``logits / T`` (same shape as input)."""
        return logits / self._temperature_tensor()


def apply_temperature_to_logits(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Divide ``logits`` by a fixed positive scalar ``temperature``."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    return logits / float(temperature)


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy (NLL) of ``logits`` [N,2] against long ``labels`` [N]."""
    return nn.functional.cross_entropy(logits, labels)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.01,
    temperature_init: float = 1.0,
    min_temperature: float = 0.05,
    max_temperature: float = 10.0,
    device: torch.device | str = "cpu",
) -> dict:
    """Fit a temperature by minimizing NLL on ``(logits, labels)``; no model weights.

    Optimizes ``log_temperature`` with Adam for ``max_iter`` steps against
    ``CrossEntropyLoss``. Model parameters are NOT involved — only the single scalar is
    updated. The returned temperature is clamped to ``[min_temperature,
    max_temperature]``.

    Parameters
    ----------
    logits:
        Tensor ``[N, 2]`` of raw model logits (detached from any model graph).
    labels:
        Integer tensor ``[N]`` of 0/1 labels (cast to long).
    max_iter, lr, temperature_init, min_temperature, max_temperature:
        Optimization / bound settings (from the experiment config).
    device:
        Device to run the fit on.

    Returns
    -------
    dict
        ``{temperature, nll_before, nll_after, n_iter, converged, min_temperature,
        max_temperature, temperature_init}``.

    Raises
    ------
    ValueError
        If ``logits`` is not ``[N, 2]``, shapes mismatch, or any logit/loss is
        non-finite.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    logits = logits.detach().to(device=device, dtype=torch.float32)
    labels = labels.detach().to(device=device, dtype=torch.long)

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"logits must have shape [N, 2]; got {tuple(logits.shape)}.")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"labels must be [N] matching logits; got {tuple(labels.shape)} vs "
            f"{tuple(logits.shape)}."
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits contain non-finite values (NaN/Inf) before fitting.")

    scaler = TemperatureScaler(
        temperature_init=temperature_init,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
    ).to(device)

    with torch.no_grad():
        nll_before = float(_nll(logits, labels).item())
    if not (nll_before == nll_before and abs(nll_before) != float("inf")):
        raise ValueError("NLL before calibration is non-finite.")

    optimizer = torch.optim.Adam(scaler.parameters(), lr=lr)

    n_iter = 0
    prev_loss = None
    converged = False
    for step in range(int(max_iter)):
        optimizer.zero_grad(set_to_none=True)
        loss = _nll(scaler(logits), labels)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"non-finite NLL at optimization step {step}.")
        loss.backward()
        optimizer.step()
        n_iter = step + 1

        cur = float(loss.item())
        if prev_loss is not None and abs(prev_loss - cur) < 1e-8:
            converged = True
            break
        prev_loss = cur

    temperature = scaler.get_temperature()
    with torch.no_grad():
        scaled = apply_temperature_to_logits(logits, temperature)
        nll_after = float(_nll(scaled, labels).item())
    if not (nll_after == nll_after and abs(nll_after) != float("inf")):
        raise ValueError("NLL after calibration is non-finite.")

    return {
        "temperature": temperature,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "n_iter": n_iter,
        "converged": converged,
        "min_temperature": float(min_temperature),
        "max_temperature": float(max_temperature),
        "temperature_init": float(temperature_init),
    }
