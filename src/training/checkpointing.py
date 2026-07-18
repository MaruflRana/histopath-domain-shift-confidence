"""Checkpoint save / load for the checkpointed development run (Milestone 4A).

A checkpoint is a single ``torch.save`` payload bundling everything needed to resume
or to re-evaluate a model:

    {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                int,
        "metrics":              dict,   # e.g. {"id_val_loss": ..., "id_val_accuracy": ...}
        "config":               dict,   # the experiment config, for provenance
    }

The run keeps two files under the experiment's checkpoint dir:
  * ``last.pt`` — overwritten every epoch (most recent state),
  * ``best.pt`` — written only when the selection metric (id_val_loss) improves.

This module is deliberately small and framework-plain: no scheduler / AMP / EMA state
yet. Later milestones can extend the payload without changing call sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Path:
    """Serialize model + optimizer + bookkeeping to ``path``.

    Parameters
    ----------
    path:
        Destination file (parent dirs are created if needed).
    model:
        Module whose ``state_dict`` is saved.
    optimizer:
        Optimizer whose ``state_dict`` is saved (so training could resume).
    epoch:
        1-based epoch this checkpoint corresponds to.
    metrics:
        Per-epoch metrics dict (stored verbatim for later inspection).
    config:
        The experiment config dict (stored for provenance).

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "config": dict(config),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint from ``path`` into ``model`` (and optionally ``optimizer``).

    Parameters
    ----------
    path:
        Checkpoint file written by :func:`save_checkpoint`.
    model:
        Module to load ``model_state_dict`` into (in place).
    optimizer:
        If given, ``optimizer_state_dict`` is loaded into it (in place). Pass ``None``
        when only the model weights are needed (e.g. evaluation-only reload).
    map_location:
        Device mapping for :func:`torch.load` (default ``"cpu"`` — portable, then the
        caller moves the model to the desired device).

    Returns
    -------
    dict
        The full checkpoint payload (``epoch``, ``metrics``, ``config`` included).
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return payload
