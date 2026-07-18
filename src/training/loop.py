"""Epoch-level train / evaluate loops for the short controlled run (Milestone 3B).

Thin wrappers over :func:`training.steps.train_step` / :func:`training.steps.eval_step`
that iterate a DataLoader for one pass and aggregate sample-weighted means. Still no
scheduler, checkpointing, or early stopping — just enough to run a few epochs and log.

Both functions return::

    {"loss_mean": float, "accuracy_mean": float, "n_samples": int, "n_batches": int}

Means are weighted by batch size so a smaller final batch does not skew the average.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.steps import eval_step, train_step  # noqa: E402


def _aggregate(step_outputs: list[dict]) -> dict:
    """Sample-weighted aggregation of per-batch step dicts."""
    n_samples = sum(o["batch_size"] for o in step_outputs)
    n_batches = len(step_outputs)
    if n_samples == 0:
        return {"loss_mean": 0.0, "accuracy_mean": 0.0, "n_samples": 0, "n_batches": 0}
    loss_mean = sum(o["loss"] * o["batch_size"] for o in step_outputs) / n_samples
    acc_mean = sum(o["accuracy"] * o["batch_size"] for o in step_outputs) / n_samples
    return {
        "loss_mean": loss_mean,
        "accuracy_mean": acc_mean,
        "n_samples": n_samples,
        "n_batches": n_batches,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Run one training pass over ``dataloader`` (weights updated per batch)."""
    outputs = [
        train_step(model, batch, criterion, optimizer, device) for batch in dataloader
    ]
    return _aggregate(outputs)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run one no-grad evaluation pass over ``dataloader`` (no weight update)."""
    outputs = [eval_step(model, batch, criterion, device) for batch in dataloader]
    return _aggregate(outputs)
