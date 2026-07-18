"""Single-batch train / eval step functions.

Batch format follows the existing dataloader output (see
:class:`data.torch_dataset.Camelyon17HFDataset`)::

    batch["image"]  -> float32 tensor [B, 3, 96, 96]
    batch["label"]  -> int64  tensor [B]
    batch[<meta>]   -> center, image_id, ... (ints; not used here)

Both steps move image/label to ``device``, run the forward pass, and compute the
loss. ``train_step`` additionally runs the backward pass and an optimizer update.
Each returns a plain dict of Python floats/ints (detached) so callers can log
without holding onto the graph.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy of ``logits`` [B, C] against integer ``labels`` [B]."""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.shape[0]


def train_step(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Run one forward + backward + optimizer update on a single batch.

    Returns ``{"loss": float, "accuracy": float, "batch_size": int}``.
    """
    model.train()

    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    logits = model(images)
    loss = criterion(logits, labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.detach().item(),
        "accuracy": _accuracy(logits.detach(), labels),
        "batch_size": int(labels.shape[0]),
    }


def eval_step(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run one no-grad forward + loss on a single batch (no weight update).

    Returns ``{"loss": float, "accuracy": float, "batch_size": int}``.
    """
    model.eval()

    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    with torch.no_grad():
        logits = model(images)
        loss = criterion(logits, labels)

    return {
        "loss": loss.item(),
        "accuracy": _accuracy(logits, labels),
        "batch_size": int(labels.shape[0]),
    }
