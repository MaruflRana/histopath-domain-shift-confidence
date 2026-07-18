"""Optimizer factory for the Camelyon17 classifier.

Milestone 3A: AdamW only. Behind a factory so schedulers / SGD / param-group
splits can be added later without changing call sites.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_SUPPORTED = ("adamw",)


def create_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Create an optimizer over ``model``'s parameters.

    Parameters
    ----------
    model:
        The module whose parameters will be optimized.
    name:
        Currently only ``adamw`` (:class:`torch.optim.AdamW`).
    lr:
        Learning rate.
    weight_decay:
        Decoupled weight decay.
    """
    if name != "adamw":
        raise ValueError(
            f"Unsupported optimizer {name!r}. Supported: {_SUPPORTED}."
        )
    return torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
