"""Loss functions for the Camelyon17 classifier.

Milestone 3A: cross-entropy only. Kept behind a small factory so later milestones
can add label smoothing / class weighting / focal loss without touching call sites.
"""

from __future__ import annotations

import torch.nn as nn

_SUPPORTED = ("cross_entropy",)


def create_loss(name: str = "cross_entropy") -> nn.Module:
    """Return a loss module by name.

    Parameters
    ----------
    name:
        Currently only ``cross_entropy`` (maps to :class:`torch.nn.CrossEntropyLoss`,
        which expects raw logits ``[B, C]`` and integer targets ``[B]``).
    """
    if name != "cross_entropy":
        raise ValueError(
            f"Unsupported loss {name!r}. Supported: {_SUPPORTED}."
        )
    return nn.CrossEntropyLoss()
