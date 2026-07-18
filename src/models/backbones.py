"""Backbone feature extractors built from ``torchvision.models``.

Milestone 2B: only ResNet-18 / ResNet-50 are supported. Each backbone has its
original classification head (``fc``) removed cleanly and replaced with
``nn.Identity`` so that calling it returns a flat feature vector ``[B, feature_dim]``
instead of class logits. The classifier head is added separately in
:mod:`models.classifier`.

``pretrained`` is honored via the modern torchvision ``weights=`` API
(``IMAGENET1K_V1``); ``pretrained=False`` builds randomly-initialized weights,
which is what the smoke test uses (no network download).
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn

# Supported backbone name -> (torchvision constructor name, default weights enum name).
# The weights enum name is resolved lazily so importing this module never triggers
# a download or requires network access.
_SUPPORTED = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
}


def available_backbones() -> tuple[str, ...]:
    """Return the tuple of supported backbone names."""
    return tuple(_SUPPORTED)


def create_backbone(name: str, pretrained: bool = False) -> Tuple[nn.Module, int]:
    """Create a backbone feature extractor and return ``(backbone, feature_dim)``.

    Parameters
    ----------
    name:
        One of ``resnet18`` / ``resnet50``.
    pretrained:
        If True, load ImageNet (``IMAGENET1K_V1``) weights via torchvision's
        ``weights=`` API (requires a one-time download). If False (default for the
        smoke test), weights are randomly initialized — no network access.

    Returns
    -------
    (backbone, feature_dim):
        ``backbone`` is an ``nn.Module`` that maps an image tensor ``[B, 3, H, W]``
        to a flat feature tensor ``[B, feature_dim]``; the original ``fc`` head has
        been replaced with ``nn.Identity``. ``feature_dim`` is the number of
        features that head consumed (512 for resnet18, 2048 for resnet50).
    """
    if name not in _SUPPORTED:
        raise ValueError(
            f"Unsupported backbone {name!r}. Supported: {available_backbones()}."
        )

    from torchvision import models

    ctor_name, weights_enum_name = _SUPPORTED[name]
    constructor = getattr(models, ctor_name)

    weights = None
    if pretrained:
        weights_enum = getattr(models, weights_enum_name)
        weights = weights_enum.IMAGENET1K_V1

    backbone = constructor(weights=weights)

    # For ResNet, ``fc`` is the final classification head; its ``in_features`` is the
    # backbone feature dimension. Remove it cleanly so forward() yields features.
    if not hasattr(backbone, "fc"):
        raise AttributeError(
            f"Backbone {name!r} has no 'fc' attribute; head removal not supported."
        )
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    return backbone, feature_dim
