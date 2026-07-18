"""Camelyon17 tumor / non-tumor patch classifier.

A thin wrapper: ``backbone`` (feature extractor from :mod:`models.backbones`) followed
by a single linear head mapping ``feature_dim -> num_classes``. Returns raw logits
``[B, num_classes]`` (no softmax — calibration/uncertainty work later needs logits).

Milestone 2B: forward pass only. No loss, optimizer, or checkpointing here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from models.backbones import create_backbone  # noqa: E402


class CamelyonClassifier(nn.Module):
    """Backbone + linear head producing class logits.

    Parameters
    ----------
    backbone_name:
        Backbone key understood by :func:`models.backbones.create_backbone`
        (``resnet18`` / ``resnet50``).
    pretrained:
        Whether the backbone loads ImageNet weights. Default False (smoke test:
        no download, random init).
    num_classes:
        Number of output logits. Default 2 (non-tumor / tumor).
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.backbone, self.feature_dim = create_backbone(backbone_name, pretrained)
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map an image batch ``[B, 3, H, W]`` to logits ``[B, num_classes]``."""
        features = self.backbone(x)          # [B, feature_dim]
        logits = self.head(features)         # [B, num_classes]
        return logits
