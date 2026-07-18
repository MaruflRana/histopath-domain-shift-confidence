"""Image transforms for Camelyon17 patches.

Milestone 2A: basic transforms only (no augmentation yet).
Pipeline: RGB PIL image -> float32 tensor in [0,1], CHW -> ImageNet normalization.

Patches are natively 96x96, so no resize is needed; a size guard is included so a
wrong-sized patch fails loudly rather than silently producing a bad tensor shape.

Implemented with torch only (no torchvision dependency) for portability. Callables
are plain objects (picklable) so they are safe with DataLoader workers on Windows.
"""

from __future__ import annotations

import numpy as np
import torch

# ImageNet channel statistics (RGB).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PATCH_SIZE = 96


class ToRGBTensorNormalize:
    """Convert an RGB PIL image to a normalized float32 CHW tensor.

    Steps:
      1. ensure RGB (defensive; dataset already converts),
      2. -> float32 numpy [H,W,3] scaled to [0,1],
      3. -> torch tensor [3,H,W],
      4. normalize per channel with (x - mean) / std.
    """

    def __init__(
        self,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        expected_size: int = PATCH_SIZE,
    ) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.expected_size = expected_size

    def __call__(self, image):
        if getattr(image, "mode", None) != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        if self.expected_size is not None and (w != self.expected_size or h != self.expected_size):
            raise ValueError(
                f"Unexpected patch size {w}x{h}; expected "
                f"{self.expected_size}x{self.expected_size}."
            )

        arr = np.asarray(image, dtype=np.uint8)  # [H, W, 3]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB array, got shape {arr.shape}.")

        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous().float().div_(255.0)
        tensor = (tensor - self.mean) / self.std
        return tensor


# Train and eval share the same basic transform for now. They are kept as separate
# objects so training-only augmentation can be added later without touching eval.
train_transform = ToRGBTensorNormalize()
eval_transform = ToRGBTensorNormalize()
