"""Train-time stain / color augmentation (Milestone 7A).

Pathology-relevant, *moderate* stain-robustness augmentation for the ERM baseline. The
dominant Camelyon17 hospital-domain shift is H&E staining / scanner colour variation, so
this augmentation perturbs colour (brightness / contrast / saturation / hue) plus applies
flips and a small rotation — the standard, low-risk robustness recipe for histopathology
patches.

Scope rules:
  * Applied ONLY to the ``train`` subset. ``id_val`` and ``ood_val`` keep the deterministic
    :data:`data.transforms.eval_transform` (no augmentation) so evaluation stays comparable
    to the plain-ERM baseline.
  * Output is identical in shape / dtype / normalization to the existing pipeline: a
    float32 ``[3, 96, 96]`` tensor normalized with ImageNet statistics. This makes it a
    drop-in ``transform`` for :class:`data.balanced_subset.BalancedSubsetDataset`.

Implemented with **torchvision transforms only** (plus a small PIL/tensor-normalizing
adapter). Classes are module-level (picklable) so they are safe with DataLoader workers on
Windows. Colour jitter is kept moderate — no extreme distortion.
"""

from __future__ import annotations

from typing import Optional

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

from data.transforms import IMAGENET_MEAN, IMAGENET_STD, PATCH_SIZE


class EnsurePILRGB:
    """Coerce the incoming sample to a PIL **RGB** image of the expected patch size.

    ``BalancedSubsetDataset`` stores decoded PIL RGB images, so the common path is a
    no-op ``convert``. A tensor input (defensive: e.g. an already-tensorized cached
    sample) is converted back to a PIL image first — a float tensor is assumed to be in
    ``[0, 1]`` (or ``[0, 255]`` if its max exceeds 1) and clamped before conversion.
    """

    def __init__(self, expected_size: Optional[int] = PATCH_SIZE) -> None:
        self.expected_size = expected_size

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.dtype != torch.uint8:
                t = t.float()
                if float(t.max()) > 1.0:
                    t = t / 255.0
                t = t.clamp(0.0, 1.0)
            image = TF.to_pil_image(t)

        if getattr(image, "mode", None) != "RGB":
            image = image.convert("RGB")

        if self.expected_size is not None:
            w, h = image.size
            if w != self.expected_size or h != self.expected_size:
                raise ValueError(
                    f"Unexpected patch size {w}x{h}; expected "
                    f"{self.expected_size}x{self.expected_size}."
                )
        return image


class StainColorAugment:
    """Moderate stain / colour augmentation → normalized float32 ``[3, 96, 96]`` tensor.

    Pipeline (torchvision ``Compose``):
      1. ensure PIL RGB (size-guarded),
      2. ``RandomHorizontalFlip`` / ``RandomVerticalFlip`` (dihedral orientation),
      3. ``RandomRotation`` (small angle),
      4. ``ColorJitter`` (brightness / contrast / saturation / hue — the stain proxy),
      5. ``ToTensor`` (→ float32 ``[3, H, W]`` in ``[0, 1]``),
      6. ``Normalize`` with ImageNet statistics (same as ``eval_transform``).

    Numerically the ``ToTensor`` + ``Normalize`` tail matches
    :class:`data.transforms.ToRGBTensorNormalize` (``/255`` then ``(x - mean) / std``), so a
    non-augmented image would map to the same tensor as the eval pipeline.
    """

    def __init__(
        self,
        *,
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        rotation_degrees: float = 15.0,
        brightness: float = 0.20,
        contrast: float = 0.20,
        saturation: float = 0.20,
        hue: float = 0.05,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        expected_size: Optional[int] = PATCH_SIZE,
    ) -> None:
        self.params = {
            "horizontal_flip_p": horizontal_flip_p,
            "vertical_flip_p": vertical_flip_p,
            "rotation_degrees": rotation_degrees,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
        }
        self.pipeline = transforms.Compose(
            [
                EnsurePILRGB(expected_size),
                transforms.RandomHorizontalFlip(p=horizontal_flip_p),
                transforms.RandomVerticalFlip(p=vertical_flip_p),
                transforms.RandomRotation(degrees=rotation_degrees),
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    def __call__(self, image) -> torch.Tensor:
        return self.pipeline(image)


def build_stain_train_transform(aug_cfg: dict) -> StainColorAugment:
    """Construct the train-time stain/colour augmentation from a config ``augmentation`` block."""
    return StainColorAugment(
        horizontal_flip_p=float(aug_cfg.get("horizontal_flip_p", 0.5)),
        vertical_flip_p=float(aug_cfg.get("vertical_flip_p", 0.5)),
        rotation_degrees=float(aug_cfg.get("rotation_degrees", 15.0)),
        brightness=float(aug_cfg.get("brightness", 0.20)),
        contrast=float(aug_cfg.get("contrast", 0.20)),
        saturation=float(aug_cfg.get("saturation", 0.20)),
        hue=float(aug_cfg.get("hue", 0.05)),
    )


class StainColorAugmentV2:
    """Tuned (v2) stain / colour augmentation → normalized float32 ``[3, 96, 96]`` tensor.

    A **stronger but still controlled** pathology-relevant recipe than
    :class:`StainColorAugment` (7A). It keeps the same ``ToTensor`` + ImageNet ``Normalize``
    tail — so a non-augmented image maps to the same tensor as the eval pipeline — and adds
    mild geometric jitter (small affine translate / scale) plus a *probabilistic*, stronger
    **contrast** perturbation and a light autocontrast, while deliberately holding hue /
    saturation distortion LOW so tumour colour cues are not destroyed. Milestone 7D.

    Pipeline (torchvision ``Compose``, all module-level classes → picklable on Windows):
      1. ensure PIL RGB (size-guarded),
      2. ``RandomHorizontalFlip`` / ``RandomVerticalFlip`` (dihedral orientation),
      3. ``RandomRotation`` (larger angle than 7A),
      4. ``RandomAffine`` (small translate + scale; no rotation here — step 3 owns rotation),
      5. ``RandomApply([ColorJitter], p)`` (brightness / contrast / saturation / hue — the
         stain proxy, applied stochastically so some patches stay un-jittered),
      6. ``RandomAutocontrast`` (light, low-probability contrast normalisation),
      7. ``ToTensor`` (→ float32 ``[3, H, W]`` in ``[0, 1]``),
      8. ``Normalize`` with ImageNet statistics (same as ``eval_transform``).

    Augmentation is applied **only to the train subset**; ``id_val`` / ``ood_val`` continue to
    use the deterministic :data:`data.transforms.eval_transform` (no augmentation).
    """

    def __init__(
        self,
        *,
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        rotation_degrees: float = 30.0,
        translate: tuple[float, float] = (0.03, 0.03),
        scale: tuple[float, float] = (0.95, 1.05),
        color_jitter_p: float = 0.85,
        brightness: float = 0.15,
        contrast: float = 0.30,
        saturation: float = 0.15,
        hue: float = 0.03,
        autocontrast_p: float = 0.25,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        expected_size: Optional[int] = PATCH_SIZE,
    ) -> None:
        self.params = {
            "horizontal_flip_p": horizontal_flip_p,
            "vertical_flip_p": vertical_flip_p,
            "rotation_degrees": rotation_degrees,
            "translate": tuple(translate),
            "scale": tuple(scale),
            "color_jitter_p": color_jitter_p,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "autocontrast_p": autocontrast_p,
        }
        self.pipeline = transforms.Compose(
            [
                EnsurePILRGB(expected_size),
                transforms.RandomHorizontalFlip(p=horizontal_flip_p),
                transforms.RandomVerticalFlip(p=vertical_flip_p),
                transforms.RandomRotation(degrees=rotation_degrees),
                transforms.RandomAffine(
                    degrees=0,
                    translate=tuple(translate),
                    scale=tuple(scale),
                ),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=brightness,
                            contrast=contrast,
                            saturation=saturation,
                            hue=hue,
                        )
                    ],
                    p=color_jitter_p,
                ),
                transforms.RandomAutocontrast(p=autocontrast_p),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    def __call__(self, image) -> torch.Tensor:
        return self.pipeline(image)


def build_stain_color_aug_v2_transform(aug_cfg: dict) -> StainColorAugmentV2:
    """Construct the tuned (v2) train-time stain/colour augmentation (Milestone 7D).

    Reads the config ``augmentation`` block; missing keys fall back to the v2 defaults
    documented on :class:`StainColorAugmentV2`. Applied to the ``train`` subset only.
    """
    translate = aug_cfg.get("translate", (0.03, 0.03))
    scale = aug_cfg.get("scale", (0.95, 1.05))
    return StainColorAugmentV2(
        horizontal_flip_p=float(aug_cfg.get("horizontal_flip_p", 0.5)),
        vertical_flip_p=float(aug_cfg.get("vertical_flip_p", 0.5)),
        rotation_degrees=float(aug_cfg.get("rotation_degrees", 30.0)),
        translate=(float(translate[0]), float(translate[1])),
        scale=(float(scale[0]), float(scale[1])),
        color_jitter_p=float(aug_cfg.get("color_jitter_p", 0.85)),
        brightness=float(aug_cfg.get("brightness", 0.15)),
        contrast=float(aug_cfg.get("contrast", 0.30)),
        saturation=float(aug_cfg.get("saturation", 0.15)),
        hue=float(aug_cfg.get("hue", 0.03)),
        autocontrast_p=float(aug_cfg.get("autocontrast_p", 0.25)),
    )


class StainSpaceAugment:
    """Pathology-specific **optical-density (HED-style) stain-space** augmentation.

    Milestone 7E-0 (transform-only smoke test). Unlike the torchvision colour-jitter
    recipes (7A / 7D), this perturbs the image in **optical-density (OD) space** — the
    physically-motivated domain for H&E stain variation (Beer–Lambert): more stain →
    higher OD → darker RGB. A mild per-channel affine perturbation in OD space mimics
    scanner / staining differences between hospitals without touching hue/saturation
    directly, so it stays pathology-plausible rather than producing neon / washed-out
    patches.

    Steps (per call):
      1. coerce input (PIL RGB / tensor / uint8 array) to a size-guarded uint8 RGB array,
      2. ``OD = -log((RGB + eps) / 255)``   (RGB 255 → OD 0; RGB 0 → large OD),
      3. per-channel ``OD' = OD * scale_c + bias_c`` with ``scale_c ~ U[od_scale_min,
         od_scale_max]`` and ``bias_c ~ U[od_bias_min, od_bias_max]`` (independent per
         channel), then ``OD'`` lower-clipped at 0 so RGB never exceeds 255,
      4. back to RGB: ``RGB' = 255 * exp(-OD') - eps``, clipped to ``[0, 255]`` and cast
         to uint8 — guaranteed valid RGB, never NaN/Inf,
      5. spatial transforms (``RandomHorizontalFlip`` / ``RandomVerticalFlip`` /
         ``RandomRotation``) on the perturbed PIL image,
      6. ``ToTensor`` (→ float32 ``[3, H, W]`` in ``[0, 1]``) + ImageNet ``Normalize``
         (identical tail to :data:`data.transforms.eval_transform`).

    Default OD ranges are deliberately **controlled** (scale ±10 %, bias ±0.03 OD) so the
    perturbation is visible but mild. Randomness uses the NumPy (OD step) and torch
    (spatial step) global RNGs, so seeding upstream makes previews reproducible. The class
    is module-level (picklable) for DataLoader workers on Windows.

    Helper split (useful for previews / smoke tests):
      * :meth:`perturb_rgb` → the augmented **uint8 RGB array** (OD perturb + spatial),
        i.e. everything BEFORE normalization — for visualization + range checks,
      * :meth:`normalize_rgb` → ``ToTensor`` + ``Normalize`` of a given uint8 RGB array,
      * :meth:`__call__` = ``normalize_rgb(perturb_rgb(image))`` → float32 ``[3, 96, 96]``.
    """

    def __init__(
        self,
        *,
        od_scale_min: float = 0.90,
        od_scale_max: float = 1.10,
        od_bias_min: float = -0.03,
        od_bias_max: float = 0.03,
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        rotation_degrees: float = 15.0,
        eps: float = 1.0,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        expected_size: Optional[int] = PATCH_SIZE,
    ) -> None:
        self.params = {
            "od_scale_min": od_scale_min,
            "od_scale_max": od_scale_max,
            "od_bias_min": od_bias_min,
            "od_bias_max": od_bias_max,
            "horizontal_flip_p": horizontal_flip_p,
            "vertical_flip_p": vertical_flip_p,
            "rotation_degrees": rotation_degrees,
            "eps": eps,
        }
        self.od_scale_min = od_scale_min
        self.od_scale_max = od_scale_max
        self.od_bias_min = od_bias_min
        self.od_bias_max = od_bias_max
        self.eps = eps
        self.expected_size = expected_size
        self._spatial = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=horizontal_flip_p),
                transforms.RandomVerticalFlip(p=vertical_flip_p),
                transforms.RandomRotation(degrees=rotation_degrees),
            ]
        )
        self._to_tensor = transforms.ToTensor()
        self._normalize = transforms.Normalize(mean=list(mean), std=list(std))

    # -- input coercion ------------------------------------------------------ #
    def _coerce_uint8_rgb(self, image) -> "np.ndarray":
        """Return a size-guarded uint8 RGB ``[H, W, 3]`` array from PIL / tensor / array."""
        import numpy as np
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            arr = np.asarray(image, dtype=np.uint8)
        elif isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.dtype != torch.uint8:
                t = t.float()
                if float(t.max()) > 1.0:
                    t = t / 255.0
                t = (t.clamp(0.0, 1.0) * 255.0)
            arr = t.round().to(torch.uint8).numpy()
            if arr.ndim == 3 and arr.shape[0] == 3:  # CHW -> HWC
                arr = np.transpose(arr, (1, 2, 0))
        else:
            arr = np.asarray(image)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).round().astype(np.uint8)

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB, got shape {arr.shape}.")
        if self.expected_size is not None:
            h, w = arr.shape[0], arr.shape[1]
            if h != self.expected_size or w != self.expected_size:
                raise ValueError(
                    f"Unexpected patch size {w}x{h}; expected "
                    f"{self.expected_size}x{self.expected_size}."
                )
        return arr

    # -- OD-space perturbation ---------------------------------------------- #
    def _od_perturb_array(self, rgb_uint8: "np.ndarray") -> "np.ndarray":
        """Apply the per-channel OD-space affine perturbation; return uint8 RGB."""
        import numpy as np

        rgb = rgb_uint8.astype(np.float64)
        od = -np.log((rgb + self.eps) / 255.0)  # >= ~0 for all valid RGB
        scale = np.random.uniform(self.od_scale_min, self.od_scale_max, size=(1, 1, 3))
        bias = np.random.uniform(self.od_bias_min, self.od_bias_max, size=(1, 1, 3))
        od_aug = od * scale + bias
        od_aug = np.clip(od_aug, 0.0, None)  # OD >= 0 => RGB <= 255 (no overshoot)
        rgb_aug = 255.0 * np.exp(-od_aug) - self.eps
        rgb_aug = np.clip(rgb_aug, 0.0, 255.0)
        return rgb_aug.round().astype(np.uint8)

    # -- public pieces ------------------------------------------------------- #
    def perturb_rgb(self, image) -> "np.ndarray":
        """OD perturb + spatial transforms → augmented uint8 RGB ``[H, W, 3]`` array."""
        import numpy as np
        from PIL import Image as PILImage

        arr = self._coerce_uint8_rgb(image)
        arr = self._od_perturb_array(arr)
        pil = PILImage.fromarray(arr, mode="RGB")
        pil = self._spatial(pil)
        return np.asarray(pil, dtype=np.uint8)

    def normalize_rgb(self, rgb_uint8: "np.ndarray") -> torch.Tensor:
        """``ToTensor`` + ImageNet ``Normalize`` of a uint8 RGB array → float32 ``[3,H,W]``."""
        from PIL import Image as PILImage

        pil = PILImage.fromarray(rgb_uint8, mode="RGB")
        return self._normalize(self._to_tensor(pil))

    def __call__(self, image) -> torch.Tensor:
        return self.normalize_rgb(self.perturb_rgb(image))


def build_stain_space_aug_transform(aug_cfg: dict) -> StainSpaceAugment:
    """Construct the OD/HED-style stain-space augmentation from a config ``augmentation`` block.

    Reads the config ``augmentation`` block; missing keys fall back to the controlled
    defaults documented on :class:`StainSpaceAugment` (OD scale ±10 %, bias ±0.03, flips,
    15° rotation). Intended for the ``train`` subset only.
    """
    return StainSpaceAugment(
        od_scale_min=float(aug_cfg.get("od_scale_min", 0.90)),
        od_scale_max=float(aug_cfg.get("od_scale_max", 1.10)),
        od_bias_min=float(aug_cfg.get("od_bias_min", -0.03)),
        od_bias_max=float(aug_cfg.get("od_bias_max", 0.03)),
        horizontal_flip_p=float(aug_cfg.get("horizontal_flip_p", 0.5)),
        vertical_flip_p=float(aug_cfg.get("vertical_flip_p", 0.5)),
        rotation_degrees=float(aug_cfg.get("rotation_degrees", 15.0)),
    )
