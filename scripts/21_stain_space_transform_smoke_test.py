"""Milestone 7E-0 / Script 21: stain-space (OD/HED-style) augmentation SMOKE TEST.

**Transform-only visual + numerical sanity check — NO training, NO model inference.** This
validates the new :class:`data.stain_transforms.StainSpaceAugment` (optical-density / HED-style
per-channel stain perturbation) before any 7E training is proposed. It loads a handful of
cached ``train`` patches, augments each a few times, saves a preview grid + a pixel/stat report,
and asserts the transform stays numerically safe (no NaN/Inf, valid RGB range, correct tensor
shape).

Strict scope (Milestone 7E-0):
  * Reads ONLY the cached balanced ``train`` subset (disk cache — no HF streaming, no
    WILDS/CodaLab). ``ood_val`` / ``ood_test`` are NEVER read. No model, no checkpoint, no
    inference, no training. No existing result CSV/checkpoint is modified.
  * Writes ONLY transform-preview artifacts under ``results/figures`` + ``results/logs``.

DEVELOPMENT transform check, NOT a result. Reproducible via a fixed seed.

Input (must already exist):
  * results/cache/balanced_train_pc1000_seed1337_shuf.pt
Outputs:
  * results/figures/exp07e0_stain_space_aug_preview.png
  * results/logs/exp07e0_stain_space_aug_smoke_report.txt
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / non-interactive; no display required
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.balanced_subset import load_or_build_balanced_examples
from data.stain_transforms import build_stain_space_aug_transform
from utils.io import ensure_results_tree, project_root, write_text
from utils.seed import set_seed

TRAIN_CACHE_REL = "results/cache/balanced_train_pc1000_seed1337_shuf.pt"
OUT_FIG = "results/figures/exp07e0_stain_space_aug_preview.png"
OUT_REPORT = "results/logs/exp07e0_stain_space_aug_smoke_report.txt"

N_PER_LABEL = 8          # 8 label-0 + 8 label-1
N_AUG_PER_IMAGE = 3      # 3 independent augmentations shown per image
SEED = 1337

# Controlled OD/HED-style augmentation settings (mild, pathology-plausible).
AUG_CFG = {
    "name": "stain_space_aug",
    "od_scale_min": 0.90,
    "od_scale_max": 1.10,
    "od_bias_min": -0.03,
    "od_bias_max": 0.03,
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.5,
    "rotation_degrees": 15,
}


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def _stats(t: torch.Tensor) -> dict:
    t = t.float()
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
    }


def main() -> int:
    ensure_results_tree()

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    emit("=" * 78)
    emit("Milestone 7E-0 - stain-space (OD/HED-style) augmentation SMOKE TEST (script 21)")
    emit("Transform-only. NO training, NO inference, NO ood_val, NO ood_test.")
    emit(f"seed={SEED}  n_per_label={N_PER_LABEL}  n_aug_per_image={N_AUG_PER_IMAGE}")
    emit("=" * 78)

    ok = True

    try:
        # --- guard: only train-cache + preview paths; never ood_val/ood_test --- #
        all_paths = [TRAIN_CACHE_REL, OUT_FIG, OUT_REPORT]
        assert not any(("ood_val" in p or "ood_test" in p) for p in all_paths), \
            "ood_val / ood_test must never be referenced"

        train_cache = _resolve(TRAIN_CACHE_REL)
        assert train_cache.exists(), f"required train cache not found: {train_cache}"

        # --- load cached balanced train subset (cache HIT; NO streaming) -------- #
        emit("")
        emit(f"--- loading cached train subset (cache HIT expected) : {train_cache} ---")
        examples = load_or_build_balanced_examples(
            "train", per_class=1000, seed=SEED,
            shuffle_collected=True, use_cache=True, verbose=True,
        )

        # --- select the first N_PER_LABEL of each label (deterministic) --------- #
        sel: dict[int, list[dict]] = {0: [], 1: []}
        for ex in examples:
            lab = int(ex["label"])
            if lab in sel and len(sel[lab]) < N_PER_LABEL:
                sel[lab].append(ex)
            if len(sel[0]) >= N_PER_LABEL and len(sel[1]) >= N_PER_LABEL:
                break

        n0, n1 = len(sel[0]), len(sel[1])
        emit("")
        emit(f"selected label-0 : {n0}")
        emit(f"selected label-1 : {n1}")

        # ordered list: 8 label-0 then 8 label-1 (row order in the grid)
        selected = [(0, ex) for ex in sel[0]] + [(1, ex) for ex in sel[1]]

        # --- build the OD/HED-style stain-space augmentation -------------------- #
        transform = build_stain_space_aug_transform(AUG_CFG)
        emit("")
        emit("augmentation settings (StainSpaceAugment):")
        for k, v in transform.params.items():
            emit(f"  {k:<18}: {v}")

        # --- deterministic augmentation + stat collection ----------------------- #
        set_seed(SEED)  # reproducible previews (numpy OD draws + torch spatial)

        orig_norm_tensors: list[torch.Tensor] = []   # normalized originals ("before")
        aug_norm_tensors: list[torch.Tensor] = []     # normalized augmented ("after")
        orig_rgbs: list[np.ndarray] = []              # uint8 RGB originals (for preview)
        aug_rgbs: list[list[np.ndarray]] = []         # uint8 RGB augmentations (for preview)

        any_nan_inf = False
        rgb_min_global, rgb_max_global = 255, 0
        bad_shape = None

        for lab, ex in selected:
            pil = ex["image"]
            orig_arr = transform._coerce_uint8_rgb(pil)  # size-guarded uint8 RGB
            orig_rgbs.append(orig_arr)
            orig_norm_tensors.append(transform.normalize_rgb(orig_arr))

            this_augs: list[np.ndarray] = []
            for _ in range(N_AUG_PER_IMAGE):
                rgb_aug = transform.perturb_rgb(pil)             # uint8 RGB (pre-norm)
                this_augs.append(rgb_aug)
                rgb_min_global = min(rgb_min_global, int(rgb_aug.min()))
                rgb_max_global = max(rgb_max_global, int(rgb_aug.max()))

                norm = transform.normalize_rgb(rgb_aug)          # float32 [3,96,96]
                if tuple(norm.shape) != (3, 96, 96):
                    bad_shape = tuple(norm.shape)
                if not bool(torch.isfinite(norm).all()):
                    any_nan_inf = True
                aug_norm_tensors.append(norm)
            aug_rgbs.append(this_augs)

        # --- aggregate stats (before vs after, over the normalized tensors) ----- #
        before = _stats(torch.stack(orig_norm_tensors))
        after = _stats(torch.stack(aug_norm_tensors))
        out_shape = tuple(aug_norm_tensors[0].shape)

        # RGB pre-normalization values are uint8 by construction; report the observed range.
        rgb_in_range = (rgb_min_global >= 0 and rgb_max_global <= 255)

        emit("")
        emit("--- numerical sanity ---")
        emit(f"output tensor shape (after full transform) : {out_shape}")
        emit(f"n normalized originals   ('before')        : {len(orig_norm_tensors)}")
        emit(f"n normalized augmentations ('after')       : {len(aug_norm_tensors)}")
        emit("")
        emit("normalized-tensor stats (ImageNet-normalized; NOT raw RGB):")
        emit(f"  before augmentation : min={before['min']:.4f} max={before['max']:.4f} "
             f"mean={before['mean']:.4f} std={before['std']:.4f}")
        emit(f"  after  augmentation : min={after['min']:.4f} max={after['max']:.4f} "
             f"mean={after['mean']:.4f} std={after['std']:.4f}")
        emit("")
        emit(f"augmented pre-normalization RGB range       : [{rgb_min_global}, {rgb_max_global}]")
        emit(f"all augmented RGB within [0,255]            : {'YES' if rgb_in_range else 'NO'}")
        emit(f"any NaN/Inf in augmented tensors            : {'YES' if any_nan_inf else 'NO'}")
        emit(f"augmented tensor shape == [3,96,96]         : {'YES' if bad_shape is None else f'NO ({bad_shape})'}")

        # --- preview grid: original + 3 augmentations per image ----------------- #
        n_rows = len(selected)                # 16
        n_cols = 1 + N_AUG_PER_IMAGE          # 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.7, n_rows * 1.7))
        col_titles = ["original", "aug 1", "aug 2", "aug 3"]
        for r, (lab, _ex) in enumerate(selected):
            imgs = [orig_rgbs[r]] + aug_rgbs[r]
            for c in range(n_cols):
                ax = axes[r, c]
                ax.imshow(imgs[c])
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(col_titles[c], fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"label {lab}", fontsize=8)
        fig.suptitle("Milestone 7E-0: OD/HED-style stain-space augmentation preview\n"
                     "(train patches; original + 3 stain-space augmentations; pre-normalization RGB)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(_resolve(OUT_FIG), dpi=130)
        plt.close(fig)

        # --- assertions --------------------------------------------------------- #
        assert train_cache.exists(), "train cache missing"
        assert n0 == N_PER_LABEL, f"selected label-0 count {n0} != {N_PER_LABEL}"
        assert n1 == N_PER_LABEL, f"selected label-1 count {n1} != {N_PER_LABEL}"
        assert bad_shape is None and out_shape == (3, 96, 96), f"bad augmented tensor shape {bad_shape or out_shape}"
        assert not any_nan_inf, "NaN/Inf occurred in augmented tensors"
        assert rgb_in_range, f"augmented RGB out of [0,255]: [{rgb_min_global},{rgb_max_global}]"
        assert _resolve(OUT_FIG).exists() and _resolve(OUT_FIG).stat().st_size > 0, "preview figure missing/empty"

        visual_pass = (bad_shape is None and not any_nan_inf and rgb_in_range
                       and out_shape == (3, 96, 96))

        emit("")
        emit("--- summary ---")
        emit(f"preview figure : {OUT_FIG}")
        emit(f"report         : {OUT_REPORT}")
        emit("ood_val usage  : NONE")
        emit("ood_test usage : NONE")
        emit(f"VISUAL SANITY  : {'PASS' if visual_pass else 'FAIL'}")

    except Exception as exc:
        ok = False
        emit("")
        emit(f"ERROR: {type(exc).__name__}: {exc}")

    emit("")
    emit("=" * 78)
    emit(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    emit("=" * 78)

    write_text(_resolve(OUT_REPORT), "\n".join(lines) + "\n")
    print(f"\nSaved report -> {_resolve(OUT_REPORT)}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
