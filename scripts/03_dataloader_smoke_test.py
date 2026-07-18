"""Milestone 2A / Script 03: PyTorch dataloader smoke test.

Builds a DataLoader for each of the four locked logical splits with a small
``max_samples`` cap, pulls one batch, and verifies tensor shapes / dtypes / value
ranges. No model, no training. ``ood_test`` is touched for shape/count sanity ONLY.

Config: max_samples=128, batch_size=16, num_workers=0 (Windows-safe).
Report -> results/logs/dataloader_smoke_test.txt
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import sys

import torch

from data.dataloaders import build_dataloader
from utils.io import ensure_results_tree, logs_dir, write_text
from utils.seed import set_seed

SPLITS = ["train", "id_val", "ood_val", "ood_test"]
MAX_SAMPLES = 128
BATCH_SIZE = 16
NUM_WORKERS = 0
EXPECTED_SHAPE = (BATCH_SIZE, 3, 96, 96)


def main() -> int:
    set_seed(42)
    ensure_results_tree()
    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    emit("=" * 64)
    emit("Milestone 2A — DataLoader smoke test (script 03)")
    emit(f"max_samples={MAX_SAMPLES}  batch_size={BATCH_SIZE}  num_workers={NUM_WORKERS}")
    emit("=" * 64)

    all_ok = True

    for split in SPLITS:
        emit("")
        emit(f"### split: {split}")
        note = "  (ood_test: shape/count sanity ONLY)" if split == "ood_test" else ""
        try:
            loader = build_dataloader(
                split_name=split,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                max_samples=MAX_SAMPLES,
                verbose=True,
            )
            dataset_len = len(loader.dataset)
            batch = next(iter(loader))

            images = batch["image"]
            labels = batch["label"]
            centers = batch["center"]

            img_shape = tuple(images.shape)
            shape_ok = img_shape == EXPECTED_SHAPE
            label_shape_ok = tuple(labels.shape) == (BATCH_SIZE,)
            dtype_ok = images.dtype == torch.float32 and labels.dtype == torch.long
            split_ok = shape_ok and label_shape_ok and dtype_ok
            all_ok = all_ok and split_ok

            # centers/labels may be lists or tensors depending on collate; normalize.
            uniq_labels = sorted(set(int(x) for x in labels.tolist()))
            uniq_centers = sorted(
                set(int(x) for x in (centers.tolist() if torch.is_tensor(centers) else centers))
            )

            emit(f"  dataset cached samples : {dataset_len}{note}")
            emit(f"  batch image shape      : {img_shape}   expected {EXPECTED_SHAPE}")
            emit(f"  batch label shape      : {tuple(labels.shape)}")
            emit(f"  image dtype            : {images.dtype}")
            emit(f"  label dtype            : {labels.dtype}")
            emit(f"  image min / max        : {images.min().item():.4f} / {images.max().item():.4f}")
            emit(f"  unique labels in batch : {uniq_labels}")
            emit(f"  unique centers in batch: {uniq_centers}")
            emit(f"  shape check            : {'OK' if shape_ok else 'FAIL'}")
            emit(f"  dtype check            : {'OK' if dtype_ok else 'FAIL'}")
            emit(f"  RESULT                 : {'PASS' if split_ok else 'FAIL'}")

        except Exception as exc:  # keep going so all splits are reported
            all_ok = False
            emit(f"  ERROR: {type(exc).__name__}: {exc}")
            emit(f"  RESULT                 : FAIL")

    emit("")
    emit("=" * 64)
    emit(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    emit("=" * 64)

    out = logs_dir() / "dataloader_smoke_test.txt"
    write_text(out, "\n".join(lines) + "\n")
    print(f"\nSaved report -> {out}", flush=True)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
