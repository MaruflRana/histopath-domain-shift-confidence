"""Milestone 2B / Script 04: model forward-pass smoke test.

Instantiates ``CamelyonClassifier(resnet18, pretrained=False, num_classes=2)`` and
pushes exactly ONE batch from each of the four locked logical splits through it,
verifying logits shape ``[B, 2]`` and finiteness. No training, loss, optimizer, or
checkpointing. ``ood_test`` is touched for a single forward-pass shape sanity ONLY.

Config: max_samples=32, batch_size=16, num_workers=0 (Windows-safe).
Report -> results/logs/model_forward_smoke_test.txt
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import sys

import torch

from data.dataloaders import build_dataloader
from models.classifier import CamelyonClassifier
from utils.io import ensure_results_tree, logs_dir, write_text
from utils.seed import set_seed

SPLITS = ["train", "id_val", "ood_val", "ood_test"]
MAX_SAMPLES = 32
BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_CLASSES = 2
BACKBONE = "resnet18"


def main() -> int:
    set_seed(42)
    ensure_results_tree()
    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emit("=" * 64)
    emit("Milestone 2B — Model forward-pass smoke test (script 04)")
    emit(
        f"backbone={BACKBONE}  pretrained=False  num_classes={NUM_CLASSES}  "
        f"max_samples={MAX_SAMPLES}  batch_size={BATCH_SIZE}  num_workers={NUM_WORKERS}"
    )
    emit(f"device={device}")
    emit("=" * 64)

    # Instantiate once; reuse across all splits. eval() mode = deterministic forward,
    # no BatchNorm running-stat updates (we are not training).
    model = CamelyonClassifier(
        backbone_name=BACKBONE, pretrained=False, num_classes=NUM_CLASSES
    ).to(device)
    model.eval()

    all_ok = True

    for split in SPLITS:
        emit("")
        emit(f"### split: {split}")
        note = "  (ood_test: forward-pass shape sanity ONLY)" if split == "ood_test" else ""
        try:
            loader = build_dataloader(
                split_name=split,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                max_samples=MAX_SAMPLES,
                verbose=True,
            )
            batch = next(iter(loader))
            images = batch["image"].to(device)
            labels = batch["label"]
            centers = batch["center"]

            b = images.shape[0]
            input_shape = tuple(images.shape)
            input_ok = input_shape == (b, 3, 96, 96)

            with torch.no_grad():
                logits = model(images)

            logits_shape = tuple(logits.shape)
            logits_ok = logits_shape == (b, NUM_CLASSES)
            finite_ok = bool(torch.isfinite(logits).all().item())

            split_ok = input_ok and logits_ok and finite_ok
            all_ok = all_ok and split_ok

            uniq_labels = sorted(set(int(x) for x in labels.tolist()))
            uniq_centers = sorted(
                set(int(x) for x in (centers.tolist() if torch.is_tensor(centers) else centers))
            )

            emit(f"  input shape            : {input_shape}{note}")
            emit(f"  logits shape           : {logits_shape}   expected ({b}, {NUM_CLASSES})")
            emit(f"  logits dtype           : {logits.dtype}")
            emit(f"  device                 : {logits.device}")
            emit(f"  labels shape           : {tuple(labels.shape)}")
            emit(f"  unique labels in batch : {uniq_labels}")
            emit(f"  unique centers in batch: {uniq_centers}")
            emit(f"  input shape check      : {'OK' if input_ok else 'FAIL'}")
            emit(f"  logits shape check     : {'OK' if logits_ok else 'FAIL'}")
            emit(f"  finite check           : {'OK' if finite_ok else 'FAIL'}")

            # Hard asserts, as required. Caught below so all splits still get reported.
            assert input_ok, f"input shape {input_shape} != ({b}, 3, 96, 96)"
            assert logits_ok, f"logits shape {logits_shape} != ({b}, {NUM_CLASSES})"
            assert finite_ok, "logits contain NaN or Inf"

            emit(f"  RESULT                 : PASS")

        except Exception as exc:  # keep going so all splits are reported
            all_ok = False
            emit(f"  ERROR: {type(exc).__name__}: {exc}")
            emit(f"  RESULT                 : FAIL")

    emit("")
    emit("=" * 64)
    emit(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    emit("=" * 64)

    out = logs_dir() / "model_forward_smoke_test.txt"
    write_text(out, "\n".join(lines) + "\n")
    print(f"\nSaved report -> {out}", flush=True)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
