"""Milestone 3A / Script 05: tiny training-loop MECHANICS smoke test.

Verifies the backward/update loop works on a single tiny TRAIN batch:
  1. load one train batch,
  2. compute cross-entropy loss,
  3. backward,
  4. optimizer step,
  5. repeat 20 steps on the SAME batch and check loss can decrease.

This is NOT full training, NOT evaluation, NOT a result. Only the ``train`` split is
used for optimization; ``id_val`` is used for a single no-grad forward/loss sanity
check. ``ood_val`` / ``ood_test`` are NOT touched.

Config: max_samples=64, batch_size=16, num_workers=0, 20 steps on one batch.
Report -> results/logs/train_mechanics_smoke_test.txt
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import sys

import torch

from data.dataloaders import build_dataloader
from models.classifier import CamelyonClassifier
from training.losses import create_loss
from training.optimizers import create_optimizer
from training.steps import eval_step, train_step
from utils.io import ensure_results_tree, logs_dir, write_text
from utils.seed import set_seed

MAX_SAMPLES = 64
BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_CLASSES = 2
BACKBONE = "resnet18"
STEPS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4


def _snapshot_params(model: torch.nn.Module) -> dict:
    """Clone current parameters so we can detect whether any changed."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _any_param_changed(before: dict, model: torch.nn.Module) -> bool:
    for name, p in model.named_parameters():
        if not torch.equal(before[name], p.detach()):
            return True
    return False


def main() -> int:
    set_seed(42)
    ensure_results_tree()
    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emit("=" * 64)
    emit("Milestone 3A — Training MECHANICS smoke test (script 05)")
    emit(
        f"backbone={BACKBONE}  pretrained=False  num_classes={NUM_CLASSES}  "
        f"optimizer=adamw  lr={LR}  weight_decay={WEIGHT_DECAY}"
    )
    emit(
        f"max_samples={MAX_SAMPLES}  batch_size={BATCH_SIZE}  "
        f"num_workers={NUM_WORKERS}  steps={STEPS}"
    )
    emit(f"device={device}")
    emit("=" * 64)

    ok = True
    warn_msgs: list[str] = []

    try:
        model = CamelyonClassifier(
            backbone_name=BACKBONE, pretrained=False, num_classes=NUM_CLASSES
        ).to(device)
        criterion = create_loss("cross_entropy")
        optimizer = create_optimizer(
            model, name="adamw", lr=LR, weight_decay=WEIGHT_DECAY
        )

        # --- one train batch, reused for all 20 steps ------------------------ #
        train_loader = build_dataloader(
            split_name="train",
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            max_samples=MAX_SAMPLES,
            verbose=True,
        )
        batch = next(iter(train_loader))

        images = batch["image"]
        labels = batch["label"]
        centers = batch["center"]
        input_shape = tuple(images.shape)
        uniq_labels = sorted(set(int(x) for x in labels.tolist()))
        uniq_centers = sorted(
            set(int(x) for x in (centers.tolist() if torch.is_tensor(centers) else centers))
        )

        emit("")
        emit(f"input shape            : {input_shape}")
        emit(f"labels present in batch: {uniq_labels}")
        emit(f"centers present in batch: {uniq_centers}")

        # snapshot params BEFORE any optimization to detect weight change later
        before = _snapshot_params(model)

        emit("")
        emit("--- optimization on the same batch ---")
        losses: list[float] = []
        finite_all = True
        for step in range(1, STEPS + 1):
            out = train_step(model, batch, criterion, optimizer, device)
            loss_val = out["loss"]
            losses.append(loss_val)
            step_finite = bool(torch.isfinite(torch.tensor(loss_val)).item())
            finite_all = finite_all and step_finite
            emit(f"  step {step:02d}  loss={loss_val:.6f}  acc={out['accuracy']:.3f}")

        initial_loss = losses[0]
        final_loss = losses[-1]
        params_changed = _any_param_changed(before, model)

        # --- optional id_val no-grad sanity (no weight update) --------------- #
        idval_loss = None
        try:
            idval_loader = build_dataloader(
                split_name="id_val",
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                max_samples=BATCH_SIZE,
                verbose=False,
            )
            idval_batch = next(iter(idval_loader))
            idval_out = eval_step(model, idval_batch, criterion, device)
            idval_loss = idval_out["loss"]
        except Exception as exc:  # sanity check is non-critical
            warn_msgs.append(f"id_val sanity skipped: {type(exc).__name__}: {exc}")

        # --- asserts -------------------------------------------------------- #
        init_finite = bool(torch.isfinite(torch.tensor(initial_loss)).item())
        final_finite = bool(torch.isfinite(torch.tensor(final_loss)).item())

        assert init_finite, "initial loss is not finite"
        assert final_finite, "final loss is not finite"
        assert finite_all, "a NaN/Inf loss occurred during optimization"
        assert params_changed, "no parameter changed after optimizer.step()"

        loss_decreased = final_loss <= initial_loss
        if not loss_decreased:
            warn_msgs.append(
                f"final loss ({final_loss:.6f}) > initial loss ({initial_loss:.6f}) "
                f"— WARN, not FAIL (mechanics still valid)"
            )

        emit("")
        emit("--- summary ---")
        emit(f"initial loss           : {initial_loss:.6f}")
        emit(f"final loss             : {final_loss:.6f}")
        emit(f"loss delta             : {final_loss - initial_loss:+.6f}")
        emit(f"loss decreased         : {'YES' if loss_decreased else 'NO (WARN)'}")
        emit(f"initial loss finite    : {'OK' if init_finite else 'FAIL'}")
        emit(f"final loss finite      : {'OK' if final_finite else 'FAIL'}")
        emit(f"all step losses finite : {'OK' if finite_all else 'FAIL'}")
        emit(f"any parameter changed  : {'YES (OK)' if params_changed else 'NO (FAIL)'}")
        if idval_loss is not None:
            emit(f"id_val no-grad loss    : {idval_loss:.6f}  (sanity only, no update)")

    except Exception as exc:
        ok = False
        emit("")
        emit(f"ERROR: {type(exc).__name__}: {exc}")

    for w in warn_msgs:
        emit(f"WARN: {w}")

    emit("")
    emit("=" * 64)
    emit(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    emit("=" * 64)

    out_path = logs_dir() / "train_mechanics_smoke_test.txt"
    write_text(out_path, "\n".join(lines) + "\n")
    print(f"\nSaved report -> {out_path}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
