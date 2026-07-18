"""Milestone 3B / Script 06: short, controlled multi-batch training run.

Trains ResNet18 for a few epochs on a capped BALANCED ``train`` subset and evaluates
each epoch on a capped BALANCED ``id_val`` subset. NOT full training, NOT a result.
No checkpointing. ``ood_val`` / ``ood_test`` are never touched.

All knobs come from configs/experiments/exp03b_short_train_resnet18.yaml.
Outputs: a per-epoch CSV log and a text report under results/logs/.
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.balanced_subset import BalancedSubsetDataset, load_or_build_balanced_examples
from data.transforms import eval_transform, train_transform
from models.classifier import CamelyonClassifier
from training.losses import create_loss
from training.loop import evaluate, train_one_epoch
from training.optimizers import create_optimizer
from utils.io import ensure_results_tree, load_yaml, project_root, write_csv, write_text
from utils.seed import set_seed

CONFIG_REL = "configs/experiments/exp03b_short_train_resnet18.yaml"


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative output path against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def main() -> int:
    root = project_root()
    cfg = load_yaml(root / CONFIG_REL)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    t_cfg = cfg["training"]
    o_cfg = cfg["output"]

    seed = int(t_cfg["seed"])
    set_seed(seed)
    ensure_results_tree()

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_per_class = int(d_cfg["train_per_class"])
    id_val_per_class = int(d_cfg["id_val_per_class"])
    batch_size = int(d_cfg["batch_size"])
    num_workers = int(d_cfg["num_workers"])
    epochs = int(t_cfg["epochs"])

    # Milestone 3C hardening knobs (with safe fallbacks if absent from config).
    max_scan_train = d_cfg.get("max_scan_train")
    max_scan_id_val = d_cfg.get("max_scan_id_val")
    progress_every = int(d_cfg.get("progress_every", 5000))
    shuffle_collected = bool(d_cfg.get("shuffle_collected", True))
    use_cache = bool(d_cfg.get("use_cache", True))

    expected_train_n = 2 * train_per_class
    expected_id_val_n = 2 * id_val_per_class

    emit("=" * 64)
    emit("Milestone 3B — Short controlled training run (script 06)")
    emit(
        f"backbone={m_cfg['backbone_name']}  pretrained={m_cfg['pretrained']}  "
        f"num_classes={m_cfg['num_classes']}"
    )
    emit(
        f"optimizer={t_cfg['optimizer']}  lr={t_cfg['learning_rate']}  "
        f"weight_decay={t_cfg['weight_decay']}  epochs={epochs}  seed={seed}"
    )
    emit(
        f"train_per_class={train_per_class} (n={expected_train_n})  "
        f"id_val_per_class={id_val_per_class} (n={expected_id_val_n})  "
        f"batch_size={batch_size}  num_workers={num_workers}"
    )
    emit(f"device={device}")
    emit("=" * 64)

    ok = True
    csv_path = _resolve(o_cfg["log_csv"])
    report_path = _resolve(o_cfg["report_txt"])

    try:
        # --- balanced subsets (train + id_val ONLY) -------------------------- #
        emit("")
        emit("--- collecting balanced subsets ---")
        train_examples = load_or_build_balanced_examples(
            "train",
            per_class=train_per_class,
            max_scan=max_scan_train,
            progress_every=progress_every,
            seed=seed,
            shuffle_collected=shuffle_collected,
            use_cache=use_cache,
            verbose=True,
        )
        id_val_examples = load_or_build_balanced_examples(
            "id_val",
            per_class=id_val_per_class,
            max_scan=max_scan_id_val,
            progress_every=progress_every,
            seed=seed,
            shuffle_collected=shuffle_collected,
            use_cache=use_cache,
            verbose=True,
        )

        train_ds = BalancedSubsetDataset(train_examples, transform=train_transform)
        id_val_ds = BalancedSubsetDataset(id_val_examples, transform=eval_transform)

        train_counts = train_ds.label_counts()
        id_val_counts = id_val_ds.label_counts()
        emit("")
        emit(f"train label counts : {dict(sorted(train_counts.items()))}  (n={len(train_ds)})")
        emit(f"id_val label counts: {dict(sorted(id_val_counts.items()))}  (n={len(id_val_ds)})")

        # --- dataloaders: train shuffle=True, id_val shuffle=False ---------- #
        gen = torch.Generator()
        gen.manual_seed(seed)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            drop_last=False, pin_memory=False, generator=gen,
        )
        id_val_loader = DataLoader(
            id_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            drop_last=False, pin_memory=False,
        )

        # --- model / loss / optimizer --------------------------------------- #
        model = CamelyonClassifier(
            backbone_name=m_cfg["backbone_name"],
            pretrained=bool(m_cfg["pretrained"]),
            num_classes=int(m_cfg["num_classes"]),
        ).to(device)
        criterion = create_loss("cross_entropy")
        optimizer = create_optimizer(
            model,
            name=t_cfg["optimizer"],
            lr=float(t_cfg["learning_rate"]),
            weight_decay=float(t_cfg["weight_decay"]),
        )

        # --- train loop ----------------------------------------------------- #
        emit("")
        emit("--- training ---")
        csv_rows: list[dict] = []
        all_losses_finite = True

        for epoch in range(1, epochs + 1):
            tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va = evaluate(model, id_val_loader, criterion, device)

            for val in (tr["loss_mean"], va["loss_mean"]):
                if not bool(torch.isfinite(torch.tensor(val)).item()):
                    all_losses_finite = False

            emit(
                f"  epoch {epoch}/{epochs}  "
                f"train_loss={tr['loss_mean']:.6f} train_acc={tr['accuracy_mean']:.4f}  "
                f"id_val_loss={va['loss_mean']:.6f} id_val_acc={va['accuracy_mean']:.4f}  "
                f"(train_n={tr['n_samples']} id_val_n={va['n_samples']})"
            )
            csv_rows.append({
                "epoch": epoch,
                "train_loss": round(tr["loss_mean"], 6),
                "train_accuracy": round(tr["accuracy_mean"], 6),
                "id_val_loss": round(va["loss_mean"], 6),
                "id_val_accuracy": round(va["accuracy_mean"], 6),
                "train_n": tr["n_samples"],
                "id_val_n": va["n_samples"],
            })

        # --- asserts -------------------------------------------------------- #
        train_n = csv_rows[-1]["train_n"]
        id_val_n = csv_rows[-1]["id_val_n"]
        train_labels_present = sorted(train_counts.keys())
        id_val_labels_present = sorted(id_val_counts.keys())

        assert all_losses_finite, "a non-finite loss occurred during training/eval"
        assert train_n == expected_train_n, f"train_n {train_n} != {expected_train_n}"
        assert id_val_n == expected_id_val_n, f"id_val_n {id_val_n} != {expected_id_val_n}"
        assert train_labels_present == [0, 1], f"train labels {train_labels_present} != [0, 1]"
        assert id_val_labels_present == [0, 1], f"id_val labels {id_val_labels_present} != [0, 1]"

        # --- save logs (no checkpoint) -------------------------------------- #
        write_csv(
            csv_path,
            csv_rows,
            header=["epoch", "train_loss", "train_accuracy",
                    "id_val_loss", "id_val_accuracy", "train_n", "id_val_n"],
        )

        emit("")
        emit("--- summary ---")
        emit(f"all losses finite      : {'OK' if all_losses_finite else 'FAIL'}")
        emit(f"train_n                : {train_n}  (expected {expected_train_n})")
        emit(f"id_val_n               : {id_val_n}  (expected {expected_id_val_n})")
        emit(f"train labels present   : {train_labels_present}")
        emit(f"id_val labels present  : {id_val_labels_present}")
        emit(f"saved CSV log          : {csv_path}")
        emit(f"saved text report      : {report_path}")

    except Exception as exc:
        ok = False
        emit("")
        emit(f"ERROR: {type(exc).__name__}: {exc}")

    emit("")
    emit("=" * 64)
    emit(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    emit("=" * 64)

    write_text(report_path, "\n".join(lines) + "\n")
    print(f"\nSaved report -> {report_path}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
