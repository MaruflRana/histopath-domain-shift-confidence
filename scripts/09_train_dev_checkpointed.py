"""Milestone 4A / Script 09: checkpointed capped DEVELOPMENT training run.

Trains ResNet18 for a small number of epochs on a larger-but-still-capped BALANCED
``train`` subset (1000/class) and, after every epoch, evaluates on a capped BALANCED
``id_val`` subset (250/class). Adds two things over script 06:

  * checkpointing — ``last.pt`` every epoch, ``best.pt`` whenever ``id_val_loss``
    improves (selection metric / mode come from the config),
  * a reload verification — after training, ``best.pt`` is reloaded into a fresh model
    and evaluated once on ``id_val`` to confirm the checkpoint round-trips.

Strict scope (Milestone 4A): ``train`` for training, ``id_val`` for model selection /
monitoring ONLY. ``ood_val`` and ``ood_test`` are NEVER read. No calibration, no
uncertainty, no selective prediction. This is a DEVELOPMENT run, NOT a paper result.

All knobs come from configs/experiments/exp04a_dev_train_resnet18.yaml.
Outputs: an incremental per-epoch CSV, a text report (results/logs/), and checkpoints
(results/checkpoints/exp04a_dev_train_resnet18/).
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.balanced_subset import (
    BalancedSubsetDataset,
    default_cache_dir,
    load_or_build_balanced_examples,
)
from data.transforms import eval_transform, train_transform
from models.classifier import CamelyonClassifier
from training.checkpointing import load_checkpoint, save_checkpoint
from training.history import append_history_csv
from training.losses import create_loss
from training.loop import evaluate, train_one_epoch
from training.optimizers import create_optimizer
from utils.io import ensure_dir, ensure_results_tree, load_yaml, project_root, write_text
from utils.seed import set_seed

CONFIG_REL = "configs/experiments/exp04a_dev_train_resnet18.yaml"


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative output path against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def _cache_status(split_name: str, per_class: int, seed: int, shuffle_collected: bool) -> tuple[Path, bool]:
    """Return (expected cache path, existed_before) for reporting load-vs-create."""
    order = "shuf" if shuffle_collected else "noshuf"
    path = default_cache_dir() / f"balanced_{split_name}_pc{per_class}_seed{seed}_{order}.pt"
    return path, path.exists()


def main() -> int:
    root = project_root()
    cfg = load_yaml(root / CONFIG_REL)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    t_cfg = cfg["training"]
    s_cfg = cfg["selection"]
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

    max_scan_train = d_cfg.get("max_scan_train")
    max_scan_id_val = d_cfg.get("max_scan_id_val")
    progress_every = int(d_cfg.get("progress_every", 10000))
    shuffle_collected = bool(d_cfg.get("shuffle_collected", True))
    use_cache = bool(d_cfg.get("use_cache", True))
    data_seed = int(d_cfg.get("seed", seed))

    sel_metric = str(s_cfg["metric"])
    sel_mode = str(s_cfg["mode"])

    expected_train_n = 2 * train_per_class
    expected_id_val_n = 2 * id_val_per_class

    csv_path = _resolve(o_cfg["log_csv"])
    report_path = _resolve(o_cfg["report_txt"])
    ckpt_dir = _resolve(o_cfg["checkpoint_dir"])
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    # Fresh incremental CSV for this run (avoid appending onto a previous run's rows).
    if csv_path.exists():
        csv_path.unlink()

    emit("=" * 72)
    emit("Milestone 4A — Checkpointed capped development training (script 09)")
    emit(f"run_name={o_cfg['run_name']}")
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
    emit(f"selection: metric={sel_metric}  mode={sel_mode}")
    emit(f"device={device}")
    emit(f"checkpoint_dir={ckpt_dir}")
    emit("=" * 72)

    ok = True

    try:
        # --- balanced subsets (train + id_val ONLY; NO ood_val / NO ood_test) --- #
        emit("")
        emit("--- collecting balanced subsets ---")

        train_cache_path, train_cache_existed = _cache_status(
            "train", train_per_class, data_seed, shuffle_collected
        )
        id_val_cache_path, id_val_cache_existed = _cache_status(
            "id_val", id_val_per_class, data_seed, shuffle_collected
        )

        train_examples = load_or_build_balanced_examples(
            "train",
            per_class=train_per_class,
            max_scan=max_scan_train,
            progress_every=progress_every,
            seed=data_seed,
            shuffle_collected=shuffle_collected,
            use_cache=use_cache,
            verbose=True,
        )
        id_val_examples = load_or_build_balanced_examples(
            "id_val",
            per_class=id_val_per_class,
            max_scan=max_scan_id_val,
            progress_every=progress_every,
            seed=data_seed,
            shuffle_collected=shuffle_collected,
            use_cache=use_cache,
            verbose=True,
        )

        train_status = "LOADED (cache HIT)" if train_cache_existed else "CREATED (cache miss -> built + cached)"
        id_val_status = "LOADED (cache HIT)" if id_val_cache_existed else "CREATED (cache miss -> built + cached)"

        train_ds = BalancedSubsetDataset(train_examples, transform=train_transform)
        id_val_ds = BalancedSubsetDataset(id_val_examples, transform=eval_transform)

        train_counts = train_ds.label_counts()
        id_val_counts = id_val_ds.label_counts()

        emit("")
        emit(f"train  cache : {train_status}")
        emit(f"             -> {train_cache_path}")
        emit(f"id_val cache : {id_val_status}")
        emit(f"             -> {id_val_cache_path}")
        emit("")
        emit(f"train  label counts : {dict(sorted(train_counts.items()))}  (n={len(train_ds)})")
        emit(f"id_val label counts : {dict(sorted(id_val_counts.items()))}  (n={len(id_val_ds)})")

        # --- dataloaders: train shuffle=True, id_val shuffle=False -------------- #
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

        # --- model / loss / optimizer ------------------------------------------ #
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

        # --- train loop with checkpointing + best selection -------------------- #
        emit("")
        emit("--- training ---")

        ensure_dir(ckpt_dir)
        all_losses_finite = True
        best_metric_value = None
        best_epoch = None
        epoch_records: list[dict] = []

        def _is_improvement(candidate: float) -> bool:
            if best_metric_value is None:
                return True
            if sel_mode == "min":
                return candidate < best_metric_value
            if sel_mode == "max":
                return candidate > best_metric_value
            raise ValueError(f"Unsupported selection mode {sel_mode!r} (want 'min'/'max').")

        for epoch in range(1, epochs + 1):
            tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va = evaluate(model, id_val_loader, criterion, device)

            for val in (tr["loss_mean"], va["loss_mean"]):
                if not math.isfinite(val):
                    all_losses_finite = False

            metrics = {
                "epoch": epoch,
                "train_loss": tr["loss_mean"],
                "train_accuracy": tr["accuracy_mean"],
                "id_val_loss": va["loss_mean"],
                "id_val_accuracy": va["accuracy_mean"],
                "train_n": tr["n_samples"],
                "id_val_n": va["n_samples"],
            }

            selection_value = float(metrics[sel_metric])
            is_best = _is_improvement(selection_value)

            # last.pt every epoch; best.pt only on improvement of the selection metric.
            save_checkpoint(last_path, model, optimizer, epoch, metrics, cfg)
            if is_best:
                best_metric_value = selection_value
                best_epoch = epoch
                save_checkpoint(best_path, model, optimizer, epoch, metrics, cfg)

            append_history_csv(
                csv_path,
                {
                    "epoch": epoch,
                    "train_loss": round(tr["loss_mean"], 6),
                    "train_accuracy": round(tr["accuracy_mean"], 6),
                    "id_val_loss": round(va["loss_mean"], 6),
                    "id_val_accuracy": round(va["accuracy_mean"], 6),
                    "train_n": tr["n_samples"],
                    "id_val_n": va["n_samples"],
                    "is_best": int(is_best),
                },
            )

            emit(
                f"  epoch {epoch}/{epochs}  "
                f"train_loss={tr['loss_mean']:.6f} train_acc={tr['accuracy_mean']:.4f}  "
                f"id_val_loss={va['loss_mean']:.6f} id_val_acc={va['accuracy_mean']:.4f}  "
                f"(train_n={tr['n_samples']} id_val_n={va['n_samples']})"
                f"{'  <- best (' + sel_metric + ')' if is_best else ''}"
            )
            epoch_records.append(metrics)

        # --- assertions --------------------------------------------------------- #
        train_n = epoch_records[-1]["train_n"]
        id_val_n = epoch_records[-1]["id_val_n"]
        train_labels_present = sorted(train_counts.keys())
        id_val_labels_present = sorted(id_val_counts.keys())

        assert all_losses_finite, "a non-finite loss occurred during training/eval"
        assert train_n == expected_train_n, f"train_n {train_n} != {expected_train_n}"
        assert id_val_n == expected_id_val_n, f"id_val_n {id_val_n} != {expected_id_val_n}"
        assert train_labels_present == [0, 1], f"train labels {train_labels_present} != [0, 1]"
        assert id_val_labels_present == [0, 1], f"id_val labels {id_val_labels_present} != [0, 1]"
        assert best_path.exists(), f"best.pt not written at {best_path}"
        assert last_path.exists(), f"last.pt not written at {last_path}"
        assert best_epoch is not None, "no best epoch was selected"

        # --- reload verification: best.pt -> fresh model -> one id_val eval ----- #
        emit("")
        emit("--- reload verification (best.pt) ---")
        fresh_model = CamelyonClassifier(
            backbone_name=m_cfg["backbone_name"],
            pretrained=bool(m_cfg["pretrained"]),
            num_classes=int(m_cfg["num_classes"]),
        )
        payload = load_checkpoint(best_path, fresh_model, optimizer=None, map_location="cpu")
        fresh_model.to(device)
        reload_eval = evaluate(fresh_model, id_val_loader, criterion, device)

        ckpt_epoch = int(payload["epoch"])
        ckpt_id_val_loss = float(payload["metrics"]["id_val_loss"])
        ckpt_id_val_acc = float(payload["metrics"]["id_val_accuracy"])
        reload_id_val_loss = float(reload_eval["loss_mean"])
        reload_id_val_acc = float(reload_eval["accuracy_mean"])

        loss_match = math.isclose(reload_id_val_loss, ckpt_id_val_loss, rel_tol=1e-5, abs_tol=1e-5)
        acc_match = math.isclose(reload_id_val_acc, ckpt_id_val_acc, rel_tol=1e-5, abs_tol=1e-5)
        reload_finite = math.isfinite(reload_id_val_loss)
        reload_ok = loss_match and acc_match and reload_finite

        assert reload_finite, "reloaded best checkpoint produced a non-finite id_val loss"

        emit(f"  best checkpoint epoch        : {ckpt_epoch}")
        emit(f"  checkpoint id_val_loss        : {ckpt_id_val_loss:.6f}")
        emit(f"  reloaded  id_val_loss         : {reload_id_val_loss:.6f}")
        emit(f"  checkpoint id_val_accuracy    : {ckpt_id_val_acc:.4f}")
        emit(f"  reloaded  id_val_accuracy     : {reload_id_val_acc:.4f}")
        emit(f"  id_val_loss matches (1e-5)    : {'OK' if loss_match else 'MISMATCH'}")
        emit(f"  id_val_accuracy matches       : {'OK' if acc_match else 'MISMATCH'}")
        emit(f"  reload verification           : {'PASS' if reload_ok else 'FAIL'}")

        if not reload_ok:
            ok = False

        # --- summary ------------------------------------------------------------ #
        emit("")
        emit("--- summary ---")
        emit(f"all losses finite      : {'OK' if all_losses_finite else 'FAIL'}")
        emit(f"train_n                : {train_n}  (expected {expected_train_n})")
        emit(f"id_val_n               : {id_val_n}  (expected {expected_id_val_n})")
        emit(f"train labels present   : {train_labels_present}")
        emit(f"id_val labels present  : {id_val_labels_present}")
        emit(f"best epoch             : {best_epoch}")
        emit(f"best {sel_metric:<16} : {best_metric_value:.6f}")
        emit(f"best checkpoint        : {best_path}")
        emit(f"last checkpoint        : {last_path}")
        emit(f"saved CSV log          : {csv_path}")
        emit(f"saved text report      : {report_path}")

    except Exception as exc:
        ok = False
        emit("")
        emit(f"ERROR: {type(exc).__name__}: {exc}")

    emit("")
    emit("=" * 72)
    emit(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    emit("=" * 72)

    write_text(report_path, "\n".join(lines) + "\n")
    print(f"\nSaved report -> {report_path}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
