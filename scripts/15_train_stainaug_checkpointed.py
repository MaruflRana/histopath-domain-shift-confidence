"""Milestone 7A / Script 15: stain/color augmentation ERM baseline — checkpointed training.

Trains ResNet18 for 5 epochs on the SAME cached balanced ``train`` subset used by the
plain-ERM 4A baseline, but with moderate **stain/color augmentation** applied to the train
data only (``id_val`` uses the deterministic eval transform). Best checkpoint is selected by
``id_val_loss`` ONLY. This is a stain-robustness / domain-generalization baseline for
reviewer risk R3.

Strict scope (Milestone 7A training):
  * ``train`` for training, ``id_val`` for model selection / monitoring ONLY.
  * ``ood_val`` is NEVER read in this script (it is used only afterwards, by script 16, for
    development-stage comparison). ``ood_test`` is NEVER read anywhere.
  * No calibration, no uncertainty, no MC-dropout, no ensemble, no new architecture.

DEVELOPMENT run, NOT a paper result. All knobs come from
configs/experiments/exp07a_stainaug_resnet18.yaml.
Outputs: per-epoch CSV, text report (results/logs/), checkpoints
(results/checkpoints/exp07a_stainaug_resnet18/).
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import math
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.balanced_subset import BalancedSubsetDataset, load_or_build_balanced_examples
from data.stain_transforms import build_stain_train_transform
from data.transforms import eval_transform
from models.classifier import CamelyonClassifier
from training.checkpointing import load_checkpoint, save_checkpoint
from training.history import append_history_csv
from training.losses import create_loss
from training.loop import evaluate, train_one_epoch
from training.optimizers import create_optimizer
from utils.io import ensure_dir, ensure_results_tree, load_yaml, project_root, write_text
from utils.seed import set_seed

CONFIG_REL = "configs/experiments/exp07a_stainaug_resnet18.yaml"

# Parse ``balanced_{split}_pc{per_class}_seed{seed}_{order}.pt`` cache filenames.
_CACHE_RE = re.compile(r"^balanced_(?P<split>.+)_pc(?P<pc>\d+)_seed(?P<seed>\d+)_(?P<order>shuf|noshuf)\.pt$")


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def _cache_params(cache_rel: str) -> tuple[str, int, int, bool]:
    """Return (split_name, per_class, seed, shuffle_collected) parsed from a cache path."""
    name = Path(cache_rel).name
    m = _CACHE_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized balanced-cache filename: {name!r}")
    return m["split"], int(m["pc"]), int(m["seed"]), (m["order"] == "shuf")


def _load_cached_subset(cache_rel: str) -> tuple[list[dict], Path]:
    """Load a balanced subset strictly from its existing disk cache (no HF streaming)."""
    path = _resolve(cache_rel)
    if not path.exists():
        raise FileNotFoundError(f"required cache not found (build via 4A/5A first): {path}")
    split_name, per_class, seed, shuffle_collected = _cache_params(cache_rel)
    examples = load_or_build_balanced_examples(
        split_name, per_class=per_class, seed=seed,
        shuffle_collected=shuffle_collected, use_cache=True, verbose=True,
    )
    return examples, path


def main() -> int:
    root = project_root()
    cfg = load_yaml(root / CONFIG_REL)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    a_cfg = cfg["augmentation"]
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

    batch_size = int(d_cfg["batch_size"])
    num_workers = int(d_cfg["num_workers"])
    epochs = int(t_cfg["epochs"])

    sel_metric = str(s_cfg["metric"])
    sel_mode = str(s_cfg["mode"])

    # Expected sizes derived from the cache filenames (train pc1000 -> 2000, id_val pc250 -> 500).
    _, train_pc, _, _ = _cache_params(d_cfg["train_cache"])
    _, id_val_pc, _, _ = _cache_params(d_cfg["id_val_cache"])
    expected_train_n = 2 * train_pc
    expected_id_val_n = 2 * id_val_pc

    csv_path = _resolve(o_cfg["log_csv"])
    report_path = _resolve(o_cfg["report_txt"])
    ckpt_dir = _resolve(o_cfg["checkpoint_dir"])
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    if csv_path.exists():
        csv_path.unlink()

    emit("=" * 72)
    emit("Milestone 7A — Stain/color augmentation ERM baseline (script 15)")
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
        f"augmentation={a_cfg['name']}  hflip={a_cfg['horizontal_flip_p']} "
        f"vflip={a_cfg['vertical_flip_p']} rot={a_cfg['rotation_degrees']} "
        f"bright={a_cfg['brightness']} contrast={a_cfg['contrast']} "
        f"sat={a_cfg['saturation']} hue={a_cfg['hue']}"
    )
    emit(f"batch_size={batch_size}  num_workers={num_workers}")
    emit(f"selection: metric={sel_metric}  mode={sel_mode}")
    emit(f"device={device}")
    emit(f"checkpoint_dir={ckpt_dir}")
    emit("scope: train (augmented) + id_val (eval transform) ONLY — ood_val/ood_test untouched.")
    emit("=" * 72)

    ok = True

    try:
        # --- balanced subsets (train + id_val ONLY; NO ood_val / NO ood_test) --- #
        emit("")
        emit("--- loading cached balanced subsets (cache HIT expected; no streaming) ---")

        train_examples, train_cache_path = _load_cached_subset(d_cfg["train_cache"])
        id_val_examples, id_val_cache_path = _load_cached_subset(d_cfg["id_val_cache"])

        # Train uses stain/color augmentation; id_val uses the deterministic eval transform.
        stain_train_transform = build_stain_train_transform(a_cfg)
        train_ds = BalancedSubsetDataset(train_examples, transform=stain_train_transform)
        id_val_ds = BalancedSubsetDataset(id_val_examples, transform=eval_transform)

        train_counts = train_ds.label_counts()
        id_val_counts = id_val_ds.label_counts()

        emit("")
        emit(f"train  cache : {train_cache_path}")
        emit(f"id_val cache : {id_val_cache_path}")
        emit(f"train  transform : StainColorAugment (train-only augmentation)")
        emit(f"id_val transform : eval_transform (no augmentation)")
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
        emit("ood_val usage          : NONE (training used train + id_val only)")
        emit("ood_test usage         : NONE")

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
