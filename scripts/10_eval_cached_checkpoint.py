"""Milestone 4B / Script 10: prediction export + classification metrics scaffold.

Loads the accepted best checkpoint from Milestone 4A, runs inference over the cached
balanced ``train`` and ``id_val`` subsets (loaded from disk — no HF streaming), writes
per-example prediction CSVs, and computes binary classification metrics (tumor = 1).

This is an evaluation scaffold that prepares for later calibration / OOD work. It does
NOT train, NOT calibrate, NOT estimate uncertainty, NOT do selective prediction, and it
NEVER reads ``ood_val`` / ``ood_test``. Not a final result.

Inputs (must already exist):
  * results/checkpoints/exp04a_dev_train_resnet18/best.pt
  * results/cache/balanced_train_pc1000_seed1337_shuf.pt
  * results/cache/balanced_id_val_pc250_seed1337_shuf.pt
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import json
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
from data.transforms import eval_transform
from evaluation.metrics_classification import compute_binary_classification_metrics
from evaluation.predict import predict_to_dataframe
from models.classifier import CamelyonClassifier
from training.checkpointing import load_checkpoint
from utils.io import ensure_dir, ensure_results_tree, project_root, write_csv, write_text
from utils.seed import set_seed

RUN_NAME = "exp04a_dev_train_resnet18"
CKPT_REL = f"results/checkpoints/{RUN_NAME}/best.pt"

BATCH_SIZE = 64
NUM_WORKERS = 0

# Metric fields (in order) flattened into the per-split CSV table.
_METRIC_FIELDS = [
    "n", "accuracy", "balanced_accuracy", "auroc", "auprc",
    "sensitivity", "specificity", "precision", "f1",
    "tn", "fp", "fn", "tp", "threshold", "positive_class",
]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def _cache_path(split_name: str, per_class: int, seed: int, shuffle_collected: bool) -> Path:
    order = "shuf" if shuffle_collected else "noshuf"
    return default_cache_dir() / f"balanced_{split_name}_pc{per_class}_seed{seed}_{order}.pt"


def main() -> int:
    root = project_root()
    ensure_results_tree()

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = _resolve(CKPT_REL)
    report_path = _resolve(f"results/logs/{RUN_NAME.split('_')[0]}_eval_cached_checkpoint_report.txt")
    # results/logs/exp04a_eval_cached_checkpoint_report.txt
    pred_dir = _resolve(f"results/predictions/{RUN_NAME}")
    metrics_dir = _resolve(f"results/metrics/{RUN_NAME}")
    train_pred_csv = pred_dir / "train_cached_predictions.csv"
    id_val_pred_csv = pred_dir / "id_val_cached_predictions.csv"
    metrics_json = metrics_dir / "classification_metrics.json"
    metrics_csv = _resolve(f"results/tables/{RUN_NAME}_classification_metrics.csv")

    emit("=" * 72)
    emit("Milestone 4B — Prediction export + classification metrics (script 10)")
    emit(f"run_name={RUN_NAME}")
    emit(f"device={device}")
    emit(f"checkpoint={ckpt_path}")
    emit("scope: cached train + id_val ONLY — no ood_val, no ood_test, no training.")
    emit("=" * 72)

    ok = True

    try:
        # --- checkpoint presence -------------------------------------------- #
        assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

        # Load payload first (need its config to rebuild the model + locate caches).
        # Instantiate a temporary model of the right type, then load weights into it.
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        m_cfg = cfg["model"]
        d_cfg = cfg["data"]
        data_seed = int(d_cfg.get("seed", 1337))
        shuffle_collected = bool(d_cfg.get("shuffle_collected", True))
        train_per_class = int(d_cfg["train_per_class"])
        id_val_per_class = int(d_cfg["id_val_per_class"])
        expected_train_n = 2 * train_per_class
        expected_id_val_n = 2 * id_val_per_class

        # Deterministic eval (seeding does not affect a no-grad forward pass, but keep
        # the project convention of an explicit seed).
        set_seed(int(cfg.get("training", {}).get("seed", 1337)))

        model = CamelyonClassifier(
            backbone_name=m_cfg["backbone_name"],
            pretrained=bool(m_cfg["pretrained"]),
            num_classes=int(m_cfg["num_classes"]),
        )
        load_checkpoint(ckpt_path, model, optimizer=None, map_location="cpu")
        model.to(device)
        emit("")
        emit(
            f"loaded checkpoint: epoch={payload.get('epoch')}  "
            f"metrics={ {k: payload['metrics'][k] for k in ('id_val_loss', 'id_val_accuracy') if k in payload.get('metrics', {})} }"
        )
        emit(
            f"model: backbone={m_cfg['backbone_name']}  pretrained={m_cfg['pretrained']}  "
            f"num_classes={m_cfg['num_classes']}"
        )

        # --- cache presence (train + id_val ONLY) --------------------------- #
        train_cache = _cache_path("train", train_per_class, data_seed, shuffle_collected)
        id_val_cache = _cache_path("id_val", id_val_per_class, data_seed, shuffle_collected)
        assert train_cache.exists(), f"train cache not found: {train_cache}"
        assert id_val_cache.exists(), f"id_val cache not found: {id_val_cache}"

        # Load cached subsets from disk (use_cache=True -> cache HIT, no HF streaming).
        train_examples = load_or_build_balanced_examples(
            "train", per_class=train_per_class, seed=data_seed,
            shuffle_collected=shuffle_collected, use_cache=True, verbose=True,
        )
        id_val_examples = load_or_build_balanced_examples(
            "id_val", per_class=id_val_per_class, seed=data_seed,
            shuffle_collected=shuffle_collected, use_cache=True, verbose=True,
        )

        train_ds = BalancedSubsetDataset(train_examples, transform=eval_transform)
        id_val_ds = BalancedSubsetDataset(id_val_examples, transform=eval_transform)

        # shuffle=False for both, so CSV `index` matches dataset position.
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            drop_last=False, pin_memory=False,
        )
        id_val_loader = DataLoader(
            id_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            drop_last=False, pin_memory=False,
        )

        # --- inference (train + id_val) ------------------------------------- #
        emit("")
        emit("--- running inference ---")
        train_df = predict_to_dataframe(model, train_loader, device, "train")
        id_val_df = predict_to_dataframe(model, id_val_loader, device, "id_val")
        emit(f"train predictions : n={len(train_df)}")
        emit(f"id_val predictions: n={len(id_val_df)}")

        # --- probability sanity (finite, [0,1], sum≈1) ---------------------- #
        def _prob_checks(df, name) -> None:
            p0 = df["prob_0"].to_numpy(dtype=float)
            p1 = df["prob_1"].to_numpy(dtype=float)
            assert bool(((p0 >= 0) & (p0 <= 1)).all()), f"{name}: prob_0 out of [0,1]"
            assert bool(((p1 >= 0) & (p1 <= 1)).all()), f"{name}: prob_1 out of [0,1]"
            assert all(math.isfinite(v) for v in p0), f"{name}: non-finite prob_0"
            assert all(math.isfinite(v) for v in p1), f"{name}: non-finite prob_1"
            s = p0 + p1
            assert bool((abs(s - 1.0) < 1e-5).all()), f"{name}: prob_0+prob_1 != 1"

        _prob_checks(train_df, "train")
        _prob_checks(id_val_df, "id_val")

        # --- save prediction CSVs ------------------------------------------- #
        ensure_dir(pred_dir)
        train_df.to_csv(train_pred_csv, index=False)
        id_val_df.to_csv(id_val_pred_csv, index=False)

        # --- classification metrics (tumor = positive) --------------------- #
        emit("")
        emit("--- classification metrics (positive class = tumor = 1) ---")
        train_metrics = compute_binary_classification_metrics(
            train_df["label"].to_numpy(), train_df["prob_1"].to_numpy(), threshold=0.5
        )
        id_val_metrics = compute_binary_classification_metrics(
            id_val_df["label"].to_numpy(), id_val_df["prob_1"].to_numpy(), threshold=0.5
        )

        def _fmt(v) -> str:
            if v is None:
                return "None"
            if isinstance(v, float):
                return f"{v:.6f}"
            return str(v)

        def _emit_metrics(name, m) -> None:
            emit(f"  [{name}]")
            for k in _METRIC_FIELDS:
                emit(f"      {k:<18}: {_fmt(m[k])}")
            if m["warnings"]:
                for w in m["warnings"]:
                    emit(f"      warning           : {w}")

        _emit_metrics("train", train_metrics)
        _emit_metrics("id_val", id_val_metrics)

        # --- save metrics (JSON + flat CSV) --------------------------------- #
        ensure_dir(metrics_dir)
        metrics_payload = {
            "run_name": RUN_NAME,
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "threshold": 0.5,
            "positive_class": 1,
            "splits": {"train": train_metrics, "id_val": id_val_metrics},
            "note": "Development evaluation on cached balanced subsets. NOT a final "
                    "result. No ood_val / ood_test used.",
        }
        write_text(metrics_json, json.dumps(metrics_payload, indent=2) + "\n")

        csv_rows = []
        for split_name, m in (("train", train_metrics), ("id_val", id_val_metrics)):
            row = {"split": split_name}
            for k in _METRIC_FIELDS:
                row[k] = m[k]
            csv_rows.append(row)
        write_csv(metrics_csv, csv_rows, header=["split", *_METRIC_FIELDS])

        # --- assertions ----------------------------------------------------- #
        assert len(train_df) == expected_train_n, f"train n {len(train_df)} != {expected_train_n}"
        assert len(id_val_df) == expected_id_val_n, f"id_val n {len(id_val_df)} != {expected_id_val_n}"
        assert train_pred_csv.exists(), "train prediction CSV missing"
        assert id_val_pred_csv.exists(), "id_val prediction CSV missing"
        assert metrics_json.exists(), "metrics JSON missing"
        assert metrics_csv.exists(), "metrics CSV missing"

        # --- summary -------------------------------------------------------- #
        emit("")
        emit("--- summary ---")
        emit(f"train predictions CSV : {train_pred_csv}")
        emit(f"id_val predictions CSV: {id_val_pred_csv}")
        emit(f"metrics JSON          : {metrics_json}")
        emit(f"metrics CSV           : {metrics_csv}")
        emit(f"text report           : {report_path}")

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
