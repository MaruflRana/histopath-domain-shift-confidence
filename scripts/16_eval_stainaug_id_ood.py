"""Milestone 7A / Script 16: evaluate the stain-aug baseline on id_val + ood_val.

Loads the best stain-augmentation checkpoint (script 15), runs inference over the cached
balanced ``id_val`` and ``ood_val`` subsets (loaded from disk — no HF streaming), writes
per-example prediction CSVs, computes binary classification metrics (tumor = 1), and prints
a side-by-side comparison against the plain-ERM 4A baseline (id_val from the 4B metrics CSV,
ood_val from the 5A calibration JSON's raw classification block).

Scope: this is the FIRST authorized ``ood_val`` read for the 7A baseline, and it is a
**development-stage comparison AFTER training** (never for training or model selection). Eval
transform only (no augmentation at eval). ``ood_test`` is NEVER read. No calibration, no
uncertainty, no MC-dropout, no ensemble. DEVELOPMENT result, NOT final performance.

Inputs (must already exist):
  * results/checkpoints/exp07a_stainaug_resnet18/best.pt   (script 15)
  * results/cache/balanced_id_val_pc250_seed1337_shuf.pt
  * results/cache/balanced_ood_val_pc1000_seed1337_noshuf.pt
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import csv as _csv
import json
import math
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.balanced_subset import BalancedSubsetDataset, load_or_build_balanced_examples
from data.transforms import eval_transform
from evaluation.metrics_classification import compute_binary_classification_metrics
from evaluation.predict import predict_to_dataframe
from models.classifier import CamelyonClassifier
from training.checkpointing import load_checkpoint
from utils.io import ensure_dir, ensure_results_tree, load_yaml, project_root, write_csv, write_text
from utils.seed import set_seed

CONFIG_REL = "configs/experiments/exp07a_stainaug_resnet18.yaml"

# Plain-ERM baseline artifacts for the comparison print-out.
PLAIN_ERM_ID_VAL_CSV = "results/tables/exp04a_dev_train_resnet18_classification_metrics.csv"
PLAIN_ERM_OOD_VAL_JSON = "results/metrics/exp05a_calibrate_ood_val_resnet18/calibration_metrics.json"

BATCH_SIZE = 64
NUM_WORKERS = 0

_METRIC_FIELDS = [
    "n", "accuracy", "balanced_accuracy", "auroc", "auprc",
    "sensitivity", "specificity", "precision", "f1",
    "tn", "fp", "fn", "tp", "threshold", "positive_class",
]

_CACHE_RE = re.compile(r"^balanced_(?P<split>.+)_pc(?P<pc>\d+)_seed(?P<seed>\d+)_(?P<order>shuf|noshuf)\.pt$")


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def _cache_params(cache_rel: str) -> tuple[str, int, int, bool]:
    name = Path(cache_rel).name
    m = _CACHE_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized balanced-cache filename: {name!r}")
    return m["split"], int(m["pc"]), int(m["seed"]), (m["order"] == "shuf")


def _load_cached_subset(cache_rel: str) -> tuple[list[dict], Path, int]:
    """Load a balanced subset strictly from its existing disk cache (no HF streaming)."""
    path = _resolve(cache_rel)
    if not path.exists():
        raise FileNotFoundError(f"required cache not found: {path}")
    split_name, per_class, seed, shuffle_collected = _cache_params(cache_rel)
    examples = load_or_build_balanced_examples(
        split_name, per_class=per_class, seed=seed,
        shuffle_collected=shuffle_collected, use_cache=True, verbose=True,
    )
    return examples, path, 2 * per_class


def _plain_erm_id_val() -> dict | None:
    """Read plain-ERM id_val (accuracy, auroc) from the 4B metrics CSV, if present."""
    path = _resolve(PLAIN_ERM_ID_VAL_CSV)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in _csv.DictReader(f):
            if row.get("split") == "id_val":
                return {"accuracy": float(row["accuracy"]), "auroc": float(row["auroc"])}
    return None


def _plain_erm_ood_val() -> dict | None:
    """Read plain-ERM ood_val (accuracy, auroc) from the 5A calibration JSON (raw), if present."""
    path = _resolve(PLAIN_ERM_OOD_VAL_JSON)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("classification", {}).get("raw")
    if not raw:
        return None
    return {"accuracy": float(raw["accuracy"]), "auroc": float(raw["auroc"])}


def main() -> int:
    root = project_root()
    cfg = load_yaml(root / CONFIG_REL)
    ensure_results_tree()

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    o_cfg = cfg["output"]
    run_name = o_cfg["run_name"]

    set_seed(int(cfg.get("training", {}).get("seed", 1337)))

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = _resolve(o_cfg["checkpoint_dir"]) / "best.pt"
    pred_dir = _resolve(o_cfg["prediction_dir"])
    id_val_pred_csv = pred_dir / "id_val_predictions.csv"
    ood_val_pred_csv = pred_dir / "ood_val_predictions.csv"
    metrics_json = _resolve(o_cfg["metrics_json"])
    metrics_csv = _resolve(o_cfg["metrics_csv"])
    report_path = _resolve(f"results/logs/{run_name}_eval_report.txt")

    emit("=" * 72)
    emit("Milestone 7A — Stain-aug baseline eval on id_val + ood_val (script 16)")
    emit(f"run_name={run_name}")
    emit(f"device={device}")
    emit(f"checkpoint={ckpt_path}")
    emit("scope: cached id_val + ood_val ONLY (eval transform). ood_val = post-training")
    emit("       development comparison (authorized). NO training, NO ood_test.")
    emit("=" * 72)

    ok = True

    try:
        assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

        model = CamelyonClassifier(
            backbone_name=m_cfg["backbone_name"],
            pretrained=bool(m_cfg["pretrained"]),
            num_classes=int(m_cfg["num_classes"]),
        )
        payload = load_checkpoint(ckpt_path, model, optimizer=None, map_location="cpu")
        model.to(device)
        emit("")
        emit(
            f"loaded checkpoint: epoch={payload.get('epoch')}  "
            f"metrics={ {k: payload['metrics'][k] for k in ('id_val_loss', 'id_val_accuracy') if k in payload.get('metrics', {})} }"
        )

        # --- cached subsets: id_val + ood_val (eval transform only) ------------- #
        id_val_examples, id_val_cache_path, expected_id_val_n = _load_cached_subset(d_cfg["id_val_cache"])
        ood_val_examples, ood_val_cache_path, expected_ood_val_n = _load_cached_subset(d_cfg["ood_val_cache"])

        id_val_ds = BalancedSubsetDataset(id_val_examples, transform=eval_transform)
        ood_val_ds = BalancedSubsetDataset(ood_val_examples, transform=eval_transform)

        id_val_loader = DataLoader(
            id_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            drop_last=False, pin_memory=False,
        )
        ood_val_loader = DataLoader(
            ood_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            drop_last=False, pin_memory=False,
        )

        emit("")
        emit(f"id_val  cache : {id_val_cache_path}  (n={len(id_val_ds)})")
        emit(f"ood_val cache : {ood_val_cache_path}  (n={len(ood_val_ds)})")

        # --- inference ---------------------------------------------------------- #
        emit("")
        emit("--- running inference ---")
        id_val_df = predict_to_dataframe(model, id_val_loader, device, "id_val")
        ood_val_df = predict_to_dataframe(model, ood_val_loader, device, "ood_val")
        emit(f"id_val predictions : n={len(id_val_df)}")
        emit(f"ood_val predictions: n={len(ood_val_df)}")

        # --- probability sanity (finite, [0,1], sum≈1) -------------------------- #
        def _prob_checks(df, name) -> None:
            p0 = df["prob_0"].to_numpy(dtype=float)
            p1 = df["prob_1"].to_numpy(dtype=float)
            assert bool(((p0 >= 0) & (p0 <= 1)).all()), f"{name}: prob_0 out of [0,1]"
            assert bool(((p1 >= 0) & (p1 <= 1)).all()), f"{name}: prob_1 out of [0,1]"
            assert all(math.isfinite(v) for v in p0), f"{name}: non-finite prob_0"
            assert all(math.isfinite(v) for v in p1), f"{name}: non-finite prob_1"
            s = p0 + p1
            assert bool((abs(s - 1.0) < 1e-5).all()), f"{name}: prob_0+prob_1 != 1"

        _prob_checks(id_val_df, "id_val")
        _prob_checks(ood_val_df, "ood_val")

        # --- save prediction CSVs ----------------------------------------------- #
        ensure_dir(pred_dir)
        id_val_df.to_csv(id_val_pred_csv, index=False)
        ood_val_df.to_csv(ood_val_pred_csv, index=False)

        # --- classification metrics (tumor = positive) ------------------------- #
        emit("")
        emit("--- classification metrics (positive class = tumor = 1) ---")
        id_val_metrics = compute_binary_classification_metrics(
            id_val_df["label"].to_numpy(), id_val_df["prob_1"].to_numpy(), threshold=0.5
        )
        ood_val_metrics = compute_binary_classification_metrics(
            ood_val_df["label"].to_numpy(), ood_val_df["prob_1"].to_numpy(), threshold=0.5
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

        _emit_metrics("id_val (stain-aug)", id_val_metrics)
        _emit_metrics("ood_val (stain-aug)", ood_val_metrics)

        # --- save metrics (JSON + flat CSV) ------------------------------------- #
        ensure_dir(metrics_json.parent)
        metrics_payload = {
            "run_name": run_name,
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "threshold": 0.5,
            "positive_class": 1,
            "splits": {"id_val": id_val_metrics, "ood_val": ood_val_metrics},
            "note": "Stain-aug ERM baseline (Milestone 7A). DEVELOPMENT comparison on capped "
                    "balanced subsets. ood_val used post-training for development comparison "
                    "only. NOT a final result. ood_test never touched.",
        }
        write_text(metrics_json, json.dumps(metrics_payload, indent=2) + "\n")

        csv_rows = []
        for split_name, m in (("id_val", id_val_metrics), ("ood_val", ood_val_metrics)):
            row = {"split": split_name}
            for k in _METRIC_FIELDS:
                row[k] = m[k]
            csv_rows.append(row)
        write_csv(metrics_csv, csv_rows, header=["split", *_METRIC_FIELDS])

        # --- comparison to plain ERM (4A/5A) ------------------------------------ #
        emit("")
        emit("--- comparison: plain ERM (4A/5A) vs stain-aug (7A) ---")
        plain_id = _plain_erm_id_val()
        plain_ood = _plain_erm_ood_val()

        def _cmp_line(label, plain, stain_acc, stain_auroc) -> None:
            if plain is None:
                emit(f"  {label:<28}: plain=UNAVAILABLE  "
                     f"stain-aug acc={stain_acc:.4f} auroc={stain_auroc:.4f}")
                return
            d_acc = stain_acc - plain["accuracy"]
            d_auc = stain_auroc - plain["auroc"]
            emit(
                f"  {label:<28}: plain acc={plain['accuracy']:.4f} auroc={plain['auroc']:.4f}  |  "
                f"stain-aug acc={stain_acc:.4f} auroc={stain_auroc:.4f}  |  "
                f"d_acc={d_acc:+.4f} d_auroc={d_auc:+.4f}"
            )

        _cmp_line("id_val", plain_id, id_val_metrics["accuracy"], id_val_metrics["auroc"])
        _cmp_line("ood_val", plain_ood, ood_val_metrics["accuracy"], ood_val_metrics["auroc"])
        emit("  (d = stain-aug - plain ERM; positive d favors stain augmentation.)")
        emit("  NOTE: development-stage comparison on capped balanced subsets - NOT final performance.")

        # --- assertions --------------------------------------------------------- #
        assert len(id_val_df) == expected_id_val_n, f"id_val n {len(id_val_df)} != {expected_id_val_n}"
        assert len(ood_val_df) == expected_ood_val_n, f"ood_val n {len(ood_val_df)} != {expected_ood_val_n}"
        assert expected_id_val_n == 500, f"expected id_val n==500, got {expected_id_val_n}"
        assert expected_ood_val_n == 2000, f"expected ood_val n==2000, got {expected_ood_val_n}"
        assert id_val_pred_csv.exists(), "id_val prediction CSV missing"
        assert ood_val_pred_csv.exists(), "ood_val prediction CSV missing"
        assert metrics_json.exists(), "metrics JSON missing"
        assert metrics_csv.exists(), "metrics CSV missing"

        # --- summary ------------------------------------------------------------ #
        emit("")
        emit("--- summary ---")
        emit(f"id_val predictions CSV : {id_val_pred_csv}")
        emit(f"ood_val predictions CSV: {ood_val_pred_csv}")
        emit(f"metrics JSON           : {metrics_json}")
        emit(f"metrics CSV            : {metrics_csv}")
        emit(f"text report            : {report_path}")
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
