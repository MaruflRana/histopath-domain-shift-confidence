"""CSV-only, resumable final summary for the authorized exp09 inference outputs.

This stage imports no dataset, model, checkpoint, or torch module. It requires the
immutable inference-complete sentinel written by script 38, validates the four saved
prediction CSV hashes and schemas, and then produces only tables, figures, metrics,
the final report, and a separate summary-complete sentinel.

It never increments the inference attempt counter. If summary generation is
interrupted after final inference completed, this script may be rerun from the saved
predictions. Existing exp09 summary files are replaceable only while the
summary-complete sentinel is absent; the sentinel makes a completed summary
idempotent and immutable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.metrics_calibration import compute_calibration_metrics  # noqa: E402
from evaluation.metrics_classification import compute_binary_classification_metrics  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "exp09_final_ood_test_frozen.yaml"
MODEL_ORDER = ("groupdro", "centerstrat_erm")
VARIANTS = ("raw", "calibrated")
MODEL_LABELS = {
    "groupdro": "7F GroupDRO (authorized final primary)",
    "centerstrat_erm": "7F center-stratified ERM (authorized matched control)",
}
CONFIDENCE_THRESHOLDS = (0.90, 0.95, 0.99)
N_ECE_BINS = 15
RANKING_TOLERANCE = 1e-12
REQUIRED_COLUMNS = (
    "image_id", "center", "patient", "slide", "node", "x_coord", "y_coord",
    "label", "logit_0", "logit_1", "prob_0", "prob_1", "pred_label",
    "confidence", "variant", "temperature",
)


class SummaryError(RuntimeError):
    """Raised when saved final-run artifacts violate the frozen schema."""


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise SummaryError(f"YAML root must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"Unreadable/malformed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SummaryError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _prediction_path(cfg: dict[str, Any], model: str, variant: str) -> Path:
    return _resolve(cfg["output"]["predictions_dir"]) / (
        f"{model}_ood_test_{variant}_predictions.csv"
    )


def _metrics_json_path(cfg: dict[str, Any]) -> Path:
    return _resolve(cfg["output"]["metrics_dir"]) / "final_metrics.json"


def _summary_output_map(cfg: dict[str, Any]) -> dict[str, Path]:
    out = cfg["output"]
    return {
        "metrics_json": _metrics_json_path(cfg),
        "classification_table": _resolve(out["classification_table"]),
        "calibration_table": _resolve(out["calibration_table"]),
        "operating_points_table": _resolve(out["operating_points_table"]),
        "high_confidence_fn_table": _resolve(out["high_confidence_fn_table"]),
        "roc_figure": _resolve(out["roc_figure"]),
        "precision_recall_figure": _resolve(out["precision_recall_figure"]),
        "reliability_figure": _resolve(out["reliability_figure"]),
        "operating_point_figure": _resolve(out["operating_point_figure"]),
        "high_confidence_fn_figure": _resolve(out["high_confidence_fn_figure"]),
        "report": _resolve(out["report"]),
    }


def _temp_path(path: Path, *, keep_suffix: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    if keep_suffix:
        return path.with_name(f".{path.stem}.{token}.tmp{path.suffix}")
    return path.with_name(f".{path.name}.{token}.tmp")


def _atomic_replace_with(path: Path, writer: Callable[[Path], None], *, keep_suffix: bool = False) -> None:
    temp = _temp_path(path, keep_suffix=keep_suffix)
    try:
        writer(temp)
        if not temp.is_file() or temp.stat().st_size <= 0:
            raise SummaryError(f"Atomic staging produced an empty output: {path}")
        with temp.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    def writer(temp: Path) -> None:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_replace_with(path, writer)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    def writer(temp: Path) -> None:
        with temp.open("x", encoding="utf-8", newline="") as handle:
            pd.DataFrame(rows).to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_replace_with(path, writer)


def _write_text_atomic(path: Path, text: str) -> None:
    def writer(temp: Path) -> None:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_replace_with(path, writer)


def _write_figure_atomic(path: Path, figure: Any) -> None:
    def writer(temp: Path) -> None:
        figure.savefig(temp, dpi=160, format=path.suffix.lstrip("."))

    _atomic_replace_with(path, writer, keep_suffix=True)


def _atomic_create_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    temp = _temp_path(path)
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise SummaryError(f"Summary sentinel already exists: {path}") from exc
    finally:
        temp.unlink(missing_ok=True)


def _validate_frame(df: pd.DataFrame, cfg: dict[str, Any], model: str, variant: str) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise SummaryError(f"{model}/{variant}: missing columns {missing}.")
    if len(df) != int(cfg["expected_rows"]):
        raise SummaryError(f"{model}/{variant}: n={len(df)} != {cfg['expected_rows']}.")
    if {int(value) for value in df["center"].unique().tolist()} != {
        int(cfg["required_center"])
    }:
        raise SummaryError(f"{model}/{variant}: center is not exactly {{2}}.")
    if {int(value) for value in df["label"].unique().tolist()} != {0, 1}:
        raise SummaryError(f"{model}/{variant}: labels are not exactly binary {{0,1}}.")
    if set(df["variant"].astype(str).unique().tolist()) != {variant}:
        raise SummaryError(f"{model}/{variant}: variant column mismatch.")
    expected_temperature = (
        1.0 if variant == "raw" else float(cfg["models"][model]["temperature"])
    )
    if not np.all(df["temperature"].to_numpy(dtype=float) == expected_temperature):
        raise SummaryError(f"{model}/{variant}: frozen temperature mismatch.")
    for column in cfg["required_metadata"]:
        if column not in df.columns or df[column].isnull().any():
            raise SummaryError(f"{model}/{variant}: missing/null metadata in {column}.")
    logits = df[["logit_0", "logit_1"]].to_numpy(dtype=float)
    probabilities = df[["prob_0", "prob_1"]].to_numpy(dtype=float)
    predictions = df["pred_label"].to_numpy(dtype=int)
    confidence = df["confidence"].to_numpy(dtype=float)
    if not np.isfinite(logits).all():
        raise SummaryError(f"{model}/{variant}: non-finite logits.")
    if not np.isfinite(probabilities).all() or not np.isfinite(confidence).all():
        raise SummaryError(f"{model}/{variant}: non-finite probability/confidence.")
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise SummaryError(f"{model}/{variant}: probability outside [0,1].")
    if not ((confidence >= 0.0) & (confidence <= 1.0)).all():
        raise SummaryError(f"{model}/{variant}: confidence outside [0,1].")
    if not (np.abs(probabilities.sum(axis=1) - 1.0) <= 1e-5).all():
        raise SummaryError(f"{model}/{variant}: probability sum error.")
    if not np.array_equal(predictions, np.argmax(logits, axis=1)):
        raise SummaryError(f"{model}/{variant}: pred_label differs from argmax.")
    if not np.array_equal(predictions, (probabilities[:, 1] >= 0.5).astype(int)):
        raise SummaryError(f"{model}/{variant}: threshold-0.5 prediction differs from argmax.")
    if not np.allclose(confidence, probabilities.max(axis=1), rtol=0.0, atol=1e-12):
        raise SummaryError(f"{model}/{variant}: confidence is not max(prob_0, prob_1).")


def _metadata_matrix(df: pd.DataFrame) -> np.ndarray:
    columns = ["image_id", "center", "patient", "slide", "node", "x_coord", "y_coord", "label"]
    return df[columns].to_numpy()


def _ranking_score(df: pd.DataFrame) -> np.ndarray:
    return (
        df["logit_1"].to_numpy(dtype=np.float64)
        - df["logit_0"].to_numpy(dtype=np.float64)
    )


def _ranking_metrics(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def _load_inference_sentinel(
    cfg: dict[str, Any],
    config_path: Path,
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(cfg["output"]["completed_run_sentinel"])
    if not path.is_file():
        raise SummaryError("Completed inference sentinel is missing; summary is forbidden.")
    sentinel = _load_json(path)
    expected = {
        "run_status": "INFERENCE_COMPLETE",
        "attempt_counter": 1,
        "run_counter": 1,
        "maximum_test_runs": 1,
        "authorization_confirmed": True,
        "row_count": int(cfg["expected_rows"]),
        "required_center": int(cfg["required_center"]),
        "test_dataset_instances": 1,
        "test_dataloader_traversals": 1,
        "models_per_batch": 2,
        "calibration_refit": False,
        "threshold_selection_on_test": False,
    }
    for key, value in expected.items():
        if sentinel.get(key) != value:
            raise SummaryError(f"Inference sentinel mismatch: {key}.")
    if sentinel.get("frozen_config_sha256") != _sha256(config_path):
        raise SummaryError("Frozen config changed after inference.")
    manifest_path = _resolve(cfg["frozen_operating_point_manifest"])
    if sentinel.get("frozen_operating_point_manifest_sha256") != _sha256(manifest_path):
        raise SummaryError("Frozen operating-point manifest changed after inference.")
    for model in MODEL_ORDER:
        if (
            sentinel.get("checkpoint_hashes", {}).get(model)
            != str(cfg["models"][model]["checkpoint_sha256"]).upper()
        ):
            raise SummaryError(f"Inference sentinel checkpoint hash mismatch: {model}.")
        if float(sentinel.get("temperatures", {}).get(model, float("nan"))) != float(
            cfg["models"][model]["temperature"]
        ):
            raise SummaryError(f"Inference sentinel temperature mismatch: {model}.")
    records = sentinel.get("prediction_files")
    if not isinstance(records, dict) or set(records) != {
        f"{model}_{variant}" for model in MODEL_ORDER for variant in VARIANTS
    }:
        raise SummaryError("Inference sentinel prediction manifest is incomplete.")
    for model in MODEL_ORDER:
        for variant in VARIANTS:
            key = f"{model}_{variant}"
            record = records[key]
            expected_path = _prediction_path(cfg, model, variant)
            if _resolve(record.get("path", "")) != expected_path:
                raise SummaryError(f"Inference sentinel prediction path mismatch: {key}.")
            if int(record.get("rows", -1)) != int(cfg["expected_rows"]):
                raise SummaryError(f"Inference sentinel row count mismatch: {key}.")
            if not expected_path.is_file():
                raise SummaryError(f"Saved prediction file is missing: {expected_path}")
            if _sha256(expected_path) != str(record.get("sha256", "")).upper():
                raise SummaryError(f"Saved prediction hash mismatch: {key}.")
    return path, sentinel


def _load_frozen_operating_points(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    path = _resolve(cfg["frozen_operating_point_manifest"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    required = {
        "model", "target_type", "target_value", "threshold_selected_on_id_val",
        "id_val_sensitivity", "id_val_specificity", "source_artifact",
        "test_selection_forbidden", "candidate_only_note",
    }
    if not manifest_rows or required - set(manifest_rows[0]):
        raise SummaryError("Frozen operating-point manifest schema is incomplete.")
    if len(manifest_rows) != 14:
        raise SummaryError(
            f"Frozen operating-point manifest has {len(manifest_rows)} rows, expected 14."
        )
    expected_keys = {
        (model, "specificity", target)
        for model in MODEL_ORDER for target in (0.80, 0.85, 0.90, 0.95)
    } | {
        (model, "sensitivity", target)
        for model in MODEL_ORDER for target in (0.80, 0.90, 0.95)
    }
    observed_keys = {
        (row["model"], row["target_type"], round(float(row["target_value"]), 2))
        for row in manifest_rows
    }
    if observed_keys != expected_keys:
        raise SummaryError("Frozen operating-point manifest key set differs from the protocol.")

    rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        if row["source_artifact"] != cfg["operating_point_source"]:
            raise SummaryError("Operating-point source artifact field mismatch.")
        if row["test_selection_forbidden"].strip().lower() != "true":
            raise SummaryError("An operating point permits test-set selection.")
        if "clinical" in row["candidate_only_note"].lower():
            raise SummaryError("Frozen manifest contains clinical-threshold language.")
        model = row["model"]
        target_type = row["target_type"]
        target_key = f"{float(row['target_value']):.2f}"
        config_item = cfg["operating_points"][model][target_type][target_key]
        for field in (
            "threshold_selected_on_id_val",
            "id_val_sensitivity",
            "id_val_specificity",
        ):
            if Decimal(row[field]) != Decimal(str(config_item[field])):
                raise SummaryError(f"Manifest/config mismatch for {model}/{target_type}/{target_key}.")
        rows.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "target_type": target_type,
            "target_value": float(row["target_value"]),
            "threshold_selected_on_id_val": float(row["threshold_selected_on_id_val"]),
            "id_val_sensitivity": float(row["id_val_sensitivity"]),
            "id_val_specificity": float(row["id_val_specificity"]),
            "id_val_FN": int(config_item["id_val_FN"]),
            "id_val_FP": int(config_item["id_val_FP"]),
        })
    return rows


def _apply_frozen_operating_points(
    frozen_rows: list[dict[str, Any]],
    predictions: dict[tuple[str, str], pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frozen in frozen_rows:
        model = frozen["model"]
        raw = predictions[(model, "raw")]
        metrics = compute_binary_classification_metrics(
            raw["label"].to_numpy(dtype=int),
            raw["prob_1"].to_numpy(dtype=float),
            threshold=frozen["threshold_selected_on_id_val"],
        )
        rows.append({
            **frozen,
            "ood_test_sensitivity": metrics["sensitivity"],
            "ood_test_specificity": metrics["specificity"],
            "ood_test_FN": metrics["fn"],
            "ood_test_FP": metrics["fp"],
            "candidate_only_note": (
                "Frozen id_val-selected raw-score candidate; explicitly non-clinical; "
                "no test-set selection or tuning."
            ),
        })
    return rows


def _high_confidence_rows(
    model: str,
    variant: str,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    y = df["label"].to_numpy(dtype=int)
    pred = df["pred_label"].to_numpy(dtype=int)
    confidence = df["confidence"].to_numpy(dtype=float)
    correct = pred == y
    total_fn = int(np.sum((y == 1) & (pred == 0)))
    rows: list[dict[str, Any]] = []
    for threshold in CONFIDENCE_THRESHOLDS:
        high = confidence >= threshold
        n_high = int(np.sum(high))
        high_errors = int(np.sum(high & ~correct))
        high_fn = int(np.sum(high & (y == 1) & (pred == 0)))
        rows.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "variant": variant,
            "confidence_threshold": threshold,
            "n_high_confidence": n_high,
            "high_confidence_errors": high_errors,
            "high_confidence_false_negatives": high_fn,
            "high_confidence_false_negative_rate": high_fn / n_high if n_high else None,
            "total_false_negatives": total_fn,
        })
    return rows


def _reliability_points(
    y: np.ndarray,
    p1: np.ndarray,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    means: list[float] = []
    fractions: list[float] = []
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        mask = (
            (p1 >= low) & (p1 <= high)
            if index == n_bins - 1 else (p1 >= low) & (p1 < high)
        )
        if np.any(mask):
            means.append(float(np.mean(p1[mask])))
            fractions.append(float(np.mean(y[mask])))
    return np.asarray(means), np.asarray(fractions)


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _existing_complete_summary(
    cfg: dict[str, Any],
    inference_sentinel_path: Path,
) -> dict[str, str] | None:
    path = _resolve(cfg["output"]["summary_completed_sentinel"])
    if not path.exists():
        return None
    sentinel = _load_json(path)
    if sentinel.get("summary_status") != "SUMMARY_COMPLETE":
        raise SummaryError("Summary sentinel exists but is not complete.")
    if sentinel.get("inference_sentinel_sha256") != _sha256(inference_sentinel_path):
        raise SummaryError("Summary sentinel references a different inference sentinel.")
    output_map = _summary_output_map(cfg)
    records = sentinel.get("outputs")
    if not isinstance(records, dict) or set(records) != set(output_map):
        raise SummaryError("Summary sentinel output manifest is incomplete.")
    for key, expected_path in output_map.items():
        record = records[key]
        if _resolve(record.get("path", "")) != expected_path:
            raise SummaryError(f"Summary sentinel path mismatch: {key}.")
        if not expected_path.is_file() or expected_path.stat().st_size <= 0:
            raise SummaryError(f"Completed summary output missing/empty: {expected_path}.")
        if _sha256(expected_path) != str(record.get("sha256", "")).upper():
            raise SummaryError(f"Completed summary output hash mismatch: {key}.")
    return {key: str(path) for key, path in output_map.items()}


def summarize(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, str]:
    """Summarize validated exp09 predictions without inference or test-set tuning."""
    config_path = _resolve(config_path)
    cfg = _load_yaml(config_path)
    inference_sentinel_path, inference_sentinel = _load_inference_sentinel(cfg, config_path)
    existing = _existing_complete_summary(cfg, inference_sentinel_path)
    if existing is not None:
        return existing

    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for model in MODEL_ORDER:
        for variant in VARIANTS:
            path = _prediction_path(cfg, model, variant)
            frame = pd.read_csv(path)
            _validate_frame(frame, cfg, model, variant)
            predictions[(model, variant)] = frame
        raw = predictions[(model, "raw")]
        calibrated = predictions[(model, "calibrated")]
        if not np.array_equal(_metadata_matrix(raw), _metadata_matrix(calibrated)):
            raise SummaryError(f"{model}: raw/calibrated metadata differ.")
        if not np.array_equal(
            raw[["logit_0", "logit_1"]].to_numpy(),
            calibrated[["logit_0", "logit_1"]].to_numpy(),
        ):
            raise SummaryError(f"{model}: raw/calibrated logits differ.")
        if not np.array_equal(
            raw["pred_label"].to_numpy(dtype=int),
            calibrated["pred_label"].to_numpy(dtype=int),
        ):
            raise SummaryError(f"{model}: argmax changed after frozen temperature application.")

    reference = _metadata_matrix(predictions[("groupdro", "raw")])
    if not np.array_equal(reference, _metadata_matrix(predictions[("centerstrat_erm", "raw")])):
        raise SummaryError("The two models do not contain aligned test rows.")

    frozen_operating_points = _load_frozen_operating_points(cfg)
    classification_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    metrics_payload: dict[str, Any] = {
        "milestone": "authorized final evaluation prepared by 8I",
        "ood_test_locked_until_authorized": True,
        "authorization_confirmed": True,
        "positive_class": int(cfg["positive_class"]),
        "logical_split": cfg["logical_split"],
        "hf_split": cfg["hf_split"],
        "required_center": int(cfg["required_center"]),
        "expected_rows": int(cfg["expected_rows"]),
        "inference_attempt_counter": int(inference_sentinel["attempt_counter"]),
        "raw_and_calibrated_reporting": True,
        "ranking_score": "logit_1_minus_logit_0 (common monotonic score)",
        "calibration_refit": False,
        "threshold_selection_on_test": False,
        "models": {},
    }
    for model in MODEL_ORDER:
        metrics_payload["models"][model] = {
            "name": MODEL_LABELS[model],
            "checkpoint_sha256": cfg["models"][model]["checkpoint_sha256"],
            "temperature": float(cfg["models"][model]["temperature"]),
            "classification": {},
            "calibration": {},
        }
        ranking_by_variant: dict[str, tuple[float, float]] = {}
        hard_by_variant: dict[str, tuple[Any, ...]] = {}
        for variant in VARIANTS:
            frame = predictions[(model, variant)]
            y = frame["label"].to_numpy(dtype=int)
            p1 = frame["prob_1"].to_numpy(dtype=float)
            auroc, auprc = _ranking_metrics(y, _ranking_score(frame))
            classification = compute_binary_classification_metrics(
                y, p1, threshold=float(cfg["default_threshold"])
            )
            classification["auroc"] = auroc
            classification["auprc"] = auprc
            calibration = compute_calibration_metrics(y, p1, n_bins=N_ECE_BINS)
            ranking_by_variant[variant] = (auroc, auprc)
            hard_by_variant[variant] = tuple(
                classification[key] for key in ("tn", "fp", "fn", "tp")
            )
            classification_rows.append({
                "model": model,
                "model_label": MODEL_LABELS[model],
                "variant": variant,
                "temperature": (
                    1.0 if variant == "raw" else float(cfg["models"][model]["temperature"])
                ),
                **{key: value for key, value in classification.items() if key != "warnings"},
                "warnings": " | ".join(classification["warnings"]),
            })
            calibration_rows.append({
                "model": model,
                "model_key": model,
                "model_label": MODEL_LABELS[model],
                "variant": variant,
                "temperature": (
                    1.0 if variant == "raw" else float(cfg["models"][model]["temperature"])
                ),
                **calibration,
            })
            metrics_payload["models"][model]["classification"][variant] = classification
            metrics_payload["models"][model]["calibration"][variant] = calibration
        raw_rank = ranking_by_variant["raw"]
        calibrated_rank = ranking_by_variant["calibrated"]
        if (
            abs(raw_rank[0] - calibrated_rank[0]) > RANKING_TOLERANCE
            or abs(raw_rank[1] - calibrated_rank[1]) > RANKING_TOLERANCE
        ):
            raise SummaryError(f"{model}: raw/calibrated ranking metrics differ.")
        if hard_by_variant["raw"] != hard_by_variant["calibrated"]:
            raise SummaryError(f"{model}: raw/calibrated hard predictions differ.")

    operating_rows = _apply_frozen_operating_points(
        frozen_operating_points, predictions
    )
    high_confidence_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for variant in VARIANTS:
            high_confidence_rows.extend(
                _high_confidence_rows(model, variant, predictions[(model, variant)])
            )

    output_map = _summary_output_map(cfg)
    _write_json_atomic(output_map["metrics_json"], metrics_payload)
    _write_csv_atomic(output_map["classification_table"], classification_rows)
    _write_csv_atomic(output_map["calibration_table"], calibration_rows)
    _write_csv_atomic(output_map["operating_points_table"], operating_rows)
    _write_csv_atomic(output_map["high_confidence_fn_table"], high_confidence_rows)

    colors = {"groupdro": "tab:red", "centerstrat_erm": "tab:blue"}
    linestyles = {"raw": "-", "calibrated": "--"}

    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for model in MODEL_ORDER:
        for variant in VARIANTS:
            frame = predictions[(model, variant)]
            score = _ranking_score(frame)
            fpr, tpr, _ = roc_curve(frame["label"], score)
            auroc = metrics_payload["models"][model]["classification"][variant]["auroc"]
            axis.plot(
                fpr, tpr,
                color=colors[model],
                linestyle=linestyles[variant],
                label=f"{model} {variant} (AUROC={auroc:.4f})",
            )
    axis.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Final ood_test ROC")
    axis.legend(fontsize=8)
    figure.tight_layout()
    _write_figure_atomic(output_map["roc_figure"], figure)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for model in MODEL_ORDER:
        for variant in VARIANTS:
            frame = predictions[(model, variant)]
            precision, recall, _ = precision_recall_curve(frame["label"], _ranking_score(frame))
            auprc = metrics_payload["models"][model]["classification"][variant]["auprc"]
            axis.plot(
                recall, precision,
                color=colors[model],
                linestyle=linestyles[variant],
                label=f"{model} {variant} (AUPRC={auprc:.4f})",
            )
    axis.set(xlabel="Recall", ylabel="Precision", title="Final ood_test precision-recall")
    axis.legend(fontsize=8)
    figure.tight_layout()
    _write_figure_atomic(output_map["precision_recall_figure"], figure)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, sharey=True)
    for axis, model in zip(axes, MODEL_ORDER):
        for variant in VARIANTS:
            frame = predictions[(model, variant)]
            means, fractions = _reliability_points(
                frame["label"].to_numpy(dtype=int),
                frame["prob_1"].to_numpy(dtype=float),
                n_bins=N_ECE_BINS,
            )
            axis.plot(
                means, fractions, marker="o", markersize=3,
                linestyle=linestyles[variant], label=variant,
            )
        axis.plot([0, 1], [0, 1], color="black", linestyle=":")
        axis.set_title(model)
        axis.set_xlabel("Mean predicted tumor probability")
        axis.legend()
    axes[0].set_ylabel("Observed tumor fraction")
    figure.suptitle("Final ood_test reliability: raw vs frozen-temperature calibrated")
    figure.tight_layout()
    _write_figure_atomic(output_map["reliability_figure"], figure)
    plt.close(figure)

    operating_frame = pd.DataFrame(operating_rows)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for model in MODEL_ORDER:
        subset = operating_frame[
            (operating_frame["model"] == model)
            & (operating_frame["target_type"] == "specificity")
        ]
        axes[0].plot(
            subset["target_value"], subset["ood_test_sensitivity"],
            marker="o", color=colors[model], label=model,
        )
        subset = operating_frame[
            (operating_frame["model"] == model)
            & (operating_frame["target_type"] == "sensitivity")
        ]
        axes[1].plot(
            subset["target_value"], subset["ood_test_specificity"],
            marker="o", color=colors[model], label=model,
        )
    axes[0].set(
        xlabel="Frozen id_val specificity target",
        ylabel="ood_test sensitivity",
        title="Non-clinical fixed-specificity candidates",
    )
    axes[1].set(
        xlabel="Frozen id_val sensitivity target",
        ylabel="ood_test specificity",
        title="Non-clinical fixed-sensitivity candidates",
    )
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    _write_figure_atomic(output_map["operating_point_figure"], figure)
    plt.close(figure)

    high_frame = pd.DataFrame(high_confidence_rows)
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    labels: list[str] = []
    values: list[int] = []
    bar_colors: list[str] = []
    for model in MODEL_ORDER:
        for variant in VARIANTS:
            subset = high_frame[
                (high_frame["model"] == model) & (high_frame["variant"] == variant)
            ]
            for row in subset.itertuples():
                labels.append(f"{model}\n{variant}\n>={row.confidence_threshold:.2f}")
                values.append(int(row.high_confidence_false_negatives))
                bar_colors.append(colors[model])
    axis.bar(np.arange(len(values)), values, color=bar_colors, alpha=0.8)
    axis.set_xticks(np.arange(len(values)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("High-confidence false negatives")
    axis.set_title("Final ood_test high-confidence missed tumors")
    figure.tight_layout()
    _write_figure_atomic(output_map["high_confidence_fn_figure"], figure)
    plt.close(figure)

    report_lines = [
        "FINAL OOD_TEST REPORT",
        "=" * 80,
        f"generated_at={datetime.now(timezone.utc).isoformat()}",
        "authorization_phrase_confirmed=True",
        "inference_attempt_counter=1",
        "test_dataset_instances=1",
        "test_dataloader_traversals=1",
        f"row_count={cfg['expected_rows']}",
        f"center={cfg['required_center']}",
        "development thresholds and temperatures were pre-frozen; nothing fit on ood_test",
        "threshold_selection_on_ood_test=false",
        "calibration_refit_on_ood_test=false",
        "ranking_score=logit_1_minus_logit_0",
        "",
        "CHECKPOINTS",
    ]
    for model in MODEL_ORDER:
        report_lines.append(
            f"{model}: {cfg['models'][model]['checkpoint_path']} "
            f"sha256={cfg['models'][model]['checkpoint_sha256']} "
            f"T={cfg['models'][model]['temperature']}"
        )
    report_lines.extend(["", "THRESHOLD-FREE AND DEFAULT-THRESHOLD COMPARISON"])
    for row in classification_rows:
        report_lines.append(
            f"{row['model']} {row['variant']}: accuracy={_format_metric(row['accuracy'])} "
            f"AUROC={_format_metric(row['auroc'])} AUPRC={_format_metric(row['auprc'])} "
            f"sensitivity={_format_metric(row['sensitivity'])} "
            f"specificity={_format_metric(row['specificity'])} "
            f"tn/fp/fn/tp={row['tn']}/{row['fp']}/{row['fn']}/{row['tp']}"
        )
    report_lines.extend(["", "RAW VS CALIBRATED RELIABILITY"])
    for row in calibration_rows:
        report_lines.append(
            f"{row['model']} {row['variant']}: ECE={_format_metric(row['ece'])} "
            f"Brier={_format_metric(row['brier'])} NLL={_format_metric(row['nll'])}"
        )
    report_lines.extend([
        "",
        "CANDIDATE ID_VAL-SELECTED OPERATING POINTS (EXPLICITLY NON-CLINICAL)",
        "All 14 thresholds were loaded from the frozen 8I manifest without test selection.",
    ])
    for row in operating_rows:
        report_lines.append(
            f"{row['model']} {row['target_type']}={row['target_value']:.2f} "
            f"threshold={row['threshold_selected_on_id_val']:.16g} "
            f"test_sensitivity={_format_metric(row['ood_test_sensitivity'])} "
            f"test_specificity={_format_metric(row['ood_test_specificity'])}"
        )
    report_lines.extend(["", "HIGH-CONFIDENCE FALSE NEGATIVES"])
    for row in high_confidence_rows:
        report_lines.append(
            f"{row['model']} {row['variant']} confidence>={row['confidence_threshold']:.2f}: "
            f"high_conf_FN={row['high_confidence_false_negatives']} "
            f"total_FN={row['total_false_negatives']}"
        )
    report_lines.extend([
        "",
        "No model, checkpoint, temperature, or threshold was selected using ood_test.",
        "No training, weight update, calibration fitting, or post-test selection occurred.",
    ])
    _write_text_atomic(output_map["report"], "\n".join(report_lines) + "\n")

    output_records: dict[str, dict[str, Any]] = {}
    for key, path in output_map.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise SummaryError(f"Required summary output missing/empty: {path}")
        output_records[key] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    summary_sentinel_path = _resolve(cfg["output"]["summary_completed_sentinel"])
    summary_sentinel = {
        "experiment": cfg["experiment"],
        "summary_status": "SUMMARY_COMPLETE",
        "summary_completed_at": datetime.now(timezone.utc).isoformat(),
        "inference_attempt_counter": 1,
        "inference_sentinel_path": str(inference_sentinel_path.relative_to(ROOT)).replace("\\", "/"),
        "inference_sentinel_sha256": _sha256(inference_sentinel_path),
        "prediction_files_validated": True,
        "thresholds_loaded_from_manifest": cfg["frozen_operating_point_manifest"],
        "threshold_selection_on_test": False,
        "calibration_refit": False,
        "outputs": output_records,
    }
    _atomic_create_json(summary_sentinel_path, summary_sentinel)
    return {key: str(path) for key, path in output_map.items()}


def main() -> int:
    try:
        result = summarize()
        print("CSV-only final summary complete or already validated.")
        for key, value in result.items():
            print(f"{key}={value}")
        return 0
    except Exception as exc:
        print(f"SUMMARY REFUSED/FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
