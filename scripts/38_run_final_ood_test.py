"""Guarded, single-traversal executor for the one authorized final ``ood_test`` run.

Without authorization, only ``--preflight-only`` is permitted. Preflight validates
the frozen config, source artifacts, checkpoint hashes, operating-point provenance,
one-shot state, output freshness, and static implementation constraints. It imports
no dataset/model stack and performs no inference.

The later execution mode requires:

    python scripts/38_run_final_ood_test.py --authorization-file <authorized.json>

Authorization and all filesystem/checkpoint guards finish before the durable
``RUN_STARTED`` record is atomically published. That record is published before the
locked loader is imported or the HF test split is accessed. The static config remains
``awaiting_user_authorization``; a valid JSON creates authorization in memory for
that invocation only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import sys
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "exp09_final_ood_test_frozen.yaml"
CHECKPOINT_MANIFEST = ROOT / "results" / "tables" / "exp08i_frozen_checkpoint_manifest.csv"
OPERATING_POINT_MANIFEST = ROOT / "results" / "tables" / "exp08i_frozen_operating_points.csv"
SUMMARY_SCRIPT = ROOT / "scripts" / "39_summarize_final_ood_test.py"

MODEL_ORDER = ("groupdro", "centerstrat_erm")
VARIANTS = ("raw", "calibrated")
META_FIELDS = ("center", "image_id", "patient", "node", "x_coord", "y_coord", "slide")
APPROVAL_FIELDS = (
    "final_model_approved",
    "matched_control_approved",
    "frozen_temperatures_approved",
    "metric_plan_approved",
    "threshold_operating_point_plan_approved",
    "no_retraining_approved",
    "no_post_test_model_selection_approved",
)
BASE_PREDICTION_COLUMNS = (
    "split", "index", "label", *META_FIELDS,
    "logit_0", "logit_1", "prob_0", "prob_1", "pred_label", "correct",
)
REQUIRED_PREDICTION_COLUMNS = (
    "image_id", "center", "patient", "slide", "node", "x_coord", "y_coord",
    "label", "logit_0", "logit_1", "prob_0", "prob_1", "pred_label",
    "confidence", "variant", "temperature",
)


class GuardError(RuntimeError):
    """Raised when a frozen-protocol execution guard fails."""


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise GuardError(f"YAML root must be a mapping: {path}")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _write_temp_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return temp


def _atomic_create_bytes(path: Path, content: bytes) -> None:
    """Atomically publish a fully-written file without replacing an existing path."""
    temp = _write_temp_bytes(path, content)
    try:
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise GuardError(f"Refusing to replace existing one-shot state: {path}") from exc
    finally:
        temp.unlink(missing_ok=True)


def _atomic_replace_bytes(path: Path, content: bytes) -> None:
    """Atomically replace a mutable status file with fully-written content."""
    temp = _write_temp_bytes(path, content)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_create_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_create_bytes(path, _json_bytes(payload))


def _atomic_replace_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_replace_bytes(path, _json_bytes(payload))


def _atomic_create_text(path: Path, text: str) -> None:
    _atomic_create_bytes(path, text.encode("utf-8"))


def _extract_authorization_phrase(template_path: Path) -> str:
    lines = template_path.read_text(encoding="utf-8").splitlines()
    in_section = False
    quote_lines: list[str] = []
    for line in lines:
        if line.strip() == "## Required authorization phrase (paste verbatim to authorize)":
            in_section = True
            continue
        if not in_section:
            continue
        if line.startswith(">"):
            quote_lines.append(line[1:].strip())
        elif quote_lines:
            break
    if not quote_lines:
        raise GuardError("Could not extract the authorization phrase from the unchanged template.")
    phrase = " ".join(quote_lines).strip()
    if phrase.startswith('"') and phrase.endswith('"'):
        phrase = phrase[1:-1]
    return phrase


def _prediction_path(cfg: dict[str, Any], model: str, variant: str) -> Path:
    return _resolve(cfg["output"]["predictions_dir"]) / (
        f"{model}_ood_test_{variant}_predictions.csv"
    )


def _metrics_json_path(cfg: dict[str, Any]) -> Path:
    return _resolve(cfg["output"]["metrics_dir"]) / "final_metrics.json"


def _all_final_output_paths(cfg: dict[str, Any]) -> list[Path]:
    out = cfg["output"]
    paths = [
        _resolve(out["fresh_output_root"]),
        _resolve(out["predictions_dir"]),
        _resolve(out["metrics_dir"]),
        _resolve(out["classification_table"]),
        _resolve(out["calibration_table"]),
        _resolve(out["operating_points_table"]),
        _resolve(out["high_confidence_fn_table"]),
        _resolve(out["roc_figure"]),
        _resolve(out["precision_recall_figure"]),
        _resolve(out["reliability_figure"]),
        _resolve(out["operating_point_figure"]),
        _resolve(out["high_confidence_fn_figure"]),
        _resolve(out["inference_report"]),
        _resolve(out["report"]),
        _resolve(out["run_state"]),
        _resolve(out["completed_run_sentinel"]),
        _resolve(out["summary_completed_sentinel"]),
        _metrics_json_path(cfg),
    ]
    paths.extend(
        _prediction_path(cfg, model, variant)
        for model in MODEL_ORDER
        for variant in VARIANTS
    )
    return paths


def _guard_outputs_fresh(cfg: dict[str, Any]) -> None:
    existing = sorted({str(path) for path in _all_final_output_paths(cfg) if path.exists()})
    if existing:
        raise GuardError(
            "Final output/attempt path already exists; one-shot/no-overwrite guard fired:\n  "
            + "\n  ".join(existing)
        )


def _flatten_config_operating_points(cfg: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        for target_type in ("specificity", "sensitivity"):
            for target_value, item in cfg["operating_points"][model][target_type].items():
                rows.append({
                    "model": model,
                    "target_type": target_type,
                    "target_value": f"{float(target_value):.2f}",
                    "threshold_selected_on_id_val": str(item["threshold_selected_on_id_val"]),
                    "id_val_sensitivity": str(item["id_val_sensitivity"]),
                    "id_val_specificity": str(item["id_val_specificity"]),
                })
    return rows


def _operating_point_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["model"], row["target_type"], f"{float(row['target_value']):.2f}"


def _guard_threshold_provenance(cfg: dict[str, Any]) -> None:
    source_path = _resolve(cfg["operating_point_source"])
    manifest_path = _resolve(cfg["frozen_operating_point_manifest"])
    if manifest_path != OPERATING_POINT_MANIFEST:
        raise GuardError("Frozen operating-point manifest path differs from the 8I path.")
    if _sha256(source_path) != str(cfg["operating_point_source_sha256"]).upper():
        raise GuardError("Accepted 8B operating-point source hash differs from the frozen config.")
    source_rows = _load_csv(source_path)
    manifest_rows = _load_csv(manifest_path)
    config_rows = _flatten_config_operating_points(cfg)
    if len(source_rows) != 14 or len(manifest_rows) != 14 or len(config_rows) != 14:
        raise GuardError(
            f"Expected 14 operating points; source={len(source_rows)}, "
            f"manifest={len(manifest_rows)}, config={len(config_rows)}."
        )
    source_by_key = {_operating_point_key(row): row for row in source_rows}
    manifest_by_key = {_operating_point_key(row): row for row in manifest_rows}
    config_by_key = {_operating_point_key(row): row for row in config_rows}
    if set(source_by_key) != set(manifest_by_key) or set(source_by_key) != set(config_by_key):
        raise GuardError("Operating-point key sets differ between 8B, 8I, and config.")
    for key, source in source_by_key.items():
        for field in (
            "threshold_selected_on_id_val",
            "id_val_sensitivity",
            "id_val_specificity",
        ):
            expected = float(source[field])
            if (
                float(manifest_by_key[key][field]) != expected
                or float(config_by_key[key][field]) != expected
            ):
                raise GuardError(f"Operating-point provenance mismatch for {key}/{field}.")
        row = manifest_by_key[key]
        if row["source_artifact"] != cfg["operating_point_source"]:
            raise GuardError(f"Operating-point source field mismatch for {key}.")
        if row["test_selection_forbidden"].strip().lower() != "true":
            raise GuardError(f"Test-selection prohibition missing for {key}.")
        if "clinical" in row["candidate_only_note"].lower():
            raise GuardError("The frozen manifest must not introduce clinical-threshold language.")


def _guard_checkpoint_manifest(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = {row["model"]: row for row in _load_csv(CHECKPOINT_MANIFEST)}
    if set(manifest) != set(MODEL_ORDER):
        raise GuardError("Checkpoint manifest must contain exactly the proposed pair.")
    verified: dict[str, dict[str, Any]] = {}
    for model in MODEL_ORDER:
        spec = cfg["models"][model]
        path = _resolve(spec["checkpoint_path"])
        if not path.is_file():
            raise GuardError(f"Frozen checkpoint missing: {path}")
        actual_hash = _sha256(path)
        actual_size = path.stat().st_size
        expected_hash = str(spec["checkpoint_sha256"]).upper()
        if actual_hash != expected_hash:
            raise GuardError(f"Checkpoint SHA256 mismatch for {model}.")
        if actual_size != int(spec["file_size_bytes"]):
            raise GuardError(f"Checkpoint file-size mismatch for {model}.")
        row = manifest[model]
        if row["checkpoint_path"] != spec["checkpoint_path"]:
            raise GuardError(f"Checkpoint manifest path mismatch for {model}.")
        if row["sha256"].upper() != expected_hash:
            raise GuardError(f"Checkpoint manifest hash mismatch for {model}.")
        if int(row["file_size_bytes"]) != actual_size:
            raise GuardError(f"Checkpoint manifest size mismatch for {model}.")
        if float(row["temperature"]) != float(spec["temperature"]):
            raise GuardError(f"Checkpoint manifest temperature mismatch for {model}.")
        if row["proposed_for_final_test"].strip().lower() != "true":
            raise GuardError(f"Checkpoint is not marked proposed for final test: {model}.")
        if row["committed_after_authorization"].strip().lower() != "false":
            raise GuardError(f"Checkpoint is marked committed before authorization: {model}.")
        verified[model] = {
            "path": str(path),
            "sha256": actual_hash,
            "file_size_bytes": actual_size,
            "temperature": float(spec["temperature"]),
        }
    return verified


def _guard_source_artifacts(cfg: dict[str, Any]) -> None:
    for path_key, hash_key in (
        ("metric_plan_source", "metric_plan_source_sha256"),
        ("threshold_policy_source", "threshold_policy_source_sha256"),
        ("calibration_policy_source", "calibration_policy_source_sha256"),
    ):
        path = _resolve(cfg[path_key])
        if _sha256(path) != str(cfg[hash_key]).upper():
            raise GuardError(f"Frozen source artifact hash mismatch: {cfg[path_key]}")


def _guard_static_config(cfg: dict[str, Any]) -> None:
    expected = {
        "status": "awaiting_user_authorization",
        "authorization_status": False,
        "dataset": "wltjr1007/Camelyon17-WILDS",
        "logical_split": "ood_test",
        "hf_split": "test",
        "required_center": 2,
        "expected_rows": 85054,
        "positive_class": 1,
        "default_threshold": 0.5,
        "raw_and_calibrated_reporting": True,
        "calibration_refit_allowed": False,
        "threshold_tuning_on_test_allowed": False,
        "training_allowed": False,
        "post_test_model_selection_allowed": False,
        "maximum_test_runs": 1,
        "run_counter": 0,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise GuardError(f"Frozen config mismatch: {key}={cfg.get(key)!r}; expected {value!r}.")
    if [float(x) for x in cfg["fixed_specificity_targets"]] != [0.80, 0.85, 0.90, 0.95]:
        raise GuardError("Fixed-specificity targets differ from the protocol.")
    if [float(x) for x in cfg["fixed_sensitivity_targets"]] != [0.80, 0.90, 0.95]:
        raise GuardError("Fixed-sensitivity targets differ from the protocol.")
    if cfg.get("authorization", {}).get("authorized") is not False:
        raise GuardError("Static config must remain unauthorized.")
    if int(cfg.get("authorization", {}).get("required_run_limit", -1)) != 1:
        raise GuardError("Authorization run limit must be one.")
    expected_models = {
        "groupdro": {
            "checkpoint_path": "results/checkpoints/exp07f_groupdro_resnet18/best.pt",
            "temperature": 2.974907,
            "role": "proposed_primary",
        },
        "centerstrat_erm": {
            "checkpoint_path": "results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt",
            "temperature": 3.496293,
            "role": "proposed_matched_control",
        },
    }
    for model, expected_spec in expected_models.items():
        spec = cfg.get("models", {}).get(model, {})
        for key, value in expected_spec.items():
            if spec.get(key) != value:
                raise GuardError(f"Frozen model mismatch: {model}.{key}.")
    expected_outputs = {
        "fresh_output_root": "results/predictions/exp09_final_ood_test",
        "predictions_dir": "results/predictions/exp09_final_ood_test",
        "metrics_dir": "results/metrics/exp09_final_ood_test",
        "classification_table": "results/tables/exp09_final_ood_test_classification.csv",
        "calibration_table": "results/tables/exp09_final_ood_test_calibration.csv",
        "operating_points_table": "results/tables/exp09_final_ood_test_frozen_operating_points.csv",
        "high_confidence_fn_table": (
            "results/tables/exp09_final_ood_test_high_confidence_false_negatives.csv"
        ),
        "roc_figure": "results/figures/exp09_final_ood_test_roc.png",
        "precision_recall_figure": "results/figures/exp09_final_ood_test_precision_recall.png",
        "reliability_figure": (
            "results/figures/exp09_final_ood_test_calibration_reliability.png"
        ),
        "operating_point_figure": "results/figures/exp09_final_ood_test_operating_points.png",
        "high_confidence_fn_figure": "results/figures/exp09_final_ood_test_high_confidence_fn.png",
        "inference_report": "results/logs/exp09_final_ood_test_inference_report.txt",
        "report": "results/logs/exp09_final_ood_test_report.txt",
        "run_state": "results/logs/exp09_final_ood_test_run_state.json",
        "completed_run_sentinel": "results/logs/exp09_final_ood_test_completed_run.json",
        "summary_completed_sentinel": (
            "results/logs/exp09_final_ood_test_summary_completed.json"
        ),
    }
    for key, value in expected_outputs.items():
        if cfg.get("output", {}).get(key) != value:
            raise GuardError(f"Frozen output path mismatch: output.{key}.")


def _guard_authorization_template(cfg: dict[str, Any]) -> str:
    path = _resolve(cfg["authorization_template"])
    if _sha256(path) != str(cfg["authorization_template_sha256"]).upper():
        raise GuardError("Authorization template changed after the config was frozen.")
    return _extract_authorization_phrase(path)


def _guard_runner_static_source() -> None:
    runner_path = Path(__file__).resolve()
    runner_source = runner_path.read_text(encoding="utf-8")
    summary_source = SUMMARY_SCRIPT.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source, filename=str(runner_path))
    summary_tree = ast.parse(summary_source, filename=str(SUMMARY_SCRIPT))

    forbidden_runner_top_level = {"torch", "datasets", "data", "models", "training", "evaluation"}
    for node in runner_tree.body:
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        for module in modules:
            if module.split(".")[0] in forbidden_runner_top_level:
                raise GuardError(f"Runner imports {module} before authorization at module scope.")

    forbidden_summary_roots = {"torch", "datasets", "data", "models", "training"}
    for node in summary_tree.body:
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        for module in modules:
            if module.split(".")[0] in forbidden_summary_roots:
                raise GuardError(f"CSV-only summary imports forbidden module {module}.")

    for script_path, tree in ((runner_path, runner_tree), (SUMMARY_SCRIPT, summary_tree)):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else ""
            )
            if name in {
                "backward", "step", "save_checkpoint", "fit_temperature",
                "train", "save",
            }:
                raise GuardError(f"Forbidden executable call {name} in {script_path.name}.")

    dataset_calls = [
        node for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Camelyon17HFDataset"
    ]
    if len(dataset_calls) != 1:
        raise GuardError("Runner must instantiate the locked test dataset exactly once.")
    loader_loops = [
        node for node in ast.walk(runner_tree)
        if isinstance(node, (ast.For, ast.AsyncFor))
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "loader"
    ]
    if len(loader_loops) != 1:
        raise GuardError("Runner must contain exactly one test DataLoader traversal.")
    if any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (
                isinstance(node, ast.Import)
                and any(alias.name == "evaluation.predict" for alias in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module == "evaluation.predict"
            )
        )
        for node in ast.walk(runner_tree)
    ):
        raise GuardError("Per-model prediction helper would cause repeated test traversals.")
    if "torch.inference_mode()" not in runner_source:
        raise GuardError("Runner must use torch.inference_mode().")
    if ".eval()" not in runner_source:
        raise GuardError("Runner must put frozen models in eval mode.")
    if "frozen_operating_point_manifest" not in summary_source:
        raise GuardError("Summary must load the frozen 8I operating-point manifest.")
    if "completed_run_sentinel" not in summary_source:
        raise GuardError("Summary must require the completed inference sentinel.")


def _offline_preflight() -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    if not CONFIG_PATH.is_file() or not SUMMARY_SCRIPT.is_file():
        raise GuardError("Frozen config or summary script is missing.")
    cfg = _load_yaml(CONFIG_PATH)
    _guard_static_config(cfg)
    phrase = _guard_authorization_template(cfg)
    _guard_source_artifacts(cfg)
    checkpoints = _guard_checkpoint_manifest(cfg)
    _guard_threshold_provenance(cfg)
    _guard_runner_static_source()
    _guard_outputs_fresh(cfg)
    return cfg, checkpoints, phrase


def _validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GuardError("user_authorized_at must be a non-empty ISO-8601 timestamp.")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise GuardError("user_authorized_at is not a valid ISO-8601 timestamp.") from exc
    return value.strip()


def _validate_authorization_file(path: Path, expected_phrase: str) -> dict[str, Any]:
    if not path.is_file():
        raise GuardError(f"Authorization JSON file missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GuardError("Authorization JSON is unreadable or malformed.") from exc
    if not isinstance(payload, dict):
        raise GuardError("Authorization JSON root must be an object.")
    if payload.get("authorized") is not True:
        raise GuardError("Authorization JSON must contain boolean authorized=true.")
    if payload.get("authorization_phrase") != expected_phrase:
        raise GuardError("Authorization phrase differs from the unchanged template.")
    for field in APPROVAL_FIELDS:
        if payload.get(field) is not True:
            raise GuardError(f"Authorization approval is not boolean true: {field}")
    _validate_timestamp(payload.get("user_authorized_at"))
    if payload.get("protocol") != "8D":
        raise GuardError("Authorization protocol must be exactly 8D.")
    if type(payload.get("run_limit")) is not int or payload.get("run_limit") != 1:
        raise GuardError("Authorization run_limit must be integer 1.")
    return payload


def _validate_prediction_frame(
    df: Any,
    cfg: dict[str, Any],
    name: str,
    *,
    expected_variant: str,
    expected_temperature: float,
) -> None:
    import numpy as np

    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in df.columns]
    if missing:
        raise GuardError(f"{name}: missing prediction columns {missing}.")
    if len(df) != int(cfg["expected_rows"]):
        raise GuardError(f"{name}: row count {len(df)} != {cfg['expected_rows']}.")
    if {int(value) for value in df["center"].unique().tolist()} != {
        int(cfg["required_center"])
    }:
        raise GuardError(f"{name}: center is not exactly {{{cfg['required_center']}}}.")
    if {int(value) for value in df["label"].unique().tolist()} != {0, 1}:
        raise GuardError(f"{name}: labels are not exactly binary {{0,1}}.")
    if set(df["variant"].astype(str).unique().tolist()) != {expected_variant}:
        raise GuardError(f"{name}: variant column mismatch.")
    if not np.all(df["temperature"].to_numpy(dtype=float) == float(expected_temperature)):
        raise GuardError(f"{name}: frozen temperature mismatch.")
    for column in cfg["required_metadata"]:
        if column not in df.columns or df[column].isnull().any():
            raise GuardError(f"{name}: missing/null metadata in {column}.")
    logits = df[["logit_0", "logit_1"]].to_numpy(dtype=float)
    probabilities = df[["prob_0", "prob_1"]].to_numpy(dtype=float)
    confidence = df["confidence"].to_numpy(dtype=float)
    predictions = df["pred_label"].to_numpy(dtype=int)
    if not np.isfinite(logits).all():
        raise GuardError(f"{name}: non-finite logits.")
    if not np.isfinite(probabilities).all() or not np.isfinite(confidence).all():
        raise GuardError(f"{name}: non-finite probability/confidence.")
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise GuardError(f"{name}: probability outside [0,1].")
    if not ((confidence >= 0.0) & (confidence <= 1.0)).all():
        raise GuardError(f"{name}: confidence outside [0,1].")
    if not (np.abs(probabilities.sum(axis=1) - 1.0) <= 1e-5).all():
        raise GuardError(f"{name}: probability sum error.")
    if not np.array_equal(predictions, np.argmax(logits, axis=1)):
        raise GuardError(f"{name}: pred_label differs from logit argmax.")
    if not np.array_equal(predictions, (probabilities[:, 1] >= 0.5).astype(int)):
        raise GuardError(f"{name}: default-threshold prediction differs from argmax.")
    if not np.allclose(confidence, probabilities.max(axis=1), rtol=0.0, atol=1e-12):
        raise GuardError(f"{name}: confidence differs from maximum class probability.")


def _make_variant_frames(
    raw_df: Any,
    temperature: float,
    cfg: dict[str, Any],
) -> tuple[Any, Any]:
    import numpy as np

    raw = raw_df.copy()
    raw["confidence"] = np.maximum(raw["prob_0"], raw["prob_1"])
    raw["variant"] = "raw"
    raw["temperature"] = 1.0

    logits = raw[["logit_0", "logit_1"]].to_numpy(dtype=np.float64)
    scaled = logits / float(temperature)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    calibrated_probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    calibrated = raw_df.copy()
    calibrated["prob_0"] = calibrated_probs[:, 0]
    calibrated["prob_1"] = calibrated_probs[:, 1]
    calibrated["pred_label"] = np.argmax(logits, axis=1).astype(int)
    calibrated["correct"] = (
        calibrated["pred_label"].to_numpy(dtype=int)
        == calibrated["label"].to_numpy(dtype=int)
    ).astype(int)
    calibrated["confidence"] = np.max(calibrated_probs, axis=1)
    calibrated["variant"] = "calibrated"
    calibrated["temperature"] = float(temperature)

    if not np.array_equal(
        raw["pred_label"].to_numpy(dtype=int),
        calibrated["pred_label"].to_numpy(dtype=int),
    ):
        raise GuardError("Frozen temperature application changed argmax predictions.")
    _validate_prediction_frame(
        raw, cfg, "raw predictions", expected_variant="raw", expected_temperature=1.0
    )
    _validate_prediction_frame(
        calibrated,
        cfg,
        "calibrated predictions",
        expected_variant="calibrated",
        expected_temperature=float(temperature),
    )
    return raw, calibrated


def _to_int_list(value: Any, torch_module: Any) -> list[int]:
    if torch_module.is_tensor(value):
        return [int(item) for item in value.detach().cpu().tolist()]
    return [int(item) for item in value]


def _stage_prediction_csv(path: Path, frame: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    return temp


def _publish_prediction_frames(frames: dict[Path, Any]) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for path, frame in frames.items():
            if path.exists():
                raise GuardError(f"FT-7: refusing prediction overwrite: {path}")
            staged.append((_stage_prediction_csv(path, frame), path))
        for _, path in staged:
            if path.exists():
                raise GuardError(f"FT-7: prediction appeared during staging: {path}")
        for temp, path in staged:
            os.replace(temp, path)
    finally:
        for temp, _ in staged:
            temp.unlink(missing_ok=True)


def _load_frozen_models(
    cfg: dict[str, Any],
    checkpoints: dict[str, dict[str, Any]],
    device: Any,
    torch_module: Any,
    classifier_type: Any,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_key in MODEL_ORDER:
        spec = cfg["models"][model_key]
        checkpoint_path = _resolve(spec["checkpoint_path"])
        payload = torch_module.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model_cfg = (payload.get("config") or {}).get("model", {})
        model = classifier_type(
            backbone_name=str(model_cfg.get("backbone_name", "resnet18")),
            # Frozen state_dict fully supplies the weights. Never trigger a network
            # download or initialize from external pretrained weights at final test.
            pretrained=False,
            num_classes=int(model_cfg.get("num_classes", 2)),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        model.eval()
        if any(module.training for module in model.modules()):
            raise GuardError(f"FT-4: {model_key} is not fully in eval mode.")
        if _sha256(checkpoint_path) != checkpoints[model_key]["sha256"]:
            raise GuardError(f"FT-1: checkpoint hash changed while loading {model_key}.")
        models[model_key] = model
    return models


def _predict_both_models_single_traversal(
    models: dict[str, Any],
    loader: Any,
    device: Any,
    torch_module: Any,
) -> dict[str, Any]:
    import pandas as pd

    rows_by_model: dict[str, list[dict[str, Any]]] = {key: [] for key in MODEL_ORDER}
    running_index = 0
    with torch_module.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].detach().cpu()
            metadata = {
                field: _to_int_list(batch[field], torch_module) for field in META_FIELDS
            }
            batch_size = int(labels.shape[0])
            for model_key in MODEL_ORDER:
                logits = models[model_key](images)
                if tuple(logits.shape) != (batch_size, 2):
                    raise GuardError(
                        f"{model_key}: logits shape {tuple(logits.shape)} != {(batch_size, 2)}."
                    )
                probabilities = torch_module.softmax(logits.to(torch_module.float64), dim=1)
                predictions = logits.argmax(dim=1)
                logits_cpu = logits.detach().to(torch_module.float64).cpu()
                probabilities_cpu = probabilities.detach().cpu()
                predictions_cpu = predictions.detach().cpu()
                for index in range(batch_size):
                    label = int(labels[index].item())
                    prediction = int(predictions_cpu[index].item())
                    row: dict[str, Any] = {
                        "split": "ood_test",
                        "index": running_index + index,
                        "label": label,
                    }
                    for field in META_FIELDS:
                        row[field] = metadata[field][index]
                    row.update({
                        "logit_0": float(logits_cpu[index, 0].item()),
                        "logit_1": float(logits_cpu[index, 1].item()),
                        "prob_0": float(probabilities_cpu[index, 0].item()),
                        "prob_1": float(probabilities_cpu[index, 1].item()),
                        "pred_label": prediction,
                        "correct": int(prediction == label),
                    })
                    rows_by_model[model_key].append(row)
            running_index += batch_size
    return {
        key: pd.DataFrame(rows_by_model[key], columns=BASE_PREDICTION_COLUMNS)
        for key in MODEL_ORDER
    }


def _inference_report_text(
    cfg: dict[str, Any],
    checkpoints: dict[str, dict[str, Any]],
    prediction_records: dict[str, dict[str, Any]],
    started_at: str,
    completed_at: str,
) -> str:
    lines = [
        "FINAL OOD_TEST INFERENCE-STAGE REPORT",
        "=" * 80,
        f"started_at={started_at}",
        f"inference_completed_at={completed_at}",
        "authorization_validated=True",
        "attempt_counter=1",
        "test_dataset_instances=1",
        "test_dataloader_traversals=1",
        "models_per_batch=2",
        f"row_count_per_model={cfg['expected_rows']}",
        f"center={cfg['required_center']}",
        "training=False",
        "calibration_fit=False",
        "threshold_selection_on_test=False",
        "summary_completed=False",
        "",
        "CHECKPOINTS",
    ]
    for model in MODEL_ORDER:
        lines.append(
            f"{model}: path={cfg['models'][model]['checkpoint_path']} "
            f"sha256={checkpoints[model]['sha256']} "
            f"temperature={cfg['models'][model]['temperature']}"
        )
    lines.extend(["", "VALIDATED PREDICTIONS"])
    for key, record in prediction_records.items():
        lines.append(
            f"{key}: path={record['path']} rows={record['rows']} sha256={record['sha256']}"
        )
    lines.append("")
    lines.append("The inference-complete sentinel is written only after these files validate.")
    return "\n".join(lines) + "\n"


def _execute_authorized_run(
    cfg: dict[str, Any],
    checkpoints: dict[str, dict[str, Any]],
    authorization: dict[str, Any],
    authorization_path: Path,
) -> int:
    effective_cfg = deepcopy(cfg)
    effective_cfg["status"] = "authorized"
    effective_cfg["authorization_status"] = True
    if effective_cfg["status"] != "authorized" or effective_cfg["authorization_status"] is not True:
        raise GuardError("Effective runtime config is not authorized.")
    if int(effective_cfg["run_counter"]) != 0:
        raise GuardError("Frozen config run counter is not zero.")
    if int(effective_cfg["maximum_test_runs"]) != 1:
        raise GuardError("Maximum test runs is not one.")
    _guard_outputs_fresh(effective_cfg)

    out = effective_cfg["output"]
    run_state_path = _resolve(out["run_state"])
    inference_sentinel_path = _resolve(out["completed_run_sentinel"])
    summary_sentinel_path = _resolve(out["summary_completed_sentinel"])
    started_at = datetime.now(timezone.utc).isoformat()
    run_state = {
        "experiment": effective_cfg["experiment"],
        "status": "RUN_STARTED",
        "attempt_counter": 1,
        "run_counter": 1,
        "maximum_test_runs": 1,
        "started_at": started_at,
        "user_authorized_at": authorization["user_authorized_at"],
        "authorization_file_sha256": _sha256(authorization_path),
        "authorization_validated": True,
        "test_access_started": False,
        "dataset_loaded": False,
        "inference_started": False,
        "inference_completed": False,
        "summary_completed": False,
    }
    _atomic_create_json(run_state_path, run_state)

    try:
        # Authorization, checkpoint, output, counter, and state guards have all
        # passed. RUN_STARTED is durable before these imports and before test access.
        import _bootstrap  # noqa: F401
        import numpy as np
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader

        from data.hf_camelyon17 import hf_id, split_spec
        from data.torch_dataset import Camelyon17HFDataset
        from data.transforms import eval_transform
        from models.classifier import CamelyonClassifier
        from utils.seed import set_seed

        set_seed(1337)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = _load_frozen_models(
            effective_cfg, checkpoints, device, torch, CamelyonClassifier
        )

        _resolve(out["predictions_dir"]).mkdir(parents=True, exist_ok=False)
        _resolve(out["metrics_dir"]).mkdir(parents=True, exist_ok=False)

        loader_spec = split_spec("ood_test")
        if hf_id() != effective_cfg["dataset"]:
            raise GuardError("FT-2: sanctioned loader dataset ID differs from the frozen config.")
        if loader_spec.get("hf_split") != effective_cfg["hf_split"]:
            raise GuardError("FT-2: sanctioned loader does not map ood_test to HF split test.")
        if {int(value) for value in loader_spec.get("center_in", [])} != {
            int(effective_cfg["required_center"])
        }:
            raise GuardError("FT-3: sanctioned loader center filter is not exactly {2}.")

        run_state.update({
            "status": "TEST_ACCESS_STARTING",
            "test_access_started": True,
            "test_access_started_at": datetime.now(timezone.utc).isoformat(),
        })
        _atomic_replace_json(run_state_path, run_state)

        dataset = Camelyon17HFDataset(
            split_name="ood_test",
            transform=eval_transform,
            max_samples=None,
            verbose=True,
        )
        run_state["dataset_loaded"] = True
        run_state["dataset_loaded_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_replace_json(run_state_path, run_state)

        if len(dataset) != int(effective_cfg["expected_rows"]):
            raise GuardError(
                f"FT-2: row count {len(dataset)} != {effective_cfg['expected_rows']}."
            )
        if dataset.spec.get("hf_split") != effective_cfg["hf_split"]:
            raise GuardError("FT-2: logical split does not map to exact HF split test.")
        if {int(value) for value in dataset.spec.get("center_in", [])} != {
            int(effective_cfg["required_center"])
        }:
            raise GuardError("FT-3: logical center filter is not exactly {2}.")
        hf_dataset = getattr(dataset, "_hf_ds", None)
        if hf_dataset is None:
            raise GuardError("FT-2: full test dataset was not materialized as required.")
        required_source_columns = {"label", *effective_cfg["required_metadata"]}
        missing_columns = required_source_columns - set(hf_dataset.column_names)
        if missing_columns:
            raise GuardError(f"FT-8: source columns missing: {sorted(missing_columns)}.")
        if {int(value) for value in hf_dataset.unique("center")} != {
            int(effective_cfg["required_center"])
        }:
            raise GuardError("FT-3: observed center set is not exactly {2}.")
        if {int(value) for value in hf_dataset.unique("label")} != {0, 1}:
            raise GuardError("FT-8: source labels are not exactly binary {0,1}.")
        for column in effective_cfg["required_metadata"]:
            if any(value is None for value in hf_dataset[column]):
                raise GuardError(f"FT-8: source metadata contains nulls in {column}.")

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )
        run_state.update({
            "status": "INFERENCE_STARTED",
            "inference_started": True,
            "inference_started_at": datetime.now(timezone.utc).isoformat(),
            "test_dataset_instances": 1,
            "test_dataloader_traversals": 1,
        })
        _atomic_replace_json(run_state_path, run_state)

        raw_frames = _predict_both_models_single_traversal(models, loader, device, torch)
        if any(len(frame) != int(effective_cfg["expected_rows"]) for frame in raw_frames.values()):
            raise GuardError("FT-2: a model did not produce exactly 85054 rows.")

        variant_frames: dict[tuple[str, str], Any] = {}
        metadata_reference = None
        for model_key in MODEL_ORDER:
            raw, calibrated = _make_variant_frames(
                raw_frames[model_key],
                float(effective_cfg["models"][model_key]["temperature"]),
                effective_cfg,
            )
            metadata_columns = [
                "image_id", "center", "patient", "slide", "node", "x_coord", "y_coord", "label"
            ]
            metadata = raw[metadata_columns].to_numpy()
            if metadata_reference is None:
                metadata_reference = metadata
            elif not np.array_equal(metadata_reference, metadata):
                raise GuardError("The two models do not contain aligned test rows.")
            if not np.array_equal(
                raw[["logit_0", "logit_1"]].to_numpy(),
                calibrated[["logit_0", "logit_1"]].to_numpy(),
            ):
                raise GuardError("Raw/calibrated logits differ.")
            variant_frames[(model_key, "raw")] = raw
            variant_frames[(model_key, "calibrated")] = calibrated

        publish_frames: dict[Path, Any] = {}
        for (model_key, variant), frame in variant_frames.items():
            ordered = list(REQUIRED_PREDICTION_COLUMNS) + [
                column for column in frame.columns
                if column not in REQUIRED_PREDICTION_COLUMNS
            ]
            publish_frames[_prediction_path(effective_cfg, model_key, variant)] = frame[ordered]
        _publish_prediction_frames(publish_frames)

        prediction_records: dict[str, dict[str, Any]] = {}
        for model_key in MODEL_ORDER:
            for variant in VARIANTS:
                path = _prediction_path(effective_cfg, model_key, variant)
                saved = pd.read_csv(path)
                expected_temperature = (
                    1.0 if variant == "raw"
                    else float(effective_cfg["models"][model_key]["temperature"])
                )
                _validate_prediction_frame(
                    saved,
                    effective_cfg,
                    f"saved {model_key}/{variant}",
                    expected_variant=variant,
                    expected_temperature=expected_temperature,
                )
                key = f"{model_key}_{variant}"
                prediction_records[key] = {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "rows": len(saved),
                    "sha256": _sha256(path),
                }

        for model_key in MODEL_ORDER:
            checkpoint_path = _resolve(effective_cfg["models"][model_key]["checkpoint_path"])
            if _sha256(checkpoint_path) != checkpoints[model_key]["sha256"]:
                raise GuardError(f"FT-1: checkpoint hash changed for {model_key}.")

        inference_completed_at = datetime.now(timezone.utc).isoformat()
        inference_report_path = _resolve(out["inference_report"])
        _atomic_create_text(
            inference_report_path,
            _inference_report_text(
                effective_cfg,
                checkpoints,
                prediction_records,
                started_at,
                inference_completed_at,
            ),
        )
        inference_sentinel = {
            "experiment": effective_cfg["experiment"],
            "run_status": "INFERENCE_COMPLETE",
            "attempt_counter": 1,
            "run_counter": 1,
            "maximum_test_runs": 1,
            "started_at": started_at,
            "inference_completed_at": inference_completed_at,
            "user_authorized_at": authorization["user_authorized_at"],
            "authorization_confirmed": True,
            "authorization_file_sha256": _sha256(authorization_path),
            "frozen_config_sha256": _sha256(CONFIG_PATH),
            "frozen_operating_point_manifest_sha256": _sha256(OPERATING_POINT_MANIFEST),
            "row_count": int(effective_cfg["expected_rows"]),
            "required_center": int(effective_cfg["required_center"]),
            "test_dataset_instances": 1,
            "test_dataloader_traversals": 1,
            "models_per_batch": 2,
            "checkpoint_hashes": {
                key: checkpoints[key]["sha256"] for key in MODEL_ORDER
            },
            "temperatures": {
                key: float(effective_cfg["models"][key]["temperature"])
                for key in MODEL_ORDER
            },
            "prediction_files": prediction_records,
            "inference_report": {
                "path": str(inference_report_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256(inference_report_path),
            },
            "calibration_refit": False,
            "threshold_selection_on_test": False,
            "summary_completed": False,
        }
        _atomic_create_json(inference_sentinel_path, inference_sentinel)
        run_state.update({
            "status": "INFERENCE_COMPLETE",
            "inference_completed": True,
            "inference_completed_at": inference_completed_at,
            "completed_run_sentinel": str(inference_sentinel_path),
        })
        _atomic_replace_json(run_state_path, run_state)

        spec = importlib.util.spec_from_file_location("final_test_summary", SUMMARY_SCRIPT)
        if spec is None or spec.loader is None:
            raise GuardError("Could not load the CSV-only final summary stage.")
        summary_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summary_module)
        summary_result = summary_module.summarize(config_path=CONFIG_PATH)

        run_state.update({
            "status": "COMPLETE",
            "summary_completed": True,
            "summary_completed_at": datetime.now(timezone.utc).isoformat(),
            "summary_completed_sentinel": str(summary_sentinel_path),
            "summary_outputs": summary_result,
        })
        _atomic_replace_json(run_state_path, run_state)
        print("AUTHORIZED FINAL INFERENCE AND CSV SUMMARY COMPLETE.", flush=True)
        print(f"Inference sentinel: {inference_sentinel_path}", flush=True)
        print(f"Summary sentinel: {summary_sentinel_path}", flush=True)
        return 0
    except Exception as exc:
        if inference_sentinel_path.exists():
            failure_status = "INFERENCE_COMPLETE_SUMMARY_FAILED"
        elif run_state.get("test_access_started"):
            failure_status = "FINAL_ATTEMPT_FAILED_NO_AUTOMATIC_RERUN"
        else:
            failure_status = "ATTEMPT_FAILED_BEFORE_TEST_ACCESS_NO_AUTOMATIC_RERUN"
        run_state.update({
            "status": failure_status,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        _atomic_replace_json(run_state_path, run_state)
        raise


def _print_preflight(
    cfg: dict[str, Any],
    checkpoints: dict[str, dict[str, Any]],
) -> None:
    print("=" * 80)
    print("Final execution-gate offline preflight: PASS")
    print("=" * 80)
    print(f"config_status={cfg['status']}")
    print(f"authorization_status={cfg['authorization_status']}")
    print("dataset_loaded=False")
    print("ood_test_accessed=False")
    print("inference_run=False")
    print("test_dataset_instances_in_implementation=1")
    print("test_dataloader_traversals_in_implementation=1")
    print("models_evaluated_per_batch=2")
    print("checkpoint_hashes:")
    for model in MODEL_ORDER:
        print(f"  {model}: {checkpoints[model]['sha256']}")
    print("frozen_operating_points=14 (runtime manifest + exact 8B provenance validated)")
    print("authorization_guard=validated_offline")
    print("atomic_RUN_STARTED=implemented")
    print("atomic_prediction_publication=implemented")
    print("inference_sentinel_before_summary=implemented")
    print("summary_resume_without_inference=implemented")
    print("output_paths=fresh")
    print("training_or_calibration_fit_calls=absent")
    print("No final-test execution occurred.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Offline validation only; never imports or loads the dataset.",
    )
    parser.add_argument(
        "--authorization-file",
        type=Path,
        help="Authorized JSON path for the later one-shot final run.",
    )
    args = parser.parse_args(argv)
    if args.preflight_only and args.authorization_file is not None:
        parser.error("Choose exactly one mode: --preflight-only or --authorization-file.")
    if not args.preflight_only and args.authorization_file is None:
        print(
            "REFUSED: execution requires --authorization-file; use --preflight-only offline.",
            file=sys.stderr,
        )
        return 2

    try:
        cfg, checkpoints, expected_phrase = _offline_preflight()
        if args.preflight_only:
            _print_preflight(cfg, checkpoints)
            return 0
        authorization_path = args.authorization_file
        assert authorization_path is not None
        authorization_path = authorization_path.resolve()
        authorization = _validate_authorization_file(authorization_path, expected_phrase)
        return _execute_authorized_run(cfg, checkpoints, authorization, authorization_path)
    except Exception as exc:
        print(f"REFUSED/FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
