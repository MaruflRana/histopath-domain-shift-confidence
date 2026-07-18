"""Milestone 5A / Script 11: OOD-val prediction export + temperature scaling.

FIRST authorized use of ``ood_val`` (HF validation, center == 1). ``ood_test`` stays
LOCKED and is never read. This script:

  1. builds/loads a capped BALANCED ``ood_val`` subset (1000/class = 2000) via the
     existing cache utility,
  2. loads the accepted Milestone 4A checkpoint (weights are NOT modified),
  3. exports RAW per-example predictions,
  4. fits a single temperature scalar on the ``ood_val`` logits (NLL minimization),
  5. applies it and exports CALIBRATED predictions,
  6. computes classification + calibration metrics BEFORE and AFTER calibration.

No training, no uncertainty, no selective prediction. NOT a final result.
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from calibration.temperature_scaling import apply_temperature_to_logits, fit_temperature
from data.balanced_subset import (
    BalancedSubsetDataset,
    default_cache_dir,
    load_or_build_balanced_examples,
)
from data.transforms import eval_transform
from evaluation.metrics_calibration import compute_calibration_metrics
from evaluation.metrics_classification import compute_binary_classification_metrics
from evaluation.predict import OUTPUT_COLUMNS, predict_to_dataframe
from models.classifier import CamelyonClassifier
from training.checkpointing import load_checkpoint
from utils.io import ensure_dir, ensure_results_tree, load_yaml, project_root, write_csv, write_text
from utils.seed import set_seed

CONFIG_REL = "configs/experiments/exp05a_calibrate_ood_val_resnet18.yaml"

_CLS_FIELDS = [
    "n", "accuracy", "balanced_accuracy", "auroc", "auprc",
    "sensitivity", "specificity", "precision", "f1",
    "tn", "fp", "fn", "tp", "threshold", "positive_class",
]
_CAL_FIELDS = ["n", "ece", "brier", "nll", "n_bins"]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def _cache_path(split_name: str, per_class: int, seed: int, shuffle_collected: bool) -> Path:
    order = "shuf" if shuffle_collected else "noshuf"
    return default_cache_dir() / f"balanced_{split_name}_pc{per_class}_seed{seed}_{order}.pt"


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main() -> int:
    root = project_root()
    cfg = load_yaml(root / CONFIG_REL)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    c_cfg = cfg["calibration"]
    o_cfg = cfg["output"]

    seed = int(d_cfg.get("seed", 1337))
    set_seed(seed)
    ensure_results_tree()

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = str(d_cfg["split"])
    per_class = int(d_cfg["per_class"])
    batch_size = int(d_cfg["batch_size"])
    num_workers = int(d_cfg["num_workers"])
    max_scan = d_cfg.get("max_scan")
    progress_every = int(d_cfg.get("progress_every", 5000))
    shuffle_collected = bool(d_cfg.get("shuffle_collected", False))
    use_cache = bool(d_cfg.get("use_cache", True))
    expected_n = 2 * per_class

    min_temp = float(c_cfg["min_temperature"])
    max_temp = float(c_cfg["max_temperature"])

    ckpt_path = _resolve(m_cfg["checkpoint_path"])
    raw_csv = _resolve(o_cfg["raw_predictions_csv"])
    cal_csv = _resolve(o_cfg["calibrated_predictions_csv"])
    metrics_json = _resolve(o_cfg["metrics_json"])
    metrics_csv = _resolve(o_cfg["metrics_csv"])
    temperature_json = _resolve(o_cfg["temperature_json"])
    report_path = _resolve(o_cfg["report_txt"])

    emit("=" * 72)
    emit("Milestone 5A — OOD-val calibration (temperature scaling) (script 11)")
    emit(f"run_name={o_cfg['run_name']}")
    emit(f"device={device}")
    emit(f"checkpoint={ckpt_path}")
    emit(f"split={split}  per_class={per_class} (n={expected_n})  batch_size={batch_size}")
    emit(f"calibration={c_cfg['method']}  max_iter={c_cfg['max_iter']}  lr={c_cfg['lr']}  "
         f"T_init={c_cfg['temperature_init']}  T_bounds=[{min_temp}, {max_temp}]")
    emit("scope: FIRST authorized ood_val use. ood_test is LOCKED and never read.")
    emit("=" * 72)

    ok = True

    # Guard: this milestone must never touch ood_test.
    assert split == "ood_val", f"Milestone 5A operates on ood_val only; got split={split!r}."

    try:
        # --- checkpoint presence + load (weights NOT modified) ------------------ #
        assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

        model = CamelyonClassifier(
            backbone_name=m_cfg["backbone_name"],
            pretrained=False,
            num_classes=int(m_cfg["num_classes"]),
        )
        payload = load_checkpoint(ckpt_path, model, optimizer=None, map_location="cpu")
        model.to(device)
        emit("")
        emit(f"loaded checkpoint: epoch={payload.get('epoch')}")

        # --- balanced ood_val subset (cache-aware) ------------------------------ #
        cache_path = _cache_path(split, per_class, seed, shuffle_collected)
        cache_existed = cache_path.exists()
        emit("")
        emit("--- collecting balanced ood_val subset ---")
        examples = load_or_build_balanced_examples(
            split,
            per_class=per_class,
            max_scan=max_scan,
            progress_every=progress_every,
            seed=seed,
            shuffle_collected=shuffle_collected,
            use_cache=use_cache,
            verbose=True,
        )
        cache_status = "LOADED (cache HIT)" if cache_existed else "CREATED (cache miss -> built + cached)"

        ds = BalancedSubsetDataset(examples, transform=eval_transform)
        counts = ds.label_counts()
        emit("")
        emit(f"ood_val cache : {cache_status}")
        emit(f"             -> {cache_path}")
        emit(f"ood_val label counts : {dict(sorted(counts.items()))}  (n={len(ds)})")

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            drop_last=False, pin_memory=False,
        )

        # --- raw predictions ---------------------------------------------------- #
        emit("")
        emit("--- raw inference (ood_val) ---")
        raw_df = predict_to_dataframe(model, loader, device, split)
        emit(f"raw predictions: n={len(raw_df)}")

        def _prob_checks(df, name) -> None:
            p0 = df["prob_0"].to_numpy(dtype=float)
            p1 = df["prob_1"].to_numpy(dtype=float)
            assert all(math.isfinite(v) for v in p0), f"{name}: non-finite prob_0"
            assert all(math.isfinite(v) for v in p1), f"{name}: non-finite prob_1"
            assert bool(((p0 >= 0) & (p0 <= 1)).all()), f"{name}: prob_0 out of [0,1]"
            assert bool(((p1 >= 0) & (p1 <= 1)).all()), f"{name}: prob_1 out of [0,1]"
            assert bool((abs((p0 + p1) - 1.0) < 1e-5).all()), f"{name}: prob_0+prob_1 != 1"

        _prob_checks(raw_df, "raw")
        ensure_dir(raw_csv.parent)
        raw_df.to_csv(raw_csv, index=False)

        # --- temperature fit on ood_val logits ---------------------------------- #
        emit("")
        emit("--- fitting temperature on ood_val logits ---")
        logits = torch.tensor(
            raw_df[["logit_0", "logit_1"]].to_numpy(), dtype=torch.float32
        )
        labels = torch.tensor(raw_df["label"].to_numpy(), dtype=torch.long)

        fit = fit_temperature(
            logits,
            labels,
            max_iter=int(c_cfg["max_iter"]),
            lr=float(c_cfg["lr"]),
            temperature_init=float(c_cfg["temperature_init"]),
            min_temperature=min_temp,
            max_temperature=max_temp,
            device=device,
        )
        temperature = float(fit["temperature"])
        emit(f"fitted temperature = {temperature:.6f}  "
             f"(NLL {fit['nll_before']:.6f} -> {fit['nll_after']:.6f}, "
             f"iters={fit['n_iter']}, converged={fit['converged']})")

        # --- calibrated predictions (same columns; recomputed from scaled logits) #
        scaled_logits = apply_temperature_to_logits(logits, temperature)
        scaled_probs = F.softmax(scaled_logits, dim=1)
        scaled_pred = scaled_logits.argmax(dim=1)

        cal_df = raw_df.copy()
        cal_df["logit_0"] = scaled_logits[:, 0].tolist()
        cal_df["logit_1"] = scaled_logits[:, 1].tolist()
        cal_df["prob_0"] = scaled_probs[:, 0].tolist()
        cal_df["prob_1"] = scaled_probs[:, 1].tolist()
        cal_df["pred_label"] = [int(v) for v in scaled_pred.tolist()]
        cal_df["correct"] = (cal_df["pred_label"] == cal_df["label"]).astype(int)
        cal_df = cal_df[OUTPUT_COLUMNS]

        _prob_checks(cal_df, "calibrated")
        ensure_dir(cal_csv.parent)
        cal_df.to_csv(cal_csv, index=False)

        # --- metrics: classification (raw vs calibrated) ------------------------ #
        y_true = raw_df["label"].to_numpy()
        raw_cls = compute_binary_classification_metrics(y_true, raw_df["prob_1"].to_numpy(), 0.5)
        cal_cls = compute_binary_classification_metrics(y_true, cal_df["prob_1"].to_numpy(), 0.5)

        # --- metrics: calibration (raw vs calibrated) --------------------------- #
        raw_cal = compute_calibration_metrics(y_true, raw_df["prob_1"].to_numpy(), n_bins=15)
        cal_cal = compute_calibration_metrics(y_true, cal_df["prob_1"].to_numpy(), n_bins=15)

        emit("")
        emit("--- classification metrics (positive = tumor = 1) ---")
        for name, m in (("raw", raw_cls), ("calibrated", cal_cls)):
            emit(f"  [{name}]")
            for k in _CLS_FIELDS:
                emit(f"      {k:<18}: {_fmt(m[k])}")

        emit("")
        emit("--- calibration metrics ---")
        for name, m in (("raw", raw_cal), ("calibrated", cal_cal)):
            emit(f"  [{name}]  ece={_fmt(m['ece'])}  brier={_fmt(m['brier'])}  "
                 f"nll={_fmt(m['nll'])}  (n={m['n']}, n_bins={m['n_bins']})")

        # --- save metrics JSON + CSV + temperature JSON ------------------------- #
        metrics_payload = {
            "run_name": o_cfg["run_name"],
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "split": split,
            "n": int(len(raw_df)),
            "temperature": temperature,
            "threshold": 0.5,
            "positive_class": 1,
            "classification": {"raw": raw_cls, "calibrated": cal_cls},
            "calibration": {"raw": raw_cal, "calibrated": cal_cal},
            "note": "FIRST authorized ood_val use (Milestone 5A). Temperature scaling on "
                    "ood_val; model weights unchanged; ood_test never touched. Development "
                    "result, NOT final performance.",
        }
        ensure_dir(metrics_json.parent)
        write_text(metrics_json, json.dumps(metrics_payload, indent=2) + "\n")

        temp_payload = {
            "run_name": o_cfg["run_name"],
            "method": c_cfg["method"],
            "temperature": temperature,
            "temperature_init": fit["temperature_init"],
            "min_temperature": fit["min_temperature"],
            "max_temperature": fit["max_temperature"],
            "nll_before": fit["nll_before"],
            "nll_after": fit["nll_after"],
            "n_iter": fit["n_iter"],
            "converged": fit["converged"],
            "fit_on": split,
            "n": int(len(raw_df)),
        }
        ensure_dir(temperature_json.parent)
        write_text(temperature_json, json.dumps(temp_payload, indent=2) + "\n")

        # flat CSV: one row per (variant, metric-family) merged into wide columns.
        csv_rows = []
        for variant, cls_m, cal_m in (
            ("raw", raw_cls, raw_cal),
            ("calibrated", cal_cls, cal_cal),
        ):
            row = {"variant": variant, "temperature": temperature}
            for k in _CLS_FIELDS:
                row[k] = cls_m[k]
            for k in ("ece", "brier", "nll", "n_bins"):
                row[k] = cal_m[k]
            csv_rows.append(row)
        header = ["variant", "temperature", *_CLS_FIELDS, "ece", "brier", "nll", "n_bins"]
        write_csv(metrics_csv, csv_rows, header=header)

        # --- assertions --------------------------------------------------------- #
        labels_present = sorted(counts.keys())
        raw_nll = raw_cal["nll"]
        cal_nll = cal_cal["nll"]

        assert len(raw_df) == expected_n, f"ood_val n {len(raw_df)} != {expected_n}"
        assert labels_present == [0, 1], f"ood_val labels {labels_present} != [0, 1]"
        assert raw_csv.exists(), "raw prediction CSV missing"
        assert cal_csv.exists(), "calibrated prediction CSV missing"
        assert metrics_json.exists() and metrics_csv.exists(), "metrics files missing"
        assert temperature_json.exists(), "temperature JSON missing"
        assert math.isfinite(temperature), "temperature is non-finite"
        assert min_temp <= temperature <= max_temp, (
            f"temperature {temperature} outside [{min_temp}, {max_temp}]"
        )
        assert math.isfinite(cal_nll), "calibrated NLL non-finite"
        assert cal_nll <= raw_nll + 1e-6, (
            f"calibrated NLL {cal_nll} > raw NLL {raw_nll} (+tol)"
        )

        # --- summary ------------------------------------------------------------ #
        emit("")
        emit("--- summary ---")
        emit(f"fitted temperature     : {temperature:.6f}")
        emit(f"raw NLL -> cal NLL     : {raw_nll:.6f} -> {cal_nll:.6f}")
        emit(f"raw ECE -> cal ECE     : {raw_cal['ece']:.6f} -> {cal_cal['ece']:.6f}")
        emit(f"raw prediction CSV     : {raw_csv}")
        emit(f"calibrated pred CSV    : {cal_csv}")
        emit(f"metrics JSON           : {metrics_json}")
        emit(f"metrics CSV            : {metrics_csv}")
        emit(f"temperature JSON       : {temperature_json}")
        emit(f"text report            : {report_path}")

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
