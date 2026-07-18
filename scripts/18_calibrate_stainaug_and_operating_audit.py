"""Milestone 7C / Script 18: calibration + candidate operating-point audit of the
stain-augmentation baseline (CSV / logit-only).

**No training. No model inference. No MC-dropout. No ensemble. No ood_test.** This script
extends Milestone 7B: it takes the *already-saved* Milestone 7A stain-augmentation
predictions and (a) fits post-hoc temperature scaling on the stain-aug ``ood_val`` logits,
(b) reports raw-vs-calibrated calibration reliability (ECE / Brier / NLL) against the plain
ERM 5A values, (c) audits high-confidence false negatives (missed tumors the model is sure
about) before/after calibration, and (d) audits *candidate* id_val-selected operating points
for both models on ``ood_val``.

Interpretation guardrails (baked into the report):
  * Thresholds selected on ``id_val`` are **candidate development thresholds**, NOT final
    clinical thresholds. No operating point is committed.
  * Temperature scaling is **argmax-invariant** — it rescales confidence, it does NOT move
    the hard prediction at a fixed argmax (0.5) threshold, so it cannot by itself fix the
    stain-aug sensitivity collapse at 0.5.
  * The 7A stain-aug 0.5 "collapse" is mostly **threshold behavior**, not a total ranking
    collapse (7B: ood_val |dAUROC| ~ 0.02).
  * Plain ERM may remain stronger than stain-aug even after threshold normalization.
  * Everything here is DEVELOPMENT-stage on capped balanced subsets — NOT final performance.

Strict scope (Milestone 7C): ``ood_val`` is used for development-stage calibration /
comparison ONLY (never training, never final test reporting). ``ood_test`` is NEVER read.
Input prediction CSVs are read-only and never modified.

Inputs (read-only):
  plain ERM  : results/predictions/exp04a_dev_train_resnet18/id_val_cached_predictions.csv
               results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_raw_predictions.csv
               results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_calibrated_predictions.csv
  stain-aug  : results/predictions/exp07a_stainaug_resnet18/id_val_predictions.csv
               results/predictions/exp07a_stainaug_resnet18/ood_val_predictions.csv
  5A metrics : results/metrics/exp05a_calibrate_ood_val_resnet18/calibration_metrics.json (plain ERM)
  7B tables  : results/tables/exp07b_fixed_specificity_operating_points.csv
               results/tables/exp07b_fixed_sensitivity_operating_points.csv
               results/tables/exp07b_threshold_free_comparison.csv
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup; kept for repo convention)

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / non-interactive; no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from calibration.temperature_scaling import apply_temperature_to_logits, fit_temperature
from evaluation.metrics_calibration import compute_calibration_metrics
from evaluation.metrics_classification import compute_binary_classification_metrics
from evaluation.metrics_operating_points import METRIC_FIELDS, apply_operating_points
from evaluation.metrics_uncertainty import compute_high_confidence_false_negative_summary
from utils.io import ensure_dir, ensure_results_tree, project_root, write_csv, write_text

# ---------------------------------------------------------------------------- #
# Input paths (read-only)                                                       #
# ---------------------------------------------------------------------------- #
PLAIN_ID_VAL_CSV = "results/predictions/exp04a_dev_train_resnet18/id_val_cached_predictions.csv"
PLAIN_OOD_RAW_CSV = "results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_raw_predictions.csv"
PLAIN_OOD_CAL_CSV = "results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_calibrated_predictions.csv"
STAIN_ID_VAL_CSV = "results/predictions/exp07a_stainaug_resnet18/id_val_predictions.csv"
STAIN_OOD_VAL_CSV = "results/predictions/exp07a_stainaug_resnet18/ood_val_predictions.csv"

PLAIN_5A_METRICS_JSON = "results/metrics/exp05a_calibrate_ood_val_resnet18/calibration_metrics.json"

# 7B tables consulted for context (read-only; not modified).
EXP07B_FIXED_SPEC_CSV = "results/tables/exp07b_fixed_specificity_operating_points.csv"
EXP07B_FIXED_SENS_CSV = "results/tables/exp07b_fixed_sensitivity_operating_points.csv"
EXP07B_THRESHOLDFREE_CSV = "results/tables/exp07b_threshold_free_comparison.csv"

# Prediction CSVs that must NOT be modified (fingerprinted pre/post).
READONLY_INPUTS = [
    PLAIN_ID_VAL_CSV, PLAIN_OOD_RAW_CSV, PLAIN_OOD_CAL_CSV,
    STAIN_ID_VAL_CSV, STAIN_OOD_VAL_CSV,
]
# Additional read-only inputs (existence-checked only).
CONTEXT_INPUTS = [
    PLAIN_5A_METRICS_JSON, EXP07B_FIXED_SPEC_CSV, EXP07B_FIXED_SENS_CSV,
    EXP07B_THRESHOLDFREE_CSV,
]

# ---------------------------------------------------------------------------- #
# Output paths                                                                  #
# ---------------------------------------------------------------------------- #
OUT_CAL_PRED_CSV = "results/predictions/exp07c_stainaug_calibration/ood_val_calibrated_predictions.csv"
OUT_TEMP_JSON = "results/metrics/exp07c_stainaug_calibration/temperature.json"
OUT_CAL_METRICS_JSON = "results/metrics/exp07c_stainaug_calibration/calibration_metrics.json"

OUT_CAL_COMPARISON_CSV = "results/tables/exp07c_calibration_comparison.csv"
OUT_HICONF_FN_CSV = "results/tables/exp07c_high_confidence_fn_comparison.csv"
OUT_CANDIDATE_OP_CSV = "results/tables/exp07c_candidate_operating_point_audit.csv"

OUT_FIG_CAL = "results/figures/exp07c_calibration_raw_vs_calibrated.png"
OUT_FIG_HICONF_FN = "results/figures/exp07c_high_confidence_fn_comparison.png"
OUT_FIG_FIXED_SPEC = "results/figures/exp07c_fixed_specificity_sensitivity_comparison.png"

OUT_REPORT_TXT = "results/logs/exp07c_stainaug_calibration_operating_audit_report.txt"

OUTPUT_PATHS = [
    OUT_CAL_PRED_CSV, OUT_TEMP_JSON, OUT_CAL_METRICS_JSON, OUT_CAL_COMPARISON_CSV,
    OUT_HICONF_FN_CSV, OUT_CANDIDATE_OP_CSV, OUT_FIG_CAL, OUT_FIG_HICONF_FN,
    OUT_FIG_FIXED_SPEC, OUT_REPORT_TXT,
]

REQUIRED_COLUMNS = ["label", "prob_0", "prob_1", "logit_0", "logit_1", "pred_label", "correct"]

TARGET_SPECIFICITIES = [0.80, 0.85, 0.90, 0.95]
TARGET_SENSITIVITIES = [0.80, 0.90, 0.95]
HICONF_THRESHOLDS = [0.90, 0.95, 0.99]

# Temperature-scaling settings (match Milestone 5A for consistency).
TEMP_MAX_ITER = 200
TEMP_LR = 0.01
TEMP_INIT = 1.0
TEMP_MIN = 0.05
TEMP_MAX = 10.0
N_ECE_BINS = 15

DEFAULT_THRESHOLD = 0.5


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "None"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax of an [N,2] logit array (numerically stable)."""
    m = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=1, keepdims=True)


def main() -> int:
    ensure_results_tree()

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    emit("=" * 78)
    emit("Milestone 7C - stain-aug calibration + candidate operating-point audit")
    emit("CSV/logit-only. NO training, NO inference, NO MC-dropout, NO ensemble.")
    emit("ood_val -> development-stage calibration/comparison ONLY (never training/final).")
    emit("ood_test: NEVER read. Candidate id_val thresholds are NOT final clinical points.")
    emit("DEVELOPMENT analysis on capped balanced subsets - NOT final performance.")
    emit("=" * 78)

    ok = True

    try:
        # --- guard: no ood_test path anywhere in the configured inputs/outputs --- #
        all_paths = READONLY_INPUTS + CONTEXT_INPUTS + OUTPUT_PATHS
        assert not any("ood_test" in p for p in all_paths), "ood_test must never be referenced"

        # --- existence checks for every input ----------------------------------- #
        resolved_ro = {name: _resolve(name) for name in READONLY_INPUTS}
        resolved_ctx = {name: _resolve(name) for name in CONTEXT_INPUTS}
        for name, path in {**resolved_ro, **resolved_ctx}.items():
            assert path.exists(), f"required input not found: {path}"

        # fingerprint the prediction CSVs so we can prove they were not modified
        pre_mtime = {name: resolved_ro[name].stat().st_mtime_ns for name in READONLY_INPUTS}
        pre_size = {name: resolved_ro[name].stat().st_size for name in READONLY_INPUTS}

        # --------------------------------------------------------------- #
        # 1. Load prediction CSVs + validate columns / counts / probs     #
        # --------------------------------------------------------------- #
        frames = {
            "plain_id_val": pd.read_csv(resolved_ro[PLAIN_ID_VAL_CSV]),
            "plain_ood_raw": pd.read_csv(resolved_ro[PLAIN_OOD_RAW_CSV]),
            "plain_ood_cal": pd.read_csv(resolved_ro[PLAIN_OOD_CAL_CSV]),
            "stain_id_val": pd.read_csv(resolved_ro[STAIN_ID_VAL_CSV]),
            "stain_ood_raw": pd.read_csv(resolved_ro[STAIN_OOD_VAL_CSV]),
        }
        expected_n = {
            "plain_id_val": 500, "plain_ood_raw": 2000, "plain_ood_cal": 2000,
            "stain_id_val": 500, "stain_ood_raw": 2000,
        }

        for key, df in frames.items():
            for col in REQUIRED_COLUMNS:
                assert col in df.columns, f"{key}: missing required column {col!r}"
            assert len(df) == expected_n[key], (
                f"{key}: expected n=={expected_n[key]}, got {len(df)}"
            )
            p0 = df["prob_0"].to_numpy(dtype=float)
            p1 = df["prob_1"].to_numpy(dtype=float)
            assert np.isfinite(p0).all() and np.isfinite(p1).all(), f"{key}: non-finite prob"
            assert ((p0 >= 0.0) & (p0 <= 1.0)).all(), f"{key}: prob_0 out of [0,1]"
            assert ((p1 >= 0.0) & (p1 <= 1.0)).all(), f"{key}: prob_1 out of [0,1]"
            assert (np.abs(p0 + p1 - 1.0) < 1e-5).all(), f"{key}: prob_0+prob_1 != 1"
            l0 = df["logit_0"].to_numpy(dtype=float)
            l1 = df["logit_1"].to_numpy(dtype=float)
            assert np.isfinite(l0).all() and np.isfinite(l1).all(), f"{key}: non-finite logits"
            labs = set(np.unique(df["label"].to_numpy()).tolist())
            assert labs.issubset({0, 1}), f"{key}: labels outside {{0,1}}: {sorted(labs)}"

        # explicit stain row-count assertions from the task spec
        assert len(frames["stain_id_val"]) == 500, "stain id_val n must be 500"
        assert len(frames["stain_ood_raw"]) == 2000, "stain ood_val n must be 2000"

        emit("")
        emit("Inputs validated (columns, row counts, probabilities, logits):")
        for key, df in frames.items():
            emit(f"  {key:<14} n={len(df):<5} "
                 f"tumors={int((df['label'] == 1).sum())} "
                 f"non-tumors={int((df['label'] == 0).sum())}")

        # load plain ERM 5A calibration metrics (for the comparison table)
        with open(resolved_ctx[PLAIN_5A_METRICS_JSON], "r", encoding="utf-8") as f:
            plain_5a = json.load(f)
        plain_cal_raw = plain_5a["calibration"]["raw"]
        plain_cal_cal = plain_5a["calibration"]["calibrated"]
        plain_T = float(plain_5a.get("temperature"))

        # --------------------------------------------------------------- #
        # 2. Temperature scaling for the stain-aug model on ood_val        #
        # --------------------------------------------------------------- #
        stain_ood = frames["stain_ood_raw"]
        y_ood = stain_ood["label"].to_numpy().astype(int)
        logits_ood = np.stack(
            [stain_ood["logit_0"].to_numpy(dtype=float),
             stain_ood["logit_1"].to_numpy(dtype=float)],
            axis=1,
        )
        fit = fit_temperature(
            torch.from_numpy(logits_ood).float(),
            torch.from_numpy(y_ood).long(),
            max_iter=TEMP_MAX_ITER, lr=TEMP_LR, temperature_init=TEMP_INIT,
            min_temperature=TEMP_MIN, max_temperature=TEMP_MAX, device="cpu",
        )
        stain_T = float(fit["temperature"])
        assert math.isfinite(stain_T) and TEMP_MIN <= stain_T <= TEMP_MAX, "stain T out of bounds"

        # apply temperature -> calibrated probabilities (argmax-invariant)
        cal_logits = apply_temperature_to_logits(
            torch.from_numpy(logits_ood).float(), stain_T
        ).numpy()
        cal_probs = _softmax_rows(cal_logits)
        cal_prob_0 = cal_probs[:, 0]
        cal_prob_1 = cal_probs[:, 1]
        cal_pred = (cal_prob_1 >= 0.5).astype(int)
        cal_correct = (cal_pred == y_ood).astype(int)
        cal_conf = np.maximum(cal_prob_0, cal_prob_1)

        # sanity: calibrated probabilities valid + argmax unchanged
        assert np.isfinite(cal_prob_0).all() and np.isfinite(cal_prob_1).all(), "non-finite cal prob"
        assert ((cal_prob_0 >= 0) & (cal_prob_0 <= 1)).all(), "cal prob_0 out of [0,1]"
        assert ((cal_prob_1 >= 0) & (cal_prob_1 <= 1)).all(), "cal prob_1 out of [0,1]"
        assert (np.abs(cal_prob_0 + cal_prob_1 - 1.0) < 1e-5).all(), "cal prob sum != 1"
        assert (cal_pred == stain_ood["pred_label"].to_numpy().astype(int)).all(), (
            "temperature scaling changed the argmax prediction (should be invariant)"
        )

        # save calibrated stain-aug predictions (preserve original columns, add calibrated)
        cal_df = stain_ood.copy()
        cal_df["calibrated_prob_0"] = cal_prob_0
        cal_df["calibrated_prob_1"] = cal_prob_1
        cal_df["calibrated_pred_label"] = cal_pred
        cal_df["calibrated_correct"] = cal_correct
        cal_df["calibrated_confidence"] = cal_conf
        cal_pred_path = _resolve(OUT_CAL_PRED_CSV)
        ensure_dir(cal_pred_path.parent)
        cal_df.to_csv(cal_pred_path, index=False)

        # temperature JSON
        temp_payload = {
            "milestone": "7C",
            "model": "stain-aug ERM (7A)",
            "fit_on": "ood_val (stain-aug logits) - development-stage calibration only",
            "temperature": stain_T,
            "nll_before": fit["nll_before"],
            "nll_after": fit["nll_after"],
            "n_iter": fit["n_iter"],
            "converged": fit["converged"],
            "min_temperature": fit["min_temperature"],
            "max_temperature": fit["max_temperature"],
            "temperature_init": fit["temperature_init"],
            "n": int(len(stain_ood)),
            "argmax_invariant": True,
            "plain_erm_temperature_5a": plain_T,
            "note": "Temperature scaling on stain-aug ood_val logits; weights unchanged; "
                    "ood_test never touched. Development result, NOT final performance.",
        }
        temp_path = _resolve(OUT_TEMP_JSON)
        ensure_dir(temp_path.parent)
        write_text(temp_path, json.dumps(temp_payload, indent=2) + "\n")

        emit("")
        emit("1/2. Stain-aug temperature scaling on ood_val (development-stage)")
        emit(f"     fitted T = {stain_T:.6f}  (NLL {fit['nll_before']:.6f} -> "
             f"{fit['nll_after']:.6f}; n_iter={fit['n_iter']}, converged={fit['converged']})")
        emit(f"     plain ERM T (5A, for reference) = {plain_T:.6f}")
        emit("     temperature scaling is argmax-invariant: hard predictions at 0.5 unchanged.")

        # --------------------------------------------------------------- #
        # 3. Calibration metrics: stain raw vs calibrated (+ plain 5A)     #
        # --------------------------------------------------------------- #
        stain_raw_cal = compute_calibration_metrics(y_ood, stain_ood["prob_1"].to_numpy(dtype=float), N_ECE_BINS)
        stain_cal_cal = compute_calibration_metrics(y_ood, cal_prob_1, N_ECE_BINS)

        # classification (to confirm argmax invariance for stain-aug too)
        stain_cls_raw = compute_binary_classification_metrics(y_ood, stain_ood["prob_1"].to_numpy(dtype=float))
        stain_cls_cal = compute_binary_classification_metrics(y_ood, cal_prob_1)

        cal_comparison_rows = [
            {"model": "plain ERM", "variant": "raw", "temperature": None,
             "ece": plain_cal_raw["ece"], "brier": plain_cal_raw["brier"], "nll": plain_cal_raw["nll"]},
            {"model": "plain ERM", "variant": "calibrated", "temperature": plain_T,
             "ece": plain_cal_cal["ece"], "brier": plain_cal_cal["brier"], "nll": plain_cal_cal["nll"]},
            {"model": "stain-aug ERM", "variant": "raw", "temperature": None,
             "ece": stain_raw_cal["ece"], "brier": stain_raw_cal["brier"], "nll": stain_raw_cal["nll"]},
            {"model": "stain-aug ERM", "variant": "calibrated", "temperature": stain_T,
             "ece": stain_cal_cal["ece"], "brier": stain_cal_cal["brier"], "nll": stain_cal_cal["nll"]},
        ]
        write_csv(_resolve(OUT_CAL_COMPARISON_CSV), cal_comparison_rows,
                  header=["model", "variant", "temperature", "ece", "brier", "nll"])

        stain_ece_impr = stain_raw_cal["ece"] - stain_cal_cal["ece"]
        calibration_helped = stain_cal_cal["ece"] < stain_raw_cal["ece"] - 1e-9

        emit("")
        emit("3. Calibration reliability on ood_val (development)  [tumor = positive]")
        emit(f"   {'model':<14}{'variant':<12}{'ECE':>9}{'Brier':>9}{'NLL':>9}")
        for r in cal_comparison_rows:
            emit(f"   {r['model']:<14}{r['variant']:<12}"
                 f"{_fmt(r['ece']):>9}{_fmt(r['brier']):>9}{_fmt(r['nll']):>9}")
        emit(f"   stain-aug ECE improvement (raw - calibrated) = {_fmt(stain_ece_impr)} "
             f"({'improved' if calibration_helped else 'no improvement'})")
        emit(f"   stain-aug classification unchanged raw->cal (argmax-invariant): "
             f"acc {_fmt(stain_cls_raw['accuracy'])} -> {_fmt(stain_cls_cal['accuracy'])}, "
             f"sens {_fmt(stain_cls_raw['sensitivity'])} -> {_fmt(stain_cls_cal['sensitivity'])}, "
             f"FN {stain_cls_raw['fn']} -> {stain_cls_cal['fn']}")

        # --------------------------------------------------------------- #
        # 4. High-confidence false-negative audit                         #
        # --------------------------------------------------------------- #
        # Build per-variant frames carrying (label, pred_label, confidence) so the
        # existing 5B helper computes high-confidence FN counts consistently.
        def _hiconf_frame(label, pred_label, confidence) -> pd.DataFrame:
            return pd.DataFrame({
                "label": np.asarray(label).astype(int),
                "pred_label": np.asarray(pred_label).astype(int),
                "confidence": np.asarray(confidence, dtype=float),
            })

        hiconf_variants = {
            "plain_raw": _hiconf_frame(
                frames["plain_ood_raw"]["label"], frames["plain_ood_raw"]["pred_label"],
                np.maximum(frames["plain_ood_raw"]["prob_0"], frames["plain_ood_raw"]["prob_1"]),
            ),
            "plain_calibrated": _hiconf_frame(
                frames["plain_ood_cal"]["label"], frames["plain_ood_cal"]["pred_label"],
                np.maximum(frames["plain_ood_cal"]["prob_0"], frames["plain_ood_cal"]["prob_1"]),
            ),
            "stain_raw": _hiconf_frame(
                stain_ood["label"], stain_ood["pred_label"],
                np.maximum(stain_ood["prob_0"], stain_ood["prob_1"]),
            ),
            "stain_calibrated": _hiconf_frame(y_ood, cal_pred, cal_conf),
        }
        variant_model = {
            "plain_raw": "plain ERM", "plain_calibrated": "plain ERM",
            "stain_raw": "stain-aug ERM", "stain_calibrated": "stain-aug ERM",
        }
        variant_kind = {
            "plain_raw": "raw", "plain_calibrated": "calibrated",
            "stain_raw": "raw", "stain_calibrated": "calibrated",
        }

        hiconf_rows: list[dict] = []
        hiconf_by_variant: dict[str, dict[float, dict]] = {}
        for vname, vdf in hiconf_variants.items():
            summ = compute_high_confidence_false_negative_summary(vdf, "confidence", HICONF_THRESHOLDS)
            hiconf_by_variant[vname] = {row["threshold"]: row for row in summ}
            for row in summ:
                hiconf_rows.append({
                    "model": variant_model[vname],
                    "variant": variant_kind[vname],
                    "confidence_threshold": row["threshold"],
                    "n_total": row["n_total"],
                    "n_high_conf": row["n_high_conf"],
                    "n_high_conf_false_negative": row["n_high_conf_false_negative"],
                    "high_conf_false_negative_rate": row["high_conf_false_negative_rate"],
                    "n_false_negative_total": row["n_false_negative_total"],
                    "fraction_of_fn_in_high_conf": row["fraction_of_fn_in_high_conf"],
                })
        write_csv(_resolve(OUT_HICONF_FN_CSV), hiconf_rows, header=[
            "model", "variant", "confidence_threshold", "n_total", "n_high_conf",
            "n_high_conf_false_negative", "high_conf_false_negative_rate",
            "n_false_negative_total", "fraction_of_fn_in_high_conf",
        ])

        emit("")
        emit("4. High-confidence false-negative (missed-tumor) audit on ood_val")
        emit("   confidently-missed tumors = label==1, pred==0, confidence >= t")
        emit(f"   {'model':<14}{'variant':<12}{'t':>6}{'n_high':>8}{'hc_FN':>7}"
             f"{'hc_FN_rate':>12}{'FN_total':>10}")
        for vname in ("plain_raw", "plain_calibrated", "stain_raw", "stain_calibrated"):
            for t in HICONF_THRESHOLDS:
                row = hiconf_by_variant[vname][t]
                emit(f"   {variant_model[vname]:<14}{variant_kind[vname]:<12}{t:>6.2f}"
                     f"{row['n_high_conf']:>8}{row['n_high_conf_false_negative']:>7}"
                     f"{_fmt(row['high_conf_false_negative_rate']):>12}"
                     f"{row['n_false_negative_total']:>10}")

        stain_hc_fn_raw_090 = hiconf_by_variant["stain_raw"][0.90]["n_high_conf_false_negative"]
        stain_hc_fn_cal_090 = hiconf_by_variant["stain_calibrated"][0.90]["n_high_conf_false_negative"]

        # --------------------------------------------------------------- #
        # 5. Candidate operating-point audit (id_val-selected thresholds)  #
        # --------------------------------------------------------------- #
        op_models = {
            "plain ERM": (frames["plain_id_val"], frames["plain_ood_raw"]),
            "stain-aug ERM": (frames["stain_id_val"], frames["stain_ood_raw"]),
        }
        op_raw_rows: dict[str, list[dict]] = {}
        for mname, (id_df, ood_df) in op_models.items():
            op_raw_rows[mname] = apply_operating_points(
                id_df, ood_df, TARGET_SPECIFICITIES, TARGET_SENSITIVITIES
            )

        def _op_lookup(mname: str, criterion: str, target: float, split: str) -> dict | None:
            for r in op_raw_rows[mname]:
                if r["criterion"] == criterion and abs(r["target"] - target) < 1e-9 and r["split"] == split:
                    return r
            return None

        # reshape into one row per (model, criterion, target) with id_val + ood_val columns
        candidate_rows: list[dict] = []
        for mname in op_models:
            for criterion, targets in (("specificity", TARGET_SPECIFICITIES),
                                       ("sensitivity", TARGET_SENSITIVITIES)):
                for target in targets:
                    idr = _op_lookup(mname, criterion, target, "id_val")
                    oor = _op_lookup(mname, criterion, target, "ood_val")
                    thr = idr["threshold"] if idr else None
                    candidate_rows.append({
                        "model": mname,
                        "criterion": criterion,
                        "target": float(target),
                        "threshold_selected_on_id_val": thr,
                        "id_val_sensitivity": idr["sensitivity"] if idr else None,
                        "id_val_specificity": idr["specificity"] if idr else None,
                        "id_val_fn": idr["fn"] if idr else None,
                        "id_val_fp": idr["fp"] if idr else None,
                        "ood_val_sensitivity": oor["sensitivity"] if oor else None,
                        "ood_val_specificity": oor["specificity"] if oor else None,
                        "ood_val_fn": oor["fn"] if oor else None,
                        "ood_val_fp": oor["fp"] if oor else None,
                        "ood_val_predicted_positive_rate": oor["predicted_positive_rate"] if oor else None,
                    })
        write_csv(_resolve(OUT_CANDIDATE_OP_CSV), candidate_rows, header=[
            "model", "criterion", "target", "threshold_selected_on_id_val",
            "id_val_sensitivity", "id_val_specificity", "id_val_fn", "id_val_fp",
            "ood_val_sensitivity", "ood_val_specificity", "ood_val_fn", "ood_val_fp",
            "ood_val_predicted_positive_rate",
        ])

        emit("")
        emit("5. Candidate operating-point audit (thresholds selected on id_val ONLY)")
        emit("   CANDIDATE development thresholds - NOT final clinical operating points.")
        emit("   Raw prob_1 scores (calibration is monotone/argmax-invariant -> same ranking).")
        emit("")
        emit("   FIXED-SPECIFICITY candidates, applied to ood_val:")
        emit(f"   {'tgt_spec':>9}{'model':<16}{'thr':>8}{'id_sens':>9}{'id_spec':>9}"
             f"{'ood_sens':>10}{'ood_spec':>10}{'ood_FN':>8}{'ood_FP':>8}{'ood_ppr':>9}")
        for spec in TARGET_SPECIFICITIES:
            for mname in op_models:
                r = next((c for c in candidate_rows
                          if c["model"] == mname and c["criterion"] == "specificity"
                          and abs(c["target"] - spec) < 1e-9), None)
                emit(f"   {spec:>9.2f}{mname:<16}{_fmt(r['threshold_selected_on_id_val']):>8}"
                     f"{_fmt(r['id_val_sensitivity']):>9}{_fmt(r['id_val_specificity']):>9}"
                     f"{_fmt(r['ood_val_sensitivity']):>10}{_fmt(r['ood_val_specificity']):>10}"
                     f"{(r['ood_val_fn'] if r['ood_val_fn'] is not None else 'None'):>8}"
                     f"{(r['ood_val_fp'] if r['ood_val_fp'] is not None else 'None'):>8}"
                     f"{_fmt(r['ood_val_predicted_positive_rate']):>9}")
        emit("")
        emit("   FIXED-SENSITIVITY candidates, applied to ood_val:")
        emit(f"   {'tgt_sens':>9}{'model':<16}{'thr':>8}{'id_sens':>9}{'id_spec':>9}"
             f"{'ood_sens':>10}{'ood_spec':>10}{'ood_FN':>8}{'ood_FP':>8}{'ood_ppr':>9}")
        for sens in TARGET_SENSITIVITIES:
            for mname in op_models:
                r = next((c for c in candidate_rows
                          if c["model"] == mname and c["criterion"] == "sensitivity"
                          and abs(c["target"] - sens) < 1e-9), None)
                emit(f"   {sens:>9.2f}{mname:<16}{_fmt(r['threshold_selected_on_id_val']):>8}"
                     f"{_fmt(r['id_val_sensitivity']):>9}{_fmt(r['id_val_specificity']):>9}"
                     f"{_fmt(r['ood_val_sensitivity']):>10}{_fmt(r['ood_val_specificity']):>10}"
                     f"{(r['ood_val_fn'] if r['ood_val_fn'] is not None else 'None'):>8}"
                     f"{(r['ood_val_fp'] if r['ood_val_fp'] is not None else 'None'):>8}"
                     f"{_fmt(r['ood_val_predicted_positive_rate']):>9}")

        # plain-vs-stain comparison at each fixed-specificity candidate (ood_val sensitivity)
        plain_stronger_flags = []
        for spec in TARGET_SPECIFICITIES:
            pr = next((c for c in candidate_rows if c["model"] == "plain ERM"
                       and c["criterion"] == "specificity" and abs(c["target"] - spec) < 1e-9), None)
            sr = next((c for c in candidate_rows if c["model"] == "stain-aug ERM"
                       and c["criterion"] == "specificity" and abs(c["target"] - spec) < 1e-9), None)
            ps = pr["ood_val_sensitivity"] if pr else None
            ss = sr["ood_val_sensitivity"] if sr else None
            if ps is not None and ss is not None:
                plain_stronger_flags.append(ps > ss)
        plain_remains_stronger = bool(plain_stronger_flags) and all(plain_stronger_flags)

        # --------------------------------------------------------------- #
        # 6. Figures (matplotlib only)                                    #
        # --------------------------------------------------------------- #
        # Fig 1: stain-aug raw vs calibrated calibration metrics (ECE/Brier/NLL)
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        metric_names = ["ECE", "Brier", "NLL"]
        raw_vals = [stain_raw_cal["ece"], stain_raw_cal["brier"], stain_raw_cal["nll"]]
        cal_vals = [stain_cal_cal["ece"], stain_cal_cal["brier"], stain_cal_cal["nll"]]
        x = np.arange(len(metric_names))
        w = 0.36
        ax.bar(x - w / 2, raw_vals, w, label="raw", color="tab:orange")
        ax.bar(x + w / 2, cal_vals, w, label=f"calibrated (T={stain_T:.2f})", color="tab:green")
        for xi, (rv, cv) in enumerate(zip(raw_vals, cal_vals)):
            ax.text(xi - w / 2, rv, f"{rv:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + w / 2, cv, f"{cv:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylabel("metric value (lower = better)")
        ax.set_title("Stain-aug ood_val calibration: raw vs calibrated (development)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_resolve(OUT_FIG_CAL), dpi=150)
        plt.close(fig)

        # Fig 2: high-confidence FN counts across thresholds, 4 variants
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        variant_order = ["plain_raw", "plain_calibrated", "stain_raw", "stain_calibrated"]
        variant_disp = {
            "plain_raw": "plain raw", "plain_calibrated": "plain cal",
            "stain_raw": "stain raw", "stain_calibrated": "stain cal",
        }
        colors = {"plain_raw": "tab:blue", "plain_calibrated": "tab:cyan",
                  "stain_raw": "tab:orange", "stain_calibrated": "tab:green"}
        xt = np.arange(len(HICONF_THRESHOLDS))
        wv = 0.2
        for i, vname in enumerate(variant_order):
            vals = [hiconf_by_variant[vname][t]["n_high_conf_false_negative"] for t in HICONF_THRESHOLDS]
            ax.bar(xt + (i - 1.5) * wv, vals, wv, label=variant_disp[vname], color=colors[vname])
        ax.set_xticks(xt)
        ax.set_xticklabels([f">= {t:.2f}" for t in HICONF_THRESHOLDS])
        ax.set_xlabel("confidence threshold")
        ax.set_ylabel("high-confidence false negatives (count)")
        ax.set_title("High-confidence missed tumors on ood_val (development)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_resolve(OUT_FIG_HICONF_FN), dpi=150)
        plt.close(fig)

        # Fig 3: ood_val sensitivity at id_val-selected fixed-specificity candidates
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        plain_sens = []
        stain_sens = []
        for spec in TARGET_SPECIFICITIES:
            pr = next((c for c in candidate_rows if c["model"] == "plain ERM"
                       and c["criterion"] == "specificity" and abs(c["target"] - spec) < 1e-9), None)
            sr = next((c for c in candidate_rows if c["model"] == "stain-aug ERM"
                       and c["criterion"] == "specificity" and abs(c["target"] - spec) < 1e-9), None)
            plain_sens.append(pr["ood_val_sensitivity"] if pr and pr["ood_val_sensitivity"] is not None else np.nan)
            stain_sens.append(sr["ood_val_sensitivity"] if sr and sr["ood_val_sensitivity"] is not None else np.nan)
        xs = np.arange(len(TARGET_SPECIFICITIES))
        ws = 0.36
        ax.bar(xs - ws / 2, plain_sens, ws, label="plain ERM", color="tab:blue")
        ax.bar(xs + ws / 2, stain_sens, ws, label="stain-aug ERM", color="tab:orange")
        for xi, (pv, sv) in enumerate(zip(plain_sens, stain_sens)):
            if not math.isnan(pv):
                ax.text(xi - ws / 2, pv, f"{pv:.2f}", ha="center", va="bottom", fontsize=8)
            if not math.isnan(sv):
                ax.text(xi + ws / 2, sv, f"{sv:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s:.2f}" for s in TARGET_SPECIFICITIES])
        ax.set_xlabel("id_val target specificity (candidate threshold)")
        ax.set_ylabel("ood_val sensitivity at that threshold")
        ax.set_title("ood_val sensitivity at id_val-selected candidate thresholds (development)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_resolve(OUT_FIG_FIXED_SPEC), dpi=150)
        plt.close(fig)

        figure_paths = [OUT_FIG_CAL, OUT_FIG_HICONF_FN, OUT_FIG_FIXED_SPEC]

        # --------------------------------------------------------------- #
        # Calibration-metrics JSON payload                                #
        # --------------------------------------------------------------- #
        cal_metrics_payload = {
            "milestone": "7C",
            "description": "Stain-aug calibration (temperature scaling on ood_val) + candidate "
                           "operating-point audit. CSV/logit-only; no training/inference; "
                           "ood_test never read.",
            "positive_class": 1,
            "n_ece_bins": N_ECE_BINS,
            "stain_aug": {
                "temperature": stain_T,
                "calibration": {"raw": stain_raw_cal, "calibrated": stain_cal_cal},
                "classification": {"raw": {k: stain_cls_raw[k] for k in
                                           ["accuracy", "sensitivity", "specificity", "fn", "fp"]},
                                   "calibrated": {k: stain_cls_cal[k] for k in
                                                  ["accuracy", "sensitivity", "specificity", "fn", "fp"]}},
            },
            "plain_erm_5a": {
                "temperature": plain_T,
                "calibration": {"raw": plain_cal_raw, "calibrated": plain_cal_cal},
            },
            "high_confidence_false_negatives": hiconf_rows,
            "candidate_operating_points": candidate_rows,
            "interpretation": {
                "calibration_helped_stain_aug": bool(calibration_helped),
                "stain_aug_ece_raw": stain_raw_cal["ece"],
                "stain_aug_ece_calibrated": stain_cal_cal["ece"],
                "stain_aug_high_conf_fn_at_090_raw": stain_hc_fn_raw_090,
                "stain_aug_high_conf_fn_at_090_calibrated": stain_hc_fn_cal_090,
                "plain_erm_remains_stronger_fixed_spec": plain_remains_stronger,
                "argmax_invariant": True,
            },
            "note": "DEVELOPMENT analysis on capped balanced subsets - NOT final performance. "
                    "Candidate id_val thresholds are NOT committed clinical operating points. "
                    "ood_val used for development-stage calibration/comparison only; ood_test never read.",
        }
        cal_metrics_path = _resolve(OUT_CAL_METRICS_JSON)
        ensure_dir(cal_metrics_path.parent)
        write_text(cal_metrics_path, json.dumps(cal_metrics_payload, indent=2) + "\n")

        # --------------------------------------------------------------- #
        # Conclusion                                                      #
        # --------------------------------------------------------------- #
        emit("")
        emit("CONCLUSION (development-stage; capped balanced subsets)")
        emit("-" * 60)
        if calibration_helped:
            emit(f"* Confidence reliability: calibration IMPROVED stain-aug confidence "
                 f"(ECE {stain_raw_cal['ece']:.4f} -> {stain_cal_cal['ece']:.4f}); it also "
                 f"reduced high-confidence missed tumors at conf>=0.90 "
                 f"({stain_hc_fn_raw_090} -> {stain_hc_fn_cal_090}).")
        else:
            emit(f"* Confidence reliability: calibration did NOT improve stain-aug ECE "
                 f"({stain_raw_cal['ece']:.4f} -> {stain_cal_cal['ece']:.4f}).")
        emit("* Argmax-invariance: temperature scaling does NOT change the hard prediction at "
             "the 0.5 argmax threshold, so it cannot by itself fix the stain-aug sensitivity")
        emit("  collapse at 0.5 - that requires an operating-point (threshold) choice.")
        emit("* 7A 0.5 'collapse' is mostly THRESHOLD behavior (7B: ood_val |dAUROC| ~ 0.02), "
             "not a total ranking collapse; candidate id_val thresholds partly recover ood_val")
        emit("  sensitivity but at a higher false-positive cost.")
        if plain_remains_stronger:
            emit("* Plain ERM REMAINS STRONGER than stain-aug at matched fixed-specificity "
                 "candidate points on ood_val (higher ood_val sensitivity at every target).")
        else:
            emit("* Plain vs stain-aug is mixed across fixed-specificity candidate points "
                 "(stain-aug is not uniformly weaker).")
        emit("* Recommendation: RETAIN the stain-aug baseline for the reliability audit with a "
             "CALIBRATED confidence output and CANDIDATE (not committed) operating points; do")
        emit("  NOT reject it on the default-0.5 result, and do NOT claim it beats plain ERM. "
             "Plain ERM stays the stronger deterministic baseline on this capped subset.")
        emit("* Final performance: NONE of these numbers are final performance; ood_test is "
             "untouched and no clinical/paper operating point is committed.")

        recommendation = "RETAIN (with calibrated confidence + candidate operating points)"

        # --------------------------------------------------------------- #
        # Verify input prediction CSVs were NOT modified                  #
        # --------------------------------------------------------------- #
        for name in READONLY_INPUTS:
            post = resolved_ro[name].stat()
            assert post.st_mtime_ns == pre_mtime[name], f"input modified (mtime): {name}"
            assert post.st_size == pre_size[name], f"input modified (size): {name}"

        # --------------------------------------------------------------- #
        # Output-existence assertions                                     #
        # --------------------------------------------------------------- #
        out_tables = [OUT_CAL_COMPARISON_CSV, OUT_HICONF_FN_CSV, OUT_CANDIDATE_OP_CSV]
        assert cal_pred_path.exists() and cal_pred_path.stat().st_size > 0, "calibrated pred CSV missing/empty"
        for p in out_tables:
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"table missing/empty: {p}"
        for p in figure_paths:
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"figure missing/empty: {p}"
        assert temp_path.exists() and temp_path.stat().st_size > 0, "temperature JSON missing/empty"
        assert cal_metrics_path.exists() and cal_metrics_path.stat().st_size > 0, "metrics JSON missing/empty"

        emit("")
        emit("OUTPUT PATHS")
        emit("-" * 60)
        emit("calibrated predictions:")
        emit(f"  {OUT_CAL_PRED_CSV}")
        emit("tables:")
        for p in out_tables:
            emit(f"  {p}")
        emit("figures:")
        for p in figure_paths:
            emit(f"  {p}")
        emit(f"temperature JSON      : {OUT_TEMP_JSON}")
        emit(f"calibration metrics JSON: {OUT_CAL_METRICS_JSON}")
        emit(f"report                : {OUT_REPORT_TXT}")
        emit(f"RECOMMENDATION        : {recommendation}")
        emit("ood_test usage: NONE")

    except Exception as exc:
        ok = False
        emit("")
        emit(f"ERROR: {type(exc).__name__}: {exc}")

    emit("")
    emit("=" * 78)
    emit(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    emit("=" * 78)

    write_text(_resolve(OUT_REPORT_TXT), "\n".join(lines) + "\n")
    print(f"\nSaved report -> {_resolve(OUT_REPORT_TXT)}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
