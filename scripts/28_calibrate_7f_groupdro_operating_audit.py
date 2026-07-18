"""Milestone 7G / Script 28: calibration + candidate operating-point audit of the two
matched Milestone 7F center-stratified models (CSV / logit-only).

**No training. No model inference. No MC-dropout. No ensemble. No CORAL/DANN. No ood_test.**
This script takes the *already-saved* Milestone 7F predictions for the two matched
center-stratified models — the **center-stratified ERM control** and the
**GroupDRO-by-center** model — and, for each model independently:

  (a) fits post-hoc temperature scaling on its ``ood_val`` logits (development-stage),
  (b) reports raw-vs-calibrated calibration reliability (ECE / Brier / NLL),
  (c) audits high-confidence false negatives (missed tumors the model is confident about)
      before/after calibration,
  (d) audits *candidate* id_val-selected operating points (fixed specificity + fixed
      sensitivity) on ``ood_val``,

and finally asks whether GroupDRO remains better than its matched ERM control after the
candidate operating-point audit.

Interpretation guardrails (baked into the report):
  * The FAIR comparison is **center-stratified ERM vs GroupDRO** (both trained on the SAME
    new center-stratified 7F-1 cache). Old plain ERM / 7A / 7D / 7E are context-only, NOT a
    controlled comparison, and are NOT re-derived here.
  * Thresholds selected on ``id_val`` are **candidate development thresholds**, NOT final
    clinical thresholds. No operating point is committed.
  * Temperature scaling is **argmax-invariant** — it rescales confidence, it does NOT move
    the hard prediction at the fixed argmax (0.5) threshold, so it cannot by itself change
    default sensitivity/specificity.
  * Everything here is DEVELOPMENT-stage on capped balanced subsets — NOT final performance.

Strict scope (Milestone 7G): ``ood_val`` is used for development-stage calibration /
comparison ONLY (never training, never final test reporting). ``ood_test`` is NEVER read.
Input prediction CSVs are read-only and never modified. Checkpoints are never touched.

Inputs (read-only):
  center-strat ERM : results/predictions/exp07f_centerstrat_erm_resnet18/id_val_predictions.csv
                     results/predictions/exp07f_centerstrat_erm_resnet18/ood_val_predictions.csv
  GroupDRO         : results/predictions/exp07f_groupdro_resnet18/id_val_predictions.csv
                     results/predictions/exp07f_groupdro_resnet18/ood_val_predictions.csv
  7F tables (ctx)  : results/tables/exp07f_overall_metrics_comparison.csv
                     results/tables/exp07f_id_val_per_center_metrics.csv
                     results/tables/exp07f_ood_val_metrics_comparison.csv
                     results/tables/exp07f_fixed_specificity_comparison.csv
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
from evaluation.metrics_operating_points import apply_operating_points
from evaluation.metrics_uncertainty import compute_high_confidence_false_negative_summary
from utils.io import ensure_dir, ensure_results_tree, project_root, write_csv, write_text

# ---------------------------------------------------------------------------- #
# Input paths (read-only)                                                       #
# ---------------------------------------------------------------------------- #
ERM_ID_VAL_CSV = "results/predictions/exp07f_centerstrat_erm_resnet18/id_val_predictions.csv"
ERM_OOD_VAL_CSV = "results/predictions/exp07f_centerstrat_erm_resnet18/ood_val_predictions.csv"
GDRO_ID_VAL_CSV = "results/predictions/exp07f_groupdro_resnet18/id_val_predictions.csv"
GDRO_OOD_VAL_CSV = "results/predictions/exp07f_groupdro_resnet18/ood_val_predictions.csv"

# 7F tables consulted for context (read-only; existence-checked only; not modified).
EXP07F_OVERALL_CSV = "results/tables/exp07f_overall_metrics_comparison.csv"
EXP07F_PERCENTER_CSV = "results/tables/exp07f_id_val_per_center_metrics.csv"
EXP07F_OOD_CSV = "results/tables/exp07f_ood_val_metrics_comparison.csv"
EXP07F_FIXEDSPEC_CSV = "results/tables/exp07f_fixed_specificity_comparison.csv"

# Prediction CSVs that must NOT be modified (fingerprinted pre/post).
READONLY_INPUTS = [
    ERM_ID_VAL_CSV, ERM_OOD_VAL_CSV, GDRO_ID_VAL_CSV, GDRO_OOD_VAL_CSV,
]
# Additional read-only inputs (existence-checked only).
CONTEXT_INPUTS = [
    EXP07F_OVERALL_CSV, EXP07F_PERCENTER_CSV, EXP07F_OOD_CSV, EXP07F_FIXEDSPEC_CSV,
]

# ---------------------------------------------------------------------------- #
# Output paths                                                                  #
# ---------------------------------------------------------------------------- #
OUT_ERM_CAL_PRED_CSV = "results/predictions/exp07g_7f_calibration/centerstrat_erm_ood_val_calibrated_predictions.csv"
OUT_GDRO_CAL_PRED_CSV = "results/predictions/exp07g_7f_calibration/groupdro_ood_val_calibrated_predictions.csv"
OUT_TEMP_JSON = "results/metrics/exp07g_7f_calibration/temperature.json"
OUT_CAL_METRICS_JSON = "results/metrics/exp07g_7f_calibration/calibration_metrics.json"

OUT_CAL_COMPARISON_CSV = "results/tables/exp07g_calibration_comparison.csv"
OUT_HICONF_FN_CSV = "results/tables/exp07g_high_confidence_fn_comparison.csv"
OUT_CANDIDATE_OP_CSV = "results/tables/exp07g_candidate_operating_point_audit.csv"

OUT_FIG_CAL = "results/figures/exp07g_calibration_raw_vs_calibrated.png"
OUT_FIG_HICONF_FN = "results/figures/exp07g_high_confidence_fn_comparison.png"
OUT_FIG_FIXED_SPEC = "results/figures/exp07g_fixed_specificity_sensitivity_comparison.png"

OUT_REPORT_TXT = "results/logs/exp07g_7f_calibration_operating_audit_report.txt"

OUTPUT_PATHS = [
    OUT_ERM_CAL_PRED_CSV, OUT_GDRO_CAL_PRED_CSV, OUT_TEMP_JSON, OUT_CAL_METRICS_JSON,
    OUT_CAL_COMPARISON_CSV, OUT_HICONF_FN_CSV, OUT_CANDIDATE_OP_CSV,
    OUT_FIG_CAL, OUT_FIG_HICONF_FN, OUT_FIG_FIXED_SPEC, OUT_REPORT_TXT,
]

REQUIRED_COLUMNS = ["label", "prob_0", "prob_1", "logit_0", "logit_1", "pred_label", "correct"]

TARGET_SPECIFICITIES = [0.80, 0.85, 0.90, 0.95]
TARGET_SENSITIVITIES = [0.80, 0.90, 0.95]
HICONF_THRESHOLDS = [0.90, 0.95, 0.99]

# Temperature-scaling settings (match Milestone 5A / 7C for consistency).
TEMP_MAX_ITER = 200
TEMP_LR = 0.01
TEMP_INIT = 1.0
TEMP_MIN = 0.05
TEMP_MAX = 10.0
N_ECE_BINS = 15

DEFAULT_THRESHOLD = 0.5

# Display names for the two matched 7F models.
ERM_NAME = "center-strat ERM (7F)"
GDRO_NAME = "GroupDRO (7F)"


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
    emit("Milestone 7G - 7F ERM vs GroupDRO calibration + candidate operating-point audit")
    emit("CSV/logit-only. NO training, NO inference, NO MC-dropout/ensemble, NO CORAL/DANN.")
    emit("ood_val -> development-stage calibration/comparison ONLY (never training/final).")
    emit("ood_test: NEVER read. Candidate id_val thresholds are NOT final clinical points.")
    emit("FAIR comparison = center-strat ERM vs GroupDRO (same 7F-1 cache). Old plain/7A/7D/7E")
    emit("are context-only, NOT controlled, and are NOT re-derived here.")
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
            "erm_id_val": pd.read_csv(resolved_ro[ERM_ID_VAL_CSV]),
            "erm_ood_val": pd.read_csv(resolved_ro[ERM_OOD_VAL_CSV]),
            "gdro_id_val": pd.read_csv(resolved_ro[GDRO_ID_VAL_CSV]),
            "gdro_ood_val": pd.read_csv(resolved_ro[GDRO_OOD_VAL_CSV]),
        }
        expected_n = {
            "erm_id_val": 450, "erm_ood_val": 2000,
            "gdro_id_val": 450, "gdro_ood_val": 2000,
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

        # explicit row-count assertions from the task spec
        assert len(frames["erm_id_val"]) == 450, "ERM id_val n must be 450"
        assert len(frames["erm_ood_val"]) == 2000, "ERM ood_val n must be 2000"
        assert len(frames["gdro_id_val"]) == 450, "GroupDRO id_val n must be 450"
        assert len(frames["gdro_ood_val"]) == 2000, "GroupDRO ood_val n must be 2000"

        # id_val center metadata (validate if present; the 7F CSVs carry it)
        id_val_center_present = {
            "erm_id_val": "center" in frames["erm_id_val"].columns,
            "gdro_id_val": "center" in frames["gdro_id_val"].columns,
        }
        for key in ("erm_id_val", "gdro_id_val"):
            if id_val_center_present[key]:
                centers = set(np.unique(frames[key]["center"].to_numpy()).tolist())
                assert centers.issubset({0, 3, 4}), (
                    f"{key}: id_val centers outside expected {{0,3,4}}: {sorted(centers)}"
                )

        emit("")
        emit("Inputs validated (columns, row counts, probabilities, logits):")
        for key, df in frames.items():
            extra = ""
            if key in id_val_center_present and id_val_center_present[key]:
                cc = sorted(set(np.unique(df["center"].to_numpy()).tolist()))
                extra = f"  centers={cc}"
            emit(f"  {key:<14} n={len(df):<5} "
                 f"tumors={int((df['label'] == 1).sum())} "
                 f"non-tumors={int((df['label'] == 0).sum())}{extra}")

        # --------------------------------------------------------------- #
        # 2. Temperature scaling on ood_val logits, for EACH model         #
        # --------------------------------------------------------------- #
        # Per-model calibration bundle: raw + calibrated arrays and fit summary.
        model_specs = [
            ("erm", ERM_NAME, "erm_ood_val", OUT_ERM_CAL_PRED_CSV),
            ("gdro", GDRO_NAME, "gdro_ood_val", OUT_GDRO_CAL_PRED_CSV),
        ]
        cal_bundle: dict[str, dict] = {}

        emit("")
        emit("2. Temperature scaling on ood_val logits (development-stage, per model)")
        for mkey, mname, ood_key, out_csv in model_specs:
            ood = frames[ood_key]
            y_ood = ood["label"].to_numpy().astype(int)
            logits_ood = np.stack(
                [ood["logit_0"].to_numpy(dtype=float),
                 ood["logit_1"].to_numpy(dtype=float)],
                axis=1,
            )
            fit = fit_temperature(
                torch.from_numpy(logits_ood).float(),
                torch.from_numpy(y_ood).long(),
                max_iter=TEMP_MAX_ITER, lr=TEMP_LR, temperature_init=TEMP_INIT,
                min_temperature=TEMP_MIN, max_temperature=TEMP_MAX, device="cpu",
            )
            T = float(fit["temperature"])
            assert math.isfinite(T) and TEMP_MIN <= T <= TEMP_MAX, f"{mkey} T out of bounds"

            cal_logits = apply_temperature_to_logits(
                torch.from_numpy(logits_ood).float(), T
            ).numpy()
            cal_probs = _softmax_rows(cal_logits)
            cal_prob_0 = cal_probs[:, 0]
            cal_prob_1 = cal_probs[:, 1]
            cal_pred = (cal_prob_1 >= DEFAULT_THRESHOLD).astype(int)
            cal_correct = (cal_pred == y_ood).astype(int)
            cal_conf = np.maximum(cal_prob_0, cal_prob_1)

            # sanity: calibrated probabilities valid + argmax unchanged
            assert np.isfinite(cal_prob_0).all() and np.isfinite(cal_prob_1).all(), f"{mkey}: non-finite cal prob"
            assert ((cal_prob_0 >= 0) & (cal_prob_0 <= 1)).all(), f"{mkey}: cal prob_0 out of [0,1]"
            assert ((cal_prob_1 >= 0) & (cal_prob_1 <= 1)).all(), f"{mkey}: cal prob_1 out of [0,1]"
            assert (np.abs(cal_prob_0 + cal_prob_1 - 1.0) < 1e-5).all(), f"{mkey}: cal prob sum != 1"
            assert (cal_pred == ood["pred_label"].to_numpy().astype(int)).all(), (
                f"{mkey}: temperature scaling changed the argmax prediction (should be invariant)"
            )

            # save calibrated predictions (preserve original columns, add calibrated)
            cal_df = ood.copy()
            cal_df["calibrated_prob_0"] = cal_prob_0
            cal_df["calibrated_prob_1"] = cal_prob_1
            cal_df["calibrated_pred_label"] = cal_pred
            cal_df["calibrated_correct"] = cal_correct
            cal_df["calibrated_confidence"] = cal_conf
            out_path = _resolve(out_csv)
            ensure_dir(out_path.parent)
            cal_df.to_csv(out_path, index=False)

            cal_bundle[mkey] = {
                "name": mname,
                "y_ood": y_ood,
                "raw_prob_1": ood["prob_1"].to_numpy(dtype=float),
                "raw_conf": np.maximum(ood["prob_0"].to_numpy(dtype=float),
                                       ood["prob_1"].to_numpy(dtype=float)),
                "raw_pred": ood["pred_label"].to_numpy().astype(int),
                "cal_prob_1": cal_prob_1,
                "cal_conf": cal_conf,
                "cal_pred": cal_pred,
                "temperature": T,
                "fit": fit,
                "out_csv": out_csv,
            }
            emit(f"   {mname:<22} T = {T:.6f}  (NLL {fit['nll_before']:.6f} -> "
                 f"{fit['nll_after']:.6f}; n_iter={fit['n_iter']}, converged={fit['converged']})")
        emit("   temperature scaling is argmax-invariant: hard predictions at 0.5 unchanged.")

        # temperature JSON (both models)
        temp_payload = {
            "milestone": "7G",
            "description": "Temperature scaling of the two matched 7F center-stratified models "
                           "(center-strat ERM + GroupDRO) on their ood_val logits. Weights "
                           "unchanged; ood_test never touched. Development result, NOT final.",
            "fit_on": "ood_val logits (development-stage OOD calibration only)",
            "argmax_invariant": True,
            "models": {
                "centerstrat_erm": {
                    "name": ERM_NAME,
                    "temperature": cal_bundle["erm"]["temperature"],
                    "nll_before": cal_bundle["erm"]["fit"]["nll_before"],
                    "nll_after": cal_bundle["erm"]["fit"]["nll_after"],
                    "n_iter": cal_bundle["erm"]["fit"]["n_iter"],
                    "converged": cal_bundle["erm"]["fit"]["converged"],
                    "n": int(len(frames["erm_ood_val"])),
                    "calibrated_predictions_csv": OUT_ERM_CAL_PRED_CSV,
                },
                "groupdro": {
                    "name": GDRO_NAME,
                    "temperature": cal_bundle["gdro"]["temperature"],
                    "nll_before": cal_bundle["gdro"]["fit"]["nll_before"],
                    "nll_after": cal_bundle["gdro"]["fit"]["nll_after"],
                    "n_iter": cal_bundle["gdro"]["fit"]["n_iter"],
                    "converged": cal_bundle["gdro"]["fit"]["converged"],
                    "n": int(len(frames["gdro_ood_val"])),
                    "calibrated_predictions_csv": OUT_GDRO_CAL_PRED_CSV,
                },
            },
            "min_temperature": TEMP_MIN,
            "max_temperature": TEMP_MAX,
            "temperature_init": TEMP_INIT,
            "note": "DEVELOPMENT temperature scaling on capped balanced ood_val subset - NOT final "
                    "performance. ood_test never read; no committed operating point.",
        }
        temp_path = _resolve(OUT_TEMP_JSON)
        ensure_dir(temp_path.parent)
        write_text(temp_path, json.dumps(temp_payload, indent=2) + "\n")

        # --------------------------------------------------------------- #
        # 3. Calibration metrics: raw vs calibrated, per model             #
        # --------------------------------------------------------------- #
        cal_comparison_rows: list[dict] = []
        cal_metrics_by_model: dict[str, dict] = {}
        for mkey in ("erm", "gdro"):
            b = cal_bundle[mkey]
            y = b["y_ood"]
            raw_cal = compute_calibration_metrics(y, b["raw_prob_1"], N_ECE_BINS)
            cal_cal = compute_calibration_metrics(y, b["cal_prob_1"], N_ECE_BINS)
            cls_raw = compute_binary_classification_metrics(y, b["raw_prob_1"])
            cls_cal = compute_binary_classification_metrics(y, b["cal_prob_1"])
            cal_metrics_by_model[mkey] = {
                "raw": raw_cal, "calibrated": cal_cal,
                "cls_raw": cls_raw, "cls_cal": cls_cal,
            }
            cal_comparison_rows.append(
                {"model": b["name"], "variant": "raw", "temperature": None,
                 "ece": raw_cal["ece"], "brier": raw_cal["brier"], "nll": raw_cal["nll"]})
            cal_comparison_rows.append(
                {"model": b["name"], "variant": "calibrated", "temperature": b["temperature"],
                 "ece": cal_cal["ece"], "brier": cal_cal["brier"], "nll": cal_cal["nll"]})
        write_csv(_resolve(OUT_CAL_COMPARISON_CSV), cal_comparison_rows,
                  header=["model", "variant", "temperature", "ece", "brier", "nll"])

        erm_cal_helped = cal_metrics_by_model["erm"]["calibrated"]["ece"] < cal_metrics_by_model["erm"]["raw"]["ece"] - 1e-9
        gdro_cal_helped = cal_metrics_by_model["gdro"]["calibrated"]["ece"] < cal_metrics_by_model["gdro"]["raw"]["ece"] - 1e-9
        calibration_helped_both = erm_cal_helped and gdro_cal_helped

        emit("")
        emit("3. Calibration reliability on ood_val (development)  [tumor = positive]")
        emit(f"   {'model':<24}{'variant':<12}{'ECE':>9}{'Brier':>9}{'NLL':>9}")
        for r in cal_comparison_rows:
            emit(f"   {r['model']:<24}{r['variant']:<12}"
                 f"{_fmt(r['ece']):>9}{_fmt(r['brier']):>9}{_fmt(r['nll']):>9}")
        for mkey, mname, helped in (("erm", ERM_NAME, erm_cal_helped),
                                    ("gdro", GDRO_NAME, gdro_cal_helped)):
            m = cal_metrics_by_model[mkey]
            emit(f"   {mname}: ECE {m['raw']['ece']:.4f} -> {m['calibrated']['ece']:.4f} "
                 f"({'improved' if helped else 'no improvement'}); classification unchanged "
                 f"raw->cal (argmax-invariant): acc {_fmt(m['cls_raw']['accuracy'])}, "
                 f"sens {_fmt(m['cls_raw']['sensitivity'])}, FN {m['cls_raw']['fn']}")

        # --------------------------------------------------------------- #
        # 4. High-confidence false-negative audit (4 variants)            #
        # --------------------------------------------------------------- #
        def _hiconf_frame(label, pred_label, confidence) -> pd.DataFrame:
            return pd.DataFrame({
                "label": np.asarray(label).astype(int),
                "pred_label": np.asarray(pred_label).astype(int),
                "confidence": np.asarray(confidence, dtype=float),
            })

        hiconf_variants = {
            "centerstrat_erm_raw": _hiconf_frame(
                cal_bundle["erm"]["y_ood"], cal_bundle["erm"]["raw_pred"], cal_bundle["erm"]["raw_conf"]),
            "centerstrat_erm_calibrated": _hiconf_frame(
                cal_bundle["erm"]["y_ood"], cal_bundle["erm"]["cal_pred"], cal_bundle["erm"]["cal_conf"]),
            "groupdro_raw": _hiconf_frame(
                cal_bundle["gdro"]["y_ood"], cal_bundle["gdro"]["raw_pred"], cal_bundle["gdro"]["raw_conf"]),
            "groupdro_calibrated": _hiconf_frame(
                cal_bundle["gdro"]["y_ood"], cal_bundle["gdro"]["cal_pred"], cal_bundle["gdro"]["cal_conf"]),
        }
        variant_model = {
            "centerstrat_erm_raw": ERM_NAME, "centerstrat_erm_calibrated": ERM_NAME,
            "groupdro_raw": GDRO_NAME, "groupdro_calibrated": GDRO_NAME,
        }
        variant_kind = {
            "centerstrat_erm_raw": "raw", "centerstrat_erm_calibrated": "calibrated",
            "groupdro_raw": "raw", "groupdro_calibrated": "calibrated",
        }
        variant_order = [
            "centerstrat_erm_raw", "centerstrat_erm_calibrated",
            "groupdro_raw", "groupdro_calibrated",
        ]

        hiconf_rows: list[dict] = []
        hiconf_by_variant: dict[str, dict[float, dict]] = {}
        for vname in variant_order:
            summ = compute_high_confidence_false_negative_summary(
                hiconf_variants[vname], "confidence", HICONF_THRESHOLDS)
            hiconf_by_variant[vname] = {row["threshold"]: row for row in summ}
            for row in summ:
                hiconf_rows.append({
                    "model": variant_model[vname],
                    "variant": variant_kind[vname],
                    "variant_key": vname,
                    "confidence_threshold": row["threshold"],
                    "n_total": row["n_total"],
                    "n_high_conf": row["n_high_conf"],
                    "n_high_conf_false_negative": row["n_high_conf_false_negative"],
                    "high_conf_false_negative_rate": row["high_conf_false_negative_rate"],
                    "n_false_negative_total": row["n_false_negative_total"],
                    "fraction_of_fn_in_high_conf": row["fraction_of_fn_in_high_conf"],
                })
        write_csv(_resolve(OUT_HICONF_FN_CSV), hiconf_rows, header=[
            "model", "variant", "variant_key", "confidence_threshold", "n_total", "n_high_conf",
            "n_high_conf_false_negative", "high_conf_false_negative_rate",
            "n_false_negative_total", "fraction_of_fn_in_high_conf",
        ])

        emit("")
        emit("4. High-confidence false-negative (missed-tumor) audit on ood_val")
        emit("   confidently-missed tumors = label==1, pred==0, confidence >= t")
        emit(f"   {'model':<24}{'variant':<12}{'t':>6}{'n_high':>8}{'hc_FN':>7}"
             f"{'hc_FN_rate':>12}{'FN_total':>10}")
        for vname in variant_order:
            for t in HICONF_THRESHOLDS:
                row = hiconf_by_variant[vname][t]
                emit(f"   {variant_model[vname]:<24}{variant_kind[vname]:<12}{t:>6.2f}"
                     f"{row['n_high_conf']:>8}{row['n_high_conf_false_negative']:>7}"
                     f"{_fmt(row['high_conf_false_negative_rate']):>12}"
                     f"{row['n_false_negative_total']:>10}")

        # --------------------------------------------------------------- #
        # 5. Candidate operating-point audit (id_val-selected thresholds)  #
        # --------------------------------------------------------------- #
        op_models = {
            ERM_NAME: (frames["erm_id_val"], frames["erm_ood_val"]),
            GDRO_NAME: (frames["gdro_id_val"], frames["gdro_ood_val"]),
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

        def _cand(mname: str, criterion: str, target: float) -> dict | None:
            return next((c for c in candidate_rows
                         if c["model"] == mname and c["criterion"] == criterion
                         and abs(c["target"] - target) < 1e-9), None)

        emit("")
        emit("5. Candidate operating-point audit (thresholds selected on id_val ONLY)")
        emit("   CANDIDATE development thresholds - NOT final clinical operating points.")
        emit("   Raw prob_1 scores (calibration is monotone/argmax-invariant -> same ranking).")
        emit("")
        emit("   FIXED-SPECIFICITY candidates, applied to ood_val:")
        emit(f"   {'tgt_spec':>9}{'model':<24}{'thr':>8}{'id_sens':>9}{'id_spec':>9}"
             f"{'ood_sens':>10}{'ood_spec':>10}{'ood_FN':>8}{'ood_FP':>8}{'ood_ppr':>9}")
        for spec in TARGET_SPECIFICITIES:
            for mname in op_models:
                r = _cand(mname, "specificity", spec)
                emit(f"   {spec:>9.2f}{mname:<24}{_fmt(r['threshold_selected_on_id_val']):>8}"
                     f"{_fmt(r['id_val_sensitivity']):>9}{_fmt(r['id_val_specificity']):>9}"
                     f"{_fmt(r['ood_val_sensitivity']):>10}{_fmt(r['ood_val_specificity']):>10}"
                     f"{(r['ood_val_fn'] if r['ood_val_fn'] is not None else 'None'):>8}"
                     f"{(r['ood_val_fp'] if r['ood_val_fp'] is not None else 'None'):>8}"
                     f"{_fmt(r['ood_val_predicted_positive_rate']):>9}")
        emit("")
        emit("   FIXED-SENSITIVITY candidates, applied to ood_val:")
        emit(f"   {'tgt_sens':>9}{'model':<24}{'thr':>8}{'id_sens':>9}{'id_spec':>9}"
             f"{'ood_sens':>10}{'ood_spec':>10}{'ood_FN':>8}{'ood_FP':>8}{'ood_ppr':>9}")
        for sens in TARGET_SENSITIVITIES:
            for mname in op_models:
                r = _cand(mname, "sensitivity", sens)
                emit(f"   {sens:>9.2f}{mname:<24}{_fmt(r['threshold_selected_on_id_val']):>8}"
                     f"{_fmt(r['id_val_sensitivity']):>9}{_fmt(r['id_val_specificity']):>9}"
                     f"{_fmt(r['ood_val_sensitivity']):>10}{_fmt(r['ood_val_specificity']):>10}"
                     f"{(r['ood_val_fn'] if r['ood_val_fn'] is not None else 'None'):>8}"
                     f"{(r['ood_val_fp'] if r['ood_val_fp'] is not None else 'None'):>8}"
                     f"{_fmt(r['ood_val_predicted_positive_rate']):>9}")

        # --- GroupDRO vs matched ERM verdict, per fixed-specificity target -------- #
        gdro_wins_spec = []
        gdro_ge_spec = []
        for spec in TARGET_SPECIFICITIES:
            er = _cand(ERM_NAME, "specificity", spec)
            gr = _cand(GDRO_NAME, "specificity", spec)
            es = er["ood_val_sensitivity"] if er else None
            gs = gr["ood_val_sensitivity"] if gr else None
            if es is not None and gs is not None:
                gdro_wins_spec.append(gs > es)
                gdro_ge_spec.append(gs >= es - 1e-12)
        gdro_wins_all_spec = bool(gdro_wins_spec) and all(gdro_wins_spec)
        gdro_ge_all_spec = bool(gdro_ge_spec) and all(gdro_ge_spec)
        n_spec_gdro_win = int(sum(gdro_wins_spec))
        n_spec_total = len(gdro_wins_spec)

        gdro_wins_sens = []
        for sens in TARGET_SENSITIVITIES:
            er = _cand(ERM_NAME, "sensitivity", sens)
            gr = _cand(GDRO_NAME, "sensitivity", sens)
            # at matched target sensitivity, the better model keeps higher ood_val specificity
            esp = er["ood_val_specificity"] if er else None
            gsp = gr["ood_val_specificity"] if gr else None
            if esp is not None and gsp is not None:
                gdro_wins_sens.append(gsp > esp)
        n_sens_gdro_win = int(sum(gdro_wins_sens))
        n_sens_total = len(gdro_wins_sens)

        # ranking (AUROC) already established in 7F; recompute here for a self-contained verdict
        erm_ood_cls = cal_metrics_by_model["erm"]["cls_raw"]
        gdro_ood_cls = cal_metrics_by_model["gdro"]["cls_raw"]
        gdro_auroc_better = (
            gdro_ood_cls["auroc"] is not None and erm_ood_cls["auroc"] is not None
            and gdro_ood_cls["auroc"] > erm_ood_cls["auroc"]
        )

        # --------------------------------------------------------------- #
        # 6. Figures (matplotlib only)                                    #
        # --------------------------------------------------------------- #
        # Fig 1: raw vs calibrated calibration metrics (ECE/Brier/NLL), both models
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
        metric_names = ["ECE", "Brier", "NLL"]
        for ax, mkey, mname in ((axes[0], "erm", ERM_NAME), (axes[1], "gdro", GDRO_NAME)):
            m = cal_metrics_by_model[mkey]
            raw_vals = [m["raw"]["ece"], m["raw"]["brier"], m["raw"]["nll"]]
            cal_vals = [m["calibrated"]["ece"], m["calibrated"]["brier"], m["calibrated"]["nll"]]
            x = np.arange(len(metric_names))
            w = 0.36
            ax.bar(x - w / 2, raw_vals, w, label="raw", color="tab:orange")
            ax.bar(x + w / 2, cal_vals, w,
                   label=f"calibrated (T={cal_bundle[mkey]['temperature']:.2f})", color="tab:green")
            for xi, (rv, cv) in enumerate(zip(raw_vals, cal_vals)):
                ax.text(xi - w / 2, rv, f"{rv:.3f}", ha="center", va="bottom", fontsize=7)
                ax.text(xi + w / 2, cv, f"{cv:.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names)
            ax.set_title(mname, fontsize=10)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("metric value (lower = better)")
        fig.suptitle("7F ood_val calibration: raw vs calibrated (development)")
        fig.tight_layout()
        fig.savefig(_resolve(OUT_FIG_CAL), dpi=150)
        plt.close(fig)

        # Fig 2: high-confidence FN counts across thresholds, 4 variants
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        variant_disp = {
            "centerstrat_erm_raw": "ERM raw", "centerstrat_erm_calibrated": "ERM cal",
            "groupdro_raw": "GroupDRO raw", "groupdro_calibrated": "GroupDRO cal",
        }
        colors = {"centerstrat_erm_raw": "tab:blue", "centerstrat_erm_calibrated": "tab:cyan",
                  "groupdro_raw": "tab:orange", "groupdro_calibrated": "tab:green"}
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
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        erm_sens = []
        gdro_sens = []
        for spec in TARGET_SPECIFICITIES:
            er = _cand(ERM_NAME, "specificity", spec)
            gr = _cand(GDRO_NAME, "specificity", spec)
            erm_sens.append(er["ood_val_sensitivity"] if er and er["ood_val_sensitivity"] is not None else np.nan)
            gdro_sens.append(gr["ood_val_sensitivity"] if gr and gr["ood_val_sensitivity"] is not None else np.nan)
        xs = np.arange(len(TARGET_SPECIFICITIES))
        ws = 0.36
        ax.bar(xs - ws / 2, erm_sens, ws, label="center-strat ERM (7F)", color="tab:blue")
        ax.bar(xs + ws / 2, gdro_sens, ws, label="GroupDRO (7F)", color="tab:orange")
        for xi, (ev, gv) in enumerate(zip(erm_sens, gdro_sens)):
            if not math.isnan(ev):
                ax.text(xi - ws / 2, ev, f"{ev:.2f}", ha="center", va="bottom", fontsize=8)
            if not math.isnan(gv):
                ax.text(xi + ws / 2, gv, f"{gv:.2f}", ha="center", va="bottom", fontsize=8)
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
            "milestone": "7G",
            "description": "Calibration (temperature scaling on ood_val) + candidate "
                           "operating-point audit of the two matched 7F center-stratified "
                           "models. CSV/logit-only; no training/inference; ood_test never read.",
            "positive_class": 1,
            "n_ece_bins": N_ECE_BINS,
            "fair_comparison": "center-strat ERM (7F) vs GroupDRO (7F) - same 7F-1 cache",
            "models": {
                "centerstrat_erm": {
                    "name": ERM_NAME,
                    "temperature": cal_bundle["erm"]["temperature"],
                    "calibration": {"raw": cal_metrics_by_model["erm"]["raw"],
                                    "calibrated": cal_metrics_by_model["erm"]["calibrated"]},
                    "classification": {
                        "raw": {k: cal_metrics_by_model["erm"]["cls_raw"][k]
                                for k in ["accuracy", "auroc", "auprc", "sensitivity", "specificity", "fn", "fp"]},
                        "calibrated": {k: cal_metrics_by_model["erm"]["cls_cal"][k]
                                       for k in ["accuracy", "auroc", "auprc", "sensitivity", "specificity", "fn", "fp"]},
                    },
                },
                "groupdro": {
                    "name": GDRO_NAME,
                    "temperature": cal_bundle["gdro"]["temperature"],
                    "calibration": {"raw": cal_metrics_by_model["gdro"]["raw"],
                                    "calibrated": cal_metrics_by_model["gdro"]["calibrated"]},
                    "classification": {
                        "raw": {k: cal_metrics_by_model["gdro"]["cls_raw"][k]
                                for k in ["accuracy", "auroc", "auprc", "sensitivity", "specificity", "fn", "fp"]},
                        "calibrated": {k: cal_metrics_by_model["gdro"]["cls_cal"][k]
                                       for k in ["accuracy", "auroc", "auprc", "sensitivity", "specificity", "fn", "fp"]},
                    },
                },
            },
            "high_confidence_false_negatives": hiconf_rows,
            "candidate_operating_points": candidate_rows,
            "interpretation": {
                "calibration_helped_centerstrat_erm": bool(erm_cal_helped),
                "calibration_helped_groupdro": bool(gdro_cal_helped),
                "calibration_helped_both": bool(calibration_helped_both),
                "groupdro_ood_auroc": gdro_ood_cls["auroc"],
                "centerstrat_erm_ood_auroc": erm_ood_cls["auroc"],
                "groupdro_ranking_better_than_erm": bool(gdro_auroc_better),
                "groupdro_wins_fixed_specificity_targets": f"{n_spec_gdro_win}/{n_spec_total}",
                "groupdro_ge_erm_all_fixed_specificity": bool(gdro_ge_all_spec),
                "groupdro_wins_fixed_sensitivity_targets": f"{n_sens_gdro_win}/{n_sens_total}",
                "groupdro_remains_better_after_candidate_audit": bool(gdro_auroc_better and gdro_ge_all_spec),
                "argmax_invariant": True,
            },
            "note": "DEVELOPMENT analysis on capped balanced subsets - NOT final performance. "
                    "Candidate id_val thresholds are NOT committed clinical operating points. "
                    "Fair comparison = 7F ERM vs 7F GroupDRO; old plain/7A/7D/7E context-only. "
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
        emit(f"* Confidence reliability: calibration "
             f"{'IMPROVED both' if calibration_helped_both else 'did NOT uniformly improve'} "
             f"7F models on ood_val "
             f"(ERM ECE {cal_metrics_by_model['erm']['raw']['ece']:.4f} -> "
             f"{cal_metrics_by_model['erm']['calibrated']['ece']:.4f}; "
             f"GroupDRO ECE {cal_metrics_by_model['gdro']['raw']['ece']:.4f} -> "
             f"{cal_metrics_by_model['gdro']['calibrated']['ece']:.4f}).")
        emit("* Argmax-invariance: temperature scaling does NOT change the hard prediction at "
             "the 0.5 argmax threshold; default sensitivity/specificity are unchanged.")
        if gdro_auroc_better and gdro_ge_all_spec:
            emit(f"* GroupDRO REMAINS BETTER than its matched center-stratified ERM control after "
                 f"the candidate operating-point audit: higher ood_val AUROC "
                 f"({gdro_ood_cls['auroc']:.4f} vs {erm_ood_cls['auroc']:.4f}) and GroupDRO ood_val "
                 f"sensitivity >= ERM at all {n_spec_total} fixed-specificity targets "
                 f"(strictly greater at {n_spec_gdro_win}/{n_spec_total}).")
        else:
            emit(f"* GroupDRO vs matched ERM after the candidate audit is MIXED "
                 f"(AUROC better={gdro_auroc_better}; fixed-spec wins {n_spec_gdro_win}/{n_spec_total}).")
        emit("* The FAIR comparison is 7F ERM vs 7F GroupDRO (same 7F-1 cache). Old plain ERM / "
             "7A / 7D / 7E are context-only, NOT controlled, and were NOT re-derived here.")
        emit("* Recommendation: RETAIN GroupDRO WITH its calibrated confidence output; report "
             "CANDIDATE (not committed) operating points only. Do NOT commit a clinical/paper")
        emit("  operating threshold and do NOT claim final performance.")
        emit("* Final performance: NONE of these numbers are final performance; ood_test is "
             "untouched and no operating point is committed.")

        if gdro_auroc_better and gdro_ge_all_spec:
            recommendation = ("RETAIN GroupDRO (calibrated confidence + candidate operating points); "
                              "GroupDRO stays better than matched ERM after the audit")
        else:
            recommendation = ("RETAIN GroupDRO with calibrated confidence; GroupDRO-vs-ERM mixed "
                              "after candidate audit")

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
        for p in (OUT_ERM_CAL_PRED_CSV, OUT_GDRO_CAL_PRED_CSV):
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"calibrated pred CSV missing/empty: {p}"
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
        emit(f"  {OUT_ERM_CAL_PRED_CSV}")
        emit(f"  {OUT_GDRO_CAL_PRED_CSV}")
        emit("tables:")
        for p in out_tables:
            emit(f"  {p}")
        emit("figures:")
        for p in figure_paths:
            emit(f"  {p}")
        emit(f"temperature JSON        : {OUT_TEMP_JSON}")
        emit(f"calibration metrics JSON: {OUT_CAL_METRICS_JSON}")
        emit(f"report                  : {OUT_REPORT_TXT}")
        emit(f"RECOMMENDATION          : {recommendation}")
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
