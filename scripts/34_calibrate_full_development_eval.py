"""Milestone 8C / Script 34: full-development calibration + reliability audit
(CSV / logit-only).

**No training. No model inference. No model-weight update. No checkpoint modification.
No MC-dropout. No ensemble. No CORAL/DANN. No ood_test.** This script takes the
*already-saved* Milestone 8B full-development prediction CSVs for the two frozen 7F models
— the **GroupDRO-by-center primary candidate** and the **center-stratified ERM matched
control** — and, for each model independently:

  (a) fits post-hoc temperature scaling on its FULL ``ood_val`` logits (development-stage),
  (b) reports raw-vs-calibrated calibration reliability (ECE / Brier / NLL) on full ood_val,
  (c) audits high-confidence false negatives (missed tumors the model is confident about)
      before/after calibration,
  (d) builds a reliability summary and decides whether GroupDRO remains better than its
      matched ERM control on the full-development metrics.

Interpretation guardrails (baked into the report):
  * The FAIR comparison is **7F center-stratified ERM vs 7F GroupDRO** (both trained on the
    SAME sound center-stratified 7F-1 cache). Old plain ERM / 7A / 7D / 7E are context-only,
    NOT a controlled comparison, and are NOT read here.
  * Temperature scaling is **argmax-invariant** — it rescales confidence, it does NOT move
    the hard prediction at the fixed argmax (0.5) threshold, so accuracy / AUROC / AUPRC /
    sensitivity / specificity are unchanged raw -> calibrated.
  * Calibration here is fit and evaluated on the SAME ``ood_val`` — this is a
    **development-stage** reliability audit only. It is **NOT** an independent-test
    calibration claim (reviewer risk R7 remains open).
  * Everything here is DEVELOPMENT-stage. No final/clinical/paper threshold is committed and
    no final-performance claim is made. ``ood_test`` is never read.

Strict scope (Milestone 8C): ``ood_val`` is used for development-stage calibration /
comparison ONLY (never training, never final test reporting). ``ood_test`` is NEVER read.
Input prediction CSVs are read-only and never modified. Checkpoints are never touched.

Inputs (read-only; saved by Milestone 8B):
  GroupDRO : results/predictions/exp08b_full_dev_eval/groupdro_id_val_predictions.csv
             results/predictions/exp08b_full_dev_eval/groupdro_ood_val_predictions.csv
  ERM      : results/predictions/exp08b_full_dev_eval/centerstrat_erm_id_val_predictions.csv
             results/predictions/exp08b_full_dev_eval/centerstrat_erm_ood_val_predictions.csv
  8B tables (ctx): results/tables/exp08b_candidate_operating_points.csv (+ passthrough)
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
from utils.io import ensure_dir, ensure_results_tree, project_root, write_csv, write_text

# ---------------------------------------------------------------------------- #
# Input paths (read-only; Milestone 8B outputs)                                #
# ---------------------------------------------------------------------------- #
PRED_DIR = "results/predictions/exp08b_full_dev_eval"
GDRO_ID_VAL_CSV = f"{PRED_DIR}/groupdro_id_val_predictions.csv"
GDRO_OOD_VAL_CSV = f"{PRED_DIR}/groupdro_ood_val_predictions.csv"
ERM_ID_VAL_CSV = f"{PRED_DIR}/centerstrat_erm_id_val_predictions.csv"
ERM_OOD_VAL_CSV = f"{PRED_DIR}/centerstrat_erm_ood_val_predictions.csv"

# 8B candidate operating-point table consulted for the passthrough (read-only).
EXP08B_CANDIDATE_OPS_CSV = "results/tables/exp08b_candidate_operating_points.csv"

# Prediction CSVs that must NOT be modified (fingerprinted pre/post).
READONLY_INPUTS = [
    GDRO_ID_VAL_CSV, GDRO_OOD_VAL_CSV, ERM_ID_VAL_CSV, ERM_OOD_VAL_CSV,
]
# Additional read-only inputs (existence-checked + fingerprinted for the passthrough).
CONTEXT_INPUTS = [EXP08B_CANDIDATE_OPS_CSV]

# ---------------------------------------------------------------------------- #
# Output paths (all under new exp08c names; no overwrite of prior milestones)  #
# ---------------------------------------------------------------------------- #
OUT_GDRO_CAL_PRED_CSV = "results/predictions/exp08c_full_dev_calibration/groupdro_ood_val_calibrated_predictions.csv"
OUT_ERM_CAL_PRED_CSV = "results/predictions/exp08c_full_dev_calibration/centerstrat_erm_ood_val_calibrated_predictions.csv"
OUT_TEMP_JSON = "results/metrics/exp08c_full_dev_calibration/temperature.json"
OUT_CAL_METRICS_JSON = "results/metrics/exp08c_full_dev_calibration/calibration_metrics.json"

OUT_CAL_COMPARISON_CSV = "results/tables/exp08c_calibration_comparison.csv"
OUT_HICONF_FN_CSV = "results/tables/exp08c_high_confidence_fn_comparison.csv"
OUT_RELIABILITY_SUMMARY_CSV = "results/tables/exp08c_reliability_summary.csv"
OUT_CANDIDATE_OP_PASSTHROUGH_CSV = "results/tables/exp08c_candidate_operating_points_with_calibration_note.csv"

OUT_FIG_CAL = "results/figures/exp08c_calibration_raw_vs_calibrated.png"
OUT_FIG_HICONF_FN = "results/figures/exp08c_high_confidence_fn_reduction.png"
OUT_FIG_ECE_BRIER_NLL = "results/figures/exp08c_ece_brier_nll_groupdro_vs_erm.png"

OUT_REPORT_TXT = "results/logs/exp08c_full_development_calibration_report.txt"

OUTPUT_PATHS = [
    OUT_GDRO_CAL_PRED_CSV, OUT_ERM_CAL_PRED_CSV, OUT_TEMP_JSON, OUT_CAL_METRICS_JSON,
    OUT_CAL_COMPARISON_CSV, OUT_HICONF_FN_CSV, OUT_RELIABILITY_SUMMARY_CSV,
    OUT_CANDIDATE_OP_PASSTHROUGH_CSV,
    OUT_FIG_CAL, OUT_FIG_HICONF_FN, OUT_FIG_ECE_BRIER_NLL, OUT_REPORT_TXT,
]

# Required columns in every 8B prediction CSV (per the Milestone 8C spec).
REQUIRED_COLUMNS = [
    "split", "label", "pred_label", "correct", "prob_0", "prob_1", "logit_0", "logit_1",
    "confidence", "center", "patient", "slide", "node", "image_id", "x_coord", "y_coord",
]

EXPECTED_N = {"id_val": 33560, "ood_val": 34904}
ALLOWED_CENTERS = {"id_val": {0, 3, 4}, "ood_val": {1}}

HICONF_THRESHOLDS = [0.90, 0.95, 0.99]

# Temperature-scaling settings (match Milestone 5A / 7C / 7G for consistency).
TEMP_MAX_ITER = 200
TEMP_LR = 0.01
TEMP_INIT = 1.0
TEMP_MIN = 0.05
TEMP_MAX = 10.0
N_ECE_BINS = 15

DEFAULT_THRESHOLD = 0.5

# Model keys / display names.
MODEL_ORDER = ("groupdro", "centerstrat_erm")
MODEL_LABEL = {
    "groupdro": "7F GroupDRO (primary candidate)",
    "centerstrat_erm": "7F center-stratified ERM (matched control)",
}


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


def _hiconf_fn_audit(label, pred_label, correct, confidence, thresholds) -> list[dict]:
    """High-confidence false-negative audit for one (model, variant).

    FN = label==1 & pred_label==0 (missed tumor). Returns, per threshold, the
    high-confidence subset size, high-confidence error count, high-confidence FN count and
    rate, the total FN count, and the fraction of ALL false negatives that are
    high-confidence.
    """
    y = np.asarray(label).astype(int)
    pred = np.asarray(pred_label).astype(int)
    corr = np.asarray(correct).astype(bool)
    conf = np.asarray(confidence, dtype=float)

    is_fn = (y == 1) & (pred == 0)
    total_fn = int(np.sum(is_fn))

    rows: list[dict] = []
    for t in thresholds:
        hi = conf >= float(t)
        n_high = int(np.sum(hi))
        hc_err = int(np.sum(hi & (~corr)))
        hc_fn = int(np.sum(hi & is_fn))
        rate = (hc_fn / n_high) if n_high > 0 else None
        frac_all_fn = (hc_fn / total_fn) if total_fn > 0 else None
        rows.append({
            "confidence_threshold": float(t),
            "n_high_confidence": n_high,
            "high_confidence_errors": hc_err,
            "high_confidence_false_negatives": hc_fn,
            "high_confidence_false_negative_rate": rate,
            "total_false_negatives": total_fn,
            "fraction_of_all_false_negatives_high_confidence": frac_all_fn,
        })
    return rows


def main() -> int:
    ensure_results_tree()

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    emit("=" * 80)
    emit("Milestone 8C - full-development calibration + reliability audit (script 34)")
    emit("CSV/logit-only. NO training, NO inference, NO weight update, NO checkpoint change,")
    emit("NO MC-dropout, NO ensemble, NO CORAL/DANN. ood_test NEVER read.")
    emit("ood_val -> development-stage calibration/comparison ONLY (never training/final).")
    emit("Calibration fit+evaluated on the SAME ood_val -> development-stage reliability audit,")
    emit("NOT an independent-test calibration claim (R7 stays open). No committed threshold.")
    emit("FAIR comparison = 7F center-stratified ERM vs 7F GroupDRO (same sound 7F-1 cache).")
    emit("DEVELOPMENT analysis on full development splits - NOT final performance.")
    emit("=" * 80)

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

        # fingerprint the read-only inputs so we can prove they were not modified
        fp_targets = READONLY_INPUTS + CONTEXT_INPUTS
        resolved_all_in = {**resolved_ro, **resolved_ctx}
        pre_mtime = {name: resolved_all_in[name].stat().st_mtime_ns for name in fp_targets}
        pre_size = {name: resolved_all_in[name].stat().st_size for name in fp_targets}

        # --------------------------------------------------------------- #
        # 1. Load prediction CSVs + validate columns / counts / probs     #
        # --------------------------------------------------------------- #
        csv_of = {
            ("groupdro", "id_val"): GDRO_ID_VAL_CSV,
            ("groupdro", "ood_val"): GDRO_OOD_VAL_CSV,
            ("centerstrat_erm", "id_val"): ERM_ID_VAL_CSV,
            ("centerstrat_erm", "ood_val"): ERM_OOD_VAL_CSV,
        }
        frames: dict[tuple[str, str], pd.DataFrame] = {}
        for (mkey, split), path_str in csv_of.items():
            df = pd.read_csv(_resolve(path_str))
            for col in REQUIRED_COLUMNS:
                assert col in df.columns, f"{mkey}/{split}: missing required column {col!r}"
            assert len(df) == EXPECTED_N[split], (
                f"{mkey}/{split}: expected n=={EXPECTED_N[split]}, got {len(df)}"
            )
            # probabilities
            p0 = df["prob_0"].to_numpy(dtype=float)
            p1 = df["prob_1"].to_numpy(dtype=float)
            assert np.isfinite(p0).all() and np.isfinite(p1).all(), f"{mkey}/{split}: non-finite prob"
            assert ((p0 >= 0.0) & (p0 <= 1.0)).all(), f"{mkey}/{split}: prob_0 out of [0,1]"
            assert ((p1 >= 0.0) & (p1 <= 1.0)).all(), f"{mkey}/{split}: prob_1 out of [0,1]"
            assert (np.abs(p0 + p1 - 1.0) < 1e-5).all(), f"{mkey}/{split}: prob_0+prob_1 != 1"
            # logits
            l0 = df["logit_0"].to_numpy(dtype=float)
            l1 = df["logit_1"].to_numpy(dtype=float)
            assert np.isfinite(l0).all() and np.isfinite(l1).all(), f"{mkey}/{split}: non-finite logits"
            # labels binary
            labs = set(np.unique(df["label"].to_numpy()).tolist())
            assert labs.issubset({0, 1}), f"{mkey}/{split}: labels outside {{0,1}}: {sorted(labs)}"
            # centers
            centers = set(int(c) for c in df["center"].unique().tolist())
            assert centers.issubset(ALLOWED_CENTERS[split]), (
                f"{mkey}/{split}: unexpected centers {sorted(centers)} (allowed {sorted(ALLOWED_CENTERS[split])})"
            )
            assert 2 not in centers, f"{mkey}/{split}: center 2 (ood_test) present"
            frames[(mkey, split)] = df

        # ood_val must be exactly center {1}
        for mkey in MODEL_ORDER:
            oc = set(int(c) for c in frames[(mkey, "ood_val")]["center"].unique().tolist())
            assert oc == {1}, f"{mkey}/ood_val: center must be exactly {{1}}, got {sorted(oc)}"

        emit("")
        emit("1. INPUTS VALIDATED (columns, row counts, probabilities, logits, centers)")
        emit("-" * 72)
        for mkey in MODEL_ORDER:
            for split in ("id_val", "ood_val"):
                df = frames[(mkey, split)]
                cc = sorted(int(c) for c in df["center"].unique().tolist())
                emit(f"  {MODEL_LABEL[mkey]:<45} {split:<8} n={len(df):>6} "
                     f"tumors={int((df['label'] == 1).sum())} centers={cc}")

        # --------------------------------------------------------------- #
        # 2. Temperature scaling on FULL ood_val logits, per model         #
        # --------------------------------------------------------------- #
        out_csv_of = {"groupdro": OUT_GDRO_CAL_PRED_CSV, "centerstrat_erm": OUT_ERM_CAL_PRED_CSV}
        cal_bundle: dict[str, dict] = {}

        emit("")
        emit("2. TEMPERATURE SCALING on FULL ood_val logits (development-stage, per model)")
        emit("-" * 72)
        for mkey in MODEL_ORDER:
            ood = frames[(mkey, "ood_val")]
            y_ood = ood["label"].to_numpy().astype(int)
            logits_ood = np.stack(
                [ood["logit_0"].to_numpy(dtype=float), ood["logit_1"].to_numpy(dtype=float)],
                axis=1,
            )
            fit = fit_temperature(
                torch.from_numpy(logits_ood).float(),
                torch.from_numpy(y_ood).long(),
                max_iter=TEMP_MAX_ITER, lr=TEMP_LR, temperature_init=TEMP_INIT,
                min_temperature=TEMP_MIN, max_temperature=TEMP_MAX, device="cpu",
            )
            T = float(fit["temperature"])
            assert math.isfinite(T) and TEMP_MIN <= T <= TEMP_MAX, f"{mkey}: T out of bounds"

            cal_logits = apply_temperature_to_logits(torch.from_numpy(logits_ood).float(), T).numpy()
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

            # save calibrated predictions (preserve original columns, add calibrated ones)
            cal_df = ood.copy()
            cal_df["calibrated_prob_0"] = cal_prob_0
            cal_df["calibrated_prob_1"] = cal_prob_1
            cal_df["calibrated_pred_label"] = cal_pred
            cal_df["calibrated_correct"] = cal_correct
            cal_df["calibrated_confidence"] = cal_conf
            out_path = _resolve(out_csv_of[mkey])
            ensure_dir(out_path.parent)
            cal_df.to_csv(out_path, index=False)

            cal_bundle[mkey] = {
                "name": MODEL_LABEL[mkey],
                "y_ood": y_ood,
                "raw_prob_1": ood["prob_1"].to_numpy(dtype=float),
                "raw_conf": ood["confidence"].to_numpy(dtype=float),
                "raw_pred": ood["pred_label"].to_numpy().astype(int),
                "raw_correct": ood["correct"].to_numpy().astype(bool),
                "cal_prob_1": cal_prob_1,
                "cal_conf": cal_conf,
                "cal_pred": cal_pred,
                "cal_correct": cal_correct.astype(bool),
                "temperature": T,
                "fit": fit,
                "out_csv": out_csv_of[mkey],
            }
            emit(f"  {MODEL_LABEL[mkey]:<45} T = {T:.6f}  (NLL {fit['nll_before']:.6f} -> "
                 f"{fit['nll_after']:.6f}; n_iter={fit['n_iter']}, converged={fit['converged']})")
        emit("  temperature scaling is argmax-invariant: hard predictions at 0.5 unchanged.")

        # temperature JSON (both models)
        temp_payload = {
            "milestone": "8C",
            "description": "Development-stage temperature scaling of the two frozen 7F models "
                           "(GroupDRO primary candidate + center-stratified ERM matched control) "
                           "on their FULL ood_val logits from Milestone 8B. Weights unchanged; "
                           "checkpoints untouched; ood_test never read. Fit+evaluated on the same "
                           "ood_val => development reliability audit, NOT an independent-test "
                           "calibration claim. Development result, NOT final.",
            "fit_on": "FULL ood_val logits (development-stage OOD calibration only)",
            "argmax_invariant": True,
            "models": {
                mkey: {
                    "name": MODEL_LABEL[mkey],
                    "temperature": cal_bundle[mkey]["temperature"],
                    "nll_before": cal_bundle[mkey]["fit"]["nll_before"],
                    "nll_after": cal_bundle[mkey]["fit"]["nll_after"],
                    "n_iter": cal_bundle[mkey]["fit"]["n_iter"],
                    "converged": cal_bundle[mkey]["fit"]["converged"],
                    "n": int(len(frames[(mkey, "ood_val")])),
                    "calibrated_predictions_csv": out_csv_of[mkey],
                }
                for mkey in MODEL_ORDER
            },
            "min_temperature": TEMP_MIN,
            "max_temperature": TEMP_MAX,
            "temperature_init": TEMP_INIT,
            "note": "DEVELOPMENT temperature scaling on the FULL ood_val split - NOT final "
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
        for mkey in MODEL_ORDER:
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
                {"model": MODEL_LABEL[mkey], "model_key": mkey, "variant": "raw",
                 "temperature": None, "ece": raw_cal["ece"], "brier": raw_cal["brier"],
                 "nll": raw_cal["nll"]})
            cal_comparison_rows.append(
                {"model": MODEL_LABEL[mkey], "model_key": mkey, "variant": "calibrated",
                 "temperature": b["temperature"], "ece": cal_cal["ece"],
                 "brier": cal_cal["brier"], "nll": cal_cal["nll"]})
        write_csv(_resolve(OUT_CAL_COMPARISON_CSV), cal_comparison_rows,
                  header=["model", "model_key", "variant", "temperature", "ece", "brier", "nll"])

        cal_helped = {}
        for mkey in MODEL_ORDER:
            m = cal_metrics_by_model[mkey]
            cal_helped[mkey] = m["calibrated"]["ece"] < m["raw"]["ece"] - 1e-9
        calibration_helped_both = all(cal_helped[mkey] for mkey in MODEL_ORDER)

        emit("")
        emit("3. CALIBRATION RELIABILITY on FULL ood_val (development)  [tumor = positive]")
        emit("-" * 72)
        emit(f"   {'model':<45}{'variant':<12}{'ECE':>9}{'Brier':>9}{'NLL':>9}")
        for r in cal_comparison_rows:
            emit(f"   {r['model']:<45}{r['variant']:<12}"
                 f"{_fmt(r['ece']):>9}{_fmt(r['brier']):>9}{_fmt(r['nll']):>9}")
        for mkey in MODEL_ORDER:
            m = cal_metrics_by_model[mkey]
            emit(f"   {MODEL_LABEL[mkey]}: ECE {m['raw']['ece']:.4f} -> {m['calibrated']['ece']:.4f} "
                 f"({'improved' if cal_helped[mkey] else 'no improvement'}); classification "
                 f"unchanged raw->cal (argmax-invariant): acc {_fmt(m['cls_raw']['accuracy'])}, "
                 f"AUROC {_fmt(m['cls_raw']['auroc'])}, sens {_fmt(m['cls_raw']['sensitivity'])}, "
                 f"FN {m['cls_raw']['fn']}")

        # --------------------------------------------------------------- #
        # 4. High-confidence false-negative audit (4 variants)            #
        # --------------------------------------------------------------- #
        variant_order = [
            ("groupdro", "raw"), ("groupdro", "calibrated"),
            ("centerstrat_erm", "raw"), ("centerstrat_erm", "calibrated"),
        ]
        hiconf_rows: list[dict] = []
        hiconf_by_variant: dict[tuple[str, str], dict[float, dict]] = {}
        for mkey, variant in variant_order:
            b = cal_bundle[mkey]
            if variant == "raw":
                audit = _hiconf_fn_audit(b["y_ood"], b["raw_pred"], b["raw_correct"],
                                         b["raw_conf"], HICONF_THRESHOLDS)
            else:
                audit = _hiconf_fn_audit(b["y_ood"], b["cal_pred"], b["cal_correct"],
                                         b["cal_conf"], HICONF_THRESHOLDS)
            hiconf_by_variant[(mkey, variant)] = {a["confidence_threshold"]: a for a in audit}
            for a in audit:
                hiconf_rows.append({
                    "model": MODEL_LABEL[mkey],
                    "model_key": mkey,
                    "variant": variant,
                    "variant_key": f"{mkey}_{variant}",
                    **a,
                })
        write_csv(_resolve(OUT_HICONF_FN_CSV), hiconf_rows, header=[
            "model", "model_key", "variant", "variant_key", "confidence_threshold",
            "n_high_confidence", "high_confidence_errors", "high_confidence_false_negatives",
            "high_confidence_false_negative_rate", "total_false_negatives",
            "fraction_of_all_false_negatives_high_confidence",
        ])

        emit("")
        emit("4. HIGH-CONFIDENCE FALSE-NEGATIVE (missed-tumor) AUDIT on FULL ood_val")
        emit("-" * 72)
        emit("   confidently-missed tumors = label==1, pred==0, confidence >= t")
        emit(f"   {'model':<45}{'variant':<12}{'t':>6}{'n_high':>9}{'hc_err':>8}"
             f"{'hc_FN':>8}{'hc_FN_rate':>12}{'FN_total':>10}")
        for mkey, variant in variant_order:
            for t in HICONF_THRESHOLDS:
                a = hiconf_by_variant[(mkey, variant)][t]
                emit(f"   {MODEL_LABEL[mkey]:<45}{variant:<12}{t:>6.2f}"
                     f"{a['n_high_confidence']:>9}{a['high_confidence_errors']:>8}"
                     f"{a['high_confidence_false_negatives']:>8}"
                     f"{_fmt(a['high_confidence_false_negative_rate']):>12}"
                     f"{a['total_false_negatives']:>10}")

        # --------------------------------------------------------------- #
        # 5. Reliability summary (4 rows)                                 #
        # --------------------------------------------------------------- #
        def _hc(mkey, variant, t, field):
            return hiconf_by_variant[(mkey, variant)][t][field]

        reliability_rows: list[dict] = []
        for mkey, variant in variant_order:
            m = cal_metrics_by_model[mkey]
            # classification is argmax-invariant -> identical raw vs calibrated; use the
            # matching variant's dict so the table is explicit.
            cls = m["cls_raw"] if variant == "raw" else m["cls_cal"]
            cal = m["raw"] if variant == "raw" else m["calibrated"]
            note = (
                "Ranking/argmax metrics (accuracy/AUROC/AUPRC/sensitivity/specificity/FN) are "
                "unchanged by temperature scaling; calibration mainly improves ECE/Brier/NLL and "
                "reduces high-confidence missed tumors. Development-stage; ood_test locked."
            )
            reliability_rows.append({
                "model": MODEL_LABEL[mkey],
                "variant": variant,
                "ood_val_n": int(cls["n"]),
                "accuracy": cls["accuracy"],
                "AUROC": cls["auroc"],
                "AUPRC": cls["auprc"],
                "sensitivity": cls["sensitivity"],
                "specificity": cls["specificity"],
                "false_negatives": cls["fn"],
                "ECE": cal["ece"],
                "Brier": cal["brier"],
                "NLL": cal["nll"],
                "high_conf_FN_ge_0p90": _hc(mkey, variant, 0.90, "high_confidence_false_negatives"),
                "high_conf_FN_ge_0p95": _hc(mkey, variant, 0.95, "high_confidence_false_negatives"),
                "high_conf_FN_ge_0p99": _hc(mkey, variant, 0.99, "high_confidence_false_negatives"),
                "interpretation_note": note,
            })
        write_csv(_resolve(OUT_RELIABILITY_SUMMARY_CSV), reliability_rows, header=[
            "model", "variant", "ood_val_n", "accuracy", "AUROC", "AUPRC", "sensitivity",
            "specificity", "false_negatives", "ECE", "Brier", "NLL",
            "high_conf_FN_ge_0p90", "high_conf_FN_ge_0p95", "high_conf_FN_ge_0p99",
            "interpretation_note",
        ])

        emit("")
        emit("5. RELIABILITY SUMMARY (FULL ood_val; development)")
        emit("-" * 72)
        emit(f"   {'model':<45}{'variant':<12}{'AUROC':>8}{'AUPRC':>8}{'sens':>7}"
             f"{'spec':>7}{'ECE':>8}{'Brier':>8}{'NLL':>8}")
        for r in reliability_rows:
            emit(f"   {r['model']:<45}{r['variant']:<12}{_fmt(r['AUROC']):>8}{_fmt(r['AUPRC']):>8}"
                 f"{_fmt(r['sensitivity']):>7}{_fmt(r['specificity']):>7}{_fmt(r['ECE']):>8}"
                 f"{_fmt(r['Brier']):>8}{_fmt(r['NLL']):>8}")

        # --------------------------------------------------------------- #
        # 6. Candidate operating-point passthrough (add calibration note)  #
        # --------------------------------------------------------------- #
        ops_df = pd.read_csv(_resolve(EXP08B_CANDIDATE_OPS_CSV))
        ops_df["calibration_affects_thresholds"] = "no_for_raw_id_val_thresholds"
        ops_df["note"] = (
            "candidate thresholds selected on id_val; not final; calibration in 8C is ood_val "
            "development-stage and does not commit a threshold."
        )
        passthrough_path = _resolve(OUT_CANDIDATE_OP_PASSTHROUGH_CSV)
        ensure_dir(passthrough_path.parent)
        ops_df.to_csv(passthrough_path, index=False)
        emit("")
        emit("6. CANDIDATE OPERATING-POINT PASSTHROUGH (8B id_val-selected; NOT committed)")
        emit("-" * 72)
        emit(f"   read : {EXP08B_CANDIDATE_OPS_CSV} ({len(ops_df)} rows)")
        emit(f"   wrote: {OUT_CANDIDATE_OP_PASSTHROUGH_CSV} (+calibration_affects_thresholds, +note)")
        emit("   candidate thresholds are id_val-selected raw-score thresholds; calibration is")
        emit("   argmax/rank-invariant and commits NO threshold.")

        # --------------------------------------------------------------- #
        # 7. GroupDRO-vs-matched-ERM verdict on full-development metrics    #
        # --------------------------------------------------------------- #
        g = cal_metrics_by_model["groupdro"]
        e = cal_metrics_by_model["centerstrat_erm"]
        g_auroc = g["cls_raw"]["auroc"]
        e_auroc = e["cls_raw"]["auroc"]
        g_auprc = g["cls_raw"]["auprc"]
        e_auprc = e["cls_raw"]["auprc"]
        gdro_ranking_better = (g_auroc is not None and e_auroc is not None and g_auroc > e_auroc)

        # reliability edge: GroupDRO has <= high-confidence FNs at every threshold, both variants
        gdro_hcfn_le = True
        for variant in ("raw", "calibrated"):
            for t in HICONF_THRESHOLDS:
                gv = hiconf_by_variant[("groupdro", variant)][t]["high_confidence_false_negatives"]
                ev = hiconf_by_variant[("centerstrat_erm", variant)][t]["high_confidence_false_negatives"]
                if gv > ev:
                    gdro_hcfn_le = False
        # calibrated-ECE edge (development reliability): GroupDRO better calibrated
        gdro_better_calibrated = (
            g["calibrated"]["ece"] <= e["calibrated"]["ece"] + 1e-12
            and g["raw"]["ece"] <= e["raw"]["ece"] + 1e-12
        )
        groupdro_remains_better = bool(gdro_ranking_better and gdro_hcfn_le)

        emit("")
        emit("7. CONTROLLED VERDICT: 7F GroupDRO vs 7F center-stratified ERM (FULL dev; after 8C)")
        emit("-" * 72)
        emit(f"   ood_val AUROC : GroupDRO {_fmt(g_auroc)}  ERM {_fmt(e_auroc)}  "
             f"(delta {_fmt((g_auroc - e_auroc) if (g_auroc is not None and e_auroc is not None) else None)})")
        emit(f"   ood_val AUPRC : GroupDRO {_fmt(g_auprc)}  ERM {_fmt(e_auprc)}  "
             f"(delta {_fmt((g_auprc - e_auprc) if (g_auprc is not None and e_auprc is not None) else None)})")
        emit(f"   ood_val ECE (raw): GroupDRO {_fmt(g['raw']['ece'])}  ERM {_fmt(e['raw']['ece'])}")
        emit(f"   ood_val ECE (cal): GroupDRO {_fmt(g['calibrated']['ece'])}  ERM {_fmt(e['calibrated']['ece'])}")
        emit(f"   GroupDRO <= ERM high-confidence FN at every threshold/variant : {gdro_hcfn_le}")
        emit(f"   GroupDRO better-calibrated (raw AND calibrated ECE)           : {gdro_better_calibrated}")
        emit("")
        emit(f"   DOES GROUPDRO REMAIN BETTER THAN THE MATCHED ERM AFTER 8C? "
             f"{'YES' if groupdro_remains_better else 'NO'}")
        emit("   (development-stage; full ood_val; NOT final performance; no committed threshold)")

        recommendation_primary = (
            "RETAIN 7F GroupDRO as the primary candidate (with development-stage temperature-scaled "
            "confidence); matched control remains 7F center-stratified ERM."
            if groupdro_remains_better else
            "GroupDRO-vs-ERM is MIXED after full-development calibration; re-examine before "
            "retaining GroupDRO as the sole primary candidate."
        )

        # --------------------------------------------------------------- #
        # 8. Figures (matplotlib only)                                    #
        # --------------------------------------------------------------- #
        color_variant = {"raw": "tab:orange", "calibrated": "tab:green"}

        # Fig 1: raw vs calibrated calibration metrics (ECE/Brier/NLL), both models
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
        metric_names = ["ECE", "Brier", "NLL"]
        for ax, mkey in zip(axes, MODEL_ORDER):
            m = cal_metrics_by_model[mkey]
            raw_vals = [m["raw"]["ece"], m["raw"]["brier"], m["raw"]["nll"]]
            cal_vals = [m["calibrated"]["ece"], m["calibrated"]["brier"], m["calibrated"]["nll"]]
            x = np.arange(len(metric_names)); w = 0.36
            ax.bar(x - w / 2, raw_vals, w, label="raw", color=color_variant["raw"])
            ax.bar(x + w / 2, cal_vals, w,
                   label=f"calibrated (T={cal_bundle[mkey]['temperature']:.2f})",
                   color=color_variant["calibrated"])
            for xi, (rv, cv) in enumerate(zip(raw_vals, cal_vals)):
                ax.text(xi - w / 2, rv, f"{rv:.3f}", ha="center", va="bottom", fontsize=7)
                ax.text(xi + w / 2, cv, f"{cv:.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(metric_names)
            ax.set_title(MODEL_LABEL[mkey], fontsize=9)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("metric value (lower = better)")
        fig.suptitle("Full ood_val calibration: raw vs calibrated (development, Milestone 8C)")
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_CAL), dpi=150); plt.close(fig)

        # Fig 2: high-confidence FN reduction across thresholds, 4 variants
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        variant_disp = {
            ("groupdro", "raw"): "GroupDRO raw", ("groupdro", "calibrated"): "GroupDRO cal",
            ("centerstrat_erm", "raw"): "ERM raw", ("centerstrat_erm", "calibrated"): "ERM cal",
        }
        variant_color = {
            ("groupdro", "raw"): "tab:orange", ("groupdro", "calibrated"): "tab:green",
            ("centerstrat_erm", "raw"): "tab:blue", ("centerstrat_erm", "calibrated"): "tab:cyan",
        }
        xt = np.arange(len(HICONF_THRESHOLDS)); wv = 0.2
        for i, key in enumerate(variant_order):
            vals = [hiconf_by_variant[key][t]["high_confidence_false_negatives"] for t in HICONF_THRESHOLDS]
            ax.bar(xt + (i - 1.5) * wv, vals, wv, label=variant_disp[key], color=variant_color[key])
        ax.set_xticks(xt); ax.set_xticklabels([f">= {t:.2f}" for t in HICONF_THRESHOLDS])
        ax.set_xlabel("confidence threshold")
        ax.set_ylabel("high-confidence missed tumors (count)")
        ax.set_title("High-confidence false negatives - full ood_val: raw vs calibrated (8C)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_HICONF_FN), dpi=150); plt.close(fig)

        # Fig 3: ECE/Brier/NLL GroupDRO vs ERM (calibrated), grouped
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        metrics3 = ["ECE", "Brier", "NLL"]
        gdro_cal = [g["calibrated"]["ece"], g["calibrated"]["brier"], g["calibrated"]["nll"]]
        erm_cal = [e["calibrated"]["ece"], e["calibrated"]["brier"], e["calibrated"]["nll"]]
        x = np.arange(len(metrics3)); w = 0.36
        ax.bar(x - w / 2, gdro_cal, w, label="7F GroupDRO (calibrated)", color="tab:red")
        ax.bar(x + w / 2, erm_cal, w, label="7F center-strat ERM (calibrated)", color="tab:blue")
        for xi, (gv, ev) in enumerate(zip(gdro_cal, erm_cal)):
            ax.text(xi - w / 2, gv, f"{gv:.3f}", ha="center", va="bottom", fontsize=7)
            ax.text(xi + w / 2, ev, f"{ev:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(metrics3)
        ax.set_ylabel("metric value (lower = better)")
        ax.set_title("Calibrated ood_val reliability: GroupDRO vs matched ERM (full dev, 8C)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_ECE_BRIER_NLL), dpi=150); plt.close(fig)

        figure_paths = [OUT_FIG_CAL, OUT_FIG_HICONF_FN, OUT_FIG_ECE_BRIER_NLL]

        # --------------------------------------------------------------- #
        # 9. Calibration-metrics JSON payload                             #
        # --------------------------------------------------------------- #
        cal_metrics_payload = {
            "milestone": "8C",
            "description": "Full-development calibration + reliability audit of the two frozen 7F "
                           "models (GroupDRO primary candidate + center-stratified ERM matched "
                           "control) on the FULL ood_val from Milestone 8B. CSV/logit-only; no "
                           "training/inference/weight-update; ood_test never read. Calibration "
                           "fit+evaluated on the same ood_val => development reliability audit, NOT "
                           "an independent-test calibration claim (R7 open).",
            "positive_class": 1,
            "n_ece_bins": N_ECE_BINS,
            "fair_comparison": "7F center-stratified ERM vs 7F GroupDRO - same sound 7F-1 cache",
            "ood_val_n": EXPECTED_N["ood_val"],
            "models": {
                mkey: {
                    "name": MODEL_LABEL[mkey],
                    "temperature": cal_bundle[mkey]["temperature"],
                    "calibration": {"raw": cal_metrics_by_model[mkey]["raw"],
                                    "calibrated": cal_metrics_by_model[mkey]["calibrated"]},
                    "classification": {
                        "raw": {k: cal_metrics_by_model[mkey]["cls_raw"][k]
                                for k in ["n", "accuracy", "auroc", "auprc", "sensitivity",
                                          "specificity", "fn", "fp", "tn", "tp"]},
                        "calibrated": {k: cal_metrics_by_model[mkey]["cls_cal"][k]
                                       for k in ["n", "accuracy", "auroc", "auprc", "sensitivity",
                                                 "specificity", "fn", "fp", "tn", "tp"]},
                    },
                }
                for mkey in MODEL_ORDER
            },
            "high_confidence_false_negatives": hiconf_rows,
            "reliability_summary": reliability_rows,
            "interpretation": {
                "calibration_helped_groupdro": bool(cal_helped["groupdro"]),
                "calibration_helped_centerstrat_erm": bool(cal_helped["centerstrat_erm"]),
                "calibration_helped_both": bool(calibration_helped_both),
                "groupdro_ood_auroc": g_auroc,
                "centerstrat_erm_ood_auroc": e_auroc,
                "groupdro_ranking_better_than_erm": bool(gdro_ranking_better),
                "groupdro_le_erm_high_confidence_fn_all": bool(gdro_hcfn_le),
                "groupdro_better_calibrated": bool(gdro_better_calibrated),
                "groupdro_remains_better_after_8c": groupdro_remains_better,
                "argmax_invariant": True,
            },
            "note": "DEVELOPMENT analysis on the FULL ood_val split - NOT final performance. "
                    "Calibration is development-stage (fit+evaluated on the same ood_val); NOT an "
                    "independent-test calibration claim. No committed clinical/paper threshold. "
                    "Fair comparison = 7F ERM vs 7F GroupDRO; old plain/7A/7D/7E context-only "
                    "(not read here). ood_test never read.",
        }
        cal_metrics_path = _resolve(OUT_CAL_METRICS_JSON)
        ensure_dir(cal_metrics_path.parent)
        write_text(cal_metrics_path, json.dumps(cal_metrics_payload, indent=2) + "\n")

        # --------------------------------------------------------------- #
        # Conclusion                                                      #
        # --------------------------------------------------------------- #
        emit("")
        emit("CONCLUSION (development-stage; FULL ood_val)")
        emit("-" * 72)
        emit(f"* Confidence reliability: calibration "
             f"{'IMPROVED both' if calibration_helped_both else 'did NOT uniformly improve'} "
             f"7F models on full ood_val "
             f"(GroupDRO ECE {g['raw']['ece']:.4f} -> {g['calibrated']['ece']:.4f}; "
             f"ERM ECE {e['raw']['ece']:.4f} -> {e['calibrated']['ece']:.4f}).")
        emit("* Argmax-invariance: temperature scaling does NOT change the hard prediction at the "
             "0.5 argmax threshold; accuracy/AUROC/AUPRC/sensitivity/specificity are unchanged.")
        if groupdro_remains_better:
            emit(f"* GroupDRO REMAINS BETTER than its matched center-stratified ERM control after "
                 f"full-development calibration: higher ood_val AUROC ({_fmt(g_auroc)} vs "
                 f"{_fmt(e_auroc)}) and <= high-confidence missed tumors at every threshold/variant.")
        else:
            emit(f"* GroupDRO vs matched ERM after full-development calibration is MIXED "
                 f"(ranking better={gdro_ranking_better}; hc-FN <= at all points={gdro_hcfn_le}).")
        emit("* Calibration is DEVELOPMENT-stage (fit + evaluated on the same ood_val) - this is a "
             "reliability audit, NOT an independent-test calibration claim (reviewer risk R7 open).")
        emit("* NO final clinical/paper threshold is committed; candidate operating points remain "
             "id_val-selected and non-committed.")
        emit("* Final performance: NONE of these numbers are final performance. ood_test (center 2, "
             "HF test split) is LOCKED and was NEVER read.")
        emit(f"* Recommendation: {recommendation_primary}")

        # --------------------------------------------------------------- #
        # Verify read-only inputs were NOT modified                       #
        # --------------------------------------------------------------- #
        for name in fp_targets:
            post = resolved_all_in[name].stat()
            assert post.st_mtime_ns == pre_mtime[name], f"input modified (mtime): {name}"
            assert post.st_size == pre_size[name], f"input modified (size): {name}"

        # --------------------------------------------------------------- #
        # Output-existence assertions                                     #
        # --------------------------------------------------------------- #
        out_tables = [OUT_CAL_COMPARISON_CSV, OUT_HICONF_FN_CSV, OUT_RELIABILITY_SUMMARY_CSV,
                      OUT_CANDIDATE_OP_PASSTHROUGH_CSV]
        for p in (OUT_GDRO_CAL_PRED_CSV, OUT_ERM_CAL_PRED_CSV):
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"calibrated pred CSV missing/empty: {p}"
        for p in out_tables:
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"table missing/empty: {p}"
        for p in figure_paths:
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"figure missing/empty: {p}"
        assert temp_path.exists() and temp_path.stat().st_size > 0, "temperature JSON missing/empty"
        assert cal_metrics_path.exists() and cal_metrics_path.stat().st_size > 0, "metrics JSON missing/empty"

        emit("")
        emit("OUTPUT PATHS")
        emit("-" * 72)
        emit("calibrated predictions:")
        emit(f"  {OUT_GDRO_CAL_PRED_CSV}")
        emit(f"  {OUT_ERM_CAL_PRED_CSV}")
        emit("tables:")
        for p in out_tables:
            emit(f"  {p}")
        emit("figures:")
        for p in figure_paths:
            emit(f"  {p}")
        emit(f"temperature JSON        : {OUT_TEMP_JSON}")
        emit(f"calibration metrics JSON: {OUT_CAL_METRICS_JSON}")
        emit(f"report                  : {OUT_REPORT_TXT}")
        emit("ood_test usage: NONE")

    except Exception as exc:
        ok = False
        emit("")
        emit(f"ERROR: {type(exc).__name__}: {exc}")
        import traceback
        emit(traceback.format_exc())

    emit("")
    emit("=" * 80)
    emit(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    emit("=" * 80)

    write_text(_resolve(OUT_REPORT_TXT), "\n".join(lines) + "\n")
    print(f"\nSaved report -> {_resolve(OUT_REPORT_TXT)}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
