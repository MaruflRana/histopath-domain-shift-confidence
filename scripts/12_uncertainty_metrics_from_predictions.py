"""Milestone 5B / Script 12: deterministic uncertainty metrics from saved predictions.

Computes uncertainty / error-detection metrics **purely from existing prediction CSVs** —
no model inference, no training, no MC-dropout, no ensembling, no selective-prediction
threshold selection, and ``ood_test`` is never read. Every number here is a closed-form
function of predictions exported in Milestones 4B (``id_val``) and 5A (``ood_val`` raw +
calibrated), so it is exactly reproducible from those files.

For each of three dataset variants:
  * ``id_val``            -> results/predictions/exp04a_dev_train_resnet18/id_val_cached_predictions.csv
  * ``ood_val_raw``       -> results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_raw_predictions.csv
  * ``ood_val_calibrated``-> results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_calibrated_predictions.csv

the script: loads the CSV, validates the required columns + probabilities, appends the
uncertainty score columns (confidence, uncertainty_confidence, entropy, margin,
uncertainty_margin, energy), computes error-detection / high-confidence-error /
high-confidence-false-negative / risk-coverage metrics, and writes enriched CSVs, a
metrics JSON, flat metric/risk-coverage/high-confidence tables, and a text report.

NOT a final result — development analysis on capped balanced subsets.
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup; kept for repo convention)

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics_uncertainty import (
    compute_error_detection_metrics,
    compute_high_confidence_error_summary,
    compute_high_confidence_false_negative_summary,
    compute_risk_coverage_curve,
    energy_score_from_logits,
    max_confidence,
    predictive_entropy,
    probability_margin,
    uncertainty_from_confidence,
    uncertainty_from_margin,
)
from utils.io import ensure_dir, ensure_results_tree, project_root, write_csv, write_text

REQUIRED_COLUMNS = [
    "split", "label", "logit_0", "logit_1", "prob_0", "prob_1", "pred_label", "correct",
]

# Uncertainty score columns evaluated for error detection / risk-coverage. Each entry:
# (column, higher_means_more_uncertain). `confidence` and `margin` are certainty scores
# (kept for the enriched CSV) and are NOT in this list.
UNCERTAINTY_COLS = [
    ("uncertainty_confidence", True),
    ("entropy", True),
    ("uncertainty_margin", True),
    ("energy", True),
]

CONF_THRESHOLDS = [0.90, 0.95, 0.99]
COVERAGE_LEVELS = [1.0, 0.95, 0.90, 0.80, 0.70, 0.50]

# variant -> (relative CSV path, expected row count)
VARIANTS = [
    (
        "id_val",
        "results/predictions/exp04a_dev_train_resnet18/id_val_cached_predictions.csv",
        500,
    ),
    (
        "ood_val_raw",
        "results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_raw_predictions.csv",
        2000,
    ),
    (
        "ood_val_calibrated",
        "results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_calibrated_predictions.csv",
        2000,
    ),
]

OUT_PRED_DIR = "results/predictions/exp05b_uncertainty"
OUT_METRICS_DIR = "results/metrics/exp05b_uncertainty"
ENRICHED_NAME = {
    "id_val": "id_val_uncertainty_scores.csv",
    "ood_val_raw": "ood_val_raw_uncertainty_scores.csv",
    "ood_val_calibrated": "ood_val_calibrated_uncertainty_scores.csv",
}

_PROB_TOL = 1e-6
_SUM_TOL = 1e-5


def _resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else project_root() / p


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main() -> int:
    root = project_root()
    ensure_results_tree()

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    pred_dir = _resolve(OUT_PRED_DIR)
    metrics_dir = _resolve(OUT_METRICS_DIR)
    metrics_json = metrics_dir / "uncertainty_metrics.json"
    metrics_csv = _resolve("results/tables/exp05b_uncertainty_metrics.csv")
    risk_csv = _resolve("results/tables/exp05b_risk_coverage.csv")
    hce_csv = _resolve("results/tables/exp05b_high_confidence_errors.csv")
    report_path = _resolve("results/logs/exp05b_uncertainty_metrics_report.txt")

    emit("=" * 72)
    emit("Milestone 5B — deterministic uncertainty metrics from saved predictions (script 12)")
    emit("scope: CSV-only. No inference, no training, no MC-dropout, no ensemble,")
    emit("       no selective thresholds. ood_test is LOCKED and never read.")
    emit("=" * 72)

    ok = True
    all_metrics: dict = {}
    metrics_rows: list[dict] = []
    risk_rows: list[dict] = []
    hce_rows: list[dict] = []

    try:
        # Guard: none of the inputs may be an ood_test artifact.
        for _, rel, _n in VARIANTS:
            assert "ood_test" not in rel.lower(), f"ood_test input is forbidden: {rel}"

        ensure_dir(pred_dir)
        ensure_dir(metrics_dir)

        for variant, rel_path, expected_n in VARIANTS:
            in_path = _resolve(rel_path)
            emit("")
            emit("-" * 72)
            emit(f"[{variant}]")
            emit(f"  input : {in_path}")

            assert in_path.exists(), f"input CSV not found: {in_path}"
            df = pd.read_csv(in_path)

            # --- column + count validation ------------------------------------- #
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            assert not missing, f"{variant}: missing required columns {missing}"
            n = int(len(df))
            emit(f"  rows  : {n}  (expected {expected_n})")
            assert n == expected_n, f"{variant}: n {n} != expected {expected_n}"

            # --- probability / logit validation -------------------------------- #
            p0 = df["prob_0"].to_numpy(dtype=float)
            p1 = df["prob_1"].to_numpy(dtype=float)
            l0 = df["logit_0"].to_numpy(dtype=float)
            l1 = df["logit_1"].to_numpy(dtype=float)
            assert np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)), f"{variant}: non-finite probs"
            assert np.all(np.isfinite(l0)) and np.all(np.isfinite(l1)), f"{variant}: non-finite logits"
            assert np.all(p0 >= -_PROB_TOL) and np.all(p0 <= 1 + _PROB_TOL), f"{variant}: prob_0 out of [0,1]"
            assert np.all(p1 >= -_PROB_TOL) and np.all(p1 <= 1 + _PROB_TOL), f"{variant}: prob_1 out of [0,1]"
            assert np.all(np.abs((p0 + p1) - 1.0) < _SUM_TOL), f"{variant}: prob_0+prob_1 != 1"

            # --- error / false-negative counts --------------------------------- #
            correct = df["correct"].to_numpy().astype(float) != 0.0
            is_error = ~correct
            is_fn = (df["label"].to_numpy() == 1) & (df["pred_label"].to_numpy() == 0)
            n_error = int(np.sum(is_error))
            n_fn = int(np.sum(is_fn))
            emit(f"  errors: {n_error}  (error_rate={n_error / n:.6f})")
            emit(f"  false negatives (label=1,pred=0): {n_fn}")

            # --- append uncertainty score columns (do NOT touch input files) ---- #
            enriched = df.copy()
            enriched["confidence"] = max_confidence(p0, p1)
            enriched["uncertainty_confidence"] = uncertainty_from_confidence(p0, p1)
            enriched["entropy"] = predictive_entropy(p0, p1)
            enriched["margin"] = probability_margin(p0, p1)
            enriched["uncertainty_margin"] = uncertainty_from_margin(p0, p1)
            enriched["energy"] = energy_score_from_logits(l0, l1)

            out_csv = pred_dir / ENRICHED_NAME[variant]
            enriched.to_csv(out_csv, index=False)
            emit(f"  enriched CSV -> {out_csv}")

            # --- error-detection metrics per uncertainty score ------------------ #
            emit("  error-detection AUROC (error = positive):")
            variant_scores: dict = {}
            for col, higher in UNCERTAINTY_COLS:
                ed = compute_error_detection_metrics(enriched, col, higher_means_more_uncertain=higher)
                variant_scores[col] = ed
                emit(
                    f"    {col:<24} auroc={_fmt(ed['error_detection_auroc'])}  "
                    f"mean(correct)={_fmt(ed['mean_uncertainty_correct'])}  "
                    f"mean(error)={_fmt(ed['mean_uncertainty_error'])}"
                )
                for w in ed["warnings"]:
                    emit(f"      WARN: {w}")
                metrics_rows.append({
                    "variant": variant,
                    "score_col": col,
                    "n": ed["n"],
                    "n_error": ed["n_error"],
                    "n_correct": ed["n_correct"],
                    "error_rate": ed["error_rate"],
                    "error_detection_auroc": ed["error_detection_auroc"],
                    "mean_uncertainty_correct": ed["mean_uncertainty_correct"],
                    "mean_uncertainty_error": ed["mean_uncertainty_error"],
                })

            # --- high-confidence error + false-negative summaries --------------- #
            hce = compute_high_confidence_error_summary(enriched, "confidence", CONF_THRESHOLDS)
            hcfn = compute_high_confidence_false_negative_summary(enriched, "confidence", CONF_THRESHOLDS)

            emit("  high-confidence errors / false negatives (by confidence >= t):")
            hcfn_by_t = {r["threshold"]: r for r in hcfn}
            for r in hce:
                fn_r = hcfn_by_t[r["threshold"]]
                emit(
                    f"    t={r['threshold']:.2f}  n_high={r['n_high_conf']:<5} "
                    f"err={r['n_high_conf_error']:<4}(rate={_fmt(r['high_conf_error_rate'])})  "
                    f"fn={fn_r['n_high_conf_false_negative']:<4}"
                    f"(rate={_fmt(fn_r['high_conf_false_negative_rate'])}, "
                    f"frac_of_all_fn={_fmt(fn_r['fraction_of_fn_in_high_conf'])})"
                )
                hce_rows.append({
                    "variant": variant,
                    "threshold": r["threshold"],
                    "n_total": r["n_total"],
                    "n_high_conf": r["n_high_conf"],
                    "high_conf_coverage": r["high_conf_coverage"],
                    "n_high_conf_error": r["n_high_conf_error"],
                    "high_conf_error_rate": r["high_conf_error_rate"],
                    "n_error_total": r["n_error_total"],
                    "n_high_conf_false_negative": fn_r["n_high_conf_false_negative"],
                    "high_conf_false_negative_rate": fn_r["high_conf_false_negative_rate"],
                    "n_false_negative_total": fn_r["n_false_negative_total"],
                    "fraction_of_fn_in_high_conf": fn_r["fraction_of_fn_in_high_conf"],
                })

            # --- risk-coverage curve (reporting only; primary score = entropy) -- #
            emit("  risk-coverage (rank by entropy, most-confident retained first):")
            rc_by_col: dict = {}
            for col, _higher in UNCERTAINTY_COLS:
                rc = compute_risk_coverage_curve(enriched, col, COVERAGE_LEVELS)
                rc_by_col[col] = rc
                for pt in rc["points"]:
                    risk_rows.append({
                        "variant": variant,
                        "uncertainty_col": col,
                        "aurc": rc["aurc"],
                        "coverage": pt["coverage"],
                        "n_selected": pt["n_selected"],
                        "n_error_selected": pt["n_error_selected"],
                        "risk": pt["risk"],
                    })
            entropy_rc = rc_by_col["entropy"]
            emit(f"    entropy AURC = {_fmt(entropy_rc['aurc'])}")
            for pt in entropy_rc["points"]:
                emit(
                    f"      coverage={pt['coverage']:.2f}  n={pt['n_selected']:<5} "
                    f"err={pt['n_error_selected']:<4} risk={_fmt(pt['risk'])}"
                )

            all_metrics[variant] = {
                "input_csv": str(in_path),
                "enriched_csv": str(out_csv),
                "n": n,
                "n_error": n_error,
                "error_rate": n_error / n,
                "n_false_negative": n_fn,
                "error_detection": variant_scores,
                "high_confidence_errors": hce,
                "high_confidence_false_negatives": hcfn,
                "risk_coverage": rc_by_col,
            }

        # --- write metrics JSON + flat tables ---------------------------------- #
        payload = {
            "run_name": "exp05b_uncertainty",
            "milestone": "5B",
            "positive_class": 1,
            "confidence_definition": "max(prob_0, prob_1)",
            "error_definition": "correct == False",
            "false_negative_definition": "label == 1 and pred_label == 0",
            "uncertainty_score_cols": [c for c, _ in UNCERTAINTY_COLS],
            "confidence_thresholds": CONF_THRESHOLDS,
            "coverage_levels": COVERAGE_LEVELS,
            "note": "Milestone 5B: uncertainty metrics computed ONLY from saved prediction "
                    "CSVs (no inference/training/MC-dropout/ensemble/selective thresholds). "
                    "ood_test never touched. DEVELOPMENT analysis on capped balanced "
                    "subsets, NOT final performance.",
            "variants": all_metrics,
        }
        write_text(metrics_json, json.dumps(payload, indent=2) + "\n")

        metrics_header = [
            "variant", "score_col", "n", "n_error", "n_correct", "error_rate",
            "error_detection_auroc", "mean_uncertainty_correct", "mean_uncertainty_error",
        ]
        write_csv(metrics_csv, metrics_rows, header=metrics_header)

        risk_header = [
            "variant", "uncertainty_col", "aurc", "coverage",
            "n_selected", "n_error_selected", "risk",
        ]
        write_csv(risk_csv, risk_rows, header=risk_header)

        hce_header = [
            "variant", "threshold", "n_total", "n_high_conf", "high_conf_coverage",
            "n_high_conf_error", "high_conf_error_rate", "n_error_total",
            "n_high_conf_false_negative", "high_conf_false_negative_rate",
            "n_false_negative_total", "fraction_of_fn_in_high_conf",
        ]
        write_csv(hce_csv, hce_rows, header=hce_header)

        # --- final assertions -------------------------------------------------- #
        assert metrics_json.exists(), "metrics JSON missing"
        assert metrics_csv.exists(), "metrics CSV missing"
        assert risk_csv.exists(), "risk-coverage CSV missing"
        assert hce_csv.exists(), "high-confidence errors CSV missing"
        for variant, _rel, _n in VARIANTS:
            assert (pred_dir / ENRICHED_NAME[variant]).exists(), f"{variant}: enriched CSV missing"

        emit("")
        emit("--- output paths ---")
        emit(f"enriched predictions dir : {pred_dir}")
        for variant, _rel, _n in VARIANTS:
            emit(f"    {ENRICHED_NAME[variant]}")
        emit(f"metrics JSON             : {metrics_json}")
        emit(f"metrics CSV              : {metrics_csv}")
        emit(f"risk-coverage CSV        : {risk_csv}")
        emit(f"high-confidence CSV      : {hce_csv}")
        emit(f"text report              : {report_path}")

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
