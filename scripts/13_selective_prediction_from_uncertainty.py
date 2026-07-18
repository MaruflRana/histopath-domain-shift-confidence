"""Milestone 5C / Script 13: selective prediction / abstention analysis from 5B CSVs.

Converts the per-example uncertainty scores exported in Milestone 5B into
selective-prediction / abstention policy tables for ``id_val`` and ``ood_val`` (raw +
calibrated). It reads **only** already-saved CSVs — no model inference, no training, no
MC-dropout, no ensembling — and ``ood_test`` is never touched.

For each of three dataset variants:
  * ``id_val``            -> results/predictions/exp05b_uncertainty/id_val_uncertainty_scores.csv
  * ``ood_val_raw``       -> results/predictions/exp05b_uncertainty/ood_val_raw_uncertainty_scores.csv
  * ``ood_val_calibrated``-> results/predictions/exp05b_uncertainty/ood_val_calibrated_uncertainty_scores.csv

it validates the required columns, computes selective-prediction metrics at fixed coverage
levels for every uncertainty score (primary interpretation = ``entropy``), computes the
false-negative-under-abstention breakdown, and writes flat tables + a metrics JSON + a text
report.

Retention = lowest-uncertainty (most-confident) rows; abstention = highest-uncertainty
rows. Because temperature scaling is monotone, ``ood_val`` raw vs calibrated share an
identical entropy ranking, so their risk–coverage rows match — but calibration still moves
where high-confidence false negatives sit (shown in Milestone 5B). DEVELOPMENT analysis on
capped balanced subsets, NOT final clinical performance; no operating threshold is committed.
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup; kept for repo convention)

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics_selective import (
    summarize_false_negatives_under_abstention,
    summarize_selective_prediction,
)
from utils.io import ensure_dir, ensure_results_tree, project_root, write_csv, write_text

REQUIRED_COLUMNS = [
    "split", "label", "prob_0", "prob_1", "pred_label", "correct",
    "confidence", "uncertainty_confidence", "entropy", "margin", "uncertainty_margin", "energy",
]

UNCERTAINTY_COLS = ["uncertainty_confidence", "entropy", "uncertainty_margin", "energy"]
PRIMARY_COL = "entropy"
COVERAGE_LEVELS = [1.00, 0.95, 0.90, 0.80, 0.70, 0.50]

# variant -> (relative CSV path, expected row count)
VARIANTS = [
    ("id_val", "results/predictions/exp05b_uncertainty/id_val_uncertainty_scores.csv", 500),
    ("ood_val_raw", "results/predictions/exp05b_uncertainty/ood_val_raw_uncertainty_scores.csv", 2000),
    ("ood_val_calibrated", "results/predictions/exp05b_uncertainty/ood_val_calibrated_uncertainty_scores.csv", 2000),
]

ALL_SCORES_CSV = "results/tables/exp05c_selective_prediction_all_scores.csv"
ENTROPY_SUMMARY_CSV = "results/tables/exp05c_selective_prediction_entropy_summary.csv"
FN_ABSTENTION_CSV = "results/tables/exp05c_false_negative_abstention_summary.csv"
METRICS_JSON = "results/metrics/exp05c_selective_prediction/selective_prediction_metrics.json"
REPORT_TXT = "results/logs/exp05c_selective_prediction_report.txt"

_ALL_SCORES_HEADER = [
    "variant", "uncertainty_col", "target_coverage", "actual_coverage", "threshold",
    "retained_n", "abstained_n", "retained_errors", "retained_risk", "retained_accuracy",
    "retained_false_negatives", "retained_false_negative_rate_over_retained",
    "retained_false_negative_rate_over_all_tumors", "abstained_errors",
    "abstained_false_negatives", "total_errors", "total_false_negatives", "total_tumors",
]
_FN_HEADER = [
    "variant", "uncertainty_col", "target_coverage", "actual_coverage", "retained_n",
    "abstained_n", "total_false_negatives", "retained_false_negatives",
    "abstained_false_negatives", "fraction_of_fn_abstained",
    "retained_false_negative_rate_over_retained",
    "retained_false_negative_rate_over_all_tumors", "total_tumors",
]


def _resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else project_root() / p


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _print_entropy_summary(emit, variant: str, summary: dict) -> None:
    emit(f"  [{variant}] entropy selective prediction "
         f"(n={summary['n']}, total_errors={summary['total_errors']}, "
         f"total_FN={summary['total_false_negatives']}, total_tumors={summary['total_tumors']}):")
    emit(f"    {'cov':>5} {'thr':>9} {'ret_n':>6} {'abs_n':>6} {'risk':>9} "
         f"{'ret_acc':>9} {'ret_FN':>7} {'abs_FN':>7}")
    for pt in summary["points"]:
        emit(f"    {pt['target_coverage']:>5.2f} {_fmt(pt['threshold']):>9} "
             f"{pt['retained_n']:>6} {pt['abstained_n']:>6} "
             f"{_fmt(pt['retained_risk']):>9} {_fmt(pt['retained_accuracy']):>9} "
             f"{pt['retained_false_negatives']:>7} {pt['abstained_false_negatives']:>7}")


def main() -> int:
    ensure_results_tree()

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    all_scores_csv = _resolve(ALL_SCORES_CSV)
    entropy_summary_csv = _resolve(ENTROPY_SUMMARY_CSV)
    fn_abstention_csv = _resolve(FN_ABSTENTION_CSV)
    metrics_json = _resolve(METRICS_JSON)
    report_path = _resolve(REPORT_TXT)

    emit("=" * 72)
    emit("Milestone 5C — selective prediction / abstention from saved uncertainty CSVs (script 13)")
    emit("scope: CSV-only. No inference, no training, no MC-dropout, no ensemble.")
    emit("       ood_val = development-stage selective-prediction analysis only.")
    emit("       ood_test is LOCKED and never read. No operating threshold committed.")
    emit(f"coverage levels : {COVERAGE_LEVELS}")
    emit(f"uncertainty cols: {UNCERTAINTY_COLS}  (primary = {PRIMARY_COL})")
    emit("=" * 72)

    ok = True
    all_scores_rows: list[dict] = []
    entropy_rows: list[dict] = []
    fn_rows: list[dict] = []
    json_variants: dict = {}

    try:
        for _, rel, _n in VARIANTS:
            assert "ood_test" not in rel.lower(), f"ood_test input is forbidden: {rel}"

        for variant, rel_path, expected_n in VARIANTS:
            in_path = _resolve(rel_path)
            emit("")
            emit("-" * 72)
            emit(f"[{variant}] input : {in_path}")
            assert in_path.exists(), f"input CSV not found: {in_path}"

            df = pd.read_csv(in_path)
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            assert not missing, f"{variant}: missing required columns {missing}"

            n = int(len(df))
            emit(f"  rows : {n}  (expected {expected_n})")
            assert n == expected_n, f"{variant}: n {n} != expected {expected_n}"

            # finiteness of uncertainty scores
            for col in UNCERTAINTY_COLS:
                vals = df[col].to_numpy(dtype=float)
                assert np.all(np.isfinite(vals)), f"{variant}: non-finite values in {col}"

            # baseline error / FN counts
            correct = df["correct"].to_numpy().astype(float) != 0.0
            n_error = int(np.sum(~correct))
            is_fn = (df["label"].to_numpy() == 1) & (df["pred_label"].to_numpy() == 0)
            n_fn = int(np.sum(is_fn))
            n_tumor = int(np.sum(df["label"].to_numpy() == 1))
            emit(f"  baseline: errors={n_error} (rate={n_error / n:.6f})  "
                 f"false_negatives={n_fn}  tumors={n_tumor}")

            # --- selective metrics for every uncertainty column ----------------- #
            per_col_summaries: dict = {}
            for col in UNCERTAINTY_COLS:
                summary = summarize_selective_prediction(df, col, COVERAGE_LEVELS, variant=variant)
                per_col_summaries[col] = summary
                all_scores_rows.extend(summary["points"])

            # --- primary (entropy) summary -------------------------------------- #
            entropy_summary = per_col_summaries[PRIMARY_COL]
            entropy_rows.extend(entropy_summary["points"])

            # --- false-negative-under-abstention (primary = entropy) ------------ #
            fn_summary = summarize_false_negatives_under_abstention(
                df, PRIMARY_COL, COVERAGE_LEVELS, variant=variant
            )
            fn_rows.extend(fn_summary)

            _print_entropy_summary(emit, variant, entropy_summary)

            emit(f"  [{variant}] false negatives under abstention (rank by {PRIMARY_COL}):")
            emit(f"    {'cov':>5} {'ret_FN':>7} {'abs_FN':>7} {'frac_abst':>10} "
                 f"{'ret_FN/ret':>11} {'ret_FN/tum':>11}")
            for r in fn_summary:
                emit(f"    {r['target_coverage']:>5.2f} {r['retained_false_negatives']:>7} "
                     f"{r['abstained_false_negatives']:>7} "
                     f"{_fmt(r['fraction_of_fn_abstained']):>10} "
                     f"{_fmt(r['retained_false_negative_rate_over_retained']):>11} "
                     f"{_fmt(r['retained_false_negative_rate_over_all_tumors']):>11}")

            json_variants[variant] = {
                "input_csv": str(in_path),
                "n": n,
                "baseline_errors": n_error,
                "baseline_false_negatives": n_fn,
                "total_tumors": n_tumor,
                "selective_prediction": {c: per_col_summaries[c] for c in UNCERTAINTY_COLS},
                "false_negative_abstention_entropy": fn_summary,
            }

        # --- write tables + JSON + report ------------------------------------- #
        write_csv(all_scores_csv, all_scores_rows, header=_ALL_SCORES_HEADER)
        write_csv(entropy_summary_csv, entropy_rows, header=_ALL_SCORES_HEADER)
        write_csv(fn_abstention_csv, fn_rows, header=_FN_HEADER)

        payload = {
            "run_name": "exp05c_selective_prediction",
            "milestone": "5C",
            "positive_class": 1,
            "retained_definition": "lowest-uncertainty (most-confident) rows, rank-based",
            "abstained_definition": "highest-uncertainty rows",
            "coverage_definition": "retained_n / total_n",
            "risk_definition": "retained_errors / retained_n",
            "error_definition": "correct == False",
            "false_negative_definition": "label == 1 and pred_label == 0",
            "uncertainty_score_cols": UNCERTAINTY_COLS,
            "primary_uncertainty_col": PRIMARY_COL,
            "coverage_levels": COVERAGE_LEVELS,
            "note": "Milestone 5C: selective-prediction / abstention tables computed ONLY "
                    "from saved Milestone 5B uncertainty CSVs (no inference/training/"
                    "MC-dropout/ensemble). ood_val used for DEVELOPMENT-stage selective "
                    "analysis only; ood_test never touched; no operating threshold "
                    "committed. NOT final clinical performance. Temperature scaling is "
                    "monotone, so ood_val raw vs calibrated share risk-coverage rows; "
                    "calibration still relocates high-confidence false negatives (5B).",
            "variants": json_variants,
        }
        ensure_dir(metrics_json.parent)
        write_text(metrics_json, json.dumps(payload, indent=2) + "\n")

        # --- assertions -------------------------------------------------------- #
        assert all_scores_csv.exists(), "all-scores coverage table missing"
        assert entropy_summary_csv.exists(), "entropy summary table missing"
        assert fn_abstention_csv.exists(), "false-negative abstention table missing"
        assert metrics_json.exists(), "metrics JSON missing"

        emit("")
        emit("--- output paths ---")
        emit(f"all-scores coverage table   : {all_scores_csv}")
        emit(f"entropy summary table       : {entropy_summary_csv}")
        emit(f"FN-abstention table         : {fn_abstention_csv}")
        emit(f"metrics JSON                : {metrics_json}")
        emit(f"text report                 : {report_path}")

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
