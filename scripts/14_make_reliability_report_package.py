"""Milestone 6A / Script 14: reliability reporting package from saved results.

Consolidates the already-accepted outputs of Milestones 4B, 5A, 5B, and 5C into a small
set of publication-style report tables + figures + a text summary. This script is
**CSV/JSON-only**: it reads existing saved artifacts, does **no** model inference, **no**
training, **no** MC-dropout / ensembling, and **never** touches ``ood_test``. It does not
modify any input file.

Inputs (read-only):
  * results/tables/exp04a_dev_train_resnet18_classification_metrics.csv        (4B: id_val)
  * results/metrics/exp05a_calibrate_ood_val_resnet18/calibration_metrics.json (5A: ood_val)
  * results/tables/exp05b_uncertainty_metrics.csv                              (5B)
  * results/tables/exp05b_risk_coverage.csv                                    (5B)
  * results/tables/exp05b_high_confidence_errors.csv                           (5B)
  * results/tables/exp05c_selective_prediction_entropy_summary.csv             (5C)
  * results/tables/exp05c_false_negative_abstention_summary.csv                (5C)

Outputs:
  tables  -> results/tables/report_table_{classification_shift,calibration,
             uncertainty,selective_prediction}.csv
  figures -> results/figures/report_fig_{domain_shift_bar,calibration_metrics,
             uncertainty_error_detection,risk_coverage_entropy,
             false_negative_abstention}.png
  log     -> results/logs/report_reliability_package_summary.txt

DEVELOPMENT analysis on capped balanced subsets — NOT final clinical performance.
``ood_test`` remains LOCKED.
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup; kept for repo convention)

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / non-interactive; no display required
import matplotlib.pyplot as plt
import pandas as pd

from utils.io import ensure_results_tree, project_root, write_csv, write_text

# ---------------------------------------------------------------------------- #
# Input paths (read-only) and output paths                                     #
# ---------------------------------------------------------------------------- #
IN_CLASSIFICATION_CSV = "results/tables/exp04a_dev_train_resnet18_classification_metrics.csv"
IN_CALIBRATION_JSON = "results/metrics/exp05a_calibrate_ood_val_resnet18/calibration_metrics.json"
IN_UNCERTAINTY_CSV = "results/tables/exp05b_uncertainty_metrics.csv"
IN_RISK_COVERAGE_CSV = "results/tables/exp05b_risk_coverage.csv"
IN_HIGH_CONF_CSV = "results/tables/exp05b_high_confidence_errors.csv"
IN_SELECTIVE_CSV = "results/tables/exp05c_selective_prediction_entropy_summary.csv"
IN_FN_ABSTENTION_CSV = "results/tables/exp05c_false_negative_abstention_summary.csv"

INPUT_PATHS = [
    IN_CLASSIFICATION_CSV, IN_CALIBRATION_JSON, IN_UNCERTAINTY_CSV, IN_RISK_COVERAGE_CSV,
    IN_HIGH_CONF_CSV, IN_SELECTIVE_CSV, IN_FN_ABSTENTION_CSV,
]

OUT_TABLE_CLASSIFICATION = "results/tables/report_table_classification_shift.csv"
OUT_TABLE_CALIBRATION = "results/tables/report_table_calibration.csv"
OUT_TABLE_UNCERTAINTY = "results/tables/report_table_uncertainty.csv"
OUT_TABLE_SELECTIVE = "results/tables/report_table_selective_prediction.csv"

OUT_FIG_DOMAIN_SHIFT = "results/figures/report_fig_domain_shift_bar.png"
OUT_FIG_CALIBRATION = "results/figures/report_fig_calibration_metrics.png"
OUT_FIG_UNCERTAINTY = "results/figures/report_fig_uncertainty_error_detection.png"
OUT_FIG_RISK_COVERAGE = "results/figures/report_fig_risk_coverage_entropy.png"
OUT_FIG_FN_ABSTENTION = "results/figures/report_fig_false_negative_abstention.png"

OUT_SUMMARY_TXT = "results/logs/report_reliability_package_summary.txt"

OUTPUT_TABLES = [
    OUT_TABLE_CLASSIFICATION, OUT_TABLE_CALIBRATION,
    OUT_TABLE_UNCERTAINTY, OUT_TABLE_SELECTIVE,
]
OUTPUT_FIGURES = [
    OUT_FIG_DOMAIN_SHIFT, OUT_FIG_CALIBRATION, OUT_FIG_UNCERTAINTY,
    OUT_FIG_RISK_COVERAGE, OUT_FIG_FN_ABSTENTION,
]

# Coverage levels reported in the consolidated selective-prediction table / figures.
REPORT_COVERAGES = [1.00, 0.90, 0.70, 0.50]
PRIMARY_UNCERTAINTY = "entropy"


def _resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else project_root() / p


def _round(v, nd: int = 6):
    """Round floats for CSV output; pass through non-floats / None unchanged."""
    if v is None:
        return None
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return v


def main() -> int:
    ensure_results_tree()

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    emit("=" * 72)
    emit("Milestone 6A — reliability reporting package from saved results (script 14)")
    emit("scope: CSV/JSON-only. No inference, no training, no MC-dropout, no ensemble.")
    emit("       Inputs are read-only; ood_test is LOCKED and never read.")
    emit("       DEVELOPMENT analysis on capped balanced subsets — NOT final performance.")
    emit("=" * 72)

    ok = True

    # Collected key numbers for the summary block.
    summary_nums: dict = {}

    try:
        # -- guard: no ood_test anywhere in the input/output contract ----------- #
        for rel in INPUT_PATHS + OUTPUT_TABLES + OUTPUT_FIGURES + [OUT_SUMMARY_TXT]:
            assert "ood_test" not in rel.lower(), f"ood_test path is forbidden: {rel}"

        # -- verify every input exists before doing anything ------------------- #
        emit("")
        emit("--- inputs (read-only) ---")
        for rel in INPUT_PATHS:
            p = _resolve(rel)
            emit(f"  {'OK ' if p.exists() else 'MISSING'} {p}")
            assert p.exists(), f"required input not found: {p}"

        # ------------------------------------------------------------------ #
        # Load sources                                                       #
        # ------------------------------------------------------------------ #
        clf_df = pd.read_csv(_resolve(IN_CLASSIFICATION_CSV))
        with open(_resolve(IN_CALIBRATION_JSON), "r", encoding="utf-8") as f:
            calib = json.load(f)
        unc_df = pd.read_csv(_resolve(IN_UNCERTAINTY_CSV))
        rc_df = pd.read_csv(_resolve(IN_RISK_COVERAGE_CSV))
        hc_df = pd.read_csv(_resolve(IN_HIGH_CONF_CSV))
        sel_df = pd.read_csv(_resolve(IN_SELECTIVE_CSV))
        fn_df = pd.read_csv(_resolve(IN_FN_ABSTENTION_CSV))

        # id_val classification row (from 4B CSV)
        id_val_clf = clf_df[clf_df["split"] == "id_val"]
        assert len(id_val_clf) == 1, "expected exactly one id_val row in 4B classification CSV"
        id_val_clf = id_val_clf.iloc[0]

        # ood_val classification (raw) lives in the 5A calibration JSON
        ood_raw_clf = calib["classification"]["raw"]
        ood_cal_clf = calib["classification"]["calibrated"]
        calib_raw = calib["calibration"]["raw"]
        calib_cal = calib["calibration"]["calibrated"]
        temperature = float(calib["temperature"])

        # ================================================================== #
        # TABLE 1 — classification under domain shift (id_val vs ood_val_raw)#
        # ================================================================== #
        classification_rows = [
            {
                "variant": "id_val",
                "n": int(id_val_clf["n"]),
                "accuracy": _round(id_val_clf["accuracy"]),
                "auroc": _round(id_val_clf["auroc"]),
                "auprc": _round(id_val_clf["auprc"]),
                "sensitivity": _round(id_val_clf["sensitivity"]),
                "specificity": _round(id_val_clf["specificity"]),
                "f1": _round(id_val_clf["f1"]),
                "fn": int(id_val_clf["fn"]),
            },
            {
                "variant": "ood_val_raw",
                "n": int(ood_raw_clf["n"]),
                "accuracy": _round(ood_raw_clf["accuracy"]),
                "auroc": _round(ood_raw_clf["auroc"]),
                "auprc": _round(ood_raw_clf["auprc"]),
                "sensitivity": _round(ood_raw_clf["sensitivity"]),
                "specificity": _round(ood_raw_clf["specificity"]),
                "f1": _round(ood_raw_clf["f1"]),
                "fn": int(ood_raw_clf["fn"]),
            },
        ]
        clf_header = ["variant", "n", "accuracy", "auroc", "auprc",
                      "sensitivity", "specificity", "f1", "fn"]
        write_csv(_resolve(OUT_TABLE_CLASSIFICATION), classification_rows, header=clf_header)

        summary_nums["id_val_acc"] = float(id_val_clf["accuracy"])
        summary_nums["ood_val_acc"] = float(ood_raw_clf["accuracy"])
        summary_nums["id_val_auroc"] = float(id_val_clf["auroc"])
        summary_nums["ood_val_auroc"] = float(ood_raw_clf["auroc"])
        summary_nums["id_val_sens"] = float(id_val_clf["sensitivity"])
        summary_nums["ood_val_sens"] = float(ood_raw_clf["sensitivity"])

        # ================================================================== #
        # TABLE 2 — calibration (ood_val raw vs calibrated)                  #
        # ================================================================== #
        calibration_rows = [
            {
                "variant": "raw",
                "temperature": _round(temperature),
                "ece": _round(calib_raw["ece"]),
                "brier": _round(calib_raw["brier"]),
                "nll": _round(calib_raw["nll"]),
            },
            {
                "variant": "calibrated",
                "temperature": _round(temperature),
                "ece": _round(calib_cal["ece"]),
                "brier": _round(calib_cal["brier"]),
                "nll": _round(calib_cal["nll"]),
            },
        ]
        calib_header = ["variant", "temperature", "ece", "brier", "nll"]
        write_csv(_resolve(OUT_TABLE_CALIBRATION), calibration_rows, header=calib_header)

        summary_nums["temperature"] = temperature
        summary_nums["ece_raw"] = float(calib_raw["ece"])
        summary_nums["ece_cal"] = float(calib_cal["ece"])
        summary_nums["brier_raw"] = float(calib_raw["brier"])
        summary_nums["brier_cal"] = float(calib_cal["brier"])
        summary_nums["nll_raw"] = float(calib_raw["nll"])
        summary_nums["nll_cal"] = float(calib_cal["nll"])

        # ================================================================== #
        # TABLE 3 — uncertainty / error detection (entropy)                  #
        # ================================================================== #
        entropy_unc = unc_df[unc_df["score_col"] == PRIMARY_UNCERTAINTY].copy()
        # AURC (entropy) per variant comes from the 5B risk_coverage table.
        entropy_rc = rc_df[rc_df["uncertainty_col"] == PRIMARY_UNCERTAINTY][
            ["variant", "aurc"]
        ].drop_duplicates("variant").set_index("variant")["aurc"].to_dict()

        uncertainty_rows = []
        for variant in ["id_val", "ood_val_raw", "ood_val_calibrated"]:
            row = entropy_unc[entropy_unc["variant"] == variant]
            assert len(row) == 1, f"expected one entropy row for {variant}"
            row = row.iloc[0]
            uncertainty_rows.append({
                "variant": variant,
                "error_detection_auroc": _round(row["error_detection_auroc"]),
                "aurc": _round(entropy_rc.get(variant)),
                "n_error": int(row["n_error"]),
                # FN counts (per variant) come from the 5C FN-abstention table @ coverage 1.0
                "fn": int(_fn_total_for(fn_df, variant)),
            })
        unc_header = ["variant", "error_detection_auroc", "aurc", "n_error", "fn"]
        write_csv(_resolve(OUT_TABLE_UNCERTAINTY), uncertainty_rows, header=unc_header)

        summary_nums["errdet_auroc_id"] = float(
            entropy_unc[entropy_unc["variant"] == "id_val"]["error_detection_auroc"].iloc[0]
        )
        summary_nums["errdet_auroc_ood_raw"] = float(
            entropy_unc[entropy_unc["variant"] == "ood_val_raw"]["error_detection_auroc"].iloc[0]
        )
        summary_nums["errdet_auroc_ood_cal"] = float(
            entropy_unc[entropy_unc["variant"] == "ood_val_calibrated"]["error_detection_auroc"].iloc[0]
        )

        # high-confidence FN reduction after calibration (from 5B high-conf table @ t=0.90)
        hc_090 = hc_df[hc_df["threshold"] == 0.90].set_index("variant")
        hc_fn_ood_raw = int(hc_090.loc["ood_val_raw", "n_high_conf_false_negative"])
        hc_fn_ood_cal = int(hc_090.loc["ood_val_calibrated", "n_high_conf_false_negative"])
        summary_nums["hc_fn_ood_raw_090"] = hc_fn_ood_raw
        summary_nums["hc_fn_ood_cal_090"] = hc_fn_ood_cal

        # ================================================================== #
        # TABLE 4 — selective prediction (entropy) at report coverages       #
        # id_val and ood_val_calibrated                                      #
        # ================================================================== #
        selective_rows = []
        for variant in ["id_val", "ood_val_calibrated"]:
            for cov in REPORT_COVERAGES:
                sel = _row_at_coverage(sel_df, variant, PRIMARY_UNCERTAINTY, cov)
                fn = _row_at_coverage(fn_df, variant, PRIMARY_UNCERTAINTY, cov)
                selective_rows.append({
                    "variant": variant,
                    "coverage": cov,
                    "retained_risk": _round(sel["retained_risk"]),
                    "retained_false_negatives": int(sel["retained_false_negatives"]),
                    "abstained_false_negatives": int(fn["abstained_false_negatives"]),
                    "fraction_of_fn_abstained": _round(fn["fraction_of_fn_abstained"]),
                })
        sel_header = ["variant", "coverage", "retained_risk", "retained_false_negatives",
                      "abstained_false_negatives", "fraction_of_fn_abstained"]
        write_csv(_resolve(OUT_TABLE_SELECTIVE), selective_rows, header=sel_header)

        # selective summary numbers (ood_val_calibrated @ 0.50)
        sel_ood_050 = _row_at_coverage(sel_df, "ood_val_calibrated", PRIMARY_UNCERTAINTY, 0.50)
        fn_ood_050 = _row_at_coverage(fn_df, "ood_val_calibrated", PRIMARY_UNCERTAINTY, 0.50)
        sel_ood_100 = _row_at_coverage(sel_df, "ood_val_calibrated", PRIMARY_UNCERTAINTY, 1.00)
        summary_nums["ood_risk_full"] = float(sel_ood_100["retained_risk"])
        summary_nums["ood_risk_cov50"] = float(sel_ood_050["retained_risk"])
        summary_nums["ood_ret_fn_cov50"] = int(sel_ood_050["retained_false_negatives"])
        summary_nums["ood_fn_total"] = int(fn_ood_050["total_false_negatives"])
        summary_nums["ood_frac_fn_abst_cov50"] = float(fn_ood_050["fraction_of_fn_abstained"])

        sel_id_050 = _row_at_coverage(sel_df, "id_val", PRIMARY_UNCERTAINTY, 0.50)
        summary_nums["id_risk_full"] = float(
            _row_at_coverage(sel_df, "id_val", PRIMARY_UNCERTAINTY, 1.00)["retained_risk"]
        )
        summary_nums["id_risk_cov50"] = float(sel_id_050["retained_risk"])

        emit("")
        emit("--- report tables written ---")
        for rel in OUTPUT_TABLES:
            emit(f"  {_resolve(rel)}")

        # ================================================================== #
        # FIGURES                                                            #
        # ================================================================== #
        _fig_domain_shift(classification_rows, _resolve(OUT_FIG_DOMAIN_SHIFT))
        _fig_calibration(calibration_rows, _resolve(OUT_FIG_CALIBRATION))
        _fig_uncertainty(uncertainty_rows, _resolve(OUT_FIG_UNCERTAINTY))
        _fig_risk_coverage(sel_df, _resolve(OUT_FIG_RISK_COVERAGE))
        _fig_fn_abstention(fn_df, _resolve(OUT_FIG_FN_ABSTENTION))

        emit("")
        emit("--- report figures written ---")
        for rel in OUTPUT_FIGURES:
            emit(f"  {_resolve(rel)}")

        # ================================================================== #
        # SUMMARY BLOCK                                                       #
        # ================================================================== #
        emit("")
        emit("=" * 72)
        emit("KEY FINDINGS (development analysis; capped balanced subsets)")
        emit("=" * 72)

        emit("")
        emit("[1] ID -> OOD degradation (id_val vs ood_val, raw):")
        emit(f"    accuracy   : {summary_nums['id_val_acc']:.4f} -> "
             f"{summary_nums['ood_val_acc']:.4f}  "
             f"(drop {summary_nums['id_val_acc'] - summary_nums['ood_val_acc']:+.4f})")
        emit(f"    AUROC      : {summary_nums['id_val_auroc']:.4f} -> "
             f"{summary_nums['ood_val_auroc']:.4f}  "
             f"(drop {summary_nums['id_val_auroc'] - summary_nums['ood_val_auroc']:+.4f})")
        emit(f"    sensitivity: {summary_nums['id_val_sens']:.4f} -> "
             f"{summary_nums['ood_val_sens']:.4f}  "
             f"(drop {summary_nums['id_val_sens'] - summary_nums['ood_val_sens']:+.4f})")

        emit("")
        emit(f"[2] Calibration on ood_val (temperature T = {summary_nums['temperature']:.4f}):")
        emit(f"    ECE  : {summary_nums['ece_raw']:.4f} -> {summary_nums['ece_cal']:.4f}  "
             f"({_pct_drop(summary_nums['ece_raw'], summary_nums['ece_cal'])})")
        emit(f"    Brier: {summary_nums['brier_raw']:.4f} -> {summary_nums['brier_cal']:.4f}  "
             f"({_pct_drop(summary_nums['brier_raw'], summary_nums['brier_cal'])})")
        emit(f"    NLL  : {summary_nums['nll_raw']:.4f} -> {summary_nums['nll_cal']:.4f}  "
             f"({_pct_drop(summary_nums['nll_raw'], summary_nums['nll_cal'])})")
        emit("    (classification metrics unchanged — temperature scaling is argmax-invariant)")

        emit("")
        emit("[3] High-confidence false negatives (missed tumors at confidence >= 0.90), ood_val:")
        emit(f"    raw -> calibrated : {summary_nums['hc_fn_ood_raw_090']} -> "
             f"{summary_nums['hc_fn_ood_cal_090']}  "
             f"(calibration removes confidently-wrong tumor misses in this subset)")

        emit("")
        emit("[4] Entropy error-detection AUROC (higher = uncertainty separates errors better):")
        emit(f"    id_val {summary_nums['errdet_auroc_id']:.4f}  |  "
             f"ood_val_raw {summary_nums['errdet_auroc_ood_raw']:.4f}  |  "
             f"ood_val_calibrated {summary_nums['errdet_auroc_ood_cal']:.4f}")
        emit("    (raw vs calibrated identical — temperature scaling preserves ranking)")

        emit("")
        emit("[5] Selective prediction (entropy abstention), retain most-confident rows:")
        emit(f"    id_val : risk {summary_nums['id_risk_full']:.4f} -> "
             f"{summary_nums['id_risk_cov50']:.4f} at 50% coverage")
        emit(f"    ood_val_calibrated : risk {summary_nums['ood_risk_full']:.4f} -> "
             f"{summary_nums['ood_risk_cov50']:.4f} at 50% coverage; "
             f"{summary_nums['ood_ret_fn_cov50']}/{summary_nums['ood_fn_total']} FNs still "
             f"retained ({summary_nums['ood_frac_fn_abst_cov50'] * 100:.1f}% of FNs abstained)")
        emit("    Confidence-based abstention cannot fully rescue a model that is")
        emit("    confidently wrong under domain shift.")

        emit("")
        emit("-" * 72)
        emit("SCOPE: development analysis only, on capped balanced subsets — NOT final")
        emit("clinical/paper performance. ood_val used for authorized OOD *analysis* only.")
        emit("ood_test remains LOCKED and was never read.")
        emit("-" * 72)

        # ------------------------------------------------------------------ #
        # Final assertions: all outputs exist and are non-empty              #
        # ------------------------------------------------------------------ #
        emit("")
        emit("--- assertions ---")
        for rel in OUTPUT_TABLES + OUTPUT_FIGURES:
            p = _resolve(rel)
            exists = p.exists() and p.stat().st_size > 0
            emit(f"  {'PASS' if exists else 'FAIL'} exists & non-empty: {p.name}")
            assert exists, f"output missing or empty: {p}"
        emit("  PASS no ood_test path referenced")
        emit(f"  PASS 4 tables + 5 figures generated")

    except Exception as exc:
        ok = False
        emit("")
        emit(f"ERROR: {type(exc).__name__}: {exc}")

    emit("")
    emit("=" * 72)
    emit(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    emit("=" * 72)

    write_text(_resolve(OUT_SUMMARY_TXT), "\n".join(lines) + "\n")
    print(f"\nSaved summary -> {_resolve(OUT_SUMMARY_TXT)}", flush=True)

    return 0 if ok else 1


# ---------------------------------------------------------------------------- #
# Small lookup helpers                                                          #
# ---------------------------------------------------------------------------- #
def _row_at_coverage(df: pd.DataFrame, variant: str, unc_col: str, coverage: float):
    """Return the single row for (variant, uncertainty_col, target_coverage)."""
    sub = df[
        (df["variant"] == variant)
        & (df["uncertainty_col"] == unc_col)
        & (df["target_coverage"].astype(float) == float(coverage))
    ]
    assert len(sub) == 1, (
        f"expected 1 row for variant={variant} col={unc_col} coverage={coverage}, "
        f"got {len(sub)}"
    )
    return sub.iloc[0]


def _fn_total_for(fn_df: pd.DataFrame, variant: str) -> int:
    """Total false-negative count for a variant (from the FN-abstention table @ coverage 1.0)."""
    row = _row_at_coverage(fn_df, variant, PRIMARY_UNCERTAINTY, 1.00)
    return int(row["total_false_negatives"])


def _pct_drop(raw: float, cal: float) -> str:
    """Signed percent change from raw -> cal (negative = reduction / improvement)."""
    if raw == 0:
        return "n/a"
    return f"{(cal - raw) / raw * 100:+.0f}%"


# ---------------------------------------------------------------------------- #
# Figures (matplotlib only)                                                     #
# ---------------------------------------------------------------------------- #
def _fig_domain_shift(rows: list[dict], out: Path) -> None:
    variants = [r["variant"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    auroc = [r["auroc"] for r in rows]

    x = range(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar([i - width / 2 for i in x], acc, width, label="accuracy", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], auroc, width, label="AUROC", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Domain shift: accuracy & AUROC (id_val vs ood_val)")
    ax.legend()
    for i, (a, u) in enumerate(zip(acc, auroc)):
        ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, u + 0.01, f"{u:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_calibration(rows: list[dict], out: Path) -> None:
    metrics = ["ece", "brier", "nll"]
    raw = [rows[0][m] for m in metrics]
    cal = [rows[1][m] for m in metrics]

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar([i - width / 2 for i in x], raw, width, label="raw", color="#C44E52")
    ax.bar([i + width / 2 for i in x], cal, width, label="calibrated", color="#55A868")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("value (lower is better)")
    ax.set_title("Calibration on ood_val: raw vs calibrated (T = %.3f)" % rows[0]["temperature"])
    ax.legend()
    for i, (rv, cv) in enumerate(zip(raw, cal)):
        ax.text(i - width / 2, rv + 0.005, f"{rv:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, cv + 0.005, f"{cv:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_uncertainty(rows: list[dict], out: Path) -> None:
    variants = [r["variant"] for r in rows]
    auroc = [r["error_detection_auroc"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    ax.bar(range(len(variants)), auroc, color=colors[: len(variants)])
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("error-detection AUROC (entropy)")
    ax.set_title("Uncertainty error detection (entropy)")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance (0.5)")
    for i, v in enumerate(auroc):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_risk_coverage(sel_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    series = [
        ("id_val", "#4C72B0", "o"),
        ("ood_val_calibrated", "#DD8452", "s"),
    ]
    for variant, color, marker in series:
        sub = sel_df[
            (sel_df["variant"] == variant)
            & (sel_df["uncertainty_col"] == PRIMARY_UNCERTAINTY)
        ].sort_values("target_coverage")
        ax.plot(sub["target_coverage"], sub["retained_risk"],
                marker=marker, color=color, label=variant)
    ax.set_xlabel("coverage (fraction retained)")
    ax.set_ylabel("retained risk (error rate)")
    ax.set_title("Risk–coverage (entropy abstention)")
    ax.invert_xaxis()  # full coverage on the left, most-selective on the right
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_fn_abstention(fn_df: pd.DataFrame, out: Path) -> None:
    variants = ["id_val", "ood_val_calibrated"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=False)
    for ax, variant in zip(axes, variants):
        sub = fn_df[
            (fn_df["variant"] == variant)
            & (fn_df["uncertainty_col"] == PRIMARY_UNCERTAINTY)
        ].sort_values("target_coverage", ascending=False)
        cov = sub["target_coverage"].to_numpy()
        retained = sub["retained_false_negatives"].to_numpy()
        abstained = sub["abstained_false_negatives"].to_numpy()
        labels = [f"{c:.2f}" for c in cov]
        x = range(len(cov))
        ax.bar(x, retained, label="retained FN", color="#C44E52")
        ax.bar(x, abstained, bottom=retained, label="abstained FN", color="#55A868")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel("coverage")
        ax.set_ylabel("false negatives (missed tumors)")
        ax.set_title(variant)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("False negatives: retained vs abstained (entropy)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
