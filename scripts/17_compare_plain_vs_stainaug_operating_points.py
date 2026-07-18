"""Milestone 7B / Script 17: threshold-normalized comparison of plain ERM vs stain-aug ERM.

**CSV-only.** This script consumes already-saved prediction CSVs (from Milestones 4A/5A for
plain ERM and 7A for the stain-augmentation baseline) and asks a single diagnostic question:

    Did the 7A stain/color augmentation baseline look worse on ``ood_val`` because its
    score *ranking* genuinely worsened, or only because the default 0.5 threshold became
    poorly placed (too conservative) for that model?

To answer it fairly we (a) report both models at the default 0.5 threshold, (b) report
threshold-free ranking metrics (AUROC / AUPRC), and (c) pick operating thresholds on
``id_val`` at matched specificity / sensitivity targets and apply the *same* threshold to
``ood_val``. If AUROC is essentially unchanged but the id_val-matched operating points
recover sensitivity on ood_val, the 7A "failure" was mostly a threshold artifact, not a
ranking collapse.

Strict scope (Milestone 7B): NO training, NO inference, NO calibration, NO MC-dropout, NO
ensemble, NO ``ood_test``. Input CSVs are read-only and never modified. ``id_val`` is used
ONLY to choose thresholds; ``ood_val`` is used ONLY for development-stage comparison after
thresholds are fixed. DEVELOPMENT analysis on capped balanced subsets — NOT final
performance.

Inputs (read-only):
  plain ERM  : results/predictions/exp04a_dev_train_resnet18/id_val_cached_predictions.csv
               results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_raw_predictions.csv
  stain-aug  : results/predictions/exp07a_stainaug_resnet18/id_val_predictions.csv
               results/predictions/exp07a_stainaug_resnet18/ood_val_predictions.csv
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
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from evaluation.metrics_operating_points import (
    METRIC_FIELDS,
    apply_operating_points,
    compute_metrics_at_threshold,
    make_threshold_sweep,
)
from utils.io import ensure_dir, ensure_results_tree, project_root, write_csv, write_text

# ---------------------------------------------------------------------------- #
# Input paths (read-only)                                                       #
# ---------------------------------------------------------------------------- #
PLAIN_ID_VAL_CSV = "results/predictions/exp04a_dev_train_resnet18/id_val_cached_predictions.csv"
PLAIN_OOD_VAL_CSV = "results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_raw_predictions.csv"
STAIN_ID_VAL_CSV = "results/predictions/exp07a_stainaug_resnet18/id_val_predictions.csv"
STAIN_OOD_VAL_CSV = "results/predictions/exp07a_stainaug_resnet18/ood_val_predictions.csv"

INPUT_PATHS = [PLAIN_ID_VAL_CSV, PLAIN_OOD_VAL_CSV, STAIN_ID_VAL_CSV, STAIN_OOD_VAL_CSV]

# ---------------------------------------------------------------------------- #
# Output paths                                                                  #
# ---------------------------------------------------------------------------- #
OUT_DEFAULT_CSV = "results/tables/exp07b_default_threshold_comparison.csv"
OUT_THRESHOLDFREE_CSV = "results/tables/exp07b_threshold_free_comparison.csv"
OUT_FIXED_SPEC_CSV = "results/tables/exp07b_fixed_specificity_operating_points.csv"
OUT_FIXED_SENS_CSV = "results/tables/exp07b_fixed_sensitivity_operating_points.csv"
OUT_SWEEPS_CSV = "results/tables/exp07b_threshold_sweeps.csv"
OUT_METRICS_JSON = "results/metrics/exp07b_operating_point_comparison/operating_point_metrics.json"
OUT_REPORT_TXT = "results/logs/exp07b_operating_point_comparison_report.txt"

OUT_FIG_ROC_ID = "results/figures/exp07b_roc_plain_vs_stainaug_id_val.png"
OUT_FIG_ROC_OOD = "results/figures/exp07b_roc_plain_vs_stainaug_ood_val.png"
OUT_FIG_PR_OOD = "results/figures/exp07b_pr_plain_vs_stainaug_ood_val.png"
OUT_FIG_TRADEOFF_OOD = "results/figures/exp07b_sensitivity_specificity_tradeoff_ood_val.png"

REQUIRED_COLUMNS = ["label", "prob_0", "prob_1", "pred_label", "correct"]

TARGET_SPECIFICITIES = [0.80, 0.85, 0.90, 0.95]
TARGET_SENSITIVITIES = [0.80, 0.90, 0.95]

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


def main() -> int:
    ensure_results_tree()

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    emit("=" * 78)
    emit("Milestone 7B - threshold-normalized comparison: plain ERM vs stain-aug ERM")
    emit("CSV-only. NO training, NO inference, NO calibration, NO MC-dropout/ensemble.")
    emit("id_val -> threshold selection ONLY; ood_val -> development comparison ONLY.")
    emit("ood_test: NEVER read. DEVELOPMENT analysis on capped subsets - NOT final perf.")
    emit("=" * 78)

    ok = True

    try:
        # --- guard: no ood_test path anywhere in the configured inputs/outputs --- #
        all_paths = INPUT_PATHS + [
            OUT_DEFAULT_CSV, OUT_THRESHOLDFREE_CSV, OUT_FIXED_SPEC_CSV, OUT_FIXED_SENS_CSV,
            OUT_SWEEPS_CSV, OUT_METRICS_JSON, OUT_REPORT_TXT, OUT_FIG_ROC_ID, OUT_FIG_ROC_OOD,
            OUT_FIG_PR_OOD, OUT_FIG_TRADEOFF_OOD,
        ]
        assert not any("ood_test" in p for p in all_paths), "ood_test must never be referenced"

        # --- load inputs (read-only) + record pre-read fingerprints -------------- #
        resolved = {name: _resolve(name) for name in INPUT_PATHS}
        for name, path in resolved.items():
            assert path.exists(), f"required input not found: {path}"
        pre_mtime = {name: resolved[name].stat().st_mtime_ns for name in INPUT_PATHS}
        pre_size = {name: resolved[name].stat().st_size for name in INPUT_PATHS}

        frames = {
            ("plain", "id_val"): pd.read_csv(resolved[PLAIN_ID_VAL_CSV]),
            ("plain", "ood_val"): pd.read_csv(resolved[PLAIN_OOD_VAL_CSV]),
            ("stain", "id_val"): pd.read_csv(resolved[STAIN_ID_VAL_CSV]),
            ("stain", "ood_val"): pd.read_csv(resolved[STAIN_OOD_VAL_CSV]),
        }

        expected_n = {"id_val": 500, "ood_val": 2000}
        model_label = {"plain": "plain ERM (4A/5A)", "stain": "stain-aug (7A)"}

        # --- validate columns / counts / probabilities -------------------------- #
        for (model, split), df in frames.items():
            tag = f"{model}/{split}"
            for col in REQUIRED_COLUMNS:
                assert col in df.columns, f"{tag}: missing required column {col!r}"
            assert len(df) == expected_n[split], (
                f"{tag}: expected n=={expected_n[split]}, got {len(df)}"
            )
            p0 = df["prob_0"].to_numpy(dtype=float)
            p1 = df["prob_1"].to_numpy(dtype=float)
            assert np.isfinite(p0).all(), f"{tag}: non-finite prob_0"
            assert np.isfinite(p1).all(), f"{tag}: non-finite prob_1"
            assert ((p0 >= 0.0) & (p0 <= 1.0)).all(), f"{tag}: prob_0 out of [0,1]"
            assert ((p1 >= 0.0) & (p1 <= 1.0)).all(), f"{tag}: prob_1 out of [0,1]"
            assert (np.abs(p0 + p1 - 1.0) < 1e-5).all(), f"{tag}: prob_0+prob_1 != 1"
            labs = set(np.unique(df["label"].to_numpy()).tolist())
            assert labs.issubset({0, 1}), f"{tag}: labels outside {{0,1}}: {sorted(labs)}"

        emit("")
        emit("Inputs validated (columns, row counts, probabilities):")
        for (model, split), df in frames.items():
            emit(f"  {model:<6} {split:<7} n={len(df):<5} "
                 f"tumors={int((df['label'] == 1).sum())} "
                 f"non-tumors={int((df['label'] == 0).sum())}")

        # ------------------------------------------------------------------ #
        # A. Default-threshold (0.5) metrics                                  #
        # ------------------------------------------------------------------ #
        default_rows: list[dict] = []
        default_by_key: dict[tuple[str, str], dict] = {}
        for (model, split), df in frames.items():
            m = compute_metrics_at_threshold(
                df["label"].to_numpy(), df["prob_1"].to_numpy(), DEFAULT_THRESHOLD
            )
            default_by_key[(model, split)] = m
            row = {"model": model, "model_label": model_label[model], "split": split}
            for k in METRIC_FIELDS:
                row[k] = m[k]
            default_rows.append(row)

        emit("")
        emit("A. Default threshold (0.50) comparison  [tumor = positive]")
        emit(f"   {'model':<16}{'split':<8}{'sens':>8}{'spec':>8}{'acc':>8}"
             f"{'bal_acc':>9}{'f1':>8}{'FN':>6}{'FP':>6}{'ppr':>8}")
        for model in ("plain", "stain"):
            for split in ("id_val", "ood_val"):
                m = default_by_key[(model, split)]
                emit(f"   {model_label[model]:<16}{split:<8}"
                     f"{_fmt(m['sensitivity']):>8}{_fmt(m['specificity']):>8}"
                     f"{_fmt(m['accuracy']):>8}{_fmt(m['balanced_accuracy']):>9}"
                     f"{_fmt(m['f1']):>8}{m['fn']:>6}{m['fp']:>6}"
                     f"{_fmt(m['predicted_positive_rate']):>8}")

        # ------------------------------------------------------------------ #
        # B. Threshold-free ranking metrics: AUROC / AUPRC                    #
        # ------------------------------------------------------------------ #
        thresholdfree_rows: list[dict] = []
        auc_by_key: dict[tuple[str, str], dict] = {}
        for (model, split), df in frames.items():
            y = df["label"].to_numpy().astype(int)
            s = df["prob_1"].to_numpy(dtype=float)
            if len(set(y.tolist())) < 2:
                auroc = auprc = None
            else:
                auroc = float(roc_auc_score(y, s))
                auprc = float(average_precision_score(y, s))
            auc_by_key[(model, split)] = {"auroc": auroc, "auprc": auprc}
            thresholdfree_rows.append({
                "model": model, "model_label": model_label[model], "split": split,
                "n": int(len(df)), "auroc": auroc, "auprc": auprc,
            })

        emit("")
        emit("B. Threshold-free ranking metrics (AUROC / AUPRC)")
        emit(f"   {'split':<8}{'plain AUROC':>13}{'stain AUROC':>13}{'dAUROC':>9}"
             f"{'plain AUPRC':>13}{'stain AUPRC':>13}{'dAUPRC':>9}")
        for split in ("id_val", "ood_val"):
            pa = auc_by_key[("plain", split)]
            sa = auc_by_key[("stain", split)]
            d_auroc = (sa["auroc"] - pa["auroc"]) if (pa["auroc"] is not None and sa["auroc"] is not None) else None
            d_auprc = (sa["auprc"] - pa["auprc"]) if (pa["auprc"] is not None and sa["auprc"] is not None) else None
            emit(f"   {split:<8}{_fmt(pa['auroc']):>13}{_fmt(sa['auroc']):>13}{_fmt(d_auroc):>9}"
                 f"{_fmt(pa['auprc']):>13}{_fmt(sa['auprc']):>13}{_fmt(d_auprc):>9}")
        emit("   (d = stain-aug - plain ERM; ~0 dAUROC => ranking preserved under the model swap.)")

        # ------------------------------------------------------------------ #
        # C/D. id_val-selected operating points applied to ood_val           #
        # ------------------------------------------------------------------ #
        op_rows: dict[str, list[dict]] = {}
        for model in ("plain", "stain"):
            rows = apply_operating_points(
                frames[(model, "id_val")], frames[(model, "ood_val")],
                TARGET_SPECIFICITIES, TARGET_SENSITIVITIES,
            )
            for r in rows:
                r["model"] = model
                r["model_label"] = model_label[model]
            op_rows[model] = rows

        def _op_lookup(model: str, criterion: str, target: float, split: str) -> dict | None:
            for r in op_rows[model]:
                if r["criterion"] == criterion and abs(r["target"] - target) < 1e-9 and r["split"] == split:
                    return r
            return None

        # ---- C. fixed specificity ---- #
        fixed_spec_rows: list[dict] = []
        for model in ("plain", "stain"):
            for r in op_rows[model]:
                if r["criterion"] == "specificity":
                    fixed_spec_rows.append(r)

        emit("")
        emit("C. id_val-selected FIXED-SPECIFICITY operating points, applied to ood_val")
        emit("   (threshold chosen on id_val so id_val specificity >= target; same t on ood_val)")
        emit(f"   {'target_spec':>11}{'model':<16}{'thr':>7}"
             f"{'id_sens':>9}{'id_spec':>9}{'ood_sens':>10}{'ood_spec':>10}{'ood_FN':>8}{'ood_acc':>9}")
        for spec in TARGET_SPECIFICITIES:
            for model in ("plain", "stain"):
                idr = _op_lookup(model, "specificity", spec, "id_val")
                oor = _op_lookup(model, "specificity", spec, "ood_val")
                thr = idr["threshold"] if idr else None
                emit(f"   {spec:>11.2f}{model_label[model]:<16}{_fmt(thr):>7}"
                     f"{_fmt(idr['sensitivity'] if idr else None):>9}"
                     f"{_fmt(idr['specificity'] if idr else None):>9}"
                     f"{_fmt(oor['sensitivity'] if oor else None):>10}"
                     f"{_fmt(oor['specificity'] if oor else None):>10}"
                     f"{(oor['fn'] if oor and oor['fn'] is not None else 'None'):>8}"
                     f"{_fmt(oor['accuracy'] if oor else None):>9}")

        # ---- D. fixed sensitivity ---- #
        fixed_sens_rows: list[dict] = []
        for model in ("plain", "stain"):
            for r in op_rows[model]:
                if r["criterion"] == "sensitivity":
                    fixed_sens_rows.append(r)

        emit("")
        emit("D. id_val-selected FIXED-SENSITIVITY operating points, applied to ood_val")
        emit("   (threshold chosen on id_val so id_val sensitivity >= target; same t on ood_val)")
        emit(f"   {'target_sens':>11}{'model':<16}{'thr':>7}"
             f"{'id_sens':>9}{'id_spec':>9}{'ood_sens':>10}{'ood_spec':>10}{'ood_FN':>8}{'ood_acc':>9}")
        for sens in TARGET_SENSITIVITIES:
            for model in ("plain", "stain"):
                idr = _op_lookup(model, "sensitivity", sens, "id_val")
                oor = _op_lookup(model, "sensitivity", sens, "ood_val")
                thr = idr["threshold"] if idr else None
                emit(f"   {sens:>11.2f}{model_label[model]:<16}{_fmt(thr):>7}"
                     f"{_fmt(idr['sensitivity'] if idr else None):>9}"
                     f"{_fmt(idr['specificity'] if idr else None):>9}"
                     f"{_fmt(oor['sensitivity'] if oor else None):>10}"
                     f"{_fmt(oor['specificity'] if oor else None):>10}"
                     f"{(oor['fn'] if oor and oor['fn'] is not None else 'None'):>8}"
                     f"{_fmt(oor['accuracy'] if oor else None):>9}")

        # ------------------------------------------------------------------ #
        # E. Threshold sweeps (both models, both splits)                     #
        # ------------------------------------------------------------------ #
        sweep_rows: list[dict] = []
        sweeps: dict[tuple[str, str], list[dict]] = {}
        for (model, split), df in frames.items():
            sw = make_threshold_sweep(df["label"].to_numpy(), df["prob_1"].to_numpy())
            sweeps[(model, split)] = sw
            for m in sw:
                row = {"model": model, "model_label": model_label[model], "split": split}
                for k in METRIC_FIELDS:
                    row[k] = m[k]
                sweep_rows.append(row)

        # ------------------------------------------------------------------ #
        # Save tables                                                        #
        # ------------------------------------------------------------------ #
        # `threshold` already appears in METRIC_FIELDS (the id_val-selected value).
        op_field_order = [
            "model", "model_label", "criterion", "target", "split", "achieved_on_id_val",
            *METRIC_FIELDS,
        ]

        write_csv(_resolve(OUT_DEFAULT_CSV), default_rows,
                  header=["model", "model_label", "split", *METRIC_FIELDS])
        write_csv(_resolve(OUT_THRESHOLDFREE_CSV), thresholdfree_rows,
                  header=["model", "model_label", "split", "n", "auroc", "auprc"])
        write_csv(_resolve(OUT_FIXED_SPEC_CSV), fixed_spec_rows, header=op_field_order)
        write_csv(_resolve(OUT_FIXED_SENS_CSV), fixed_sens_rows, header=op_field_order)
        write_csv(_resolve(OUT_SWEEPS_CSV), sweep_rows,
                  header=["model", "model_label", "split", *METRIC_FIELDS])

        # ------------------------------------------------------------------ #
        # Figures (matplotlib only)                                          #
        # ------------------------------------------------------------------ #
        def _roc_points(model: str, split: str):
            df = frames[(model, split)]
            y = df["label"].to_numpy().astype(int)
            s = df["prob_1"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, s)
            return fpr, tpr

        # ROC id_val
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        for model, color in (("plain", "tab:blue"), ("stain", "tab:orange")):
            fpr, tpr = _roc_points(model, "id_val")
            ax.plot(fpr, tpr, color=color,
                    label=f"{model_label[model]} (AUROC {_fmt(auc_by_key[(model, 'id_val')]['auroc'])})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title("ROC - id_val (development)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(_resolve(OUT_FIG_ROC_ID), dpi=150)
        plt.close(fig)

        # ROC ood_val
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        for model, color in (("plain", "tab:blue"), ("stain", "tab:orange")):
            fpr, tpr = _roc_points(model, "ood_val")
            ax.plot(fpr, tpr, color=color,
                    label=f"{model_label[model]} (AUROC {_fmt(auc_by_key[(model, 'ood_val')]['auroc'])})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title("ROC - ood_val (development)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(_resolve(OUT_FIG_ROC_OOD), dpi=150)
        plt.close(fig)

        # PR ood_val
        from sklearn.metrics import precision_recall_curve

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        for model, color in (("plain", "tab:blue"), ("stain", "tab:orange")):
            df = frames[(model, "ood_val")]
            y = df["label"].to_numpy().astype(int)
            s = df["prob_1"].to_numpy(dtype=float)
            prec, rec, _ = precision_recall_curve(y, s)
            ax.plot(rec, prec, color=color,
                    label=f"{model_label[model]} (AUPRC {_fmt(auc_by_key[(model, 'ood_val')]['auprc'])})")
        ax.set_xlabel("Recall (sensitivity)")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall - ood_val (development)")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(_resolve(OUT_FIG_PR_OOD), dpi=150)
        plt.close(fig)

        # sensitivity/specificity vs threshold - ood_val
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        for model, sc, lc in (("plain", "tab:blue", "tab:cyan"), ("stain", "tab:orange", "tab:red")):
            sw = sweeps[(model, "ood_val")]
            ts = [r["threshold"] for r in sw]
            sens = [(r["sensitivity"] if r["sensitivity"] is not None else np.nan) for r in sw]
            spec = [(r["specificity"] if r["specificity"] is not None else np.nan) for r in sw]
            ax.plot(ts, sens, color=sc, label=f"{model_label[model]} sensitivity")
            ax.plot(ts, spec, color=lc, linestyle="--", label=f"{model_label[model]} specificity")
        ax.axvline(DEFAULT_THRESHOLD, color="k", linewidth=0.8, alpha=0.5, label="threshold 0.5")
        ax.set_xlabel("Decision threshold on prob_1")
        ax.set_ylabel("Rate")
        ax.set_title("Sensitivity / specificity vs threshold - ood_val")
        ax.legend(loc="center left", fontsize=7)
        fig.tight_layout()
        fig.savefig(_resolve(OUT_FIG_TRADEOFF_OOD), dpi=150)
        plt.close(fig)

        figure_paths = [OUT_FIG_ROC_ID, OUT_FIG_ROC_OOD, OUT_FIG_PR_OOD, OUT_FIG_TRADEOFF_OOD]

        # ------------------------------------------------------------------ #
        # Interpretation                                                     #
        # ------------------------------------------------------------------ #
        d_auroc_ood = None
        pa = auc_by_key[("plain", "ood_val")]["auroc"]
        sa = auc_by_key[("stain", "ood_val")]["auroc"]
        if pa is not None and sa is not None:
            d_auroc_ood = sa - pa

        # sensitivity recovery for stain-aug at the highest specificity target that plain
        # ERM's default 0.5 roughly corresponds to; use the 0.90 fixed-specificity point.
        stain_def_ood_sens = default_by_key[("stain", "ood_val")]["sensitivity"]
        plain_def_ood_sens = default_by_key[("plain", "ood_val")]["sensitivity"]
        stain_sens_spec90 = None
        r = _op_lookup("stain", "specificity", 0.90, "ood_val")
        if r is not None:
            stain_sens_spec90 = r["sensitivity"]

        ranking_worsened = (d_auroc_ood is not None and d_auroc_ood <= -0.03)
        ranking_preserved = (d_auroc_ood is not None and abs(d_auroc_ood) < 0.03)

        # Did matching the operating point recover a large chunk of the lost sensitivity?
        recovered = (
            stain_sens_spec90 is not None
            and stain_def_ood_sens is not None
            and (stain_sens_spec90 - stain_def_ood_sens) >= 0.10
        )

        emit("")
        emit("INTERPRETATION (development-stage; capped balanced subsets)")
        emit("-" * 60)

        if ranking_worsened:
            emit("* Ranking: stain-aug WORSENED ood_val ranking (dAUROC "
                 f"{_fmt(d_auroc_ood)} <= -0.03). The augmentation degraded score ordering,")
            emit("  not merely the threshold placement.")
        elif ranking_preserved:
            emit("* Ranking: stain-aug LARGELY PRESERVED ood_val ranking (|dAUROC| "
                 f"{_fmt(d_auroc_ood)} < 0.03). Score ordering is close to plain ERM;")
            emit("  the raw AUROC gap is small relative to the default-threshold gap.")
        else:
            emit(f"* Ranking: mixed - ood_val dAUROC = {_fmt(d_auroc_ood)} "
                 "(a modest ranking change, not a clean collapse or a clean tie).")

        emit(f"* Default threshold: at 0.50, stain-aug ood_val sensitivity = "
             f"{_fmt(stain_def_ood_sens)} vs plain {_fmt(plain_def_ood_sens)} "
             "-> the 0.5 point is far more")
        emit("  conservative for the stain-aug model (it pushes scores below 0.5 under shift).")
        if stain_sens_spec90 is not None:
            emit(f"* Operating-point matching: at an id_val-selected spec>=0.90 threshold, "
                 f"stain-aug ood_val sensitivity = {_fmt(stain_sens_spec90)} "
                 f"(vs {_fmt(stain_def_ood_sens)} at 0.5).")
            if recovered:
                emit("  Matching the operating point RECOVERS a substantial part of the lost "
                     "sensitivity -> much of the")
                emit("  apparent 7A 'collapse' is a THRESHOLD artifact, not a ranking failure.")
            else:
                emit("  Matching the operating point does NOT recover much sensitivity -> the "
                     "loss is not merely threshold placement.")

        # Overall recommendation
        emit("")
        if ranking_worsened and not recovered:
            recommendation = (
                "REJECT (as-is): stain augmentation degraded OOD ranking and operating-point "
                "matching does not rescue it on this capped subset."
            )
        elif (ranking_preserved or not ranking_worsened) and recovered:
            recommendation = (
                "TUNE / RETAIN with a calibrated operating point: the ranking is largely intact "
                "and the sensitivity collapse is mostly a 0.5-threshold artifact. Do NOT reject "
                "on the default-threshold result alone; the natural next step is calibration + "
                "operating-point selection (still development-stage), not abandonment."
            )
        else:
            recommendation = (
                "TUNE (inconclusive as-is): the default-threshold result overstates the failure, "
                "but operating-point matching only partially helps. Prefer threshold/calibration "
                "tuning and a stronger stain-robustness variant over outright rejection."
            )
        emit(f"RECOMMENDATION: {recommendation}")
        emit("(Development-stage diagnosis only - NOT a final-performance or superiority claim.)")

        # ------------------------------------------------------------------ #
        # Metrics JSON                                                        #
        # ------------------------------------------------------------------ #
        def _op_json(model: str) -> list[dict]:
            out = []
            for r in op_rows[model]:
                out.append({k: r[k] for k in
                            ["criterion", "target", "split", "achieved_on_id_val", *METRIC_FIELDS]})
            return out

        metrics_payload = {
            "milestone": "7B",
            "description": "Threshold-normalized comparison of plain ERM vs stain-aug ERM "
                           "(CSV-only; id_val selects thresholds, ood_val is development comparison).",
            "positive_class": 1,
            "default_threshold": DEFAULT_THRESHOLD,
            "target_specificities": TARGET_SPECIFICITIES,
            "target_sensitivities": TARGET_SENSITIVITIES,
            "inputs": {name: str(resolved[name]) for name in INPUT_PATHS},
            "default_threshold_metrics": {
                f"{model}_{split}": {k: default_by_key[(model, split)][k]
                                     for k in ["n", *METRIC_FIELDS]}
                for model in ("plain", "stain") for split in ("id_val", "ood_val")
            },
            "threshold_free": {
                f"{model}_{split}": auc_by_key[(model, split)]
                for model in ("plain", "stain") for split in ("id_val", "ood_val")
            },
            "operating_points": {"plain": _op_json("plain"), "stain": _op_json("stain")},
            "interpretation": {
                "ood_val_delta_auroc_stain_minus_plain": d_auroc_ood,
                "ranking_worsened": bool(ranking_worsened),
                "ranking_preserved": bool(ranking_preserved),
                "stain_default_ood_sensitivity": stain_def_ood_sens,
                "plain_default_ood_sensitivity": plain_def_ood_sens,
                "stain_ood_sensitivity_at_id_spec90": stain_sens_spec90,
                "operating_point_recovers_sensitivity": bool(recovered),
                "recommendation": recommendation,
            },
            "note": "DEVELOPMENT analysis on capped balanced subsets - NOT final performance. "
                    "ood_val used for development-stage comparison only after id_val threshold "
                    "selection. ood_test never read.",
        }
        json_path = _resolve(OUT_METRICS_JSON)
        ensure_dir(json_path.parent)
        write_text(json_path, json.dumps(metrics_payload, indent=2) + "\n")

        # ------------------------------------------------------------------ #
        # Verify input CSVs were NOT modified                                #
        # ------------------------------------------------------------------ #
        for name in INPUT_PATHS:
            post = resolved[name].stat()
            assert post.st_mtime_ns == pre_mtime[name], f"input modified (mtime): {name}"
            assert post.st_size == pre_size[name], f"input modified (size): {name}"

        # ------------------------------------------------------------------ #
        # Output-existence assertions                                        #
        # ------------------------------------------------------------------ #
        out_tables = [OUT_DEFAULT_CSV, OUT_THRESHOLDFREE_CSV, OUT_FIXED_SPEC_CSV,
                      OUT_FIXED_SENS_CSV, OUT_SWEEPS_CSV]
        for p in out_tables:
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"table missing/empty: {p}"
        for p in figure_paths:
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"figure missing/empty: {p}"
        assert json_path.exists() and json_path.stat().st_size > 0, "metrics JSON missing/empty"

        # ------------------------------------------------------------------ #
        # Output paths summary                                               #
        # ------------------------------------------------------------------ #
        emit("")
        emit("OUTPUT PATHS")
        emit("-" * 60)
        emit("tables:")
        for p in out_tables:
            emit(f"  {p}")
        emit("figures:")
        for p in figure_paths:
            emit(f"  {p}")
        emit(f"metrics JSON: {OUT_METRICS_JSON}")
        emit(f"report      : {OUT_REPORT_TXT}")
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
