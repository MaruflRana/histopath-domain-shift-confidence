"""Milestone 8B / Script 33: summarize the full development-scale evaluation.

CSV-only analysis of the four prediction CSVs written by ``scripts/32`` — NO model
inference, NO training, NO calibration fitting, NO MC-dropout, NO ensemble, NO CORAL/DANN,
NO ``ood_test``. Positive class = tumor (1).

Computes, for the two frozen 7F models (primary GroupDRO candidate + matched center-strat
ERM control) on the full/larger development splits (id_val {0,3,4}, ood_val {1}):

  1. Overall classification metrics (both models, both splits).
  2. Per-center id_val metrics (center 0/3/4).
  3. Candidate operating points: thresholds selected on full id_val, applied to full
     ood_val (fixed specificity 0.80/0.85/0.90/0.95; fixed sensitivity 0.80/0.90/0.95).
     *** CANDIDATE points only — NO clinical threshold committed. ***
  4. High-confidence false-negative audit (RAW only) at confidence 0.90/0.95/0.99.

Writes JSON + 4 tables + report + 5 matplotlib figures under new exp08b names, and prints
the required blocks plus whether GroupDRO still beats the matched ERM at full scale.

DEVELOPMENT-stage analysis on full/larger development splits — NOT final performance.
"""

import _bootstrap  # noqa: F401

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from evaluation.metrics_classification import compute_binary_classification_metrics
from evaluation.metrics_operating_points import (
    METRIC_FIELDS,
    apply_operating_points,
    compute_metrics_at_threshold,
)
from utils.io import ensure_results_tree, project_root, write_csv, write_text

RUN_NAME = "exp08b_full_dev_eval"
PRED_DIR = f"results/predictions/{RUN_NAME}"

MODEL_ORDER = ("groupdro", "centerstrat_erm")
MODEL_LABEL = {
    "groupdro": "7F GroupDRO (primary candidate)",
    "centerstrat_erm": "7F center-stratified ERM (matched control)",
}
SPLITS = ("id_val", "ood_val")
EXPECTED_N = {"id_val": 33560, "ood_val": 34904}
ALLOWED_CENTERS = {"id_val": {0, 3, 4}, "ood_val": {1}}

TARGET_SPECIFICITIES = [0.80, 0.85, 0.90, 0.95]
TARGET_SENSITIVITIES = [0.80, 0.90, 0.95]
CONF_THRESHOLDS = [0.90, 0.95, 0.99]
DEFAULT_THRESHOLD = 0.5

# outputs
OUT_METRICS_JSON = f"results/metrics/{RUN_NAME}/full_dev_metrics.json"
OUT_OVERALL_CSV = "results/tables/exp08b_overall_metrics.csv"
OUT_PERCENTER_CSV = "results/tables/exp08b_id_val_per_center_metrics.csv"
OUT_OPS_CSV = "results/tables/exp08b_candidate_operating_points.csv"
OUT_HCFN_CSV = "results/tables/exp08b_high_confidence_false_negatives.csv"
OUT_REPORT_TXT = "results/logs/exp08b_full_development_eval_report.txt"

OUT_FIG_ROC = "results/figures/exp08b_ood_val_roc_groupdro_vs_erm.png"
OUT_FIG_PR = "results/figures/exp08b_ood_val_pr_groupdro_vs_erm.png"
OUT_FIG_FIXED_SPEC = "results/figures/exp08b_candidate_fixed_specificity_sensitivity.png"
OUT_FIG_PERCENTER = "results/figures/exp08b_id_val_per_center_accuracy.png"
OUT_FIG_HCFN = "results/figures/exp08b_high_confidence_fn_comparison.png"

_CLS_FIELDS = [
    "n", "accuracy", "balanced_accuracy", "auroc", "auprc",
    "sensitivity", "specificity", "precision", "f1", "tn", "fp", "fn", "tp",
]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def _pred_path(model_key: str, split: str) -> str:
    return f"{PRED_DIR}/{model_key}_{split}_predictions.csv"


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "None"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _prob_checks(df, name) -> None:
    p0 = df["prob_0"].to_numpy(dtype=float)
    p1 = df["prob_1"].to_numpy(dtype=float)
    assert np.isfinite(p0).all() and np.isfinite(p1).all(), f"{name}: non-finite prob"
    assert ((p0 >= 0) & (p0 <= 1)).all() and ((p1 >= 0) & (p1 <= 1)).all(), f"{name}: prob out of [0,1]"
    assert (np.abs(p0 + p1 - 1.0) < 1e-5).all(), f"{name}: prob_0+prob_1 != 1"


def _per_center_metrics(df) -> list[dict]:
    """Per-center id_val metrics from a prediction DataFrame (has 'center')."""
    rows = []
    for c in sorted(int(x) for x in df["center"].unique().tolist()):
        sub = df[df["center"] == c]
        y = sub["label"].to_numpy()
        p1 = sub["prob_1"].to_numpy(dtype=float)
        both = set(int(v) for v in np.unique(y).tolist()) == {0, 1}
        m = compute_binary_classification_metrics(y, p1, DEFAULT_THRESHOLD)
        # mean cross-entropy on the true class (loss) when logits/probs available
        p_true = np.where(y == 1, p1, 1.0 - p1)
        p_true = np.clip(p_true, 1e-7, 1.0)
        loss = float(-np.log(p_true).mean())
        rows.append({
            "center": int(c), "n": int(len(sub)),
            "label0": int((y == 0).sum()), "label1": int((y == 1).sum()),
            "accuracy": m["accuracy"],
            "auroc": m["auroc"] if both else None,
            "auprc": m["auprc"] if both else None,
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "loss": loss,
        })
    return rows


def _high_conf_fn_audit(df, thresholds) -> list[dict]:
    """RAW high-confidence false-negative audit. FN = label==1 & pred_label==0."""
    y = df["label"].to_numpy().astype(int)
    pred = df["pred_label"].to_numpy().astype(int)
    conf = df["confidence"].to_numpy(dtype=float)
    correct = (pred == y)
    total_fn = int(np.sum((y == 1) & (pred == 0)))
    out = []
    for t in thresholds:
        hi = conf >= t
        n_high = int(np.sum(hi))
        hc_err = int(np.sum(hi & (~correct)))
        hc_fn = int(np.sum(hi & (y == 1) & (pred == 0)))
        rate = (hc_fn / n_high) if n_high > 0 else None
        out.append({
            "confidence_threshold": float(t),
            "n_high_confidence": n_high,
            "high_confidence_errors": hc_err,
            "high_confidence_false_negatives": hc_fn,
            "high_confidence_false_negative_rate": rate,
            "total_false_negatives": total_fn,
        })
    return out


def main() -> int:
    ensure_results_tree()
    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    emit("=" * 80)
    emit("Milestone 8B - summarize full development-scale evaluation (script 33)")
    emit("CSV-only. No inference, no training, no calibration, no MC-dropout, no ensemble,")
    emit("no CORAL/DANN. ood_test never read. Positive class = tumor (1).")
    emit("=" * 80)

    ok = True
    try:
        # --- guard: no ood_test anywhere ---------------------------------------- #
        all_out = [OUT_METRICS_JSON, OUT_OVERALL_CSV, OUT_PERCENTER_CSV, OUT_OPS_CSV,
                   OUT_HCFN_CSV, OUT_REPORT_TXT, OUT_FIG_ROC, OUT_FIG_PR,
                   OUT_FIG_FIXED_SPEC, OUT_FIG_PERCENTER, OUT_FIG_HCFN]
        all_in = [_pred_path(m, s) for m in MODEL_ORDER for s in SPLITS]
        assert not any("ood_test" in str(p) for p in all_out + all_in), "ood_test must never be referenced"

        # --- load the four prediction CSVs -------------------------------------- #
        preds = {}
        for key in MODEL_ORDER:
            for split in SPLITS:
                p = _resolve(_pred_path(key, split))
                assert p.exists(), f"missing prediction CSV (run script 32 first): {p}"
                df = pd.read_csv(p)
                _prob_checks(df, f"{key}/{split}")
                centers = set(int(c) for c in df["center"].unique().tolist())
                labels = set(int(v) for v in df["label"].unique().tolist())
                assert labels.issubset({0, 1}), f"{key}/{split}: non-binary labels"
                assert centers.issubset(ALLOWED_CENTERS[split]), \
                    f"{key}/{split}: unexpected centers {centers}"
                assert 2 not in centers, f"{key}/{split}: center 2 (ood_test) present"
                assert len(df) == EXPECTED_N[split], \
                    f"{key}/{split}: n {len(df)} != {EXPECTED_N[split]}"
                assert "confidence" in df.columns, f"{key}/{split}: missing confidence column"
                preds[(key, split)] = df

        emit("")
        emit("ROW COUNTS")
        emit("-" * 60)
        for key in MODEL_ORDER:
            for split in SPLITS:
                df = preds[(key, split)]
                emit(f"  {MODEL_LABEL[key]:<45} {split:<8} n={len(df):>6} "
                     f"centers={sorted(int(c) for c in df['center'].unique())}")

        # --- 1. overall metrics ------------------------------------------------- #
        emit("")
        emit("1. OVERALL CLASSIFICATION METRICS (tumor = positive, threshold 0.5)")
        overall = {}
        overall_rows = []
        for key in MODEL_ORDER:
            for split in SPLITS:
                df = preds[(key, split)]
                m = compute_binary_classification_metrics(
                    df["label"].to_numpy(), df["prob_1"].to_numpy(dtype=float), threshold=0.5)
                overall[(key, split)] = m
                row = {"model": key, "model_label": MODEL_LABEL[key], "split": split}
                for k in _CLS_FIELDS:
                    row[k] = m[k]
                overall_rows.append(row)

        for split in SPLITS:
            emit(f"  [{split}]")
            for key in MODEL_ORDER:
                m = overall[(key, split)]
                emit(f"    {MODEL_LABEL[key]:<45} acc={_fmt(m['accuracy'])} "
                     f"auroc={_fmt(m['auroc'])} auprc={_fmt(m['auprc'])} "
                     f"sens={_fmt(m['sensitivity'])} spec={_fmt(m['specificity'])} "
                     f"prec={_fmt(m['precision'])} f1={_fmt(m['f1'])} "
                     f"tn/fp/fn/tp={m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}")

        # --- 2. per-center id_val ----------------------------------------------- #
        emit("")
        emit("2. PER-CENTER id_val METRICS (tumor = positive, threshold 0.5)")
        percenter = {}
        percenter_rows = []
        for key in MODEL_ORDER:
            rows = _per_center_metrics(preds[(key, "id_val")])
            percenter[key] = rows
            for r in rows:
                percenter_rows.append({"model": key, "model_label": MODEL_LABEL[key], **r})
                emit(f"  [{MODEL_LABEL[key]}] center {r['center']}: n={r['n']} "
                     f"(l0={r['label0']},l1={r['label1']}) acc={_fmt(r['accuracy'])} "
                     f"auroc={_fmt(r['auroc'])} auprc={_fmt(r['auprc'])} "
                     f"sens={_fmt(r['sensitivity'])} spec={_fmt(r['specificity'])} "
                     f"loss={_fmt(r['loss'])}")

        def _worst_center(key):
            return max(percenter[key], key=lambda r: (r["loss"] if r["loss"] is not None else -1))

        # --- 3. candidate operating points (id_val -> ood_val) ------------------ #
        emit("")
        emit("3. CANDIDATE OPERATING POINTS (threshold selected on FULL id_val -> applied to FULL ood_val)")
        emit("   *** CANDIDATE development points only - NO clinical threshold committed ***")
        ops_by_key = {}
        for key in MODEL_ORDER:
            ops_by_key[key] = apply_operating_points(
                preds[(key, "id_val")], preds[(key, "ood_val")],
                TARGET_SPECIFICITIES, TARGET_SENSITIVITIES)

        def _op(key, criterion, target, split):
            for r in ops_by_key[key]:
                if (r["criterion"] == criterion and abs(r["target"] - target) < 1e-9
                        and r["split"] == split):
                    return r
            return None

        ops_rows = []
        for key in MODEL_ORDER:
            for criterion, targets in (("specificity", TARGET_SPECIFICITIES),
                                       ("sensitivity", TARGET_SENSITIVITIES)):
                for tgt in targets:
                    idr = _op(key, criterion, tgt, "id_val")
                    oor = _op(key, criterion, tgt, "ood_val")
                    thr = idr["threshold"] if idr else None
                    ops_rows.append({
                        "model": key, "model_label": MODEL_LABEL[key],
                        "target_type": criterion, "target_value": tgt,
                        "threshold_selected_on_id_val": thr,
                        "id_val_sensitivity": idr["sensitivity"] if idr else None,
                        "id_val_specificity": idr["specificity"] if idr else None,
                        "id_val_FN": idr["fn"] if idr else None,
                        "id_val_FP": idr["fp"] if idr else None,
                        "ood_val_sensitivity": oor["sensitivity"] if oor else None,
                        "ood_val_specificity": oor["specificity"] if oor else None,
                        "ood_val_FN": oor["fn"] if oor else None,
                        "ood_val_FP": oor["fp"] if oor else None,
                        "ood_val_predicted_positive_rate": oor["predicted_positive_rate"] if oor else None,
                        "candidate_only_note": "CANDIDATE development operating point "
                                               "(id_val-selected); NOT a committed clinical threshold.",
                    })

        emit("   Fixed-specificity (id_val-selected) -> ood_val sensitivity:")
        emit(f"     {'target_spec':>11}  {'model':<45}{'thr':>8}{'ood_sens':>10}"
             f"{'ood_spec':>10}{'ood_FN':>8}{'ood_FP':>8}{'ood_PPR':>9}")
        for spec in TARGET_SPECIFICITIES:
            for key in MODEL_ORDER:
                idr = _op(key, "specificity", spec, "id_val")
                oor = _op(key, "specificity", spec, "ood_val")
                thr = idr["threshold"] if idr else None
                emit(f"     {spec:>11.2f}  {MODEL_LABEL[key]:<45}{_fmt(thr):>8}"
                     f"{_fmt(oor['sensitivity'] if oor else None):>10}"
                     f"{_fmt(oor['specificity'] if oor else None):>10}"
                     f"{(oor['fn'] if oor and oor['fn'] is not None else 'None'):>8}"
                     f"{(oor['fp'] if oor and oor['fp'] is not None else 'None'):>8}"
                     f"{_fmt(oor['predicted_positive_rate'] if oor else None):>9}")

        emit("   Fixed-sensitivity (id_val-selected) -> ood_val:")
        emit(f"     {'target_sens':>11}  {'model':<45}{'thr':>8}{'ood_sens':>10}{'ood_spec':>10}")
        for sens in TARGET_SENSITIVITIES:
            for key in MODEL_ORDER:
                idr = _op(key, "sensitivity", sens, "id_val")
                oor = _op(key, "sensitivity", sens, "ood_val")
                thr = idr["threshold"] if idr else None
                emit(f"     {sens:>11.2f}  {MODEL_LABEL[key]:<45}{_fmt(thr):>8}"
                     f"{_fmt(oor['sensitivity'] if oor else None):>10}"
                     f"{_fmt(oor['specificity'] if oor else None):>10}")

        # --- 4. high-confidence FN audit (raw) ---------------------------------- #
        emit("")
        emit("4. HIGH-CONFIDENCE FALSE-NEGATIVE AUDIT (RAW; label==1 & pred==0 & confidence>=t)")
        hcfn = {}
        hcfn_rows = []
        for key in MODEL_ORDER:
            for split in SPLITS:
                audit = _high_conf_fn_audit(preds[(key, split)], CONF_THRESHOLDS)
                hcfn[(key, split)] = audit
                for a in audit:
                    hcfn_rows.append({"model": key, "model_label": MODEL_LABEL[key],
                                      "split": split, **a})
        for split in SPLITS:
            emit(f"  [{split}]")
            for key in MODEL_ORDER:
                for a in hcfn[(key, split)]:
                    emit(f"    {MODEL_LABEL[key]:<45} t={a['confidence_threshold']:.2f} "
                         f"n_high={a['n_high_confidence']:>6} hc_err={a['high_confidence_errors']:>6} "
                         f"hc_FN={a['high_confidence_false_negatives']:>6} "
                         f"hc_FN_rate={_fmt(a['high_confidence_false_negative_rate'])} "
                         f"total_FN={a['total_false_negatives']}")

        # --- GroupDRO vs matched ERM verdict (full-scale) ----------------------- #
        g_ood = overall[("groupdro", "ood_val")]
        e_ood = overall[("centerstrat_erm", "ood_val")]
        g_id = overall[("groupdro", "id_val")]
        e_id = overall[("centerstrat_erm", "id_val")]

        def _delta(a, b):
            return (a - b) if (a is not None and b is not None) else None

        d_auroc = _delta(g_ood["auroc"], e_ood["auroc"])
        d_auprc = _delta(g_ood["auprc"], e_ood["auprc"])
        d_sens = _delta(g_ood["sensitivity"], e_ood["sensitivity"])

        n_ge, n_tgt = 0, 0
        for spec in TARGET_SPECIFICITIES:
            g = _op("groupdro", "specificity", spec, "ood_val")
            e = _op("centerstrat_erm", "specificity", spec, "ood_val")
            gv = g["sensitivity"] if (g and g["sensitivity"] is not None) else None
            ev = e["sensitivity"] if (e and e["sensitivity"] is not None) else None
            if gv is not None and ev is not None:
                n_tgt += 1
                if gv >= ev:
                    n_ge += 1

        ood_ranking_better = (d_auroc is not None and d_auroc > 0)
        fixedspec_better = (n_tgt > 0 and n_ge > n_tgt / 2)
        groupdro_beats_erm = bool(ood_ranking_better and fixedspec_better)

        emit("")
        emit("CONTROLLED COMPARISON: 7F GroupDRO vs 7F center-stratified ERM (full/larger dev splits)")
        emit("-" * 78)
        emit(f"  id_val  AUROC : GroupDRO {_fmt(g_id['auroc'])}  ERM {_fmt(e_id['auroc'])}")
        emit(f"  id_val  acc   : GroupDRO {_fmt(g_id['accuracy'])}  ERM {_fmt(e_id['accuracy'])}")
        emit(f"  ood_val AUROC : GroupDRO {_fmt(g_ood['auroc'])}  ERM {_fmt(e_ood['auroc'])}  (delta {_fmt(d_auroc)})")
        emit(f"  ood_val AUPRC : GroupDRO {_fmt(g_ood['auprc'])}  ERM {_fmt(e_ood['auprc'])}  (delta {_fmt(d_auprc)})")
        emit(f"  ood_val sens@0.5 : GroupDRO {_fmt(g_ood['sensitivity'])}  ERM {_fmt(e_ood['sensitivity'])}  (delta {_fmt(d_sens)})")
        emit(f"  matched fixed-spec ood_val sensitivity wins (GroupDRO >= ERM): {n_ge}/{n_tgt}")
        emit("")
        emit(f"  DOES GROUPDRO STILL BEAT THE MATCHED ERM AT FULL/LARGER DEVELOPMENT SCALE? "
             f"{'YES' if groupdro_beats_erm else 'NO'}")
        emit("  (development-stage; full/larger dev splits; NOT final performance; no committed threshold)")

        # --- figures ------------------------------------------------------------ #
        color = {"groupdro": "tab:red", "centerstrat_erm": "tab:blue"}

        # ROC ood_val
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        for key in MODEL_ORDER:
            df = preds[(key, "ood_val")]
            y = df["label"].to_numpy().astype(int); s = df["prob_1"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=color[key], linewidth=2.0,
                    label=f"{MODEL_LABEL[key]} (AUROC {_fmt(overall[(key, 'ood_val')]['auroc'])})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title("ROC - full ood_val (development): GroupDRO vs matched ERM")
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_ROC), dpi=150); plt.close(fig)

        # PR ood_val
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        for key in MODEL_ORDER:
            df = preds[(key, "ood_val")]
            y = df["label"].to_numpy().astype(int); s = df["prob_1"].to_numpy(dtype=float)
            prec, rec, _ = precision_recall_curve(y, s)
            ax.plot(rec, prec, color=color[key], linewidth=2.0,
                    label=f"{MODEL_LABEL[key]} (AUPRC {_fmt(overall[(key, 'ood_val')]['auprc'])})")
        ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall - full ood_val (development): GroupDRO vs matched ERM")
        ax.legend(loc="lower left", fontsize=7)
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_PR), dpi=150); plt.close(fig)

        # candidate fixed-specificity ood_val sensitivity
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        x = np.arange(len(TARGET_SPECIFICITIES)); width = 0.38
        for i, key in enumerate(MODEL_ORDER):
            vals = []
            for spec in TARGET_SPECIFICITIES:
                oor = _op(key, "specificity", spec, "ood_val")
                vals.append(oor["sensitivity"] if (oor and oor["sensitivity"] is not None) else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width, color=color[key], label=MODEL_LABEL[key])
        ax.set_xticks(x); ax.set_xticklabels([f"{s:.2f}" for s in TARGET_SPECIFICITIES])
        ax.set_xlabel("id_val-selected target specificity")
        ax.set_ylabel("ood_val sensitivity (candidate)")
        ax.set_title("Candidate fixed-specificity ood_val sensitivity\n(GroupDRO vs matched ERM; full dev; NOT committed)")
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_FIXED_SPEC), dpi=150); plt.close(fig)

        # per-center id_val accuracy
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        centers = sorted({r["center"] for r in percenter["groupdro"]})
        x = np.arange(len(centers)); width = 0.38
        for i, key in enumerate(MODEL_ORDER):
            lut = {r["center"]: r for r in percenter[key]}
            accs = [lut[c]["accuracy"] if lut[c]["accuracy"] is not None else np.nan for c in centers]
            ax.bar(x + (i - 0.5) * width, accs, width, color=color[key], label=MODEL_LABEL[key])
        ax.set_xticks(x); ax.set_xticklabels([f"center {c}" for c in centers])
        ax.set_ylabel("id_val accuracy"); ax.set_ylim(0.0, 1.02)
        ax.set_title("Per-center id_val accuracy (full dev): GroupDRO vs matched ERM")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_PERCENTER), dpi=150); plt.close(fig)

        # high-confidence FN comparison (ood_val, raw)
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        x = np.arange(len(CONF_THRESHOLDS)); width = 0.38
        for i, key in enumerate(MODEL_ORDER):
            vals = [a["high_confidence_false_negatives"] for a in hcfn[(key, "ood_val")]]
            ax.bar(x + (i - 0.5) * width, vals, width, color=color[key], label=MODEL_LABEL[key])
        ax.set_xticks(x); ax.set_xticklabels([f"{t:.2f}" for t in CONF_THRESHOLDS])
        ax.set_xlabel("confidence threshold")
        ax.set_ylabel("high-confidence missed tumors (ood_val, raw)")
        ax.set_title("High-confidence false negatives - full ood_val (raw)")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_HCFN), dpi=150); plt.close(fig)

        figure_paths = [OUT_FIG_ROC, OUT_FIG_PR, OUT_FIG_FIXED_SPEC, OUT_FIG_PERCENTER, OUT_FIG_HCFN]

        # --- save tables -------------------------------------------------------- #
        write_csv(_resolve(OUT_OVERALL_CSV), overall_rows,
                  header=["model", "model_label", "split", *_CLS_FIELDS])
        write_csv(_resolve(OUT_PERCENTER_CSV), percenter_rows,
                  header=["model", "model_label", "center", "n", "label0", "label1",
                          "accuracy", "auroc", "auprc", "sensitivity", "specificity", "loss"])
        write_csv(_resolve(OUT_OPS_CSV), ops_rows,
                  header=["model", "model_label", "target_type", "target_value",
                          "threshold_selected_on_id_val",
                          "id_val_sensitivity", "id_val_specificity", "id_val_FN", "id_val_FP",
                          "ood_val_sensitivity", "ood_val_specificity", "ood_val_FN", "ood_val_FP",
                          "ood_val_predicted_positive_rate", "candidate_only_note"])
        write_csv(_resolve(OUT_HCFN_CSV), hcfn_rows,
                  header=["model", "model_label", "split", "confidence_threshold",
                          "n_high_confidence", "high_confidence_errors",
                          "high_confidence_false_negatives",
                          "high_confidence_false_negative_rate", "total_false_negatives"])

        # --- metrics JSON ------------------------------------------------------- #
        payload = {
            "milestone": "8B",
            "description": "Full/larger development-scale evaluation of the two frozen 7F models "
                           "(primary GroupDRO candidate + matched center-stratified ERM control) on "
                           "id_val {0,3,4} and ood_val {1}. Inference-only; CSV summary. Candidate "
                           "operating points only (id_val-selected); no committed threshold.",
            "positive_class": 1, "default_threshold": DEFAULT_THRESHOLD,
            "splits": {s: {"expected_n": EXPECTED_N[s],
                           "n": int(len(preds[("groupdro", s)])),
                           "centers": sorted(int(c) for c in preds[("groupdro", s)]["center"].unique())}
                       for s in SPLITS},
            "overall_metrics": {f"{key}_{split}": {k: overall[(key, split)][k] for k in ["n", *_CLS_FIELDS]}
                                for key in MODEL_ORDER for split in SPLITS},
            "per_center_id_val": {key: percenter[key] for key in MODEL_ORDER},
            "candidate_operating_points": ops_rows,
            "high_confidence_false_negatives": {f"{key}_{split}": hcfn[(key, split)]
                                                for key in MODEL_ORDER for split in SPLITS},
            "controlled_comparison": {
                "delta_ood_auroc_gdro_minus_erm": d_auroc,
                "delta_ood_auprc_gdro_minus_erm": d_auprc,
                "delta_ood_default_sensitivity_gdro_minus_erm": d_sens,
                "fixedspec_gdro_ge_erm": f"{n_ge}/{n_tgt}",
                "ood_ranking_better": bool(ood_ranking_better),
                "fixedspec_better": bool(fixedspec_better),
                "groupdro_beats_matched_erm_full_scale": groupdro_beats_erm,
                "worst_center_id_val": {key: _worst_center(key) for key in MODEL_ORDER},
            },
            "note": "DEVELOPMENT-stage evaluation on full/larger development splits - NOT final "
                    "performance. ood_val used for development-stage OOD analysis only. No "
                    "calibration fitted, no MC-dropout/ensemble/CORAL/DANN, no committed operating "
                    "point. ood_test never read.",
        }
        write_text(_resolve(OUT_METRICS_JSON), json.dumps(payload, indent=2) + "\n")

        # --- assertions --------------------------------------------------------- #
        for p in (OUT_METRICS_JSON, OUT_OVERALL_CSV, OUT_PERCENTER_CSV, OUT_OPS_CSV, OUT_HCFN_CSV):
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"output missing/empty: {p}"
        for p in figure_paths:
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"figure missing/empty: {p}"

        # --- output paths ------------------------------------------------------- #
        emit("")
        emit("OUTPUT PATHS")
        emit("-" * 60)
        emit(f"  metrics json : {OUT_METRICS_JSON}")
        emit("  tables:")
        for p in (OUT_OVERALL_CSV, OUT_PERCENTER_CSV, OUT_OPS_CSV, OUT_HCFN_CSV):
            emit(f"    {p}")
        emit("  figures:")
        for p in figure_paths:
            emit(f"    {p}")
        emit(f"  report       : {OUT_REPORT_TXT}")
        emit("  ood_test usage: NONE")

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
    print(f"\nSaved report -> {OUT_REPORT_TXT}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
