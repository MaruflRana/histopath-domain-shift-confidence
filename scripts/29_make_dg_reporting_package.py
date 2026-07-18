"""Milestone 7H / Script 29: consolidated domain-generalization (DG) reporting package.

**Reporting only. NO training. NO model inference. NO calibration fitting. NO MC-dropout.
NO ensemble. NO CORAL/DANN. NO ood_test. NO committed operating threshold. NO final
performance claim.**

This script consolidates the *already-saved* development-stage results through Milestone
7G into clean, paper-facing DG tables + figures + documentation. It reads existing CSV/JSON
artifacts read-only and writes new consolidated artifacts. It performs no computation on raw
images, no logits, no model forward passes — every number is copied/derived from saved
result files.

Interpretation guardrails (baked into every artifact):
  * The FAIR CONTROLLED comparison is **7F center-stratified ERM vs 7F GroupDRO** — both
    trained on the SAME sound center-stratified 7F-1 cache, id_val selection only, ood_val
    read only after training.
  * **Old plain ERM / 7A / 7D / 7E are CONTEXT-ONLY** relative to 7F: they were trained on
    the OLD confounded balanced cache (and a different id_val subset), so old-vs-new numbers
    are context, NOT a controlled comparison. GroupDRO does NOT beat the old plain ERM in a
    controlled comparison; that reference gap must not be read as a controlled loss.
  * `ood_val` is DEVELOPMENT-stage only (never training, never final test reporting).
  * `ood_test` is LOCKED and is NEVER read.
  * Candidate id_val-selected operating points are development candidates, NOT committed
    clinical thresholds.

Inputs (read-only): see INPUT_FILES below.

Outputs:
  Tables  (results/tables/exp07h_*.csv):
    1. exp07h_consolidated_model_comparison.csv
    2. exp07h_controlled_dg_comparison.csv
    3. exp07h_context_vs_controlled_warning_table.csv
    4. exp07h_groupdro_calibration_summary.csv
    5. exp07h_groupdro_operating_point_summary.csv
    6. exp07h_reviewer_risk_update.csv
  Figures (results/figures/exp07h_*.png):
    1. exp07h_ood_auroc_auprc_model_comparison.png
    2. exp07h_controlled_groupdro_gain.png
    3. exp07h_calibration_ece_brier_nll.png
    4. exp07h_high_confidence_fn_reduction.png
    5. exp07h_fixed_specificity_sensitivity.png
  Docs (docs/):
    1. DG_RESULTS_INTERPRETATION.md   (generated)
    2. DG_MANUSCRIPT_UPDATE.md        (generated)
  Report (results/logs/):
    exp07h_dg_reporting_package_summary.txt
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup; kept for repo convention)

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / non-interactive; no display required
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
LOGS = ROOT / "results" / "logs"
METRICS = ROOT / "results" / "metrics"
DOCS = ROOT / "docs"

# --------------------------------------------------------------------------------------
# Input artifacts (read-only). The parsed set is the primary source of every number; the
# broader "context" set is fingerprinted to guarantee we never mutate any saved result.
# --------------------------------------------------------------------------------------
PARSED_INPUTS = [
    TABLES / "exp07f_ood_val_metrics_comparison.csv",
    TABLES / "exp07f_overall_metrics_comparison.csv",
    TABLES / "exp07f_fixed_specificity_comparison.csv",
    TABLES / "exp07g_calibration_comparison.csv",
    TABLES / "exp07g_high_confidence_fn_comparison.csv",
    TABLES / "exp07g_candidate_operating_point_audit.csv",
    METRICS / "exp07g_7f_calibration" / "calibration_metrics.json",
    METRICS / "exp07g_7f_calibration" / "temperature.json",
]

# Additional saved artifacts consolidated by reference (fingerprinted read-only, not all
# re-parsed here — they are provenance for the reporting package).
CONTEXT_INPUTS = [
    TABLES / "report_table_classification_shift.csv",
    TABLES / "report_table_calibration.csv",
    TABLES / "report_table_uncertainty.csv",
    TABLES / "report_table_selective_prediction.csv",
    LOGS / "report_reliability_package_summary.txt",
    TABLES / "exp07b_default_threshold_comparison.csv",
    TABLES / "exp07b_threshold_free_comparison.csv",
    TABLES / "exp07b_fixed_specificity_operating_points.csv",
    TABLES / "exp07d_default_threshold_comparison.csv",
    TABLES / "exp07d_threshold_free_comparison.csv",
    TABLES / "exp07d_fixed_specificity_comparison.csv",
    TABLES / "exp07e_default_threshold_comparison.csv",
    TABLES / "exp07e_threshold_free_comparison.csv",
    TABLES / "exp07e_fixed_specificity_comparison.csv",
    TABLES / "exp07f0_cache_metadata_summary.csv",
    TABLES / "exp07f0_group_label_counts.csv",
    TABLES / "exp07f0_dg_method_feasibility.csv",
    TABLES / "exp07f1_center_stratified_cache_availability.csv",
    TABLES / "exp07f1_center_stratified_cache_summary.csv",
    TABLES / "exp07f1_center_stratified_label_center_counts.csv",
    TABLES / "exp07f_id_val_per_center_metrics.csv",
]

# Existing docs updated by hand this milestone (script only VERIFIES they carry a 7H marker).
UPDATED_DOCS = [
    DOCS / "CLAIMS_AND_LIMITATIONS.md",
    DOCS / "REVIEWER_RISK_AUDIT.md",
    DOCS / "FIGURE_TABLE_MANIFEST.md",
]

WARNINGS: list[str] = []
ERRORS: list[str] = []


def _fingerprint(paths):
    fp = {}
    for p in paths:
        if p.exists():
            st = p.stat()
            fp[str(p)] = (st.st_size, st.st_mtime_ns)
    return fp


def f(x, nd=4):
    """Format a float safely (None -> 'n/a')."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.{nd}f}"


def main():
    for d in (TABLES, FIGURES, LOGS, DOCS):
        d.mkdir(parents=True, exist_ok=True)

    # ---- guardrail: ood_test must never appear in any input/output path ----
    all_paths = PARSED_INPUTS + CONTEXT_INPUTS + UPDATED_DOCS
    for p in all_paths:
        assert "ood_test" not in str(p).lower(), f"ood_test referenced in path: {p}"

    # ---- read-only fingerprint (before) ----
    fp_before = _fingerprint(PARSED_INPUTS + CONTEXT_INPUTS)

    # ---- required parsed inputs must exist ----
    missing = [str(p) for p in PARSED_INPUTS if not p.exists()]
    assert not missing, f"required inputs missing: {missing}"

    # ==================================================================================
    # LOAD
    # ==================================================================================
    ood = pd.read_csv(TABLES / "exp07f_ood_val_metrics_comparison.csv").set_index("model")
    calib = pd.read_csv(TABLES / "exp07g_calibration_comparison.csv")
    hcfn = pd.read_csv(TABLES / "exp07g_high_confidence_fn_comparison.csv")
    ops = pd.read_csv(TABLES / "exp07g_candidate_operating_point_audit.csv")
    with open(METRICS / "exp07g_7f_calibration" / "calibration_metrics.json") as fh:
        calib_json = json.load(fh)
    with open(METRICS / "exp07g_7f_calibration" / "temperature.json") as fh:
        temp_json = json.load(fh)

    def orow(key):
        return ood.loc[key]

    # high-confidence FN at conf>=0.90 by model/variant
    def hc_fn(model_name, variant):
        m = hcfn[(hcfn["model"] == model_name) & (hcfn["variant"] == variant)
                 & (hcfn["confidence_threshold"] == 0.9)]
        return int(m["n_high_conf_false_negative"].iloc[0])

    erm_hc_raw = hc_fn("center-strat ERM (7F)", "raw")
    erm_hc_cal = hc_fn("center-strat ERM (7F)", "calibrated")
    gd_hc_raw = hc_fn("GroupDRO (7F)", "raw")
    gd_hc_cal = hc_fn("GroupDRO (7F)", "calibrated")

    cal_erm = calib_json["models"]["centerstrat_erm"]["calibration"]
    cal_gd = calib_json["models"]["groupdro"]["calibration"]
    t_erm = calib_json["models"]["centerstrat_erm"]["temperature"]
    t_gd = calib_json["models"]["groupdro"]["temperature"]

    # ==================================================================================
    # TABLE 1 — consolidated model comparison (context + controlled)
    # ==================================================================================
    model_map = [
        ("old_plain_erm_context", "plain_old", "context_only",
         "old confounded balanced cache (pc1000; center<->label confound; center 4 absent)", 500,
         "Context-only reference; strongest ood_val ranking but NOT a controlled DG comparator (different cache)."),
        ("stain_aug_7a_context", "stain7a_old", "context_only",
         "old confounded balanced cache (pc1000) + moderate torchvision stain aug", 500,
         "Context-only; stain-aug lever, did not beat plain-ERM ranking."),
        ("stain_aug_7d_context", "stain7d_old", "context_only",
         "old confounded balanced cache (pc1000) + tuned torchvision stain aug v2", 500,
         "Context-only; stain-aug lever, did not beat plain-ERM ranking."),
        ("stain_space_7e_context", "stain7e_old", "context_only",
         "old confounded balanced cache (pc1000) + OD/HED stain-space aug", 500,
         "Context-only; strongest stain-aug at fixed-spec operating points but lowest ood_val AUROC."),
        ("centerstrat_erm_7f_controlled", "erm7f", "controlled",
         "NEW center-stratified sound cache (7F-1; centers {0,3,4} x labels balanced)", 450,
         "Matched control for GroupDRO (same sound cache). The fair DG baseline."),
        ("groupdro_7f_controlled", "groupdro7f", "controlled",
         "NEW center-stratified sound cache (7F-1; centers {0,3,4} x labels balanced)", 450,
         "First DG lever to beat its matched control on ood_val ranking (controlled vs 7F ERM)."),
    ]
    t1_rows = []
    for label, key, role, cache, idn, note in model_map:
        r = orow(key)
        t1_rows.append({
            "model": label,
            "comparison_role": role,
            "training_cache_type": cache,
            "id_val_n": idn,
            "ood_val_n": int(r["n"]),
            "ood_val_accuracy": round(float(r["accuracy"]), 6),
            "ood_val_auroc": round(float(r["auroc"]), 6),
            "ood_val_auprc": round(float(r["auprc"]), 6),
            "ood_val_sensitivity_default_0p5": round(float(r["sensitivity"]), 6),
            "ood_val_specificity_default_0p5": round(float(r["specificity"]), 6),
            "ood_val_fn_default_0p5": int(r["fn"]),
            "interpretation_note": note,
        })
    t1 = pd.DataFrame(t1_rows)
    t1.to_csv(TABLES / "exp07h_consolidated_model_comparison.csv", index=False)

    # ==================================================================================
    # TABLE 2 — controlled DG comparison (7F ERM vs 7F GroupDRO)
    # ==================================================================================
    erm = orow("erm7f")
    gd = orow("groupdro7f")

    def mkrow(metric, erm_v, gd_v, higher_better, note):
        delta = gd_v - erm_v
        return {
            "metric": metric,
            "centerstrat_erm": round(erm_v, 6),
            "groupdro": round(gd_v, 6),
            "delta_groupdro_minus_erm": round(delta, 6),
            "interpretation": note,
        }

    t2_rows = [
        mkrow("ood_val AUROC", float(erm["auroc"]), float(gd["auroc"]), True,
              "GroupDRO higher -> better OOD ranking (controlled)."),
        mkrow("ood_val AUPRC", float(erm["auprc"]), float(gd["auprc"]), True,
              "GroupDRO higher -> better OOD precision-recall (controlled)."),
        mkrow("default accuracy (0.5)", float(erm["accuracy"]), float(gd["accuracy"]), True,
              "GroupDRO higher; both low under shift (development)."),
        mkrow("default sensitivity (0.5)", float(erm["sensitivity"]), float(gd["sensitivity"]), True,
              "GroupDRO higher; both very low at the fixed 0.5 argmax."),
        mkrow("default specificity (0.5)", float(erm["specificity"]), float(gd["specificity"]), True,
              "Equal specificity at 0.5 (both 0.943)."),
        mkrow("default FN (0.5)", float(erm["fn"]), float(gd["fn"]), False,
              "GroupDRO fewer missed tumors at 0.5 (lower is better)."),
        mkrow("ECE raw", cal_erm["raw"]["ece"], cal_gd["raw"]["ece"], False,
              "GroupDRO better-calibrated raw (lower ECE)."),
        mkrow("ECE calibrated", cal_erm["calibrated"]["ece"], cal_gd["calibrated"]["ece"], False,
              "GroupDRO better-calibrated after temperature scaling (lower ECE)."),
        mkrow("Brier raw", cal_erm["raw"]["brier"], cal_gd["raw"]["brier"], False,
              "GroupDRO lower Brier raw."),
        mkrow("Brier calibrated", cal_erm["calibrated"]["brier"], cal_gd["calibrated"]["brier"], False,
              "GroupDRO lower Brier calibrated."),
        mkrow("NLL raw", cal_erm["raw"]["nll"], cal_gd["raw"]["nll"], False,
              "GroupDRO much lower NLL raw."),
        mkrow("NLL calibrated", cal_erm["calibrated"]["nll"], cal_gd["calibrated"]["nll"], False,
              "GroupDRO lower NLL calibrated."),
        mkrow("high-conf FN (conf>=0.90) raw", float(erm_hc_raw), float(gd_hc_raw), False,
              "GroupDRO fewer confidently-missed tumors raw (lower is better)."),
        mkrow("high-conf FN (conf>=0.90) calibrated", float(erm_hc_cal), float(gd_hc_cal), False,
              "Calibration cuts confidently-missed tumors for both; GroupDRO fewer."),
    ]
    t2 = pd.DataFrame(t2_rows)
    t2.to_csv(TABLES / "exp07h_controlled_dg_comparison.csv", index=False)

    # ==================================================================================
    # TABLE 3 — context vs controlled warning table
    # ==================================================================================
    t3_rows = [
        {
            "comparison": "old plain ERM vs 7F GroupDRO",
            "valid_as_controlled_comparison": "no",
            "reason": "Different training cache (old confounded pc1000 vs new sound center-stratified 7F-1) and a different id_val subset; not matched.",
            "allowed_interpretation": "Context/reference only: old plain ERM has higher ood_val AUROC (0.7696 vs 0.7129), reported as context, not as a controlled loss for GroupDRO.",
            "forbidden_interpretation": "Claiming GroupDRO beats OR loses to plain ERM in a controlled comparison; deriving any DG-superiority conclusion from this pair.",
        },
        {
            "comparison": "7A/7D/7E vs 7F GroupDRO",
            "valid_as_controlled_comparison": "no",
            "reason": "Stain-aug models trained on the old confounded cache with augmentation; not matched to the 7F sound cache.",
            "allowed_interpretation": "Context only: situates GroupDRO among prior development levers (stain augmentation never beat plain-ERM ranking).",
            "forbidden_interpretation": "Treating stain-aug-vs-GroupDRO deltas as a controlled DG benchmark result.",
        },
        {
            "comparison": "7F centerstrat ERM vs 7F GroupDRO",
            "valid_as_controlled_comparison": "yes",
            "reason": "Both trained on the SAME sound center-stratified 7F-1 cache, matched init/seed/schedule, id_val selection only, ood_val read only after training.",
            "allowed_interpretation": "The FAIR controlled DG result: GroupDRO improves ood_val AUROC/AUPRC and fixed-specificity sensitivity over its matched ERM control (development-stage).",
            "forbidden_interpretation": "Claiming final/clinical performance, ood_test conclusions, or committed operating thresholds.",
        },
    ]
    t3 = pd.DataFrame(t3_rows)
    t3.to_csv(TABLES / "exp07h_context_vs_controlled_warning_table.csv", index=False)

    # ==================================================================================
    # TABLE 4 — GroupDRO / ERM calibration summary (7G)
    # ==================================================================================
    def calib_row(name, temp, cal):
        ece_d = cal["calibrated"]["ece"] - cal["raw"]["ece"]
        brier_d = cal["calibrated"]["brier"] - cal["raw"]["brier"]
        nll_d = cal["calibrated"]["nll"] - cal["raw"]["nll"]
        concl = ("Temperature scaling improved confidence reliability (ECE/Brier/NLL all down); "
                 "argmax-invariant (hard predictions unchanged).")
        return {
            "model": name,
            "temperature": round(temp, 6),
            "ECE_raw": round(cal["raw"]["ece"], 6),
            "ECE_calibrated": round(cal["calibrated"]["ece"], 6),
            "ECE_delta": round(ece_d, 6),
            "Brier_raw": round(cal["raw"]["brier"], 6),
            "Brier_calibrated": round(cal["calibrated"]["brier"], 6),
            "Brier_delta": round(brier_d, 6),
            "NLL_raw": round(cal["raw"]["nll"], 6),
            "NLL_calibrated": round(cal["calibrated"]["nll"], 6),
            "NLL_delta": round(nll_d, 6),
            "confidence_reliability_conclusion": concl,
        }

    t4 = pd.DataFrame([
        calib_row("center-strat ERM (7F)", t_erm, cal_erm),
        calib_row("GroupDRO (7F)", t_gd, cal_gd),
    ])
    t4.to_csv(TABLES / "exp07h_groupdro_calibration_summary.csv", index=False)

    # ==================================================================================
    # TABLE 5 — GroupDRO operating-point summary (fixed specificity candidates)
    # ==================================================================================
    spec_ops = ops[ops["criterion"] == "specificity"].copy()
    t5_rows = []
    for target in [0.8, 0.85, 0.9, 0.95]:
        for model_name in ["center-strat ERM (7F)", "GroupDRO (7F)"]:
            m = spec_ops[(spec_ops["model"] == model_name)
                         & (spec_ops["target"] == target)].iloc[0]
            t5_rows.append({
                "target_specificity": target,
                "model": model_name,
                "threshold_selected_on_id_val": round(float(m["threshold_selected_on_id_val"]), 6),
                "id_val_sensitivity": round(float(m["id_val_sensitivity"]), 6),
                "id_val_specificity": round(float(m["id_val_specificity"]), 6),
                "ood_val_sensitivity": round(float(m["ood_val_sensitivity"]), 6),
                "ood_val_specificity": round(float(m["ood_val_specificity"]), 6),
                "ood_val_FN": int(m["ood_val_fn"]),
                "ood_val_FP": int(m["ood_val_fp"]),
                "candidate_only_note": "CANDIDATE development operating point (id_val-selected); NOT a committed clinical threshold.",
            })
    t5 = pd.DataFrame(t5_rows)
    t5.to_csv(TABLES / "exp07h_groupdro_operating_point_summary.csv", index=False)

    # ==================================================================================
    # TABLE 6 — reviewer risk update
    # ==================================================================================
    t6_rows = [
        {
            "risk_id": "R1",
            "risk": "Capped balanced subsets are not a real/final evaluation",
            "previous_severity": "HIGH",
            "current_status_after_7G": "OPEN (unchanged): all 7F/7G results are on capped balanced subsets (id_val n=450/500, ood_val n=2000).",
            "residual_severity": "HIGH",
            "mitigation_completed": "Every 7H artifact carries the development / capped-subset scope tag.",
            "remaining_action": "Full-split evaluation + final locked ood_test run (gated).",
            "manuscript_disclosure": "State capped-subset development scope in Limitations; frame as reliability/DG audit, not performance.",
        },
        {
            "risk_id": "R3",
            "risk": "No domain-generalization baseline",
            "previous_severity": "HIGH",
            "current_status_after_7G": "PARTLY MITIGATED (reduced): a dedicated DG method (GroupDRO-by-center) was implemented and, in the controlled 7F setup, beat its matched center-stratified ERM control on ood_val AUROC/AUPRC and 4/4 fixed-specificity sensitivity targets.",
            "residual_severity": "MEDIUM",
            "mitigation_completed": "GroupDRO implemented (src/training/groupdro.py); matched controlled comparison (7F) + calibration audit (7G) done; consolidated in 7H.",
            "remaining_action": "Only GroupDRO tested; add CORAL/DANN and full/final evaluation before any superiority claim (all gated).",
            "manuscript_disclosure": "Report GroupDRO as the first positive DG lever in the controlled setup; disclose that CORAL/DANN and final test remain future work.",
        },
        {
            "risk_id": "R4",
            "risk": "No full ood_val / no full-test evaluation yet",
            "previous_severity": "HIGH",
            "current_status_after_7G": "OPEN (unchanged): ood_test remains LOCKED and unread; all OOD numbers are the capped balanced ood_val subset.",
            "residual_severity": "HIGH",
            "mitigation_completed": "Reserved-test discipline maintained; ood_test never touched in 7A-7H.",
            "remaining_action": "Single final locked ood_test evaluation + full ood_val (gated, one-time).",
            "manuscript_disclosure": "Present reserved-test discipline as a protocol feature; no generalization number until the locked run.",
        },
        {
            "risk_id": "R7",
            "risk": "Calibration fitted and evaluated on the same ood_val subset",
            "previous_severity": "HIGH",
            "current_status_after_7G": "OPEN (unchanged): 7G temperature scaling for both 7F models was fit AND evaluated on the same capped ood_val subset (optimistic).",
            "residual_severity": "MEDIUM-HIGH",
            "mitigation_completed": "Disclosed as a limitation; calibration claims kept narrow ('improves calibration on this subset'); argmax-invariance stated.",
            "remaining_action": "Separate calibration vs evaluation folds (or fit-on-one-hospital/eval-on-another) before any 'calibration generalizes' claim (gated).",
            "manuscript_disclosure": "Prominent Limitations disclosure; narrow calibration claim only.",
        },
        {
            "risk_id": "R8",
            "risk": "Deterministic-only uncertainty",
            "previous_severity": "MEDIUM",
            "current_status_after_7G": "OPEN (unchanged): only single-forward-pass entropy/confidence + temperature scaling; no MC-dropout/ensemble.",
            "residual_severity": "MEDIUM",
            "mitigation_completed": "Scope limited to deterministic uncertainty; model-based uncertainty explicitly deferred/optional.",
            "remaining_action": "MC-dropout / deep ensembles (deferred/optional; gated).",
            "manuscript_disclosure": "Scope uncertainty claims to deterministic; present ID-vs-OOD gap as motivation for model-based uncertainty.",
        },
        {
            "risk_id": "R9",
            "risk": "Possible class-order / cache sampling bias",
            "previous_severity": "MEDIUM",
            "current_status_after_7G": "PARTLY MITIGATED: 7F-1 center-stratified rebuild removed the center<->label confound for DG training (per-(center,label) quota sampling across {0,3,4}); provenance documented.",
            "residual_severity": "MEDIUM",
            "mitigation_completed": "Center-stratified sound caches built (7F-1); label-ordered-stream handling + seeds documented.",
            "remaining_action": "Full-split evaluation to remove subsetting entirely (gated).",
            "manuscript_disclosure": "Document seeded balanced caching + center-stratified rebuild; report seeds and cache provenance.",
        },
    ]
    t6 = pd.DataFrame(t6_rows)
    t6.to_csv(TABLES / "exp07h_reviewer_risk_update.csv", index=False)

    # ==================================================================================
    # FIGURES
    # ==================================================================================
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

    # Fig 1 — ood_val AUROC & AUPRC across all 6 models (context vs controlled via hatch)
    order = ["plain_old", "stain7a_old", "stain7d_old", "stain7e_old", "erm7f", "groupdro7f"]
    disp = ["plain ERM\n(context)", "7A stain\n(context)", "7D stain\n(context)",
            "7E stain\n(context)", "7F ERM\n(CONTROLLED)", "7F GroupDRO\n(CONTROLLED)"]
    is_ctrl = [False, False, False, False, True, True]
    aurocs = [float(ood.loc[k, "auroc"]) for k in order]
    auprcs = [float(ood.loc[k, "auprc"]) for k in order]
    x = range(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar([i - w / 2 for i in x], aurocs, w, label="ood_val AUROC",
                color="#4C72B0", edgecolor="black")
    b2 = ax.bar([i + w / 2 for i in x], auprcs, w, label="ood_val AUPRC",
                color="#DD8452", edgecolor="black")
    for bars in (b1, b2):
        for bar, ctrl in zip(bars, is_ctrl):
            if ctrl:
                bar.set_hatch("//")
    ax.set_xticks(list(x))
    ax.set_xticklabels(disp, fontsize=8)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("ood_val AUROC / AUPRC — context-only vs CONTROLLED (hatched = controlled 7F)\n"
                 "Development-stage, capped balanced subset; ood_test LOCKED", fontsize=9)
    ax.legend(loc="lower left")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=6)
    fig.savefig(FIGURES / "exp07h_ood_auroc_auprc_model_comparison.png")
    plt.close(fig)

    # Fig 2 — controlled GroupDRO minus ERM gains
    gain_auroc = float(gd["auroc"]) - float(erm["auroc"])
    gain_auprc = float(gd["auprc"]) - float(erm["auprc"])
    gain_sens = float(gd["sensitivity"]) - float(erm["sensitivity"])
    # fixed-specificity sensitivity @0.90
    erm_fs90 = float(spec_ops[(spec_ops["model"] == "center-strat ERM (7F)")
                              & (spec_ops["target"] == 0.9)]["ood_val_sensitivity"].iloc[0])
    gd_fs90 = float(spec_ops[(spec_ops["model"] == "GroupDRO (7F)")
                             & (spec_ops["target"] == 0.9)]["ood_val_sensitivity"].iloc[0])
    gain_fs90 = gd_fs90 - erm_fs90
    labels = ["AUROC", "AUPRC", "default\nsensitivity", "fixed-spec\nsensitivity @0.90"]
    gains = [gain_auroc, gain_auprc, gain_sens, gain_fs90]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, gains, color="#55A868", edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("GroupDRO minus center-strat ERM (ood_val)")
    ax.set_title("CONTROLLED DG gain: 7F GroupDRO - 7F center-strat ERM (same sound cache)\n"
                 "Positive = GroupDRO better; development-stage; ood_test LOCKED", fontsize=9)
    for bar, g in zip(bars, gains):
        ax.annotate(f"+{g:.3f}" if g >= 0 else f"{g:.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom" if g >= 0 else "top", fontsize=8)
    fig.savefig(FIGURES / "exp07h_controlled_groupdro_gain.png")
    plt.close(fig)

    # Fig 3 — raw vs calibrated ECE/Brier/NLL for 7F ERM and GroupDRO
    metrics_names = ["ECE", "Brier", "NLL"]
    erm_raw = [cal_erm["raw"]["ece"], cal_erm["raw"]["brier"], cal_erm["raw"]["nll"]]
    erm_cal = [cal_erm["calibrated"]["ece"], cal_erm["calibrated"]["brier"], cal_erm["calibrated"]["nll"]]
    gd_raw = [cal_gd["raw"]["ece"], cal_gd["raw"]["brier"], cal_gd["raw"]["nll"]]
    gd_cal = [cal_gd["calibrated"]["ece"], cal_gd["calibrated"]["brier"], cal_gd["calibrated"]["nll"]]
    x = range(len(metrics_names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - 1.5 * w for i in x], erm_raw, w, label="ERM raw", color="#C44E52", edgecolor="black")
    ax.bar([i - 0.5 * w for i in x], erm_cal, w, label="ERM calibrated", color="#C44E52",
           edgecolor="black", hatch="//", alpha=0.7)
    ax.bar([i + 0.5 * w for i in x], gd_raw, w, label="GroupDRO raw", color="#4C72B0", edgecolor="black")
    ax.bar([i + 1.5 * w for i in x], gd_cal, w, label="GroupDRO calibrated", color="#4C72B0",
           edgecolor="black", hatch="//", alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("value (lower = better)")
    ax.set_title("7F calibration: raw vs calibrated (temperature scaling; argmax-invariant)\n"
                 "Development-stage on capped ood_val; ood_test LOCKED", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(FIGURES / "exp07h_calibration_ece_brier_nll.png")
    plt.close(fig)

    # Fig 4 — high-confidence FN (conf>=0.90) before vs after calibration
    models4 = ["7F ERM", "7F GroupDRO"]
    raw_hc = [erm_hc_raw, gd_hc_raw]
    cal_hc = [erm_hc_cal, gd_hc_cal]
    x = range(len(models4))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar([i - w / 2 for i in x], raw_hc, w, label="raw", color="#C44E52", edgecolor="black")
    b2 = ax.bar([i + w / 2 for i in x], cal_hc, w, label="calibrated", color="#55A868", edgecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models4)
    ax.set_ylabel("high-confidence FN count (conf >= 0.90)")
    ax.set_title("High-confidence missed tumors (conf>=0.90): raw vs calibrated\n"
                 "Development-stage on capped ood_val; ood_test LOCKED", fontsize=9)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{int(bar.get_height())}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8)
    fig.savefig(FIGURES / "exp07h_high_confidence_fn_reduction.png")
    plt.close(fig)

    # Fig 5 — ood_val sensitivity vs target id_val specificity, ERM vs GroupDRO
    targets = [0.8, 0.85, 0.9, 0.95]
    erm_sens = [float(spec_ops[(spec_ops["model"] == "center-strat ERM (7F)")
                               & (spec_ops["target"] == t)]["ood_val_sensitivity"].iloc[0]) for t in targets]
    gd_sens = [float(spec_ops[(spec_ops["model"] == "GroupDRO (7F)")
                              & (spec_ops["target"] == t)]["ood_val_sensitivity"].iloc[0]) for t in targets]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(targets, erm_sens, "o-", label="7F center-strat ERM", color="#C44E52")
    ax.plot(targets, gd_sens, "s-", label="7F GroupDRO", color="#4C72B0")
    for t, v in zip(targets, erm_sens):
        ax.annotate(f"{v:.3f}", (t, v), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7)
    for t, v in zip(targets, gd_sens):
        ax.annotate(f"{v:.3f}", (t, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xlabel("target id_val specificity (candidate operating point)")
    ax.set_ylabel("ood_val sensitivity")
    ax.set_title("Candidate fixed-specificity operating points (CONTROLLED 7F)\n"
                 "id_val-selected candidates only, NOT committed; development-stage; ood_test LOCKED", fontsize=9)
    ax.legend()
    ax.set_xticks(targets)
    fig.savefig(FIGURES / "exp07h_fixed_specificity_sensitivity.png")
    plt.close(fig)

    # ==================================================================================
    # DOC 1 — DG_RESULTS_INTERPRETATION.md
    # ==================================================================================
    doc1 = f"""# DG RESULTS INTERPRETATION (Milestone 7H)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology:
Calibration, Uncertainty, and Selective Prediction

_Generated by `scripts/29_make_dg_reporting_package.py` (Milestone 7H) from saved
CSV/JSON artifacts. Reporting only — no training, no inference, no calibration fitting.
All numbers are DEVELOPMENT-stage on capped balanced subsets. `ood_test` (center 2) is
LOCKED and was never read._

---

## One-paragraph summary

Under hospital-domain shift (train centers {{0,3,4}} → OOD-val center 1), the reliability of
an ID-trained tumor/non-tumor patch classifier degrades sharply. Within a **fair controlled
setup** — a matched center-stratified ERM control vs a GroupDRO-by-center model, both trained
on the same sound center-stratified cache (Milestone 7F-1) — **GroupDRO is the first
domain-generalization lever to beat its matched control on OOD ranking**: ood_val AUROC
{f(gd['auroc'])} vs {f(erm['auroc'])} (+{f(gd['auroc'] - erm['auroc'])}), AUPRC {f(gd['auprc'])}
vs {f(erm['auprc'])} (+{f(gd['auprc'] - erm['auprc'])}), and higher ood_val sensitivity at all
four id_val-selected fixed-specificity candidate points. Temperature scaling (Milestone 7G)
improves confidence reliability for both models (all ECE/Brier/NLL down) and sharply reduces
high-confidence missed tumors, but is argmax-invariant. GroupDRO is also the better-calibrated
model. This is a development-stage reliability/DG audit, not a final performance result.

## Why the 7F-1 center-stratified rebuild was necessary

The original capped balanced `train` cache was built by scanning the label-ordered HF stream
to fill per-class quotas. Milestone 7F-0 found this made the cache **center↔label confounded**:
center 0 was ~9% tumor, center 3 was 100% tumor, and source center 4 was entirely absent. A
center-grouped DG objective on that cache would learn *center == label* rather than a
domain-invariant tumor signal — runnable but **not sound**. Milestone 7F-1 rebuilt sound caches
by per-`(center,label)` quota sampling across centers {{0,3,4}} (train 300/cell → n=1800,
900/900, 600 per center; id_val 75/cell → n=450), preserving all metadata and the serializer
schema, on new filenames (no overwrite).

## Why the old caches were invalid for DG training

Center-based GroupDRO/CORAL/DANN require each center group to carry both labels. The old
confounded cache violated this (single-class centers; missing center 4), so any center-grouped
objective trained on it is unsound. The old plain ERM / 7A / 7D / 7E models were all trained on
that old cache (and a different id_val subset).

## Why matched center-stratified ERM is the fair control

Because the old plain-ERM checkpoint used the old confounded cache and a different id_val
subset, it is **not** a matched baseline for GroupDRO. The fair control is an ERM trained on the
**same** sound center-stratified 7F-1 cache with matched init/seed/schedule and id_val selection
only. That is exactly the 7F center-stratified ERM. Only `7F ERM vs 7F GroupDRO` isolates the
GroupDRO objective from cache/subset differences.

## Controlled GroupDRO result (7F ERM vs 7F GroupDRO)

| metric | center-strat ERM | GroupDRO | Δ (GroupDRO − ERM) |
|---|---:|---:|---:|
| ood_val AUROC | {f(erm['auroc'])} | {f(gd['auroc'])} | +{f(gd['auroc'] - erm['auroc'])} |
| ood_val AUPRC | {f(erm['auprc'])} | {f(gd['auprc'])} | +{f(gd['auprc'] - erm['auprc'])} |
| default sensitivity (0.5) | {f(erm['sensitivity'])} | {f(gd['sensitivity'])} | +{f(gd['sensitivity'] - erm['sensitivity'])} |
| default FN (0.5) | {int(erm['fn'])} | {int(gd['fn'])} | {int(gd['fn'] - erm['fn'])} |

GroupDRO wins all four id_val-selected fixed-specificity targets (ood_val sensitivity at
spec 0.80/0.85/0.90/0.95: {f(gd_sens[0],3)}/{f(gd_sens[1],3)}/{f(gd_sens[2],3)}/{f(gd_sens[3],3)}
vs ERM {f(erm_sens[0],3)}/{f(erm_sens[1],3)}/{f(erm_sens[2],3)}/{f(erm_sens[3],3)}). GroupDRO did
**not** improve worst-center id_val behavior (it up-weighted center 4).

## Calibration result (7G)

| model | T | ECE raw→cal | Brier raw→cal | NLL raw→cal |
|---|---:|---|---|---|
| center-strat ERM (7F) | {f(t_erm)} | {f(cal_erm['raw']['ece'])}→{f(cal_erm['calibrated']['ece'])} | {f(cal_erm['raw']['brier'])}→{f(cal_erm['calibrated']['brier'])} | {f(cal_erm['raw']['nll'])}→{f(cal_erm['calibrated']['nll'])} |
| GroupDRO (7F) | {f(t_gd)} | {f(cal_gd['raw']['ece'])}→{f(cal_gd['calibrated']['ece'])} | {f(cal_gd['raw']['brier'])}→{f(cal_gd['calibrated']['brier'])} | {f(cal_gd['raw']['nll'])}→{f(cal_gd['calibrated']['nll'])} |

Calibration improved all three metrics for both models but is **argmax-invariant** (default hard
predictions unchanged). High-confidence (conf≥0.90) missed tumors fell ERM {erm_hc_raw}→{erm_hc_cal}
and GroupDRO {gd_hc_raw}→{gd_hc_cal}. GroupDRO is better-calibrated than its matched ERM both raw
and calibrated.

## Operating-point result

Candidate id_val-selected fixed-specificity operating points show GroupDRO ood_val sensitivity ≥
ERM at all four targets (strictly greater at 4/4). These are **candidate development operating
points only — no threshold is committed**.

## Context-only comparison caveat

Old plain ERM (ood_val AUROC {f(orow('plain_old')['auroc'])}), 7A, 7D, and 7E are **context-only**
relative to 7F: they used the old confounded cache and a different id_val subset. The old plain
ERM's higher ood_val AUROC vs GroupDRO is a **reference gap, not a controlled loss** — it does not
invalidate the controlled GroupDRO result and must never be reported as GroupDRO losing a
controlled comparison. The only controlled comparison is 7F ERM vs 7F GroupDRO.

## What this supports

- In the controlled 7F setup, GroupDRO improves ood_val AUROC/AUPRC and fixed-specificity
  sensitivity over its matched center-stratified ERM control (development-stage).
- GroupDRO has fewer high-confidence false negatives than matched ERM after calibration in this
  development subset, and is the better-calibrated model.
- GroupDRO is retained as the first positive DG lever in the controlled 7F setup.

## What this does NOT support

- Any claim that GroupDRO beats the old plain ERM in a controlled comparison (different caches).
- Any clinical-readiness, final-generalization, or ood_test conclusion.
- Any committed/deployable operating threshold.
- Any superiority over untested DG methods (CORAL/DANN not run) or model-based uncertainty
  (MC-dropout/ensemble not run).

## `ood_test` statement

`ood_test` (center 2) remains **LOCKED** and unread. It may be used only once, for the single
final locked evaluation, after explicit written authorization. No 7H artifact references it.
"""
    (DOCS / "DG_RESULTS_INTERPRETATION.md").write_text(doc1, encoding="utf-8")

    # ==================================================================================
    # DOC 2 — DG_MANUSCRIPT_UPDATE.md
    # ==================================================================================
    doc2 = f"""# DG MANUSCRIPT UPDATE (Milestone 7H)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology:
Calibration, Uncertainty, and Selective Prediction

_Generated by `scripts/29_make_dg_reporting_package.py` (Milestone 7H). Manuscript-planning
guidance only — development-stage, capped balanced subsets; `ood_test` LOCKED and unread. No
final-performance claim; no committed operating threshold._

---

## Suggested manuscript section updates

### Methods — center-stratified cache construction
Describe the label-ordered HF stream and the center↔label confound in the naive balanced cache
(center 0 ~9% tumor, center 3 100% tumor, center 4 absent). Introduce the per-`(center,label)`
quota-sampling rebuild across centers {{0,3,4}} (train 300/cell → n=1800; id_val 75/cell → n=450),
which yields sound center groups for DG training. Note: same locked split mapping; new cache
filenames; no overwrite.

### Methods — GroupDRO training objective
State the group-DRO objective over center groups {{0,3,4}}: per-group mean CE, exponentiated-
gradient weight update `q_g ← q_g·exp(step·detached_L_g)` then renormalize, absent groups carried
forward; robust loss `Σ_g q_g·L_g`. Matched ERM control trained on the same cache with matched
init/seed/schedule; id_val_loss selection only; ood_val read only after training.

### Experiments — controlled ERM vs GroupDRO comparison
Emphasize the matched design and why it (not old plain ERM) is the fair comparator. Report the
context-only status of old plain ERM / 7A / 7D / 7E explicitly.

### Results — matched DG comparison
Report Table `exp07h_controlled_dg_comparison.csv`: GroupDRO ood_val AUROC {f(gd['auroc'])} vs
{f(erm['auroc'])} (+{f(gd['auroc'] - erm['auroc'])}), AUPRC {f(gd['auprc'])} vs {f(erm['auprc'])},
and 4/4 fixed-specificity sensitivity wins.

### Results — calibration and high-confidence false negatives
Report Table `exp07h_groupdro_calibration_summary.csv` and Fig
`exp07h_high_confidence_fn_reduction.png`: temperature scaling improves ECE/Brier/NLL for both
models (argmax-invariant), high-confidence missed tumors ERM {erm_hc_raw}→{erm_hc_cal}, GroupDRO
{gd_hc_raw}→{gd_hc_cal}.

### Discussion — why old plain ERM remains context-only
The old plain ERM's higher ood_val AUROC ({f(orow('plain_old')['auroc'])}) is a reference gap from
a different (confounded) training cache and id_val subset — not a controlled loss for GroupDRO.
Frame the controlled result (7F ERM vs 7F GroupDRO) as the scientific claim.

### Limitations
Development-stage; capped balanced subsets (id_val n=450/500, ood_val n=2000); single OOD-val
hospital (center 1); calibration fitted and evaluated on the same ood_val subset; no final locked
`ood_test` evaluation yet; only GroupDRO tested (no CORAL/DANN); deterministic uncertainty +
temperature scaling only (no MC-dropout/ensemble); patch-level only.

## Suggested figure/table placement

- **Main:** `exp07h_consolidated_model_comparison.csv` (Table), `exp07h_controlled_dg_comparison.csv`
  (Table), `exp07h_ood_auroc_auprc_model_comparison.png` (Fig), `exp07h_controlled_groupdro_gain.png`
  (Fig), `exp07h_calibration_ece_brier_nll.png` (Fig).
- **Supplementary:** `exp07h_context_vs_controlled_warning_table.csv`,
  `exp07h_groupdro_calibration_summary.csv`, `exp07h_groupdro_operating_point_summary.csv`,
  `exp07h_reviewer_risk_update.csv`, `exp07h_high_confidence_fn_reduction.png`,
  `exp07h_fixed_specificity_sensitivity.png`.

## Suggested wording for the core claim

> "In a controlled setup on matched center-stratified development data, a GroupDRO-by-center
> objective improved out-of-distribution (center-1) ranking and confidence reliability over its
> matched ERM control (ood_val AUROC {f(gd['auroc'])} vs {f(erm['auroc'])}); it was also better
> calibrated. Results are development-stage on capped balanced subsets; the reserved test hospital
> (center 2) was not evaluated, and no operating point was committed."
"""
    (DOCS / "DG_MANUSCRIPT_UPDATE.md").write_text(doc2, encoding="utf-8")

    # ==================================================================================
    # REPORT
    # ==================================================================================
    lines = []
    lines.append("MILESTONE 7H — CONSOLIDATED DG REPORTING PACKAGE (summary)")
    lines.append("=" * 78)
    lines.append("Reporting only: no training, no inference, no calibration fitting, no MC-dropout,")
    lines.append("no ensemble, no CORAL/DANN, no ood_test, no committed threshold, no final claim.")
    lines.append("")
    lines.append("CONTROLLED GroupDRO result (7F center-strat ERM vs 7F GroupDRO; same sound cache):")
    lines.append(f"  ood_val AUROC : ERM {f(erm['auroc'])}  ->  GroupDRO {f(gd['auroc'])}   (+{f(gd['auroc'] - erm['auroc'])})")
    lines.append(f"  ood_val AUPRC : ERM {f(erm['auprc'])}  ->  GroupDRO {f(gd['auprc'])}   (+{f(gd['auprc'] - erm['auprc'])})")
    lines.append(f"  default sens  : ERM {f(erm['sensitivity'])}  ->  GroupDRO {f(gd['sensitivity'])}")
    lines.append(f"  default FN    : ERM {int(erm['fn'])}  ->  GroupDRO {int(gd['fn'])}")
    lines.append(f"  fixed-spec ood_val sensitivity (0.80/0.85/0.90/0.95):")
    lines.append(f"      ERM      {f(erm_sens[0],3)}/{f(erm_sens[1],3)}/{f(erm_sens[2],3)}/{f(erm_sens[3],3)}")
    lines.append(f"      GroupDRO {f(gd_sens[0],3)}/{f(gd_sens[1],3)}/{f(gd_sens[2],3)}/{f(gd_sens[3],3)}   (GroupDRO wins 4/4)")
    lines.append("")
    lines.append("CALIBRATION result (7G; temperature scaling; argmax-invariant):")
    lines.append(f"  center-strat ERM T={f(t_erm)}: ECE {f(cal_erm['raw']['ece'])}->{f(cal_erm['calibrated']['ece'])}, "
                 f"Brier {f(cal_erm['raw']['brier'])}->{f(cal_erm['calibrated']['brier'])}, "
                 f"NLL {f(cal_erm['raw']['nll'])}->{f(cal_erm['calibrated']['nll'])}")
    lines.append(f"  GroupDRO       T={f(t_gd)}: ECE {f(cal_gd['raw']['ece'])}->{f(cal_gd['calibrated']['ece'])}, "
                 f"Brier {f(cal_gd['raw']['brier'])}->{f(cal_gd['calibrated']['brier'])}, "
                 f"NLL {f(cal_gd['raw']['nll'])}->{f(cal_gd['calibrated']['nll'])}")
    lines.append("  -> calibration improved all three metrics for BOTH; GroupDRO better-calibrated raw+cal.")
    lines.append("")
    lines.append("HIGH-CONFIDENCE FN (conf>=0.90) raw -> calibrated:")
    lines.append(f"  center-strat ERM: {erm_hc_raw} -> {erm_hc_cal}")
    lines.append(f"  GroupDRO        : {gd_hc_raw} -> {gd_hc_cal}")
    lines.append("")
    lines.append("FIXED-SPECIFICITY candidate operating points: GroupDRO ood_val sensitivity >= ERM at 4/4")
    lines.append("  (candidate development thresholds only; NONE committed).")
    lines.append("")
    lines.append("CONTEXT-ONLY caveat:")
    lines.append(f"  old plain ERM ood_val AUROC {f(orow('plain_old')['auroc'])}, 7A {f(orow('stain7a_old')['auroc'])}, "
                 f"7D {f(orow('stain7d_old')['auroc'])}, 7E {f(orow('stain7e_old')['auroc'])} are CONTEXT-ONLY")
    lines.append("  (old confounded cache + different id_val subset). GroupDRO does NOT beat old plain ERM in a")
    lines.append("  CONTROLLED comparison; that reference gap is not a controlled loss. Only 7F ERM vs 7F")
    lines.append("  GroupDRO is a controlled comparison.")
    lines.append("")
    lines.append("REVIEWER-RISK update:")
    lines.append("  R3 no-DG-baseline: HIGH -> PARTLY MITIGATED (GroupDRO implemented + beat matched ERM); residual MEDIUM.")
    lines.append("  R1 capped subsets: OPEN (HIGH).  R4 no full/final test: OPEN (HIGH).")
    lines.append("  R7 calibration same-subset: OPEN (MEDIUM-HIGH).  R8 deterministic-only: OPEN (MEDIUM).")
    lines.append("  R9 sampling/cache bias: PARTLY MITIGATED (7F-1 rebuild); residual MEDIUM.")
    lines.append("")
    lines.append("SCOPE: DEVELOPMENT analysis only on capped balanced subsets — NOT final performance.")
    lines.append("       ood_val used for development-stage comparison only; ood_test remains LOCKED and unread.")
    lines.append("")

    # ---- read-only fingerprint (after) ----
    fp_after = _fingerprint(PARSED_INPUTS + CONTEXT_INPUTS)
    changed = [k for k in fp_before if fp_before.get(k) != fp_after.get(k)]

    # ==================================================================================
    # ASSERTIONS
    # ==================================================================================
    assertions = []

    def check(name, ok):
        assertions.append((name, bool(ok)))
        if not ok:
            ERRORS.append(f"ASSERTION FAILED: {name}")

    # no ood_test usage anywhere
    check("no ood_test in any input/output/doc path",
          all("ood_test" not in str(p).lower() for p in all_paths))
    # inputs read-only / unmodified
    check("input artifacts unmodified (size+mtime fingerprint)", not changed)
    # 6 tables saved
    t_out = [
        TABLES / "exp07h_consolidated_model_comparison.csv",
        TABLES / "exp07h_controlled_dg_comparison.csv",
        TABLES / "exp07h_context_vs_controlled_warning_table.csv",
        TABLES / "exp07h_groupdro_calibration_summary.csv",
        TABLES / "exp07h_groupdro_operating_point_summary.csv",
        TABLES / "exp07h_reviewer_risk_update.csv",
    ]
    check("all 6 tables saved (non-empty)",
          all(p.exists() and p.stat().st_size > 0 for p in t_out))
    # 5 figures saved
    f_out = [
        FIGURES / "exp07h_ood_auroc_auprc_model_comparison.png",
        FIGURES / "exp07h_controlled_groupdro_gain.png",
        FIGURES / "exp07h_calibration_ece_brier_nll.png",
        FIGURES / "exp07h_high_confidence_fn_reduction.png",
        FIGURES / "exp07h_fixed_specificity_sensitivity.png",
    ]
    check("all 5 figures saved (non-empty)",
          all(p.exists() and p.stat().st_size > 0 for p in f_out))
    # 2 new docs saved
    new_docs = [DOCS / "DG_RESULTS_INTERPRETATION.md", DOCS / "DG_MANUSCRIPT_UPDATE.md"]
    check("2 new docs saved (non-empty)",
          all(p.exists() and p.stat().st_size > 0 for p in new_docs))
    # 3 existing docs updated (carry a Milestone 7H marker)
    marker_ok = []
    for p in UPDATED_DOCS:
        txt = p.read_text(encoding="utf-8") if p.exists() else ""
        marker_ok.append(("7H" in txt) or ("exp07h" in txt.lower()))
    check("3 existing docs carry a 7H marker (updated)", all(marker_ok))
    # content sanity: controlled result direction
    check("GroupDRO ood_val AUROC > ERM ood_val AUROC (controlled)",
          float(gd["auroc"]) > float(erm["auroc"]))
    check("GroupDRO ood_val AUPRC > ERM ood_val AUPRC (controlled)",
          float(gd["auprc"]) > float(erm["auprc"]))
    check("GroupDRO fixed-spec ood_val sensitivity >= ERM at all 4 targets",
          all(g >= e for g, e in zip(gd_sens, erm_sens)))
    check("calibration reduced ECE for both models",
          cal_erm["calibrated"]["ece"] < cal_erm["raw"]["ece"]
          and cal_gd["calibrated"]["ece"] < cal_gd["raw"]["ece"])
    # table row-count sanity
    check("table1 has 6 model rows", len(t1) == 6)
    check("table2 has 14 metric rows", len(t2) == 14)
    check("table3 has 3 comparison rows", len(t3) == 3)
    check("table5 has 8 operating-point rows", len(t5) == 8)
    check("table6 has 6 reviewer-risk rows", len(t6) == 6)

    all_pass = all(ok for _, ok in assertions)

    lines.append("ASSERTIONS:")
    for name, ok in assertions:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    lines.append("")
    lines.append(f"WARNINGS: {len(WARNINGS)}")
    for w_ in WARNINGS:
        lines.append(f"  - {w_}")
    lines.append(f"ERRORS: {len(ERRORS)}")
    for e_ in ERRORS:
        lines.append(f"  - {e_}")
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if all_pass and not ERRORS else 'FAIL'}")

    report_txt = "\n".join(lines)
    (LOGS / "exp07h_dg_reporting_package_summary.txt").write_text(report_txt, encoding="utf-8")
    print(report_txt)

    if not (all_pass and not ERRORS):
        sys.exit(1)


if __name__ == "__main__":
    main()
