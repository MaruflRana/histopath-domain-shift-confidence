# FIGURE / TABLE MANIFEST

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in
Histopathology: Calibration, Uncertainty, and Selective Prediction

Current post-test reporting inventory after Milestone 9B. The machine-readable source is
`results/tables/exp09b_manuscript_table_figure_manifest.csv`. Historical 6A/7H/8B/8C artifacts
remain available for provenance and supplementary context, but the items below are the preferred
final manuscript set.

All captions must distinguish:

- full-development `ood_val` center 1 from final held-out `ood_test` center 2;
- predeclared GroupDRO primary from matched ERM control;
- raw from frozen-temperature calibrated reliability;
- candidate/non-clinical operating points from clinical thresholds.

## Recommended main-paper tables

| Item | Artifact | Purpose | Required caveat |
|---|---|---|---|
| Table 1 | `results/tables/exp09b_development_vs_final_comparison.csv` | Full development `id_val`, development center 1, and final center 2 metrics for both frozen models. | Development and final rows must not be conflated. |
| Table 2 | `results/tables/exp09b_final_model_comparison.csv` | Metric-level final GroupDRO-versus-ERM comparison and deltas. | GroupDRO remains the predeclared primary; ERM remains the matched control. |
| Table 3 | `results/tables/exp09b_final_calibration_summary.csv` | Raw versus frozen-temperature calibrated ECE, Brier score, and NLL. | No calibration refit; hard predictions and total FNs unchanged. |

## Recommended supplementary tables

| Item | Artifact | Purpose | Required caveat |
|---|---|---|---|
| Table S1 | `results/tables/exp09b_operating_point_transfer.csv` | Transfer of all 14 frozen `id_val`-selected operating points to final center 2. | Candidate/non-clinical; no test-set threshold selection. |
| Table S2 | `results/tables/exp09b_high_confidence_fn_summary.csv` | High-confidence FN burden at 0.90/0.95/0.99, raw and calibrated. | Calibration changes confidence, not total errors. |
| Table S3 | `results/tables/exp09b_final_claims_audit.csv` | Machine-readable allowed/conditional/forbidden claim map. | Wording is governed by `docs/FINAL_CLAIMS_AND_LIMITATIONS.md`. |
| Table S4 | `results/tables/exp09b_manuscript_table_figure_manifest.csv` | Machine-readable package inventory and placement. | Administrative/reproducibility artifact. |

## Recommended main-paper figures

| Item | Artifact | Purpose | Required caveat |
|---|---|---|---|
| Figure 1 | `results/figures/exp09b_development_test_reversal.png` | Shows the positive development GroupDRO AUROC difference and negative final difference. | Does not imply universal ERM superiority. |
| Figure 2 | `results/figures/exp09b_development_to_final_auroc_auprc.png` | AUROC and AUPRC across development center 1 and final held-out center 2. | Scope-tag both stages. |
| Figure 3 | `results/figures/exp09b_final_default_threshold_metrics.png` | Final accuracy, sensitivity, specificity, precision, and F1 at 0.5. | The 0.5 threshold is a reporting default, not clinically validated. |
| Figure 4 | `results/figures/exp09b_final_calibration_raw_vs_calibrated.png` | Held-out ECE, Brier score, and NLL before/after frozen temperature scaling. | Temperatures were frozen; hard predictions unchanged. |

## Recommended supplementary figures

| Item | Artifact | Purpose | Required caveat |
|---|---|---|---|
| Figure S1 | `results/figures/exp09b_operating_point_transfer.png` | Nominal development targets versus achieved final sensitivity/specificity. | Candidate/non-clinical thresholds selected on `id_val` only. |
| Figure S2 | `results/figures/exp09b_high_confidence_fn_raw_vs_calibrated.png` | Raw/calibrated high-confidence FN counts at 0.90/0.95/0.99. | Confidence correction is not error correction. |

## Existing final-test figures retained unchanged

These Milestone 9A figures are valid final-test source figures and were not modified in 9B:

- `results/figures/exp09_final_ood_test_roc.png`
- `results/figures/exp09_final_ood_test_precision_recall.png`
- `results/figures/exp09_final_ood_test_calibration_reliability.png`
- `results/figures/exp09_final_ood_test_operating_points.png`
- `results/figures/exp09_final_ood_test_high_confidence_fn.png`

They are supplementary candidates when curve-level detail is needed.

## Development provenance retained

- Full-development controlled metrics: `results/tables/exp08b_overall_metrics.csv`
- Full-development per-center metrics: `results/tables/exp08b_id_val_per_center_metrics.csv`
- Frozen operating-point selection source:
  `results/tables/exp08b_candidate_operating_points.csv`
- Development calibration: `results/tables/exp08c_calibration_comparison.csv`
- Development high-confidence FN audit:
  `results/tables/exp08c_high_confidence_fn_comparison.csv`
- Development reliability summary: `results/tables/exp08c_reliability_summary.csv`

These artifacts are development evidence, not substitutes for final center-2 reporting.

## Current state

Milestone **9B** is complete. The final manuscript package exists. No exp09 prediction, metric,
authorization, run-state, sentinel, or existing final figure was modified. No second `ood_test`
inference is authorized. The next action is literature/citation verification and journal-template
formatting.
