# CLAIMS AND LIMITATIONS

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in
Histopathology: Calibration, Uncertainty, and Selective Prediction

This is the current post-test claim-governance summary after Milestone 9B. The detailed manuscript
map is `docs/FINAL_CLAIMS_AND_LIMITATIONS.md`; its machine-readable counterpart is
`results/tables/exp09b_final_claims_audit.csv`. Historical development-only claim maps are
superseded for active manuscript use but remain traceable through `docs/MILESTONE_LOG.md` and the
8E source artifacts.

## Scope that must accompany every result

- **Development:** full `ood_val`, center 1, n=34,904, used for development assessment.
- **Final held-out:** one authorized `ood_test` run, center 2, n=85,054 per model.
- **Roles:** 7F GroupDRO is the predeclared primary; 7F center-stratified ERM is the matched
  control. The roles do not change because of the final result.
- **Operating points:** all 14 thresholds were selected on `id_val` before test access and remain
  candidate/non-clinical.
- **Calibration:** temperatures were frozen before test access and applied without refitting.

## Allowed claims

| Claim | Required wording/caveat | Supporting artifact |
|---|---|---|
| GroupDRO was stronger than matched ERM on development center 1 AUROC. | “On full-development `ood_val` center 1, GroupDRO AUROC was 0.8956 versus 0.8673 for matched ERM.” Do not call this final performance. | `results/tables/exp08b_overall_metrics.csv` |
| Matched ERM outperformed the predeclared GroupDRO primary on final center 2 ranking and several default-threshold metrics. | Name ERM as the matched control and GroupDRO as the predeclared primary. Report the unfavorable result directly. | `results/tables/exp09_final_ood_test_classification.csv` |
| The development GroupDRO advantage reversed on the reserved hospital. | Limit the conclusion to the observed development center 1 and final center 2 comparison. | `results/tables/exp09b_development_vs_final_comparison.csv` |
| GroupDRO retained higher specificity and slightly higher precision on center 2. | Present alongside its worse AUROC, AUPRC, accuracy, sensitivity, F1, and FN burden. | `results/tables/exp09b_final_model_comparison.csv` |
| Frozen temperature scaling improved held-out ECE, Brier score, and NLL for both models. | State that temperatures were pre-frozen, not refit, and that hard predictions and total FNs did not change. | `results/tables/exp09b_final_calibration_summary.csv` |
| Calibrated ERM remained better calibrated than calibrated GroupDRO on center 2. | This is a one-hospital held-out result, not universal calibration validity. | `results/tables/exp09_final_ood_test_calibration.csv` |
| Frozen development-selected operating-point targets transferred poorly. | Describe thresholds as candidate/non-clinical and state that no threshold was selected on `ood_test`. | `results/tables/exp09b_operating_point_transfer.csv` |
| Calibration sharply reduced high-confidence false negatives. | Add: total false negatives were unchanged; confidence correction is not error correction. | `results/tables/exp09b_high_confidence_fn_summary.csv` |
| Strict reserved-test discipline exposed development-to-test model-selection risk. | Frame this as a protocol/reproducibility contribution, not proof that one learning algorithm is universally superior. | `docs/FINAL_RESULTS_INTERPRETATION.md` |

## Conditional claims

| Claim | Condition |
|---|---|
| Hospital shift degraded model ranking, sensitivity, and reliability. | Name the exact observed centers and metrics; do not generalize to all hospitals. |
| GroupDRO may have adapted to center-1-specific heterogeneity. | Mark explicitly as a hypothesis requiring additional multi-center validation. |
| Validation-center overfitting may explain the reversal. | Mark explicitly as a hypothesis; no post-test experiment was conducted to establish causality. |
| Calibration may support safer confidence communication. | State that it did not repair classification errors or validate deployment. |
| Candidate thresholds illustrate operating-point instability. | Do not call any threshold deployable, optimal, or clinically validated. |

## Forbidden claims

- Clinical deployment readiness, clinical safety, or suitability for autonomous triage.
- A clinically validated or committed operating threshold.
- Robustness or calibration validity across all hospitals.
- Universal ERM superiority or universal GroupDRO failure.
- Post-test promotion of matched ERM to “predeclared primary.”
- Independent multi-center external validation.
- WSI-level or patient-level effectiveness; the evaluation is patch-level.
- Superiority over CORAL, DANN, MC-dropout, ensembles, or any other untested method.
- Calibration improved accuracy, sensitivity, hard predictions, or total false-negative count.
- Thresholds were selected, optimized, or tuned on `ood_test`.
- A second `ood_test` run, replacement final result, or exploratory post-test model selection.
- Clinical conclusions from the capped training cache or a claim that the cache represents the
  full multi-hospital distribution.

## Current limitations that must remain visible

- One final reserved hospital within Camelyon17; no external non-Camelyon cohort.
- Patch-level rather than WSI/patient-level analysis.
- ResNet-18 and a capped, center-stratified training cache.
- Poor final default-threshold sensitivity and large false-negative burdens for both models.
- Residual held-out miscalibration after temperature scaling.
- Poor transport of development-selected operating points.
- Center identity is an imperfect proxy for all domain variation.
- No clinically validated threshold.
- No CORAL/DANN comparison and no model-based uncertainty; both remain optional future work.
- The one-shot final run cannot be repeated without violating the frozen protocol.

The latest completed milestone is **9B**. The final manuscript package exists, `ood_test` must not
be accessed again, and the next action is literature/citation verification followed by
journal-template formatting.
