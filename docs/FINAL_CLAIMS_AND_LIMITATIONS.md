# Final Post-Test Claims and Limitations

This is the authoritative post-Milestone-9A claim map for the final manuscript. The machine-readable counterpart is `results/tables/exp09b_final_claims_audit.csv`.

## Allowed claims

| Claim | Supporting artifact | Exact allowed wording | Required caveat | Manuscript location |
|---|---|---|---|---|
| Final held-out center-2 metrics for both frozen models | `results/metrics/exp09_final_ood_test/final_metrics.json` | On the single reserved center-2 evaluation, the frozen GroupDRO primary and matched ERM control achieved the reported exp09 metrics. | Single hospital, patch-level, one authorized attempt; not all-hospital or clinical performance. | Results |
| Matched ERM outperformed the predeclared GroupDRO primary on center 2 | `results/tables/exp09b_final_model_comparison.csv` | The matched control outperformed the predeclared primary on AUROC, AUPRC, accuracy, sensitivity, F1, and false-negative count on the reserved center. | ERM remains the matched control; no post-test model selection or retrospective primary relabeling. | Abstract, Results, Discussion |
| Development advantage reversed on the reserved hospital | `results/tables/exp09b_development_vs_final_comparison.csv` | The GroupDRO advantage observed on development center 1 did not generalize to center 2. | This is a hospital-specific result, not universal ERM superiority or universal GroupDRO failure. | Abstract, Results, Discussion |
| Frozen temperature scaling improved held-out reliability | `results/tables/exp09b_final_calibration_summary.csv` | Applying the pre-frozen temperatures improved held-out ECE, Brier score, and NLL for both models on center 2. | No refit; hard predictions and total false negatives were unchanged; no universal calibration-validity claim. | Results, Discussion |
| Operating-point targets transferred poorly | `results/tables/exp09b_operating_point_transfer.csv` | The frozen `id_val`-selected operating targets were not reliably preserved under the center-2 hospital shift. | Candidate/non-clinical thresholds; no threshold selected on test. | Results, Discussion |
| Calibration reduced high-confidence false negatives but not total false negatives | `results/tables/exp09b_high_confidence_fn_summary.csv` | Calibration sharply reduced confidently stated missed tumors while leaving the underlying total number of missed tumors unchanged. | Confidence correction is not error correction. | Results, Discussion |
| Reserved-test discipline exposed model-selection risk | `results/logs/exp09_final_ood_test_report.txt` | The reserved-hospital protocol exposed a development-to-test reversal that would have been hidden by reporting `ood_val` as final performance. | Methodological contribution within this study; not proof that every protocol will show a reversal. | Discussion, Conclusion |

## Conditional claims

| Claim | Current status | Allowed wording now | Requirement before stronger wording |
|---|---|---|---|
| ERM should replace GroupDRO for clinical use | Conditional and currently unsupported | The center-2 result motivates further independent validation of the matched ERM control. | New independent multi-hospital validation and a separately validated operating policy. |
| Frozen calibration is transportable | Conditional | The pre-frozen temperatures improved calibration metrics on one reserved hospital. | Evaluation on multiple independent hospitals without refitting. |
| A threshold can support clinical triage | Conditional and currently unsupported | The development-selected thresholds were unstable on center 2. | Independent threshold selection, clinical utility analysis, and prospective validation. |
| Model-based uncertainty improves safety | Conditional and untested | Model-based uncertainty remains a possible future direction. | A new preregistered study implementing and evaluating the methods. |

## Forbidden claims

| Forbidden claim | Why forbidden |
|---|---|
| Clinical deployment, triage, or diagnostic readiness | Patch-level, single-dataset evaluation with poor sensitivity and no validated clinical workflow. |
| Robustness to all hospitals or populations | Only one reserved test hospital was evaluated. |
| Universal ERM superiority | ERM won on center 2, while GroupDRO won on development center 1. |
| Universal GroupDRO failure | GroupDRO improved development center 1 and worst-center source validation. |
| Independent multi-center calibration validity | Frozen temperatures were tested on one reserved center only. |
| A clinically validated threshold | All 14 operating points remain candidate/non-clinical. |
| Calibration reduced classification errors or total false negatives | Temperature scaling was argmax-invariant. |
| Whole-slide or patient-level effectiveness | The evaluation unit was a patch. |
| Superiority over CORAL, DANN, MC-dropout, ensembles, or other untested methods | These methods were not evaluated. |
| Post-test selection of ERM as the new predeclared primary | The primary and matched-control roles were frozen before center-2 access. |
| Reopening development and replacing the completed final result | The one-shot final inference is complete and cannot be rerun or superseded through test-driven iteration. |

## Universal final-manuscript caveats

1. The final result is a single reserved-hospital, patch-level evaluation on Camelyon17-WILDS center 2.
2. GroupDRO was the predeclared primary and ERM the matched control; their roles do not change after observing the result.
3. The development center-1 result and final center-2 result must be reported separately.
4. Temperatures and thresholds were frozen before test access; no calibration fitting or threshold tuning used center 2.
5. Calibration changed confidence magnitudes, not hard predictions or total false negatives.
6. Candidate operating points are non-clinical and did not reliably preserve their nominal targets.
7. ResNet-18, capped center-stratified training, patch-level analysis, one benchmark dataset, and one reserved hospital limit generalization.
8. No second `ood_test` inference is authorized.
