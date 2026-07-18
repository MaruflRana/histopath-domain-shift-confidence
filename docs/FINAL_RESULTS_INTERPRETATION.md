# Final Results Interpretation

## The development signal and final result are different findings

The full-development comparison on `ood_val` center 1 favored the predeclared GroupDRO primary candidate. GroupDRO achieved AUROC 0.895609 and AUPRC 0.895795, compared with AUROC 0.867271 and AUPRC 0.875723 for the matched center-stratified ERM control. At the default threshold of 0.5, GroupDRO also had higher sensitivity (0.640672 versus 0.607380) and fewer false negatives (6,271 versus 6,852). These values were valid development evidence, but center 1 had already been used for model assessment, calibration, and operating-point analysis; it was not an unbiased final hospital.

The single authorized evaluation on reserved `ood_test` center 2 reversed that ordering. GroupDRO achieved AUROC 0.6633704256200204 and AUPRC 0.63641261236421, whereas the matched ERM control achieved AUROC 0.6984352121958427 and AUPRC 0.6556283801091156. At threshold 0.5, GroupDRO accuracy was 0.5337197545089002, sensitivity 0.11056505278999224, specificity 0.9568744562278082, precision 0.7194002447980417, F1 0.19167193200578847, and false negatives 37,825. ERM accuracy was 0.5711547957768006, sensitivity 0.24107037881816257, specificity 0.9012392127354386, precision 0.7093827843897038, F1 0.3598518752522859, and false negatives 32,275.

The controlled final verdict is therefore negative for the predeclared primary: the matched ERM control outperformed the predeclared GroupDRO primary on AUROC, AUPRC, accuracy, sensitivity, F1, and false-negative count on the reserved center. GroupDRO retained higher specificity and slightly higher precision. ERM must still be described as the matched control; the result does not authorize a post-test model switch or retrospective relabeling of the primary.

## The development-to-test reversal is the central scientific narrative

GroupDRO minus ERM AUROC changed from approximately +0.0283 on development center 1 to −0.0351 on final center 2. This supports the narrow statement that the development GroupDRO advantage did not generalize to center 2. It does not support universal ERM superiority or universal GroupDRO failure. Instead, it demonstrates hospital-specific instability in model ordering and shows why a development OOD hospital cannot substitute for a reserved final hospital.

The negative model result and the positive protocol contribution should be kept separate:

- **Negative model result:** GroupDRO did not retain its development advantage on the reserved hospital.
- **Positive protocol contribution:** the locked, one-shot reserved-hospital design revealed that reversal and prevented development-stage evidence from being misreported as final performance.

## Frozen calibration improved confidence reliability, not classification

The pre-frozen temperatures were applied without refitting: GroupDRO T=2.974907 and ERM T=3.496293. For GroupDRO, ECE improved from 0.4006439581474795 to 0.2583399203510463, Brier score from 0.41233859378344173 to 0.3096680541395694, and NLL from 1.9441627836686814 to 0.8743126064429281. For ERM, ECE improved from 0.3084182197890404 to 0.1489841758680231, Brier score from 0.3446598909605201 to 0.2555235818479269, and NLL from 1.4183276475805955 to 0.7130964662716416.

The allowed interpretation is that frozen temperature scaling improved held-out calibration metrics for both models on center 2. Calibrated ERM remained better calibrated than calibrated GroupDRO. Temperature scaling was argmax-invariant: accuracy, sensitivity, specificity, confusion counts, and total false negatives did not change. This is held-out evidence for the pre-frozen temperatures on one hospital, not proof of universal or multi-center calibration validity.

## Development-selected operating points were unstable across hospitals

All 14 thresholds were selected on `id_val` and frozen before test access. They were not optimized on center 1 or center 2. Their final behavior shows that nominal development targets were not reliably preserved under the new hospital shift.

The fixed-sensitivity family transferred especially poorly. GroupDRO thresholds nominally targeting sensitivity 0.80, 0.90, and 0.95 achieved final sensitivities 0.1430, 0.2884, and 0.4748. ERM achieved 0.3573, 0.5223, and 0.6692. Fixed-specificity results were closer to their nominal targets in some cases, but were not exact and were accompanied by low sensitivity. These results illustrate operating-point instability rather than identifying a better post-test threshold. Every threshold remains candidate/non-clinical.

## Calibration reduced confidently stated missed tumors

GroupDRO high-confidence false negatives changed from 29,775 to 5,479 at confidence ≥0.90, from 25,620 to 755 at ≥0.95, and from 15,485 to 1 at ≥0.99 after calibration. ERM changed from 19,142 to 1,828, from 14,626 to 513, and from 7,354 to 0 at the same thresholds.

These reductions show that calibration corrected confidence magnitudes assigned to many errors. They do not indicate that tumors were newly detected: total false negatives remained 37,825 for GroupDRO and 32,275 for ERM. The concise interpretation is: **confidence correction is not error correction**.

## Claims supported by the completed study

- Final held-out center-2 metrics for both frozen models may be reported.
- The matched ERM control outperformed the predeclared GroupDRO primary on center 2 for ranking, accuracy, sensitivity, F1, and false-negative count.
- The development GroupDRO advantage reversed on the reserved hospital.
- Frozen temperature scaling improved held-out ECE, Brier score, and NLL for both models without changing hard predictions.
- Development-selected operating targets were not reliably preserved on center 2.
- Calibration reduced high-confidence false negatives but not total false negatives.
- Strict reserved-test discipline exposed model-selection and operating-policy risk.

## Claims not supported

- Clinical deployment or triage readiness.
- Robustness to all hospitals or patient populations.
- Universal superiority of ERM or universal failure of GroupDRO.
- Independent multi-center calibration validity.
- A clinically validated threshold or abstention policy.
- Whole-slide or patient-level effectiveness.
- Superiority over CORAL, DANN, ensembles, MC-dropout, or any untested method.
- Any post-test model switch based on the center-2 result.
