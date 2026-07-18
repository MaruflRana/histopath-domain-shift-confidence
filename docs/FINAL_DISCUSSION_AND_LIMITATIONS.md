# Final Discussion and Limitations

## Evidence-supported interpretation

The central empirical finding is a development-to-test reversal. GroupDRO outperformed its matched ERM control on the full development OOD hospital, center 1, but the matched control outperformed the predeclared GroupDRO primary on the reserved center-2 evaluation. This result demonstrates that model ordering can depend on the hospital used for evaluation. It also validates the study's decision to keep a separate hospital unread until the model pair, temperatures, metrics, and operating policies were frozen.

The final result is not evidence that ERM is universally better than GroupDRO. GroupDRO improved development center-1 discrimination and improved worst-center performance within the source-center `id_val` analysis, especially on center 4. Conversely, center 2 favored ERM for AUROC, AUPRC, accuracy, sensitivity, F1, and false-negative count. The supported conclusion is therefore hospital-specific: the development GroupDRO advantage did not generalize to the reserved hospital.

## Why GroupDRO may have helped center 1 but not center 2

Several explanations are plausible, but the present artifacts do not identify a causal mechanism.

**Evidence-supported observations** are that GroupDRO optimized center-grouped losses over source hospitals 0, 3, and 4; it improved the worst source-validation center; it also improved development center-1 performance; and its advantage reversed on center 2.

**Hypotheses requiring future investigation** include:

- Center 1 may share staining, acquisition, or tissue-composition characteristics with the source-center variation emphasized by GroupDRO, whereas center 2 may differ along features not represented by the source grouping.
- Grouping solely by hospital center may be too coarse. Within-center scanner, stain batch, slide, patient, or tissue differences could be more relevant than center identity.
- Repeated development decisions using center 1 may have produced validation-center overfitting at the project level, even though test labels were never used for training.
- The short, capped, center-balanced training cache may have limited the representation needed for robust transfer to a qualitatively different hospital.

These hypotheses should not be presented as established explanations without targeted image, metadata, or representation analyses.

## Sensitivity and the false-negative burden

Both models had poor sensitivity at the default threshold on center 2, with GroupDRO sensitivity 0.1106 and ERM sensitivity 0.2411. The corresponding false-negative counts were 37,825 and 32,275. GroupDRO's higher specificity reflects a strongly conservative positive-call pattern rather than an overall clinical advantage. The high false-negative burden is incompatible with a clinical-readiness claim and makes threshold-free discrimination, threshold-specific behavior, and confidence reliability all necessary parts of interpretation.

The default threshold of 0.5 was predeclared for reporting, not clinically validated. A different threshold could trade sensitivity against specificity, but the final test cannot be used to choose it. Any future clinical operating policy would require independent threshold selection and prospective validation outside the completed one-shot test.

## Calibration cannot repair classification errors

Frozen temperature scaling improved ECE, Brier score, and NLL for both models on center 2 and reduced the number of errors assigned extreme confidence. This is a meaningful confidence-reliability result: many missed tumors were no longer described by the model as highly certain negatives.

However, temperature scaling is argmax-invariant. It left every hard prediction and the total number of false negatives unchanged. Calibration therefore cannot compensate for weak sensitivity or poor discrimination. The high-confidence false-negative audit should be described as a correction of stated confidence, not a recovery of missed tumors. In short, confidence correction is not error correction.

## Operating-point instability

Thresholds selected on `id_val` did not reliably preserve their nominal sensitivity or specificity targets on the reserved hospital. The fixed-sensitivity family showed especially large underachievement, while fixed-specificity targets were closer in some cases but still shifted and often accompanied by very low sensitivity. This demonstrates that operating thresholds are transportability-sensitive and should not be assumed to transfer across hospitals.

No threshold was selected on `ood_test`, and the completed study validates no clinical operating point. The 14 thresholds remain descriptive candidate/non-clinical points. Future deployment-oriented work would require threshold selection on data independent of the final evaluation and validation across multiple target hospitals.

## Scope limitations

### One reserved hospital

The final evaluation covers one unseen hospital, center 2. This is stronger than reusing the development OOD hospital, but it does not establish robustness across all hospitals, laboratories, scanners, or patient populations.

### Patch-level evaluation

Metrics are computed at the 96×96-patch level. Patch-level discrimination and calibration do not directly establish whole-slide, lymph-node, patient-level, or clinical-workflow effectiveness. Correlation among patches from the same slide or patient is not converted into a patient-level decision in this study.

### Model and training limitations

Both models use a ResNet-18 backbone and short training on capped, center-stratified caches rather than all available source patches. The cache was designed to make the GroupDRO comparison sound by balancing center-label cells, but it may omit variation important for transfer. Results should not be interpreted as the performance ceiling of ERM, GroupDRO, or modern histopathology architectures.

### Limited domain-generalization breadth

GroupDRO was the only dedicated DG objective in the final controlled pair. CORAL and DANN were not implemented, and no conclusion can be made about their relative performance. The study also does not test newer foundation models, stain-normalization pipelines, or larger architectures.

### Deterministic uncertainty only

The project reports softmax-derived confidence, entropy-based development analyses, and temperature scaling. It does not include MC-dropout, deep ensembles, or other model-based uncertainty methods. Their absence limits claims about epistemic uncertainty, but they are not required to interpret the completed deterministic reliability result.

### Calibration design

The temperatures were fit during development on center 1 and then applied unchanged to center 2. Improvement on center 2 supports transfer of these specific frozen scalars to one held-out hospital. It does not prove independent multi-center calibration validity, and no calibration parameter was estimated from center 2.

### No external non-Camelyon cohort

All evidence comes from Camelyon17-WILDS. Center 2 provides an internal reserved domain within that benchmark, not an external dataset from a separate study or health system.

### No clinical threshold or workflow

The study does not validate a clinical threshold, abstention policy, triage rule, or human-in-the-loop workflow. It reports candidate operating behavior to expose instability, not to recommend deployment.

## Implications for future multi-center validation

Future work should prioritize independent multi-hospital validation rather than reopening model selection on the completed center-2 result. A stronger design would include multiple target hospitals, a calibration set distinct from all evaluation hospitals, patient- or slide-level aggregation, confidence intervals that respect clustering, and prospective or temporally separated validation. Broader DG comparisons and model-based uncertainty could be evaluated in a new preregistered development cycle, but they must not replace or reinterpret the completed one-shot result.

The immediate implication is methodological: development OOD performance, calibration, and operating thresholds are not substitutes for a reserved-hospital evaluation. The final manuscript should preserve both sides of the evidence—the encouraging center-1 development result and the unfavorable center-2 final result—because their disagreement is the most informative outcome of the study.
