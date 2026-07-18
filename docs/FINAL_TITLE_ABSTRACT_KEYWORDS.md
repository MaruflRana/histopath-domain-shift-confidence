# Final Title, Abstract, and Keywords

## Title options

1. **Hospital Domain Shift Can Reverse Development Gains in Histopathology Tumor Detection: A Reserved-Center Study of Calibration and Operating-Point Transfer**
2. **A Reserved-Hospital Evaluation of Calibration, Operating-Point Transfer, and Domain Generalization in Histopathology**
3. **Development-to-Test Reversal Under Hospital Shift: A Confidence-Aware Histopathology Study**
4. **Confidence Under Hospital Shift: A Single-Shot Reserved-Center Evaluation of Tumor Detection**
5. **Calibration Helps Confidence, Not Errors: Reserved-Hospital Evaluation in Histopathology**
6. **Testing Development-Stage Domain Generalization on a Reserved Histopathology Hospital**
7. **Hospital-Specific Generalization and Reliability in Patch-Level Tumor Detection**
8. **From Development Center to Reserved Center: Reliability Limits in Histopathology Classification**

## Best three

1. **Hospital Domain Shift Can Reverse Development Gains in Histopathology Tumor Detection: A Reserved-Center Study of Calibration and Operating-Point Transfer**
2. **Development-to-Test Reversal Under Hospital Shift: A Confidence-Aware Histopathology Study**
3. **A Reserved-Hospital Evaluation of Calibration, Operating-Point Transfer, and Domain Generalization in Histopathology**

## Recommended title

**Hospital Domain Shift Can Reverse Development Gains in Histopathology Tumor Detection: A Reserved-Center Study of Calibration and Operating-Point Transfer**

This title states the central result without claiming GroupDRO superiority, clinical readiness, or universal failure of a method.

## Final abstract

Hospital domain shift can change not only discrimination but also confidence reliability and the operating behavior of histopathology classifiers. We evaluated this problem using a locked Camelyon17-WILDS hospital split, a center-stratified empirical-risk-minimization (ERM) control, and a Group Distributionally Robust Optimization (GroupDRO) model grouped by source hospital. Model development used centers 0, 3, and 4 for training and in-distribution validation and center 1 for out-of-distribution development. Center 2 was reserved for one explicitly authorized final evaluation. On full development center 1, the predeclared GroupDRO primary candidate exceeded the matched ERM control in AUROC (0.8956 versus 0.8673). This ordering reversed on the reserved center: ERM achieved higher AUROC (0.6984 versus 0.6634), AUPRC (0.6556 versus 0.6364), accuracy (0.5712 versus 0.5337), sensitivity (0.2411 versus 0.1106), and fewer false negatives (32,275 versus 37,825). GroupDRO retained higher specificity (0.9569 versus 0.9012) and slightly higher precision.

Pre-frozen temperature scaling was applied without refitting on the test hospital. It improved ECE, Brier score, and negative log-likelihood for both models, while leaving hard predictions and total false-negative counts unchanged. Calibration also sharply reduced the number of missed tumors assigned high confidence, showing that confidence correction is not classification-error correction. Fourteen thresholds selected only on in-distribution validation data did not reliably preserve their nominal sensitivity or specificity targets on the reserved hospital and remained non-clinical candidate operating points.

The negative model result is paired with a positive protocol finding: strict separation of development and reserved hospitals exposed a development-to-test reversal that would have been hidden if the development OOD center had been treated as final performance. These results support hospital-specific external validation, explicit calibration auditing, and predeclared operating policies before clinical translation.

## Keywords

- Histopathology
- Domain shift
- Hospital generalization
- GroupDRO
- Calibration
- Temperature scaling
- Selective prediction
- False negatives
- External validation
- Camelyon17-WILDS

## One-sentence contribution

This study shows that a development-stage GroupDRO advantage can reverse on a strictly reserved hospital while frozen temperature scaling improves confidence reliability without correcting the underlying classification errors.

## Clinical-scope disclaimer

The study is a patch-level, single-dataset methodological evaluation and does not establish clinical deployment readiness, patient-level effectiveness, or a validated clinical operating threshold.
