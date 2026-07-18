# FINAL-TEST OUTPUT SCHEMA (Milestone 8D)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology.

> **Schema specification only — NO final-test outputs are created now.** This defines the outputs a
> later authorized final-test milestone would produce (under a fresh run name, e.g. `exp09_*`),
> consistent with the 8B/8C schemas. Milestone 8D wrote none of these; `ood_test` is untouched.

---

## 1. Prediction CSV schema (per model, per variant)

One row per `ood_test` patch. Required columns:

| column | type | notes |
|--------|------|-------|
| `image_id` | int | provenance |
| `center` | int | must be 2 for `ood_test` |
| `patient` | int | metadata |
| `slide` | int | metadata |
| `node` | int | metadata |
| `x_coord` | int | metadata |
| `y_coord` | int | metadata |
| `label` | int | 0 = non-tumor, 1 = tumor |
| `logit_0`, `logit_1` | float | raw model logits |
| `prob_0`, `prob_1` | float | softmax probabilities; `prob_0+prob_1≈1` (±1e-5) |
| `pred_label` | int | argmax (threshold 0.5) |
| `confidence` | float | `max(prob_0, prob_1)` |
| `variant` | str | `raw` or `calibrated` |
| `temperature` | float | 1.0 for raw; frozen 8C T for calibrated |

## 2. Metrics JSON schema (raw and calibrated)

```
{
  "milestone": "<final-test milestone>",
  "ood_test_locked_until_authorized": true,
  "positive_class": 1,
  "models": {
    "groupdro": {
      "name": "7F GroupDRO (primary)",
      "temperature": 2.974907,
      "classification": { "raw": {...}, "calibrated": {...} },
      "calibration":    { "raw": {...}, "calibrated": {...} }
    },
    "centerstrat_erm": {
      "name": "7F center-stratified ERM (matched control)",
      "temperature": 3.496293,
      "classification": { "raw": {...}, "calibrated": {...} },
      "calibration":    { "raw": {...}, "calibrated": {...} }
    }
  }
}
```
`classification` blocks: accuracy, balanced_accuracy, auroc, auprc, sensitivity, specificity,
precision, f1, tn, fp, fn, tp. `calibration` blocks: ece, brier, nll, n_bins.

## 3. Threshold table schema

Columns: `model, model_label, target_type (specificity|sensitivity), target_value,
threshold_selected_on_id_val, id_val_sensitivity, id_val_specificity, id_val_FN, id_val_FP,
ood_test_sensitivity, ood_test_specificity, ood_test_FN, ood_test_FP, candidate_only_note`.
Thresholds selected on `id_val` only.

## 4. Calibration table schema

Columns: `model, model_key, variant (raw|calibrated), temperature, ece, brier, nll`.

## 5. High-confidence FN table schema

Columns: `model, model_label, variant, confidence_threshold (0.90|0.95|0.99), n_high_confidence,
high_confidence_errors, high_confidence_false_negatives, high_confidence_false_negative_rate,
total_false_negatives`.

## 6. Report schema (text)

Must state: authorization phrase confirmed; frozen checkpoints + hashes; `ood_test` row count ==
85054 and center == {2}; raw + calibrated metrics; candidate operating
points (id_val-selected); high-confidence FN audit; single-run confirmation; explicit
"development thresholds/temperatures were pre-frozen; nothing fit on `ood_test`" statement.

## 7. Required metadata columns (all outputs)

`center, patient, slide, node, image_id, x_coord, y_coord` must be preserved (no dropped columns).
