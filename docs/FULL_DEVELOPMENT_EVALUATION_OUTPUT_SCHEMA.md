# FULL DEVELOPMENT-SCALE EVALUATION OUTPUT SCHEMA (Milestone 8A)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology:
Calibration, Uncertainty, and Selective Prediction

_Milestone 8A — **planning only**. Defines the **expected output files and schemas** for the
later execution milestone (8B). 8A writes **no** prediction/metrics artifacts of the kinds
below; it only specifies them. `ood_test` (center 2) is **LOCKED and unused**; no schema below
includes an `ood_test` split value._

---

## 0. Conventions

- Positive class = **tumor = 1**; negative = non-tumor = 0.
- `split` ∈ {`id_val`, `ood_val`} only. **`ood_test` is never a permitted value** in any 8B
  output.
- Suggested run root for 8B: `results/predictions/exp08b_full_dev_eval/`,
  `results/metrics/exp08b_full_dev_eval/`, `results/tables/exp08b_*`, `results/logs/exp08b_*`.
  New run names only — **no overwrite** of existing 4A/5A/7A–7F artifacts.
- Schemas below **extend** the existing prediction schema used through Milestone 7F
  (`split,index,label,center,image_id,patient,node,x_coord,y_coord,slide,logit_0,logit_1,
  prob_0,prob_1,pred_label,correct`) by adding a `confidence` column.

## 1. Prediction CSV schema (per model × per split)

File (per required model, per split), e.g.
`results/predictions/exp08b_full_dev_eval/<model>/<split>_predictions.csv`.

**Required columns:**

| column | type | definition |
|---|---|---|
| `split` | str | `id_val` or `ood_val` (never `ood_test`) |
| `label` | int {0,1} | ground-truth (0 non-tumor, 1 tumor) |
| `pred_label` | int {0,1} | argmax at threshold 0.5 (or stated operating threshold) |
| `correct` | bool | `pred_label == label` |
| `prob_0` | float [0,1] | softmax P(non-tumor) |
| `prob_1` | float [0,1] | softmax P(tumor); `prob_0+prob_1 ≈ 1` |
| `logit_0` | float | raw logit, class 0 |
| `logit_1` | float | raw logit, class 1 |
| `confidence` | float [0,1] | `max(prob_0, prob_1)` |
| `center` | int | hospital id (metadata) |
| `patient` | int | patient id (metadata) |
| `slide` | int | slide id (metadata) |
| `node` | int | node id (metadata) |
| `image_id` | int | patch/image id (metadata) |
| `x_coord` | int | patch x coordinate (metadata) |
| `y_coord` | int | patch y coordinate (metadata) |

Optional helper columns permitted (not required): `index`, `entropy`, `margin`,
`calibrated_prob_0`, `calibrated_prob_1`, `calibrated_confidence`. All 15 metadata/prob/logit
fields plus `confidence` above are **mandatory**. Missing any of them is a **stop** condition
(see stop rules).

## 2. Metrics JSON schema (per model × per split)

File: `results/metrics/exp08b_full_dev_eval/<model>/classification_metrics.json`.

```
{
  "model": "<name>",
  "split": "id_val" | "ood_val",
  "n": <int>,
  "threshold": 0.5,
  "positive_class": 1,
  "accuracy": <float>, "balanced_accuracy": <float>,
  "auroc": <float|null>, "auprc": <float|null>,
  "sensitivity": <float|null>, "specificity": <float|null>,
  "precision": <float|null>, "f1": <float|null>,
  "tn": <int>, "fp": <int>, "fn": <int>, "tp": <int>,
  "warnings": [ ... ]
}
```

`auroc`/`auprc`/ratios may be `null` with a warning if a split is single-class (should not
happen for id_val/ood_val). Reuse `src/evaluation/metrics_classification.py`.

## 3. Calibration JSON / table schema (development-only)

File: `results/metrics/exp08b_full_dev_eval/<model>/calibration_metrics.json` and companion
table `results/tables/exp08b_calibration_comparison.csv`.

JSON:
```
{
  "model": "<name>", "split": "ood_val",
  "temperature": <float>,           # reused 7G value or dev-only refit; NOT a final-test claim
  "temperature_source": "reused_7g" | "dev_refit_ood_val",
  "raw":        {"ece": <float>, "brier": <float>, "nll": <float>},
  "calibrated": {"ece": <float>, "brier": <float>, "nll": <float>},
  "argmax_invariant": true
}
```

Table `exp08b_calibration_comparison.csv` columns: `model, variant(raw|calibrated), ece, brier,
nll`. Calibration is **development-only**; no independent-test calibration claim (R7 open).

## 4. Operating-point table schema

File: `results/tables/exp08b_operating_points.csv`. One row per (model × target × target_type).

Columns: `model, target_type(specificity|sensitivity), target_value, threshold_selected_on_id_val,
id_val_sensitivity, id_val_specificity, ood_val_sensitivity, ood_val_specificity, ood_val_FN,
ood_val_FP, candidate_only_note`.

`candidate_only_note` must state: "CANDIDATE development operating point (id_val-selected); NOT a
committed clinical threshold." Reuse `src/evaluation/metrics_operating_points.py` (thresholds
selected on `id_val` only, applied to `ood_val`).

## 5. Per-center `id_val` table schema

File: `results/tables/exp08b_id_val_per_center_metrics.csv`. One row per (model × center) plus an
overall row.

Columns: `model, center(0|3|4|overall), n, accuracy, auroc, auprc, sensitivity, specificity, fn,
loss_if_available, note`. Purpose: worst-center id_val behavior (as in 7F, where GroupDRO
up-weighted center 4).

## 6. High-confidence false-negative table schema

File: `results/tables/exp08b_high_confidence_fn.csv`. Rows per (model × variant × threshold).

Columns: `model, variant(raw|calibrated), confidence_threshold(0.90|0.95|0.99),
high_conf_fn_count, note`. Definition: `label==1 & pred_label==0 & confidence≥t`.

## 7. Optional uncertainty / selective-prediction tables

Optional in 8B (deterministic only; **no** MC-dropout / ensemble):
- `results/tables/exp08b_uncertainty_metrics.csv` — error-detection AUROC / AURC by score
  (entropy/confidence/margin/energy), per split.
- `results/tables/exp08b_risk_coverage.csv` and `exp08b_selective_prediction.csv` — risk vs
  coverage; retained FN under abstention. Reuse `src/evaluation/metrics_uncertainty.py` /
  `metrics_selective.py`.

## 8. Report / log schema

File: `results/logs/exp08b_full_development_evaluation_report.txt`. Plain-text sections:
`RUN CONFIG` (models, checkpoints, splits, sizes, batch size, seed), `EXECUTION ORDER`
(id_val→ood_val), `RESULTS` (classification / AUROC-AUPRC / operating points / calibration /
high-conf FN / per-center), `CONTROLLED COMPARISON` (7F ERM vs 7F GroupDRO), `CONTEXT-ONLY`
(if run), `ASSERTIONS` (PASS/FAIL lines incl. no-`ood_test`, row counts, prob validity, binary
labels, metadata present, no-overwrite), `WARNINGS`, `ERRORS`, and an explicit
`ood_test LOCKED` statement. Mirrors the 4B/5A/7F report style.

## 9. `ood_test` statement

No output schema in this document contains an `ood_test` split value or path. `ood_test`
(center 2) is **LOCKED and unread** and is produced by **no** 8A or 8B artifact.
