# FINAL MANUSCRIPT CONSISTENCY AUDIT

**Milestone:** 9B
**Date:** 2026-07-17
**Mode:** saved-artifact-only manuscript/results integration
**Verdict:** **PASS — internally consistent and ready for literature verification and
journal-template formatting**

## Scope and non-execution statement

This audit used only saved CSV, JSON, log, document, and figure artifacts. It loaded no dataset,
imported no Hugging Face test split, read no center-2 image, ran no model inference, accessed no
checkpoint file, trained no model, fit no calibration, and tuned no threshold. Script 38 was not
rerun. The authorized final inference attempt count remains 1.

## Required consistency checks

| # | Check | Result | Evidence |
|---:|---|---|---|
| 1 | Every final metric matches saved exp09 artifacts. | PASS | Exact center-2 classification, calibration, operating-point, and high-confidence-FN values were generated from `results/tables/exp09_final_ood_test_*.csv` and checked against `results/metrics/exp09_final_ood_test/final_metrics.json`. |
| 2 | Every development metric matches accepted 8B/8C artifacts. | PASS | Development rows trace to `results/tables/exp08b_overall_metrics.csv`, `exp08b_id_val_per_center_metrics.csv`, `exp08b_candidate_operating_points.csv`, and the accepted 8C calibration/reliability tables. |
| 3 | No final number is attributed to development or vice versa. | PASS | Tables and prose label full-development `ood_val` center 1 separately from final held-out `ood_test` center 2. |
| 4 | GroupDRO remains the predeclared primary. | PASS | Manuscript, claims map, figures, and context documents use “predeclared GroupDRO primary” or equivalent role-preserving wording. |
| 5 | ERM remains the matched control. | PASS | No document relabels matched ERM as the predeclared primary. |
| 6 | The final reversal is reported honestly. | PASS | GroupDRO development AUROC 0.895609 versus ERM 0.867271; final GroupDRO AUROC 0.6633704256200204 versus ERM 0.6984352121958427. |
| 7 | No post-test model selection is implied. | PASS | The matched control’s final advantage is reported without changing model roles or selecting a replacement primary. |
| 8 | Calibration is not described as changing hard predictions. | PASS | Raw/calibrated argmax invariance is explicit in the manuscript, calibration table, claims map, and captions. |
| 9 | Calibration is not described as reducing total false negatives. | PASS | Total FN remains 37,825 for GroupDRO and 32,275 for ERM; only high-confidence FN counts fall. |
| 10 | Operating points remain candidate/non-clinical. | PASS | All 14 rows retain their frozen `id_val` provenance and non-clinical caveat; no test threshold was selected. |
| 11 | No clinical-readiness claim exists. | PASS | Clinical readiness and validated-threshold claims are explicitly forbidden; the manuscript states the study does not establish clinical readiness. |
| 12 | No universal DG conclusion exists. | PASS | The manuscript does not claim universal ERM superiority, universal GroupDRO failure, or robustness across all hospitals. |
| 13 | No second `ood_test` run occurred. | PASS | Run state and sentinels retain attempt/run counter 1; the 9B script has no dataset/model import and reports `second_inference_attempt=false`. |
| 14 | Authorization/run/sentinel provenance is documented. | PASS | Authorization SHA256, run timestamps, one dataset instance, one traversal, row count, center, checkpoint hashes, temperatures, and sentinel paths are recorded in `docs/FINAL_REPRODUCIBILITY_CHECKLIST.md`. |
| 15 | Existing exp09 predictions/metrics remain unchanged. | PASS | Eighteen immutable authorization, prediction, metric, table, sentinel, and existing exp09-figure hashes matched their pre-9B values after package generation. |

## Table and figure checks

- All seven requested `exp09b_*` tables exist, are non-empty, and have the required schemas.
- The development-versus-final table has 6 rows; the final comparison uses an explicit long format
  with 24 model-metric rows; calibration has 4 rows; operating-point transfer has 14 rows;
  high-confidence FN has 12 rows.
- All six requested `exp09b_*` figures exist and were visually inspected.
- The first operating-point draft clipped low fixed-sensitivity results below 0.55. This genuine
  presentation defect was corrected before acceptance. The final achieved-metric axes use 0–1,
  exposing rather than hiding the poor transfer.
- Existing Milestone 9A figures were inspected and retained unchanged.
- No seaborn was used.

## Scientific conclusion check

The package preserves the binding interpretation:

- GroupDRO appeared stronger than matched ERM on development center 1.
- The matched ERM control outperformed the predeclared GroupDRO primary on final center-2 AUROC,
  AUPRC, accuracy, sensitivity, F1, and false-negative count.
- GroupDRO retained higher specificity and slightly higher precision.
- The development GroupDRO advantage did not generalize to center 2.
- Frozen temperature scaling improved held-out ECE, Brier score, and NLL for both models without
  correcting the underlying classification errors.
- Development-selected operating-point targets were unstable under the new hospital shift.
- The negative model result is distinct from the positive protocol contribution: the strict
  reserved-center design exposed a reversal that development-only reporting would have hidden.

## Claims and limitation check

Allowed, conditional, and forbidden claims are synchronized across:

- `docs/FINAL_MANUSCRIPT_DRAFT.md`
- `docs/FINAL_RESULTS_INTERPRETATION.md`
- `docs/FINAL_DISCUSSION_AND_LIMITATIONS.md`
- `docs/FINAL_CLAIMS_AND_LIMITATIONS.md`
- `docs/CLAIMS_AND_LIMITATIONS.md`
- `results/tables/exp09b_final_claims_audit.csv`

The following remain forbidden: clinical deployment readiness, a validated clinical threshold,
all-hospital robustness, universal ERM superiority, universal GroupDRO failure, WSI/patient-level
effectiveness, independent multi-center calibration validity, and superiority over untested
CORAL/DANN/model-based uncertainty methods.

## Operational validation

- `scripts/40_make_final_manuscript_package.py` imports only standard saved-artifact libraries,
  pandas, NumPy, and matplotlib.
- Static inspection found no torch, torchvision, datasets, project data/model/training import,
  dataset loader call, optimizer, backward call, model training call, or temperature-fit call.
- The permitted package command completed with `OVERALL=PASS`.
- Final context state is Milestone 9B in `CLAUDE.md`, `AGENTS.md`,
  `docs/SESSION_STATE.md`, and `docs/NEXT_ACTIONS.md`.
- `CLAUDE.md` and `AGENTS.md` satisfy their character limits.

## Remaining work

The manuscript is not yet submission-final because external references and venue metadata were not
invented in 9B. The next gated work is:

1. verify citations listed in `docs/LITERATURE_CITATION_GAPS.md`;
2. select a venue and apply its manuscript template;
3. complete author, ethics, funding, conflict, contribution, and data/code-availability metadata;
4. perform final language and submission-format QA.

No additional `ood_test` inference or exploratory model development is required or authorized.
