# Submission Package README

## Current submission package

Milestone 9F finalized the Journal of Pathology Informatics Original Research package at
`submission/jpi/` for first and corresponding author Jishan Islam Maruf and second author Ishtiak
Al Mamoon. The authoritative upload map is
`submission/jpi/JPI_Submission_Package_README.md`. The double-anonymization audit and final
scientific-integrity audit both pass. The abstract is 221 words, there are seven keywords and five
compliant highlights, and all standard declarations—including AI-use disclosure—are complete.

All ten coauthor confirmations are recorded in `docs/COAUTHOR_CONFIRMATION_CHECKLIST.md`. Jishan
Islam Maruf remains the principal contributor. Ishtiak Al Mamoon's CRediT roles are limited to
Supervision, Validation, and Writing – review and editing. Public data/code statements name the
GitHub repository; blinded files withhold its identifying address.

Milestone 9E published the sanitized, submission-aligned project to the public GitHub repository
`https://github.com/MaruflRana/histopath-domain-shift-confidence` on branch `main`. The initial
public commit is `9907d6efb546c529fa5226fd0182edac17878b86`. The public repository intentionally
excludes datasets, checkpoints, raw predictions, private authorization/run-state/sentinel files,
certificates, local archives, redundant TIFF copies, and local identity-bearing editorial
documents. Accepted exp09 evidence remained immutable, and no dataset,
model, checkpoint, image, HF split, or inference path was accessed during publication.

## Recommended positioning

The paper should be positioned as a methodological and empirical study of **reserved-hospital evaluation, confidence reliability, and operating-policy transportability under hospital domain shift**. Its central scientific result is not GroupDRO superiority. The main narrative is that a development-stage GroupDRO advantage reversed on the strictly reserved hospital, while frozen temperature scaling improved confidence reliability without correcting classification errors.

The positive contribution is the protocol discipline: the one-shot reserved-center design exposed a model-ordering reversal and operating-point instability that would have been hidden if development `ood_val` had been reported as final performance.

## Primary manuscript files

- `docs/FINAL_MANUSCRIPT_WITH_CITATIONS.md`
- `references/final_references.bib`
- `docs/FINAL_MANUSCRIPT_DRAFT.md`
- `docs/FINAL_TITLE_ABSTRACT_KEYWORDS.md`
- `docs/FINAL_RESULTS_INTERPRETATION.md`
- `docs/FINAL_DISCUSSION_AND_LIMITATIONS.md`
- `docs/FINAL_CLAIMS_AND_LIMITATIONS.md`
- `docs/VERIFIED_LITERATURE_REVIEW.md`
- `docs/FINAL_CITATION_AUDIT.md`
- `docs/FINAL_NOVELTY_STATEMENT.md`
- `docs/TARGET_JOURNAL_SHORTLIST.md`
- `docs/RECOMMENDED_SUBMISSION_STRATEGY.md`
- `docs/FINAL_REPRODUCIBILITY_CHECKLIST.md`
- `docs/FINAL_MANUSCRIPT_CONSISTENCY_AUDIT.md`

## Main-paper table candidates

- `results/tables/exp09b_development_vs_final_comparison.csv`
- `results/tables/exp09b_final_model_comparison.csv`
- `results/tables/exp09b_final_calibration_summary.csv`
- `results/tables/exp09b_high_confidence_fn_summary.csv`

## Supplementary table candidates

- `results/tables/exp09b_operating_point_transfer.csv`
- `results/tables/exp09b_final_claims_audit.csv`
- `results/tables/exp09b_manuscript_table_figure_manifest.csv`
- `results/tables/exp08b_id_val_per_center_metrics.csv`
- `results/tables/exp08b_candidate_operating_points.csv`
- `results/tables/exp08c_calibration_comparison.csv`
- `results/tables/exp08c_high_confidence_fn_comparison.csv`

## Main-paper figure candidates

- `results/figures/exp09b_development_test_reversal.png`
- `results/figures/exp09b_development_to_final_auroc_auprc.png`
- `results/figures/exp09b_final_default_threshold_metrics.png`
- `results/figures/exp09b_final_calibration_raw_vs_calibrated.png`
- `results/figures/exp09b_high_confidence_fn_raw_vs_calibrated.png`

## Supplementary figure candidates

- `results/figures/exp09b_operating_point_transfer.png`
- `results/figures/exp09_final_ood_test_roc.png`
- `results/figures/exp09_final_ood_test_precision_recall.png`
- `results/figures/exp09_final_ood_test_calibration_reliability.png`
- `results/figures/exp09_final_ood_test_operating_points.png`
- `results/figures/exp09_final_ood_test_high_confidence_fn.png`

## Reproducibility and provenance artifacts

- Frozen protocol: `docs/FROZEN_FINAL_EVAL_PROTOCOL_DRAFT.md`
- Final execution checklist: `docs/FINAL_TEST_EXECUTION_CHECKLIST.md`
- Authorization record: `configs/experiments/exp09_final_ood_test_authorization.json`
- Run state: `results/logs/exp09_final_ood_test_run_state.json`
- Inference sentinel: `results/logs/exp09_final_ood_test_completed_run.json`
- Summary sentinel: `results/logs/exp09_final_ood_test_summary_completed.json`
- Inference report: `results/logs/exp09_final_ood_test_inference_report.txt`
- Final report: `results/logs/exp09_final_ood_test_report.txt`
- Final metrics: `results/metrics/exp09_final_ood_test/final_metrics.json`
- Milestone 9B integration report: `results/logs/exp09b_final_manuscript_integration_report.txt`
- Packaging script: `scripts/40_make_final_manuscript_package.py`
- Verified-reference table: `results/tables/exp09c_verified_references.csv`
- Citation-to-claim map: `results/tables/exp09c_citation_to_claim_map.csv`
- Journal comparison: `results/tables/exp09c_journal_comparison.csv`
- Citation audit script: `scripts/41_verify_literature_and_citations.py`

## Exact final-test provenance

- Dataset: `wltjr1007/Camelyon17-WILDS`
- HF split: `test`
- Logical split: `ood_test`
- Center: `{2}`
- Rows/model: 85,054
- Positive class: tumor = 1
- Inference attempt count: 1
- Dataset instances: 1
- Dataloader traversals: 1
- Models per batch: 2
- Run start: `2026-07-16T17:16:25.455291+00:00`
- Inference complete: `2026-07-16T17:35:17.051997+00:00`
- Summary complete: `2026-07-16T17:35:20.395673+00:00`
- No training, calibration fitting, test-threshold selection, or post-test model selection.
- No second `ood_test` inference is authorized.

## Verified target-journal order

1. **Journal of Pathology Informatics** — best specialist fit for pathology informatics,
   validation experience, and the negative-model/positive-protocol result.
2. **Journal of Medical Imaging** — strong imaging-methodology audience, with higher risk because
   the study introduces no new algorithm.
3. **PLOS ONE** — safer broad fallback whose editorial criteria suit rigorous negative findings.
4. Scientific Reports.
5. BMC Medical Informatics and Decision Making.
6. Journal of Imaging Informatics in Medicine.
7. Artificial Intelligence in Medicine — high desk-rejection risk because its official scope
   expects stronger methodological novelty than application of established algorithms.

JPI formatting has been applied. See `docs/TARGET_JOURNAL_SHORTLIST.md` for the venue research and
`docs/JPI_FINAL_SUBMISSION_AUDIT.md` for the completed package audit.

## Remaining work before submission

1. Open the prepared DOCX files in Microsoft Word and perform a final human visual check.
2. Re-verify current JPI APC/waiver and submission-system requirements.
3. Upload the title page, anonymized manuscript, highlights, declarations, figures, captions,
   supplementary material, and CLAIM checklist using the package README order.
4. Enter the prepared metadata and declarations in the portal; no ORCID or repository DOI exists.
5. Inspect the portal-generated proof before manual submission.

## Hard submission guardrails

- Do not rerun or re-access `ood_test`.
- Do not reopen model development based on the final test.
- Do not relabel ERM as the predeclared primary.
- Do not hide the unfavorable GroupDRO result.
- Do not claim clinical readiness, all-hospital robustness, or a validated clinical threshold.
- Do not add untested exploratory methods to the current manuscript package.
