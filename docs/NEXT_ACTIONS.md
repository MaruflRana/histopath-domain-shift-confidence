# NEXT ACTIONS

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in
Histopathology: Calibration, Uncertainty, and Selective Prediction

Nothing in this file is authorized to start until the user explicitly approves it.

---

## Immediate status

**Milestones 1 through 9F are complete and accepted; the latest completed milestone is
Milestone 9F.**

Milestone 9F finalized the approved author order Jishan Islam Maruf then Ishtiak Al Mamoon.
Jishan remains first/corresponding author and principal contributor; Ishtiak's roles are limited
to Supervision, Validation, and Writing – review and editing. All ten confirmations are complete.
Affected public, identity-bearing, and blinded submission files were rebuilt and audited. The
reviewer ZIP remained anonymous and was not rebuilt. Protected exp09 hashes, result tables, and
scientific figures remained unchanged; no dataset/model/test action occurred.

Milestone 9E published the sanitized project to the public repository
`https://github.com/MaruflRana/histopath-domain-shift-confidence` on `main`; the initial public
commit is `9907d6efb546c529fa5226fd0182edac17878b86`. Publication audits passed. Datasets,
checkpoints, raw predictions, caches, authorization/run-state/sentinel files, certificates,
local archives, and redundant TIFF copies were excluded. Accepted exp09 evidence remained
immutable, no `ood_test` rerun occurred, and repository publication does not reopen development.

Milestone 9D created the original Journal of Pathology Informatics Original Research submission
package under `submission/jpi/`; Milestone 9F superseded its initial author metadata. The abstract is 221 words,
there are exactly seven keywords and five compliant highlights, all declarations are complete,
the required AI-use disclosure is included, and both double-anonymization and final
scientific-integrity audits pass.

No dataset was loaded. No HF split, center-2 image, model, checkpoint, prediction path, or
inference path was accessed. No training, calibration fitting, threshold tuning, authorization
change, sentinel change, post-test model selection, or second `ood_test` run occurred. Existing
exp09 predictions, metrics, tables, figures, authorization, run state, and sentinels remained
unchanged.

## Binding final interpretation

- GroupDRO remains the predeclared primary; center-stratified ERM remains the matched control.
- GroupDRO was stronger on full-development center 1 AUROC (0.8956 vs 0.8673).
- On final center 2, matched ERM outperformed GroupDRO on AUROC (0.698435 vs 0.663370),
  AUPRC (0.655628 vs 0.636413), accuracy (0.571155 vs 0.533720), sensitivity
  (0.241070 vs 0.110565), F1, and false-negative count (32,275 vs 37,825).
- GroupDRO retained higher specificity (0.956874 vs 0.901239) and slightly higher precision
  (0.719400 vs 0.709383).
- Frozen temperature scaling improved held-out ECE, Brier score, and NLL without changing hard
  predictions or total false negatives.
- The 14 frozen `id_val`-selected operating points transferred poorly and remain
  candidate/non-clinical.
- No clinical readiness, universal model ranking, or all-hospital robustness claim is allowed.

## JPI submission package

- `submission/jpi/JPI_Submission_Package_README.md`
- `submission/jpi/JPI_Title_Page.docx`
- `submission/jpi/JPI_Anonymized_Manuscript.docx`
- `submission/jpi/JPI_Anonymized_Manuscript.pdf`
- `submission/jpi/JPI_Cover_Letter.docx`
- `submission/jpi/JPI_Highlights.docx`
- `submission/jpi/JPI_Author_Declarations.docx`
- `submission/jpi/JPI_Supplementary_Material.docx`
- `submission/jpi/JPI_CLAIM_Checklist.docx`
- `submission/jpi/JPI_Anonymization_Audit.md`
- `submission/jpi/JPI_Reproducibility_Code_Package.zip`
- `docs/JPI_FINAL_SUBMISSION_AUDIT.md`
- `results/logs/exp09d_jpi_submission_package_report.txt`

Supporting verified manuscript sources:

- `docs/FINAL_MANUSCRIPT_WITH_CITATIONS.md`
- `references/final_references.bib`
- `docs/VERIFIED_LITERATURE_REVIEW.md`
- `docs/FINAL_CITATION_AUDIT.md`
- `docs/CITATION_INTEGRATION_CHANGELOG.md`
- `docs/FINAL_NOVELTY_STATEMENT.md`
- `docs/TARGET_JOURNAL_SHORTLIST.md`
- `docs/RECOMMENDED_SUBMISSION_STRATEGY.md`
- `results/tables/exp09c_verified_references.csv`
- `results/tables/exp09c_citation_to_claim_map.csv`
- `results/tables/exp09c_journal_comparison.csv`
- `results/tables/exp09c_literature_novelty_comparison.csv`
- `results/tables/exp09c_unresolved_citation_gaps.csv`

## Exact next recommendation

Proceed with **human review and manual JPI submission**:

1. Open every DOCX in Microsoft Word and confirm page breaks, editable tables, superscript
   citations, and absence of tracked changes/comments.
2. Review the PDF, all six TIFF figures, captions, and supplementary material at 100% zoom.
3. Re-verify current JPI APC/waiver, file-designation, and declaration fields on the official
   submission portal.
4. Enter the prepared author, affiliation, corresponding-author, funding, conflict, ethics,
   consent, data/code, CRediT, and AI-use fields exactly.
5. Upload in the order documented in `submission/jpi/JPI_Submission_Package_README.md`, inspect
   the portal-generated proof, and submit manually.

## Permanent guardrails

- Do not access or infer on `ood_test` again. The single authorized attempt is complete and the
  attempt counter is 1.
- Do not modify exp09 predictions, metrics, authorization files, run state, or sentinels.
- Do not relabel matched ERM as the predeclared primary or perform post-test model switching.
- Do not tune thresholds or fit calibration on the final test.
- Do not reopen model development because GroupDRO underperformed on center 2.
- Do not claim clinical readiness, a clinically validated threshold, robustness across all
  hospitals, universal ERM superiority, universal GroupDRO failure, or WSI/patient-level
  effectiveness.
- Model-based uncertainty, MC-dropout, ensembles, CORAL, and DANN remain deferred/optional and
  are not required for submission.
- Do not create another journal version or scientific rewrite without explicit approval.

Historical milestone detail belongs in `docs/MILESTONE_LOG.md`.
