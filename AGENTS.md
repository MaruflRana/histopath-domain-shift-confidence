# AGENTS.md

**Project/manuscript:** Hospital Domain Shift Can Reverse Development Gains in Histopathology
Tumor Detection: A Reserved-Center Study of Calibration and Operating-Point Transfer

This is the agent-onboarding entry point. Read this file, then `CLAUDE.md`,
`docs/SESSION_STATE.md`, and `docs/NEXT_ACTIONS.md` before acting. Full chronological history is in
`docs/MILESTONE_LOG.md`.

---

## Start here

1. `PROJECT_PROTOCOL.md` — locked data protocol.
2. `CLAUDE.md` — operating contract and no-go rules.
3. `docs/SESSION_STATE.md` — current accepted state and final results.
4. `docs/NEXT_ACTIONS.md` — the gated next work.
5. `submission/jpi/JPI_Submission_Package_README.md` — JPI upload map and manual steps.
6. `submission/jpi/JPI_Anonymized_Manuscript.docx` — primary editable manuscript.
7. `submission/jpi/JPI_Title_Page.docx` — identity-bearing title page.
8. `docs/JPI_FINAL_SUBMISSION_AUDIT.md` — final format/scientific QA.
9. `docs/MILESTONE_LOG.md` — full history.

## Environment

- Windows; Python **3.12.10** in `.venv`; PyTorch **2.9.0+cu128**; CUDA **True**;
  GPU **RTX 3060**; `datasets` **5.0.0**.
- Interpreter: `.venv/Scripts/python.exe`. Run from project root.

## Dataset and locked splits

- Hugging Face dataset **`wltjr1007/Camelyon17-WILDS`** only. Never use the
  WILDS/CodaLab download.
- `label`: 0 = non-tumor, 1 = tumor. Images may be RGBA and must be converted to RGB.

| Logical split | HF split | Centers | Rows |
|---|---|---:|---:|
| `train` | `train` | {0,3,4} | 302,436 |
| `id_val` | `validation` | {0,3,4} | 33,560 |
| `ood_val` | `validation` | {1} | 34,904 |
| `ood_test` | `test` | {2} | 85,054 |

## Hard rules

- No random splits.
- `ood_val` is development-only: never training data and never final-test reporting.
- `ood_test` was accessed exactly once in Milestone 9A. Do not access it again, rerun inference,
  reset the one-shot state, or create a replacement final result.
- Use saved exp09 artifacts for all manuscript and submission work.
- Do not switch the final model pair or relabel the matched ERM control as the predeclared primary.
- Do not train, tune thresholds, refit calibration, or perform post-test model selection.
- Do not claim clinical readiness, a validated clinical threshold, all-hospital robustness, or
  WSI/patient-level effectiveness.
- MC-dropout, ensembles, CORAL, and DANN remain deferred/optional and are not required for the
  manuscript.
- Do not begin a new milestone without explicit user approval.

## Conventions

- The pipeline is script-based. Scripts that use project/HF modules import `_bootstrap` first.
- Seed through `src/utils/seed.py`; config belongs in `configs/`; artifacts belong in `results/`.
- Report negative and unfavorable findings faithfully.
- Do not modify accepted checkpoints, predictions, metrics, authorization records, or run
  sentinels.

## Current accepted state

**Milestones 1 through 9G are complete; the latest completed milestone is Milestone 9G.**
Milestone 9G applied the exact revised manuscript title, added a deterministic locked-workflow
schematic as Figure 1, and renumbered the six accepted scientific figures as Figures 2-7. Only
affected JPI materials were rebuilt; anonymization and scientific-integrity audits passed.
Existing figure pixels, numerical results, author order/roles, and protected exp09 evidence were
unchanged. No dataset/model/test action or generative-image model was used. Journal submission is
intentionally on hold; no portal action is currently authorized.

Milestone 9F finalized the approved two-author order (**Jishan Islam Maruf**, then **Ishtiak Al
Mamoon**), retained Jishan Islam Maruf as first/corresponding author and principal contributor,
updated public and blinded availability wording, rebuilt only affected JPI documents, and passed
anonymization, reviewer-ZIP, scientific-integrity, and GitHub publication audits. The repository
remains unlicensed. No dataset, HF split, image, checkpoint, inference, training, calibration
fitting, threshold tuning, or second `ood_test` attempt occurred; all protected hashes and
table/figure tree digests remained unchanged.

Milestone 9E published the sanitized project to the public GitHub repository
`https://github.com/MaruflRana/histopath-domain-shift-confidence` on branch `main`. The initial
public commit is `9907d6efb546c529fa5226fd0182edac17878b86`. Security, privacy, portability,
large-file, staged-file, and remote-content audits passed. Datasets, checkpoints, raw
predictions, caches, authorization/run-state/sentinel files, certificates, local archives, and
redundant TIFF submission copies were excluded. Accepted exp09 evidence retained its recorded
SHA256 hashes; no dataset, model, checkpoint, image, HF split, or inference path was accessed,
and `ood_test` was not rerun.

Milestone 9D created and audited the original JPI Original Research submission package at
`submission/jpi/`; its initial author metadata was superseded by the finalized Milestone 9F
two-author record. The title page, double-anonymized manuscript, PDF proof,
highlights, cover letter, declarations, editable tables, figure copies, supplement, CLAIM
checklist, seven-figure main package, and sanitized code archive are complete. The abstract is 221 words; the keyword count
is 7; all five highlights are at most 85 characters; anonymization and scientific-integrity audits
both pass. No dataset, model, checkpoint, inference, training, calibration fitting, threshold
tuning, or exp09 modification occurred.

The single authorized final run remains attempt count **1** on HF split `test`, center **{2}**, with
**85,054 rows per model**. No second `ood_test` run is authorized.

The final controlled result is unfavorable to the predeclared primary:

- matched ERM vs GroupDRO AUROC: **0.698435 vs 0.663370**;
- AUPRC: **0.655628 vs 0.636413**;
- sensitivity at 0.5: **0.241070 vs 0.110565**;
- false negatives: **32,275 vs 37,825**.

GroupDRO retained higher specificity (**0.956874 vs 0.901239**) and slightly higher precision
(**0.719400 vs 0.709383**). The development GroupDRO advantage on center 1 did not generalize to
center 2.

Frozen temperature scaling improved held-out ECE, Brier score, and NLL for both models without
changing hard predictions or total false negatives. The 14 frozen `id_val`-selected operating
points transferred poorly and remain candidate/non-clinical.

Primary package entry points:

- `submission/jpi/JPI_Submission_Package_README.md`
- `submission/jpi/JPI_Anonymized_Manuscript.docx`
- `submission/jpi/JPI_Anonymized_Manuscript.pdf`
- `submission/jpi/JPI_Title_Page.docx`
- `submission/jpi/JPI_Anonymization_Audit.md`
- `docs/JPI_FINAL_SUBMISSION_AUDIT.md`
- `docs/FINAL_MANUSCRIPT_WITH_CITATIONS.md`
- `references/final_references.bib`
- `docs/VERIFIED_LITERATURE_REVIEW.md`
- `docs/FINAL_CITATION_AUDIT.md`
- `docs/TARGET_JOURNAL_SHORTLIST.md`
- `docs/RECOMMENDED_SUBMISSION_STRATEGY.md`
- `docs/FINAL_TITLE_ABSTRACT_KEYWORDS.md`
- `docs/FINAL_RESULTS_INTERPRETATION.md`
- `docs/FINAL_DISCUSSION_AND_LIMITATIONS.md`
- `docs/FINAL_CLAIMS_AND_LIMITATIONS.md`
- `docs/SUBMISSION_PACKAGE_README.md`
- `docs/FINAL_REPRODUCIBILITY_CHECKLIST.md`
- `docs/FINAL_MANUSCRIPT_CONSISTENCY_AUDIT.md`
- `results/tables/exp09b_manuscript_table_figure_manifest.csv`
- `results/tables/exp09c_verified_references.csv`
- `results/tables/exp09c_citation_to_claim_map.csv`
- `results/tables/exp09c_journal_comparison.csv`

Recommended journal order: **Journal of Pathology Informatics**, **Journal of Medical Imaging**,
then **PLOS ONE** as the safer fallback.

**Next action:** Journal submission is intentionally on hold. No portal action is currently
authorized. The next action is a future manual journal review when separately approved. Do not
rerun `ood_test`, reopen model development, add exploratory methods, or alter the scientific
result.
