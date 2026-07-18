# CLAUDE.md — Agent Operating Contract

**Project/manuscript:** Hospital Domain Shift Can Reverse Development Gains in Histopathology
Tumor Detection: A Reserved-Center Study of Calibration and Operating-Point Transfer

This file is the active operating contract for any AI agent working in this repository.
Chronological history belongs in `docs/MILESTONE_LOG.md`; `PROJECT_PROTOCOL.md` is the locked
source of truth for data. When instructions disagree, follow the stricter rule.

---

## 0. TL;DR for a new agent

- The pipeline is script-based (`scripts/00…44`), Windows-native, and uses Hugging Face.
- Data comes only from `wltjr1007/Camelyon17-WILDS`. Never use WILDS/CodaLab download.
- The four logical splits in §4 are locked. Never invent random splits.
- **Latest completed milestone = 9G.** Milestones 1 through 9G are complete and accepted.
- Milestone 9G applied the exact revised manuscript title, added a deterministic locked-workflow
  schematic as Figure 1, renumbered the six accepted scientific figures as Figures 2-7, rebuilt
  only affected submission documents, and repeated anonymization/scientific-integrity audits.
  Existing scientific figure pixels and all accepted exp09 evidence remained unchanged. No
  dataset/model/test action or generative-image model was used. JPI submission remains on hold.
- Milestone 9F finalized author order Jishan Islam Maruf then Ishtiak Al Mamoon. Jishan remains
  first/corresponding author and principal contributor; Ishtiak's roles are limited to
  Supervision, Validation, and Writing – review and editing. Public/blinded availability wording
  and affected submission files were rebuilt and audited. No scientific/test action occurred.
- Milestone 9E published the sanitized project to the public GitHub repository
  `https://github.com/MaruflRana/histopath-domain-shift-confidence` on `main`. The initial public
  commit is `9907d6efb546c529fa5226fd0182edac17878b86`. Publication audits passed; datasets,
  checkpoints, raw predictions, authorization/run-state/sentinel files, caches, certificates,
  local archives, and redundant TIFF copies were excluded. Accepted exp09 hashes remained
  unchanged and no dataset/model/test action occurred.
- Milestone 9D created the original Journal of Pathology Informatics (JPI) Original Research
  submission package under `submission/jpi/`; Milestone 9F superseded its initial author metadata
  with the finalized two-author record. It includes the title page, double-anonymized manuscript,
  PDF proof, highlights, cover letter, declarations, editable tables, seven figure pairs,
  supplementary material, CLAIM checklist, sanitized code archive, and PASS audits.
- The JPI abstract is **221 words**, there are exactly **7** keywords, and all five highlights
  pass the 85-character limit (68, 69, 63, 64, 69). The AI-use disclosure is included.
- Milestone 9D was formatting/QA only: no dataset, image data, model, checkpoint, inference,
  training, calibration fitting, threshold tuning, authorization, or sentinel access.
- The single explicitly authorized `ood_test` inference completed once in Milestone 9A
  (attempt count 1; center 2; 85,054 rows/model). Do not access or infer on `ood_test` again.
- Final pair used unchanged: 7F GroupDRO-by-center predeclared primary and 7F
  center-stratified ERM matched control, with frozen temperatures 2.974907 and 3.496293 and
  all 14 frozen `id_val`-selected candidate operating points.
- Honest final result: matched ERM beat GroupDRO on center-2 AUROC
  (**0.698435 vs 0.663370**), AUPRC (**0.655628 vs 0.636413**), sensitivity
  (**0.241070 vs 0.110565**), and false negatives (**32,275 vs 37,825**). GroupDRO retained
  higher specificity and slightly higher precision.
- Frozen temperature scaling improved ECE/Brier/NLL for both models without changing hard
  predictions or total false negatives. Candidate operating points transferred poorly and
  remain non-clinical.
- **Journal submission is intentionally on hold. No portal action is currently authorized.** The
  next action is a future human journal review when separately authorized. Do not reopen model
  development or add exploratory methods.
- Model-based uncertainty, MC-dropout, ensembles, CORAL, and DANN remain deferred/optional.
- Current package entry points:
  `submission/jpi/JPI_Submission_Package_README.md`,
  `submission/jpi/JPI_Anonymized_Manuscript.docx`,
  `submission/jpi/JPI_Title_Page.docx`,
  `submission/jpi/JPI_Anonymization_Audit.md`,
  `docs/JPI_FINAL_SUBMISSION_AUDIT.md`,
  `docs/SESSION_STATE.md`, and `docs/NEXT_ACTIONS.md`.

---

## 1. Environment (validated)

- Windows; Python **3.12.10** in `.venv`; PyTorch **2.9.0+cu128**; CUDA **True**;
  NVIDIA GeForce RTX 3060; torchvision **0.24.0+cu128**; `datasets` **5.0.0**.
- Interpreter: `.venv/Scripts/python.exe`. Run commands from the project root.

---

## 2. Dataset (LOCKED)

- Loader: Hugging Face `datasets` only.
- Dataset identifier: `wltjr1007/Camelyon17-WILDS`.
- Forbidden: `wilds.get_dataset(..., download=True)` and CodaLab/WILDS auto-download.
- Fields: `image, label, center, image_id, patient, node, x_coord, y_coord, slide`.
- Label 0 = non-tumor; label 1 = tumor. Images may be RGBA and must be converted to RGB.
- Streaming is the default access mode; a full local copy is not required.

---

## 3. Repository layout

```text
PROJECT_PROTOCOL.md                  # locked data protocol
CLAUDE.md / AGENTS.md                # active agent contract
docs/                                # state, manuscript, audit, and submission docs
references/final_references.bib      # verified Milestone 9C bibliography
configs/                             # dataset and experiment configs
src/{utils,data,models,training,calibration,evaluation}/
scripts/00_check_env.py … 44_make_methodology_workflow_figure.py, _bootstrap.py
results/{tables,figures,logs,cache,checkpoints,predictions,metrics}/
```

Read `docs/SESSION_STATE.md` for exact current state, `docs/NEXT_ACTIONS.md` for the gated next
work, and `docs/MILESTONE_LOG.md` for full history.

---

## 4. Locked split mapping

| Logical split | HF split | Center filter | Expected total |
|---|---|---|---:|
| `train` | `train` | {0,3,4} | 302,436 |
| `id_val` | `validation` | {0,3,4} | 33,560 |
| `ood_val` | `validation` | {1} | 34,904 |
| `ood_test` | `test` | {2} | 85,054 |

Do not change this mapping. It is encoded in `configs/dataset/camelyon17_hf.yaml` and enforced
by `src/data/hf_camelyon17.py`.

---

## 5. No-go rules

1. Do not create random splits.
2. `ood_val` is development-only. It is never training data and never final-test reporting.
3. `ood_test` was used exactly once in Milestone 9A. Do not access it again, rerun inference,
   reset the attempt state, or create a replacement final result.
4. Use saved exp09 artifacts for all subsequent reporting and manuscript work.
5. Do not call WILDS/CodaLab download.
6. Do not relabel the matched ERM control as the predeclared primary or perform post-test model
   selection.
7. Do not train, fine-tune, tune thresholds, refit calibration, replace checkpoints, or add a
   new method in response to the final result.
8. Do not claim clinical readiness, a validated clinical threshold, all-hospital robustness,
   universal ERM superiority, universal GroupDRO failure, or WSI/patient-level effectiveness.
9. Keep the pipeline script-based. Notebooks are inspection-only.
10. Do not start another milestone without explicit user approval.

---

## 6. Known issues and fixes

- CodaLab/WILDS download is unreliable; use HF streaming only.
- Norton TLS interception was fixed by `src/utils/ssl_setup.py`, invoked through
  `scripts/_bootstrap.py` before HF imports. Every data-using script imports `_bootstrap` first.
- The HF stream is label ordered. Naive early batches are single-class.
- Balanced subset caches avoid repeated long streams. For sound DG-by-center training, use the
  center-stratified caches. The old center/label-confounded caches must not be used for DG
  training.
- Transient HF network errors may be retried by `datasets`; they are not automatically a
  scientific failure.

---

## 7. Reproducibility and conventions

- Seed through `src/utils/seed.py::set_seed`.
- Config belongs in `configs/`; generated artifacts belong in `results/`.
- Numbered scripts write reports under `results/logs/`.
- Scripts that use project/HF data modules import `_bootstrap` first.
- Report failures, skips, and unfavorable scientific findings faithfully.
- Do not modify accepted checkpoints, exp09 predictions/metrics, authorization records,
  run-state files, or sentinels.

---

## 8. Current accepted state

**Milestones 1 through 9G are complete and accepted.**

Milestone 9G applied the exact approved title, added a deterministic methodology workflow as
Figure 1, and renumbered the six accepted scientific figures as Figures 2-7. The affected JPI
documents, PDF proof, captions, manifest, checklists, and audits were rebuilt. Existing scientific
figure pixels and all accepted exp09 evidence remained unchanged; no dataset, HF split, image,
checkpoint, inference, training, calibration fitting, threshold tuning, second `ood_test`
attempt, or generative-image model was used. Journal submission remains on hold.

Milestone 9F finalized the approved coauthorship and publication metadata. Final author order is
Jishan Islam Maruf followed by Ishtiak Al Mamoon; Jishan remains first author, corresponding
author, and principal contributor. Ishtiak's exact CRediT roles are Supervision, Validation, and
Writing – review and editing. The title page, cover letter, declarations, anonymized manuscript,
supplement, CLAIM checklist, public metadata, and availability statements were updated. Blinded
files withhold the identifying GitHub address and passed anonymity/metadata audits. The reviewer
ZIP remained anonymous and did not require rebuilding. All nine protected hashes plus the complete
result-table and scientific-figure tree digests remained unchanged. No dataset, HF split, image,
checkpoint, inference, training, calibration fitting, threshold tuning, or second `ood_test`
attempt occurred. The manuscript remains unsubmitted and model development remains closed.

Milestone 9E sanitized, initialized, audited, and published the repository at
`https://github.com/MaruflRana/histopath-domain-shift-confidence` as a public GitHub repository
with default branch `main`. The initial publication commit is
`9907d6efb546c529fa5226fd0182edac17878b86`. The repository contains reproducible code,
documentation, submission-aligned materials, aggregate results, and PNG figures while excluding
datasets, checkpoints, raw predictions, caches, private governance/run-state records,
certificates, local archives, and redundant TIFF copies. Security, privacy, portability,
large-file, staged-file, and remote audits passed. Accepted exp09 evidence remained immutable;
no HF split, image, model, checkpoint, or inference path was accessed and `ood_test` was not
rerun. Repository publication does not authorize reopening model development.

Milestone 9D formatted the accepted 9C manuscript for JPI without changing scientific content.
The package is at `submission/jpi/`; its title page and double-anonymized manuscript are complete,
the anonymization audit and final scientific-integrity audit both pass, and all standard
declarations—including the required AI-use disclosure—are complete. The anonymized manuscript
uses 30 first-appearance AMA-style references, five editable main tables, and six accepted
scientific figures copied losslessly into 300-dpi TIFF/PNG submission files. A sanitized,
credential-scanned local reproducibility ZIP is prepared but not uploaded.

Milestone 9B produced the venue-neutral final manuscript/results package from saved 8B/8C/9A
artifacts. Milestone 9C then:

- verified **46** references against DOI, PubMed/PMC, official proceedings, publisher, or
  official dataset/project records;
- created `references/final_references.bib`;
- replaced all 22 literature-gap markers in
  `docs/FINAL_MANUSCRIPT_WITH_CITATIONS.md`;
- produced a claim-level citation map and a verified literature review;
- recorded **0 unresolved citation gaps**, **0 duplicate DOI records**, and
  **0 duplicate title records**;
- preserved every scientific number and the unfavorable development-to-test reversal;
- recommended the submission order:
  1. Journal of Pathology Informatics,
  2. Journal of Medical Imaging,
  3. PLOS ONE,
  followed by Scientific Reports, BMC Medical Informatics and Decision Making,
  Journal of Imaging Informatics in Medicine, and Artificial Intelligence in Medicine.

Milestones 9C and 9D performed no experiment or model action. They did not load a dataset, access HF test
or center-2 images, run inference, train, fit calibration, tune thresholds, access checkpoints,
modify exp09 evidence, or create/modify authorization or sentinel records.

### Final locked result

- Attempt count: **1**.
- HF split: `test`; center: `{2}`; rows/model: **85,054**; labels: 42,527/42,527.
- GroupDRO primary: AUROC 0.663370, AUPRC 0.636413, sensitivity 0.110565,
  specificity 0.956874, precision 0.719400, FN 37,825.
- Matched ERM control: AUROC 0.698435, AUPRC 0.655628, sensitivity 0.241070,
  specificity 0.901239, precision 0.709383, FN 32,275.
- The matched control outperformed the predeclared primary on ranking, accuracy,
  sensitivity, F1, and false-negative count. The development GroupDRO advantage did not
  generalize to center 2.
- Frozen calibration improved held-out ECE, Brier score, and NLL for both models without
  changing hard predictions or total false negatives.
- No threshold was selected on `ood_test`; all 14 points remain candidate/non-clinical.

### Frozen checkpoint artifacts

- GroupDRO primary:
  `results/checkpoints/exp07f_groupdro_resnet18/best.pt`
- Center-stratified ERM matched control:
  `results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt`

The exact artifacts were authorized and used once. No replacement pair and no second final-test
run are allowed.

### Current reviewer risks

- Single benchmark and one reserved external hospital.
- Patch-level rather than WSI/patient-level evaluation.
- Capped center-stratified training cache and ResNet-18.
- GroupDRO/ERM only; no CORAL, DANN, MC-dropout, or ensemble comparison.
- Low final sensitivity and high false-negative burden.
- Residual miscalibration despite improvement.
- Poor operating-point transfer and no clinically validated threshold.
- No prospective workflow, reader study, or external non-Camelyon cohort.
- No prospective workflow, WSI/patient endpoint, or independent non-Camelyon validation is available.

### Next action (gated)

Journal submission is intentionally on hold. No portal action is currently authorized. When a
future human journal review is separately authorized, open the generated DOCX files in Microsoft
Word and inspect the seven figures, PDF proof, page breaks, and file designations. Portal/APC and
waiver checks remain future human actions; submission must not be inferred from this milestone.

Do not rerun `ood_test`, reopen model development, add exploratory methods, or rewrite the science
in response to portal review. Model-based uncertainty remains deferred/optional and is not
required for submission completion.
