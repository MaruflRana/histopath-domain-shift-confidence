# Journal of Pathology Informatics Final Submission Audit

**Milestone:** 9G
**Article type:** Original Research Article
**Target:** Journal of Pathology Informatics
**Overall verdict:** **ON HOLD — READY FOR FUTURE HUMAN REVIEW**

Milestone 9G applied the exact approved title, added a deterministic methodology workflow as
Figure 1, and renumbered the six accepted scientific figures as Figures 2-7. Only affected
documents and figure copies were rebuilt. No dataset, HF split, histopathology image, checkpoint,
model, inference, training, calibration fitting, or threshold-tuning path was accessed.

## Final authorship

- Author order: Jishan Islam Maruf; Ishtiak Al Mamoon.
- First, corresponding, and principal-contributor author: Jishan Islam Maruf.
- Second author: Ishtiak Al Mamoon; not a corresponding author.
- Shared affiliation: Department of Computer Science and Engineering,
  IUBAT—International University of Business Agriculture and Technology.
- Ishtiak Al Mamoon CRediT: Supervision, Validation, and Writing – review and editing.
- All ten coauthor confirmations: approved.

## Scientific-integrity checks

| # | Check | Result | Evidence |
|---:|---|---|---|
| 1 | Every scientific number matches accepted 8B, 8C, and 9A evidence | PASS | Source-driven build from the accepted 9C manuscript and exp09b tables; binding full-precision final metrics validated before formatting |
| 2 | Development center 1 and final center 2 are never confused | PASS | Separate Methods, Results, tables, captions, and scope notes |
| 3 | GroupDRO remains the predeclared primary candidate | PASS | Roles retained throughout manuscript and captions |
| 4 | ERM remains the matched control | PASS | No post-test relabeling |
| 5 | Final reversal is reported honestly | PASS | ERM advantages and GroupDRO specificity/precision advantages both retained |
| 6 | No post-test model selection is implied | PASS | One-shot protocol and fixed roles stated explicitly |
| 7 | Calibration is confidence correction, not error correction | PASS | Discussion and figure captions use this distinction |
| 8 | Total false-negative counts are unchanged by calibration | PASS | GroupDRO 37,825; ERM 32,275 in text and tables |
| 9 | Frozen operating points remain candidate/non-clinical | PASS | Methods, Results, Figure 6, captions, and supplement |
| 10 | No clinical-readiness claim appears | PASS | Clinical readiness is expressly disclaimed |
| 11 | No universal model claim appears | PASS | Neither universal ERM superiority nor universal GroupDRO failure is claimed |
| 12 | No WSI/patient-level effectiveness claim appears | PASS | Patch-level limitation is explicit |
| 13 | Ethics statement is supported and bounded | PASS | Public, de-identified secondary analysis; no IUBAT exemption claimed |
| 14 | AI assistance is disclosed and approved by both authors | PASS | Exact final declaration before References and in identity-bearing declarations |
| 15 | No generative-AI figure creation occurred | PASS | Figure 1 was rendered deterministically with matplotlib; Figures 2-7 are pixel-identical copies of accepted exp09b figures |
| 16 | Author order and contribution roles are exact | PASS | Jishan first/corresponding/principal; Ishtiak second with only the approved three CRediT roles |
| 17 | Public and blinded availability statements are separated | PASS | Public URL appears only in public/identity-bearing files; blinded wording withholds the identifying address |
| 18 | Competing-interest wording is final | PASS | Both authors declare no known competing interests or relevant personal relationships |

## Journal-format and package checks

| # | Check | Result | Evidence |
|---:|---|---|---|
| 19 | Citations are defined and numbered correctly | PASS | 30 references, first-appearance order 1–30, no missing or duplicate list number |
| 20 | Abstract is no more than 250 words | PASS | 221 words |
| 21 | Keyword count is exactly seven | PASS | Seven binding English keywords |
| 22 | Five highlights meet the 85-character limit | PASS | 68, 69, 63, 64, and 69 characters |
| 23 | Double-anonymization audit passes | PASS | Manuscript, supplement, CLAIM checklist, and reviewer ZIP; see `submission/jpi/JPI_Anonymization_Audit.md` |
| 24 | Affected DOCX files open successfully | PASS | Ten rebuilt DOCX files parsed and passed OOXML structure checks |
| 25 | PDF proof renders without clipping | PASS | 12 pages rendered and visually inspected; Figure 1 is readable at manuscript width |
| 26 | Existing scientific figures remain unchanged | PASS | The 52-file pre-9G figure tree excluding the three new workflow outputs retained digest `A141388875BE48C6B64465A98EAEE767F07491E3674BB803E1407781EF608EAB` |
| 27 | All tables remain unchanged | PASS | 99-file result-table tree digest matches the pre-9F baseline |
| 28 | No placeholders remain in final package metadata | PASS | No pending coauthor or availability marker detected |
| 29 | No exp09 evidence artifact changed | PASS | All nine protected hashes match accepted baselines |
| 30 | No dataset or model action occurred | PASS | Targeted build imports no project data/model module and records every prohibited action as false |
| 31 | No second `ood_test` attempt occurred | PASS | Attempt count remains one; scripts 38 and 39 were not run |

## References, tables, and figures

- References: 30 manuscript-cited records, converted from verified BibTeX metadata into first-appearance AMA-style numbering. DOI links were retained where present.
- Main tables: 5 editable tables embedded in the manuscript and duplicated in `JPI_Tables.docx` for portal flexibility.
- Figures: 7. Figure 1 is the deterministic methodology workflow; Figures 2-7 are the six accepted exp09b scientific figures copied without resampling or pixel alteration. TIFF and PNG copies are tagged 300 dpi; dimensions, hashes, old/new numbering, and visual status are recorded in `JPI_Figure_Manifest.csv`.
- Supplementary material: locked split mapping, checkpoint-hash/temperature provenance, all 14 frozen operating points, one-shot provenance, high-confidence false negatives, reproducibility summary, expanded limitations, and inventory.
- CLAIM 2024 checklist: 44 items with explicit Yes, No, or Not applicable status; no patient-level or clinical item was fabricated as complete.

## Code-release QA

The sanitized local archive contains 91 files. It excludes images, caches, `.venv`, credentials,
checkpoints, raw patch-level predictions, authorization records, run sentinels, and JPI build
helpers containing author metadata. Its 9F identity/repository/credential/path scan passed, so it
was not rebuilt. Archive SHA256:

`27FF9ED54AA4C4A1898216B8E098D6716DB0709FC81D6845E835FA6ABDE1ED2C`

## Operational warning

Microsoft Word, LibreOffice, and Pandoc were unavailable in the environment. The editable DOCX files were validated structurally with `python-docx`. The PDF proof was generated by a document-only fallback renderer using ReportLab and visually inspected page by page. A final human opening in Microsoft Word and the JPI previewer remains required before upload because pagination can differ across Word/rendering engines.

`python-docx` 1.2.0 and its dependency `lxml` 6.1.1 were installed in the project virtual environment for this milestone; no unrelated package was installed.

## Final disposition

The title and methodology-figure update is complete and the package has not been submitted.
Journal submission is intentionally on hold. No portal action is currently authorized. A future
human Microsoft Word and journal review requires separate authorization. Do not rerun `ood_test`
or reopen model development.
