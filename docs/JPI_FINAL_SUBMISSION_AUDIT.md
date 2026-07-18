# Journal of Pathology Informatics Final Submission Audit

**Milestone:** 9D
**Article type:** Original Research Article
**Target:** Journal of Pathology Informatics
**Overall verdict:** **PASS — ready for human pre-upload review**

This was a formatting and quality-assurance milestone using saved manuscript and result artifacts only. No dataset, image, checkpoint, model, or inference path was accessed.

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
| 9 | Frozen operating points remain candidate/non-clinical | PASS | Methods, Results, Figure 5, captions, and supplement |
| 10 | No clinical-readiness claim appears | PASS | Clinical readiness is expressly disclaimed |
| 11 | No universal model claim appears | PASS | Neither universal ERM superiority nor universal GroupDRO failure is claimed |
| 12 | No WSI/patient-level effectiveness claim appears | PASS | Patch-level limitation is explicit |
| 13 | Ethics statement is supported and bounded | PASS | Public, de-identified secondary analysis; no IUBAT exemption claimed |
| 14 | AI assistance is disclosed | PASS | Full required declaration before References; summary on title page and cover letter |
| 15 | No generative-AI figure creation occurred | PASS | Submission figures are pixel-identical copies of accepted exp09b figures |

## Journal-format and package checks

| # | Check | Result | Evidence |
|---:|---|---|---|
| 16 | Citations are defined and numbered correctly | PASS | 30 references, first-appearance order 1–30, no missing or duplicate list number |
| 17 | Abstract is no more than 250 words | PASS | 221 words |
| 18 | Keyword count is exactly seven | PASS | Seven binding English keywords |
| 19 | Five highlights meet the 85-character limit | PASS | 68, 69, 63, 64, and 69 characters |
| 20 | Double-anonymization audit passes | PASS | See `submission/jpi/JPI_Anonymization_Audit.md` |
| 21 | All DOCX files open successfully | PASS | Ten editable DOCX files parsed successfully |
| 22 | PDF proof renders without clipping | PASS | 11 pages rendered and visually inspected |
| 23 | All figures pass visual inspection | PASS | Six PNG and six TIFF copies; source pixels unchanged; 300 dpi metadata |
| 24 | All tables are readable and cited | PASS | Five editable main tables, sequentially cited |
| 25 | No placeholders remain | PASS | No unresolved citation, TODO, TBD, or insertion marker detected |
| 26 | No exp09 evidence artifact changed | PASS | Authorization, metrics, tables, and run/summary state hashes match accepted baselines |
| 27 | No dataset or model action occurred | PASS | Build helpers import no `torch`, `torchvision`, `datasets`, or project data/model module |
| 28 | No second `ood_test` attempt occurred | PASS | Attempt count remains one; scripts 38 and 39 were not run |

## References, tables, and figures

- References: 30 manuscript-cited records, converted from verified BibTeX metadata into first-appearance AMA-style numbering. DOI links were retained where present.
- Main tables: 5 editable tables embedded in the manuscript and duplicated in `JPI_Tables.docx` for portal flexibility.
- Figures: 6 accepted exp09b scientific figures copied without resampling or pixel alteration. TIFF and PNG copies are tagged 300 dpi; dimensions and sizes are recorded in `JPI_Figure_Manifest.csv`.
- Supplementary material: locked split mapping, checkpoint-hash/temperature provenance, all 14 frozen operating points, one-shot provenance, high-confidence false negatives, reproducibility summary, expanded limitations, and inventory.
- CLAIM 2024 checklist: 44 items with explicit Yes, No, or Not applicable status; no patient-level or clinical item was fabricated as complete.

## Code-release QA

The sanitized local archive contains 91 files and is reproducibly timestamped. It excludes images, caches, `.venv`, credentials, checkpoints, raw patch-level predictions, authorization records, run sentinels, and JPI build helpers containing author metadata. Credential/path scanning passed. Archive SHA256:

`27FF9ED54AA4C4A1898216B8E098D6716DB0709FC81D6845E835FA6ABDE1ED2C`

## Operational warning

Microsoft Word, LibreOffice, and Pandoc were unavailable in the environment. The editable DOCX files were validated structurally with `python-docx`. The PDF proof was generated by a document-only fallback renderer using ReportLab and visually inspected page by page. A final human opening in Microsoft Word and the JPI previewer remains required before upload because pagination can differ across Word/rendering engines.

`python-docx` 1.2.0 and its dependency `lxml` 6.1.1 were installed in the project virtual environment for this milestone; no unrelated package was installed.

## Final disposition

The package is scientifically and structurally complete. The only remaining action is a human visual review in the intended desktop editor and manual entry/upload through the JPI submission system. Do not rerun `ood_test` or reopen model development.
