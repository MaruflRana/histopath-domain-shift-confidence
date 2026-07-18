# Journal of Pathology Informatics submission package

## Submission identity

- Journal: Journal of Pathology Informatics (Elsevier)
- Article type: Original Research Article
- Title: When Development Gains Do Not Transfer: Confidence-Aware Tumor Detection Under Reserved-Hospital Shift
- Corresponding author: Jishan Islam Maruf (jishanislammaruf62@gmail.com)
- Abstract: 221 words
- Main tables: 5
- Main figures: 6
- Supplementary files: 2 (`JPI_Supplementary_Material.docx` and `JPI_CLAIM_Checklist.docx`)
- Official author guide: https://www.sciencedirect.com/journal/journal-of-pathology-informatics/publish/guide-for-authors
- CLAIM 2024 guideline: https://pubs.rsna.org/doi/10.1148/ryai.240300

## File roles and recommended upload order

1. `JPI_Title_Page.docx` — title page with author identity, correspondence, counts, declarations, CRediT, and acknowledgements.
2. `JPI_Anonymized_Manuscript.docx` — primary editable double-anonymized manuscript with editable tables and AMA-style references.
3. `JPI_Anonymized_Manuscript.pdf` — review proof only; do not use as the sole source file.
4. `JPI_Highlights.docx` — five highlights, each ≤85 characters.
5. `JPI_Cover_Letter.docx` — editor-facing cover letter.
6. `JPI_Declaration_of_Interest.docx` and `JPI_Author_Declarations.docx` — declaration files.
7. `figures/Figure_1.tiff` through `Figure_6.tiff` — figure submission files; PNG copies are for review.
8. `JPI_Figure_Captions.docx` — editable captions.
9. `JPI_Tables.docx` — separate editable tables if requested by the submission system; the same main tables are embedded in the manuscript.
10. `JPI_Supplementary_Material.docx` — technical provenance, thresholds, reproducibility, and expanded limitations.
11. `JPI_CLAIM_Checklist.docx` — completed CLAIM 2024 reporting checklist.
12. `JPI_Reproducibility_Code_Package.zip` — sanitized local code archive; upload only if the journal permits or requests code supplements.

## Manual submission-system fields

Enter the exact title, sole author identity, affiliation, corresponding-author address/email/telephone, seven keywords, funding declaration, competing-interest declaration, ethics/consent statements, data/code availability statements, and generative-AI disclosure from the prepared files. Select Original Research Article and double-anonymized review. No ORCID is supplied.

The current article-processing charge and any waiver eligibility must be re-verified on the official JPI/Elsevier pages immediately before submission because charges and policies may change. This package intentionally records no unverified APC amount.

## Final pre-upload inspection

Open every DOCX and the PDF proof; inspect page breaks, tables, superscript citations, figure files, captions, and anonymization. Confirm that the submission portal has not exposed title-page identity to reviewers. Verify the current journal declarations, APC/waiver information, and required file designations. Do not rerun `ood_test` or reopen model development.

Milestone 9D automated and visual QA passed. See `JPI_Anonymization_Audit.md` and
`../../docs/JPI_FINAL_SUBMISSION_AUDIT.md`. Because Word/LibreOffice was unavailable in the build
environment, the PDF proof used a document-only fallback renderer; open the DOCX in Microsoft Word
and inspect the JPI portal-generated proof before submission.
