# JPI Double-Anonymization Audit

**Verdict: PASS**

Audited files:

- `JPI_Anonymized_Manuscript.docx`
- `JPI_Anonymized_Manuscript.pdf`

## Visible-content checks

| Check | Result |
|---|---|
| Author name absent | PASS |
| Email and telephone absent | PASS |
| Affiliation and postal address absent | PASS |
| IUBAT/institution name absent | PASS |
| Acknowledgements absent | PASS |
| Named CRediT statement absent | PASS |
| Personal Windows username and local file paths absent | PASS |
| Identity-bearing text absent from the PDF proof | PASS |

Bibliographic institution names were permitted where they form part of a cited source. No author identity appeared through that exception.

## File-structure and metadata checks

| Check | Result | Evidence |
|---|---|---|
| DOCX creator metadata generic | PASS | `Anonymous` |
| DOCX last-modified-by metadata generic | PASS | `Anonymous` |
| PDF author metadata generic | PASS | `Anonymous` |
| Custom document properties absent | PASS | No `docProps/custom.xml` part |
| Comments absent | PASS | No comments part or relationship |
| Tracked changes absent | PASS | No insertion, deletion, or move revision markup |
| Hidden text absent | PASS | No `vanish` or `webHidden` markup |
| Personal paths absent from DOCX XML | PASS | No Windows user path detected |

The DOCX opened successfully as an editable document. The 11-page PDF proof opened and rendered successfully. Every page was visually inspected; no clipping, overlap, blank unintended page, or identity leakage was observed.

## Scope note

The title page, cover letter, declaration-of-interest file, and author-declarations file intentionally retain the author identity and were not treated as blinded files.
