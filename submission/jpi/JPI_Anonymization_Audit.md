# JPI Double-Anonymization Audit

**Milestone:** 9F
**Verdict:** **PASS**

Audited blinded files:

- `JPI_Anonymized_Manuscript.docx`
- `JPI_Anonymized_Manuscript.pdf`
- `JPI_Anonymized_Manuscript.md`
- `JPI_Supplementary_Material.docx`
- `JPI_CLAIM_Checklist.docx`
- `JPI_Reproducibility_Code_Package.zip`

## Visible-content checks

| Check | Result |
|---|---|
| Jishan Islam Maruf absent | PASS |
| Ishtiak Al Mamoon absent | PASS |
| Both author emails absent | PASS |
| IUBAT and shared affiliation absent | PASS |
| `MaruflRana` and the exact GitHub URL absent | PASS |
| Telephone, room, extension, and postal identity absent | PASS |
| Acknowledgements and named CRediT statements absent | PASS |
| Personal Windows username and local paths absent | PASS |
| Blinded data/code wording exact | PASS |
| Identifying repository address explicitly withheld | PASS |

Bibliographic institution names remain permitted only when they form part of a cited source. No
author identity appeared through that exception.

## DOCX/PDF structure and metadata

| Check | Result | Evidence |
|---|---|---|
| DOCX creator metadata generic | PASS | `Anonymous` |
| DOCX last-modified-by metadata generic | PASS | `Anonymous` |
| PDF author metadata generic | PASS | `Anonymous` |
| Custom document properties absent | PASS | No `docProps/custom.xml` part |
| Comments absent | PASS | No comments part or relationship |
| Tracked changes absent | PASS | No insertion, deletion, or move revision markup |
| Hidden text absent | PASS | No `vanish` markup |
| Personal paths absent | PASS | No local Windows path detected |
| Anonymized manuscript page count | PASS | 11 pages |
| Rendered page layout | PASS | All 11 pages inspected; no clipping, overlap, or blank unintended page |

## Reviewer code ZIP

The existing 91-entry reviewer ZIP was scanned without extraction. It contained no author name,
email, IUBAT reference, `MaruflRana`, exact repository URL, local path, credential, certificate,
authorization file, run sentinel/state file, checkpoint, or raw prediction. Its SHA256 remained:

`27FF9ED54AA4C4A1898216B8E098D6716DB0709FC81D6845E835FA6ABDE1ED2C`

The ZIP therefore did not require rebuilding.

## Scope note

The title page, cover letter, declaration-of-interest file, and author-declarations file are
identity-bearing editorial files and intentionally contain the finalized author metadata. The
title page and cover letter remain local and excluded from GitHub under repository policy.
