# Final Citation Audit

**Milestone:** 9C
**Audit date:** 2026-07-17
**Verdict:** **PASS**

## Inventory

- Verified BibTeX entries: **46**
- Unique citation keys used in `docs/FINAL_MANUSCRIPT_WITH_CITATIONS.md`: **38**
- Unique citation keys used across the manuscript and verified literature review: **46**
- Explicitly labeled background/review-only entries: **8**
- Unresolved literature citation gaps: **0**
- Duplicate DOI records: **0**
- Duplicate normalized-title records: **0**

## Audit checks

| Check | Result | Evidence |
|---|---|---|
| Every manuscript citation key exists in BibTeX | PASS | Offline key-set validation |
| Every BibTeX record has title, author, and year | PASS | `scripts/41_verify_literature_and_citations.py` |
| Every bibliography row is marked verified | PASS | 46/46 rows in `exp09c_verified_references.csv` |
| DOI/title duplicates absent | PASS | Normalized duplicate scan |
| No fabricated placeholder reference | PASS | Metadata checked against DOI, PubMed/PMC, proceedings, publisher, or official project pages |
| No `[CITATION NEEDED: ...]` marker remains | PASS | Zero markers in cited manuscript |
| No `[CITATION UNRESOLVED: ...]` marker remains | PASS | Zero markers in cited manuscript |
| Unresolved gaps are explicitly tabulated | PASS | Header-only unresolved table; 0 unresolved rows |
| No citation supports this project's numerical result | PASS | Result sections cite no external source for project-generated values |
| Citation scope matches nearby wording | PASS | Claim-level map and manual review |
| Unused bibliography entries are labeled | PASS | Eight entries labeled background or verified-review-only |
| Manuscript scientific numbers are unchanged | PASS | Numeric-token multiset identical after citation markup is removed |
| GroupDRO/ERM roles are unchanged | PASS | GroupDRO remains predeclared primary; ERM remains matched control |
| Unfavorable final result is preserved | PASS | Development-to-test reversal retained |
| No clinical-readiness or universal-model claim introduced | PASS | Manual claim scan |
| Journal facts use official sources | PASS | Official publisher/journal pages recorded with verification date |

## Wording adjustments reviewed

Three areas were narrowed for evidentiary accuracy:

1. High-confidence errors are described as warranting separate audit because confidence
   can influence reliance; no direct patient-harm or review-omission effect is attributed
   to this dataset.
2. Group-label limitations are conditional on whether known groups represent future
   variation; no cited paper is used to claim the causal reason for center-2 failure.
3. External-validation guidance is described as supporting transparent, separated, and
   prespecified evaluation. Retaining one completely locked hospital is presented as
   this study's design recommendation rather than a universal reporting mandate.

These changes narrow interpretation only. No project result or conclusion was
strengthened.

## Metadata and source policy

- Peer-reviewed articles were checked using DOI/publisher records and PubMed/PMC where
  available.
- Conference papers were checked against official PMLR, NeurIPS, or OpenReview records.
- The official WILDS page is intentionally typed as an authoritative project/dataset
  record rather than a peer-reviewed article.
- Reporting guidelines are labeled as guidelines; reviews and commentaries are labeled
  by publication type.
- APCs were recorded only when a current official amount was found. Otherwise, the
  shortlist explicitly requires re-verification at submission.

## Offline verification result

` .venv/Scripts/python.exe scripts/41_verify_literature_and_citations.py `

returned:

```text
PASS: Milestone 9C citation audit
bibtex_entries=46
manuscript_citation_keys=38
package_citation_keys=46
background_or_review_only_entries=8
unresolved_citation_gaps=0
duplicate_doi_records=0
duplicate_title_records=0
dataset_loaded=false
ood_test_accessed=false
inference_run=false
```

## Remaining non-citation actions

The ethics/IRB statement, author list and contributions, funding, conflicts,
acknowledgments, public code repository, archival DOI, and mirror-specific dataset link
still require author confirmation. These are submission metadata tasks, not unresolved
literature citations.
