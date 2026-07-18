# Recommended Submission Strategy

## First choice: Journal of Pathology Informatics

Submit first to the Journal of Pathology Informatics as a Research Article. The
specialist audience is the best match for a pathology-informatics paper whose main
contribution is rigorous hospital-shift evaluation rather than a new architecture.

### Framing

Lead with the reserved-hospital reliability result:

> A predeclared GroupDRO development advantage reversed on a separately reserved
> hospital, while frozen calibration improved confidence reliability without
> correcting classification errors or stabilizing operating points.

The cover letter should emphasize:

- pathology-specific hospital shift;
- the matched, predeclared GroupDRO/ERM comparison;
- the one-shot final evaluation and no post-test model switching;
- the negative result as a validation lesson;
- directly reusable reliability and governance reporting.

### Expected reviewer concerns

- one benchmark and one reserved target hospital;
- patch-level rather than whole-slide/patient-level endpoints;
- capped source cache and ResNet-18;
- no broader DG-method comparison;
- low sensitivity and high false-negative burden;
- whether the frozen protocol is enough novelty for a Research Article.

### Changes before submission

- Reduce the abstract to no more than 250 words.
- Select no more than seven keywords.
- Prepare an anonymized manuscript and separate title page.
- Complete ethics, authorship, funding, competing-interest, data, code, and AI-use
  declarations.
- Attach CLAIM and a short reproducibility supplement.

## Second choice: Journal of Medical Imaging

Use the same scientific story, but foreground imaging-evaluation methodology and the
difference among discrimination, calibration, and operating-point transport.

### Expected reviewer concerns

- no novel imaging algorithm;
- limited architecture and benchmark breadth;
- absence of uncertainty baselines or statistical intervals;
- patch-level scope.

### Changes before submission

- Tighten the methods contribution around the predeclared evaluation design.
- Add a compact protocol schematic as a supplementary figure if permitted.
- Make code and artifact provenance accessible to reviewers.
- Verify the live SPIE word, figure, and data-policy requirements immediately before
  formatting.

## Safer fallback: PLOS ONE

PLOS ONE is the safest fallback because it evaluates research for technical validity
and explicitly accommodates negative or null findings. Preserve the unfavorable
GroupDRO result and the full limitations section.

### Expected reviewer concerns

- absence of confidence intervals or cluster-aware uncertainty;
- generalizability beyond one benchmark;
- data/code accessibility;
- potential dependence among patches from the same slide or patient.

### Changes before submission

- Provide a complete Data Availability Statement and public code/reproducibility
  archive if author permissions allow.
- Include the claim map, frozen-protocol checklist, and final-test provenance as
  supplementary material.
- Explain why no post-test bootstrap or subgroup analysis was performed under the
  one-shot governance rule.

## Supplementary material

Recommended supplementary package:

1. Locked split and protocol table.
2. Checkpoint, temperature, threshold, and authorization provenance.
3. Full development-versus-final comparison table.
4. All 14 operating-point transfer rows.
5. High-confidence false-negative table.
6. CLAIM/TRIPOD+AI applicability checklist.
7. Reproducibility checklist and one-shot run-state summary.

Do not include raw restricted imagery or imply that the Hugging Face mirror can be
redistributed by the authors.

## Code-release recommendation

Release the analysis and reporting code before or at submission if institutional and
co-author approval permits. Exclude large checkpoints or derived predictions if
licensing, privacy, or repository constraints are unresolved; publish hashes,
configuration, schemas, and instructions regardless. Archive a tagged release with a
DOI when possible.

## Submission sequence rule

Do not re-enter model development after editorial feedback. Revisions may improve
wording, literature, reporting, code packaging, and non-inferential analyses of
already saved aggregate artifacts. No new center-2 inference, threshold selection, or
post-test model replacement is allowed.
