# Citation Integration Changelog

## Scope

Milestone 9C created `docs/FINAL_MANUSCRIPT_WITH_CITATIONS.md` from the accepted
Milestone 9B manuscript. The integration used verified literature only. No scientific
result, model role, protocol decision, checkpoint, temperature, threshold, or final
interpretation changed.

## Abstract and keywords

- No external citation was added to the abstract.
- All result values and the development-to-test reversal were preserved.
- Keywords were unchanged.

## Introduction

- Replaced the histopathology domain-shift marker with Stacke, Howard, and Tellez.
- Replaced the external-validation marker with Kleppe, Varoquaux and Cheplygina, and
  Steyerberg and Harrell.
- Replaced the calibration/operating-point marker with Van Calster, Leeflang, and
  related calibration evidence.
- Narrowed the high-confidence-error sentence. The draft suggested that a confident
  false negative could be less likely to trigger review. The cited human-factors
  literature supports altered reliance and automation bias, but not that exact
  workflow consequence in this dataset. The cited version therefore says that
  confident errors warrant a separate audit.
- Added verified domain-generalization and GroupDRO citations without changing the
  study's claims.

## Related work

- Added primary computational-pathology domain-shift, stain-normalization, external
  validation, CAMELYON17, WILDS, GroupDRO, calibration, proper-scoring-rule, and
  selective-classification sources.
- Narrowed the group-label limitation to a conditional statement: group robustness
  depends on whether the specified groups capture variation relevant to the future
  shift.
- Explicitly separated selective-prediction background from this study's
  non-clinical confidence audit.

## Materials and methods

- Added the CAMELYON17 challenge paper, WILDS benchmark paper, and official WILDS
  dataset page to the dataset description.
- Added the original GroupDRO source to the loss and exponentiated-gradient update.
- Added the original temperature-scaling source to the argmax-invariance statement.
- No method, metric, split, count, temperature, threshold, or run-state wording was
  changed.

## Results

- No literature citation was used to support a project result.
- Every numerical value was retained exactly from the accepted manuscript.
- Minus signs and confidence-threshold symbols were normalized typographically only.

## Discussion

- Added direct evidence for known site-specific stain/scanner effects while retaining
  the manuscript's statement that explanations for center 2 remain hypotheses.
- Added calibration-under-shift literature to limit the generality of the held-out
  temperature-scaling result.
- Added diagnostic-test and decision-analysis sources to distinguish threshold
  transport from clinical utility.
- Replaced the unqualified recommendation that a reserved institution “should” be
  preregistered with a source-accurate statement: established guidance emphasizes
  transparent partitioning, external validation, and complete reporting; the
  one-institution lock is this study's design recommendation, not a universal rule
  imposed by those guidelines.

## Limitations

- Added whole-slide and digital-pathology validation sources to the patch-level
  limitation.
- Preserved all limitations concerning one hospital, one benchmark, capped caches,
  ResNet-18, untested DG methods, absent model-based uncertainty, calibration scope,
  poor sensitivity, and the prohibition on rerunning center 2.

## Conclusion and availability statements

- Added responsible-clinical-AI and reporting-guideline citations to the conclusion
  without adding a clinical-readiness claim.
- Replaced the unresolved dataset-license marker with the official WILDS statement
  that its standardized Camelyon17 variant is public-domain/CC0.
- Retained an author action to verify the Hugging Face mirror-specific link and
  wording before submission.

## Unresolved items

There are zero unresolved literature citation markers. Author-supplied ethics,
conflict-of-interest, funding, contribution, acknowledgments, repository URL, and
archival metadata remain administrative placeholders rather than literature gaps.
