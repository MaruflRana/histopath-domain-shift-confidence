# Literature and Citation Gaps

This checklist identifies manuscript statements that require external literature verification. It intentionally contains no invented references or bibliographic metadata. During the next milestone, replace each `[CITATION NEEDED: topic]` marker only with a verified source.

## Histopathology domain shift

- Evidence that staining, scanner, laboratory workflow, tissue preparation, and site-specific acquisition can shift histopathology image distributions.
- Evidence that cross-site performance degradation is a recognized problem in computational pathology.
- Reviews of stain variability, domain shift, and generalization in digital pathology.
- Studies showing that model ordering or performance can vary across external hospitals.
- Appropriate scope for claims about external validation in computational pathology.

## Camelyon17 and WILDS

- Original Camelyon17 dataset description, task definition, institutions, labels, and evaluation design.
- WILDS benchmark paper and the intended Camelyon17 distribution-shift formulation.
- Documentation confirming the hospital-center split roles used by the benchmark.
- Any caveat about patch-level sampling, metadata, and benchmark prevalence needed for methods reporting.
- Licensing and permitted data-use wording for the final data-availability statement.

## GroupDRO

- Original Group Distributionally Robust Optimization method.
- The exponentiated-gradient group-weight update and worst-group objective.
- Applications or evaluations of GroupDRO under domain shift.
- Limitations of group-defined robustness when groups incompletely represent future shifts.
- Evidence that group labels may be too coarse or miss within-group heterogeneity.

## Calibration and temperature scaling

- Original temperature-scaling calibration paper.
- Definitions and standard use of expected calibration error, Brier score, and negative log-likelihood.
- Evidence that temperature scaling is argmax-invariant.
- Literature on calibration degradation under distribution shift.
- Literature on calibration transfer or failure across domains/hospitals.
- Cautions about fitting and evaluating calibration on the same split.

## Selective prediction and abstention

- Foundational selective-classification or risk-coverage literature.
- Definitions of coverage, selective risk, and abstention.
- Evidence that confidence-based abstention can fail under distribution shift.
- Literature distinguishing threshold selection from final evaluation.
- Clinical-AI literature on uncertainty-aware review or referral, without implying that this study validates such a workflow.

## High-confidence clinical errors

- Clinical-AI or diagnostic-safety literature explaining why confidently wrong predictions can be especially harmful.
- Evidence concerning false-negative consequences in metastatic breast-cancer pathology or lymph-node assessment.
- Literature on confidence communication and automation bias.
- Appropriate language for discussing patch-level missed tumors without claiming patient-level harm.

## External validation and reproducibility

- Guidance for external validation of medical imaging and clinical AI models.
- Reporting guidelines applicable to diagnostic prediction models, computational pathology, or AI in medicine.
- Literature supporting preregistration, locked protocols, or separation of model development from final evaluation.
- Recommendations for multi-site, temporal, or prospective validation.
- Guidance on clustered uncertainty or patient/slide-level resampling when patch observations are correlated.
- Reproducibility and transparent reporting standards for machine-learning studies.

## Operating-point transportability

- Evidence that sensitivity/specificity thresholds may not transport across prevalence, case mix, acquisition, or institutions.
- Literature distinguishing discrimination, calibration, and clinical utility.
- Guidance on selecting and validating clinical operating thresholds.
- Decision-curve or clinical-utility literature, if the final venue expects utility analysis.
- Studies of threshold instability under dataset shift.

## Manuscript sections containing citation markers

- Introduction: domain shift, computational pathology reliability, and external validation.
- Related work: WILDS/Camelyon17, GroupDRO, calibration, selective prediction, and medical-AI reporting.
- Methods: dataset provenance, GroupDRO objective, temperature scaling, ECE/Brier/NLL, and high-confidence error motivation.
- Discussion: hospital-specific shift hypotheses, external-validation discipline, calibration limitations, and operating-point transportability.
- Ethics/data availability: dataset licensing and benchmark governance.

## Verification rules for the next milestone

1. Prefer original methods papers, benchmark papers, authoritative reporting guidelines, and systematic reviews.
2. Verify title, authors, venue, year, DOI/URL, and the exact claim supported.
3. Do not cite a paper for a stronger claim than it establishes.
4. Keep the center-2 scientific results sourced to local accepted artifacts, not external literature.
5. Record unresolved citation gaps rather than inserting plausible but unverified references.
