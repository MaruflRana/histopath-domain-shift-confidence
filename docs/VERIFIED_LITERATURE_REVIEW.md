# Verified Literature Review

## Hospital domain shift in histopathology

Digital histopathology can encode institution-specific signals arising from stain,
scanner, preparation, and acquisition practices. Stacke et al. directly quantified
histopathology domain shift, while Howard et al. demonstrated site-specific digital
histology signatures that affected model accuracy and bias
[@Stacke2021HistopathologyDomainShift; @Howard2021SiteSpecificHistology]. Tellez et
al., Macenko et al., and Vahadane et al. establish the methodological background for
stain normalization and augmentation, but do not imply that one normalization policy
will transfer to every target site [@Tellez2019StainNormalization;
@Macenko2009HistologyNormalization; @Vahadane2016ColorNormalization].

This manuscript differs from work that treats a single development OOD site as the
endpoint. Its defensible contribution is the observed reversal between a repeatedly
used development hospital and a separately reserved hospital. It does not establish
the causal source of center-2 shift, because no post-test image or subgroup
exploration was conducted.

## CAMELYON17 and WILDS

The CAMELYON17 challenge evaluated automated detection of lymph-node metastases
across five medical centers [@Bandi2019Camelyon17]. WILDS standardized a patch-level
Camelyon17 benchmark in which hospitals define domains and the held-out hospital
tests distribution-shift performance [@Koh2021WILDS; @WILDSCamelyon17Dataset].

The present work uses that domain structure but adds a project-level governance
layer: explicit separation of development and final hospitals, predeclared
model/control roles, frozen calibration and thresholds, and a one-attempt execution
record. The benchmark still represents one disease task and one source collection;
it is not an independent non-Camelyon clinical cohort.

## GroupDRO and domain generalization

GroupDRO optimizes a worst-group-weighted objective when training-group identities are
known [@Sagawa2020GroupDRO]. DomainBed and the broader domain-generalization literature
show that model-selection protocol and the relationship between observed and future
domains are central to credible DG claims [@Gulrajani2021DomainBed;
@Zhou2023DomainGeneralizationSurvey]. Piratla et al. further illustrate that robustness
claims depend on how group distributions are represented
[@Piratla2022GroupRobustness].

This study does not introduce a new DG algorithm. Its contribution is an honest,
controlled test of a predeclared GroupDRO candidate against matched ERM. The negative
final result does not prove universal ERM superiority or universal GroupDRO failure;
it shows that the center-1 development advantage did not transport to center 2.

## Calibration under shift

Temperature scaling is a simple post-hoc calibration method that preserves argmax
predictions [@Guo2017Calibration]. ECE, Brier score, and NLL capture related but
non-identical aspects of probabilistic performance [@Brier1950ProbabilityForecasts;
@Gneiting2007ProperScoringRules; @VanCalster2019Calibration]. Ovadia et al. and
Minderer et al. show that uncertainty and calibration can degrade under distribution
shift [@Ovadia2019UncertaintyShift; @Minderer2021Calibration].

The manuscript adds a held-out transfer check in which temperatures fitted before
test access improved all three calibration metrics on center 2 without refitting.
That finding is specific to two models and one reserved hospital. It does not validate
the temperatures universally, and it does not imply that calibration repaired
classification errors.

## Selective prediction

Selective classification formalizes abstention and risk-coverage tradeoffs
[@ElYaniv2010SelectiveClassification; @Geifman2017SelectiveClassification].
SelectiveNet jointly learns prediction and selection in a dedicated architecture
[@Geifman2019SelectiveNet].

This project did not implement or validate a clinical abstention system. Selective
prediction is relevant only as conceptual background for auditing highly confident
missed tumors. Calling the frozen confidence thresholds “clinical rejection
thresholds” would exceed the evidence.

## High-confidence clinical errors

Human-factors studies show that decision aids can change clinician reliance and that
automation bias is a real concern [@Goddard2012AutomationBias;
@Gaube2021ClinicalDecisionAids]. In pathology, experimental work has also examined
human-AI interaction and imperfect assistants [@Kiani2020HistopathologyAssistant;
@Tschandl2020HumanComputerCollaboration]. These sources motivate auditing confident
errors, but they do not establish a patient-outcome effect for the present models.

The manuscript's defensible finding is narrower: frozen calibration sharply reduced
the number of false negatives whose predicted confidence exceeded 0.90, 0.95, or
0.99, while total false negatives were unchanged. Confidence correction is not error
correction.

## External validation and reporting

External validation assesses model performance in data distinct from development
and is essential for evaluating transportability [@Bleeker2003ExternalValidation;
@Steyerberg2016Validation]. TRIPOD and TRIPOD+AI, CLAIM, and related guidance promote
transparent reporting of datasets, model evaluation, and limitations
[@Collins2015TRIPOD; @Collins2024TRIPODAI; @Tejani2024CLAIM;
@Park2018AIEvaluation]. STARD, DECIDE-AI, and sample-size guidance address adjacent
diagnostic and evaluation stages [@Bossuyt2015STARD; @Vasey2022DECIDEAI;
@Riley2021ExternalValidationSampleSize].

The one-shot reserved-center design is stricter than merely reporting an external
split after repeated access. Its contribution is empirical evidence that this
discipline exposed a model-ranking reversal. It remains a retrospective patch-level
benchmark evaluation, not a prospective clinical study.

## Operating-point transportability

Sensitivity and specificity can vary with spectrum, setting, and threshold
application, and calibration does not by itself establish clinical usefulness
[@Leeflang2013SensitivitySpecificity; @VanCalster2016CalibrationHierarchy].
Decision-curve analysis is one framework for evaluating consequences once a clinical
decision context and threshold range are justified [@Vickers2006DecisionCurve].

The frozen `id_val` thresholds failed to preserve several nominal targets on center
2. This supports a claim of operating-point instability across these hospitals, not
a claim that a replacement threshold was found. No center-2 threshold was optimized.

## Digital-pathology translation

Clinical-grade pathology systems commonly require whole-slide aggregation, workflow
integration, and technical validation beyond patch classification
[@Campanella2019ClinicalGradePathology; @Lu2021WeaklySupervisedPathology;
@Pantanowitz2013WSIValidation]. Broader reviews likewise emphasize the gap between
retrospective algorithm performance and clinical impact [@Bera2019DigitalPathologyAI;
@Echle2021CancerPathology; @Wiens2019DoNoHarm; @Kelly2019ClinicalImpact].

Accordingly, this manuscript's novelty is methodological and empirical rather than
clinical: it demonstrates how a reserved-hospital protocol, calibration transfer
audit, operating-point audit, and high-confidence false-negative analysis can reveal
risks hidden by development-only reporting.
