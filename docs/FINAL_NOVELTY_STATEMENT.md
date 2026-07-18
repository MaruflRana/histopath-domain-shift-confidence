# Final Novelty Statement

**Current manuscript title:** Hospital Domain Shift Can Reverse Development Gains in Histopathology Tumor Detection: A Reserved-Center Study of Calibration and Operating-Point Transfer

## Recommended statement

This study's novelty is methodological and empirical rather than architectural. It
combines a strictly reserved-hospital evaluation with a predeclared GroupDRO primary
candidate and matched ERM control, then evaluates whether development-stage model
ranking, frozen temperature scaling, and `id_val`-selected operating points transfer
to the reserved hospital. The one-shot protocol exposed a development-to-test
reversal, documented held-out calibration improvement without test refitting,
demonstrated poor operating-point transportability, and quantified the distinction
between reducing high-confidence false negatives and reducing total false negatives.

## What is new in this package

- A controlled, predeclared GroupDRO-versus-ERM comparison whose roles were not
  changed after the final result.
- Explicit separation of the OOD development hospital from the single reserved final
  hospital.
- Application of frozen temperatures without center-2 refitting, with raw and
  calibrated reliability reported together.
- Application of 14 frozen, development-selected candidate operating points without
  test-set threshold selection.
- A high-confidence false-negative audit that shows confidence correction without
  error correction.
- Durable one-shot provenance that prevents a favorable rerun or post-test model
  switch.

## What is not claimed

The study does not introduce a neural architecture, DG objective, calibration
algorithm, uncertainty method, or clinical decision rule. It does not claim
state-of-the-art performance, universal ERM superiority, universal GroupDRO failure,
robustness to all hospitals, whole-slide or patient-level effectiveness, or clinical
utility.

## Conservative one-sentence contribution

A predeclared, one-shot reserved-hospital evaluation showed that a GroupDRO
development advantage reversed against matched ERM, while frozen calibration
improved confidence reliability but neither corrected missed tumors nor stabilized
development-selected operating points.
