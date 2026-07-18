# Reproducibility record

This document separates the reproducible development workflow from the completed, non-repeatable reserved-hospital evaluation. It contains no private authorization text or run-state payload.

## Locked split mapping

| Logical split | Hugging Face split | Centers | Rows | Permitted role |
|---|---|---:|---:|---|
| `train` | `train` | 0, 3, 4 | 302,436 | Training only |
| `id_val` | `validation` | 0, 3, 4 | 33,560 | ID selection and operating-point provenance |
| `ood_val` | `validation` | 1 | 34,904 | OOD development and calibration fitting; never training |
| `ood_test` | `test` | 2 | 85,054 | One completed reserved final evaluation; never rerun |

The dataset source is `wltjr1007/Camelyon17-WILDS` through Hugging Face `datasets`. No random split is permitted.

## Frozen model roles

- Predeclared primary: 7F GroupDRO by center.
- Matched control: 7F center-stratified ERM.
- The matched ERM control was not relabeled as the primary after the final result.

The accepted local checkpoints are excluded. Their immutable fingerprints are:

| Model | Local artifact size | SHA256 |
|---|---:|---|
| GroupDRO | 134,250,665 bytes | `CE0DC65DC6106648F2ABB77C603746A1DB2B856F9672B01C2E8F540BEEAD8502` |
| Center-stratified ERM | 134,250,601 bytes | `8BF990BE517A41AA74D111E0A4F4111A05A9FE416919261D979F2FD65AE974FF` |

## Frozen calibration and operating points

- GroupDRO temperature: 2.974907.
- Center-stratified ERM temperature: 3.496293.
- Calibration was fit during development and never refit on `ood_test`.
- The operating-point family contains four fixed-specificity and three fixed-sensitivity candidates per model, for 14 total thresholds.
- Thresholds were selected on `id_val` and frozen before final evaluation.
- Public provenance is in `results/tables/exp08i_frozen_operating_points.csv` and final transfer results are in `results/tables/exp09_final_ood_test_frozen_operating_points.csv`.
- No threshold was selected, changed, or tuned on center 2. Every operating point is candidate and non-clinical.

## Single-run provenance

The final run began at 2026-07-16T17:16:25.455291+00:00, completed inference at 2026-07-16T17:35:17.051997+00:00, and completed its CSV-only summary at 2026-07-16T17:35:20.395673+00:00. It used the Hugging Face `test` split restricted to center 2, processed 85,054 rows per model in one dataloader traversal, and preserved the frozen pair, temperatures, and thresholds. Attempt count is one.

No training, optimizer step, backward pass, checkpoint write, calibration fitting, threshold tuning, or post-test model selection occurred during the final run. The private authorization and run-state records are intentionally excluded.

## Expected public aggregate outputs

The accepted final public evidence includes:

- `results/metrics/exp09_final_ood_test/final_metrics.json`
- `results/tables/exp09_final_ood_test_classification.csv`
- `results/tables/exp09_final_ood_test_calibration.csv`
- `results/tables/exp09_final_ood_test_frozen_operating_points.csv`
- `results/tables/exp09_final_ood_test_high_confidence_false_negatives.csv`
- `results/figures/exp09_final_ood_test_roc.png`
- `results/figures/exp09_final_ood_test_precision_recall.png`
- `results/figures/exp09_final_ood_test_calibration_reliability.png`
- `results/figures/exp09_final_ood_test_operating_points.png`
- `results/figures/exp09_final_ood_test_high_confidence_fn.png`
- manuscript-facing exp09b tables and figures

The public artifact classification is maintained in `docs/PUBLIC_ARTIFACT_MANIFEST.md`.

## Permitted workflow

1. Create the environment documented in `README.md`.
2. Obtain the dataset independently under its terms and keep it outside Git.
3. Read `PROJECT_PROTOCOL.md` before using the numbered scripts.
4. For independent development, use centers 0, 3, and 4 for training, center 1 for OOD development, and establish a new genuinely reserved evaluation source. Do not use center 2 to modify the accepted project.
5. Rebuild reporting/manuscript artifacts only from committed aggregate evidence where the corresponding script supports that input.

Scripts 38 and 39 document the guarded final-evaluation implementation and must not be run for the accepted project. Raw final predictions and logits must not be regenerated, reconstructed, or committed.

## Excluded artifacts and regeneration

- Dataset patches, source images, Hugging Face caches, and balanced subset caches are excluded by data-access and size policy.
- Checkpoints are excluded; only hashes and sizes are public.
- Raw patch-level predictions and logits are excluded; aggregate metrics and tables are public.
- Authorization, run state, completion sentinels, and local logs are excluded as governance or machine-local records.
- TIFF submission copies are regenerated locally from accepted PNG/source figures by the JPI package builder.
- `JPI_Reproducibility_Code_Package.zip` is a redundant local build artifact; the repository itself is the public code source.
- The identity-bearing title page and cover letter remain local because they contain a private telephone number.

## Limits of reproducibility

Without redistributed data, weights, and row-level predictions, the exact accepted inference is not replayable from the public repository alone. This is intentional and consistent with the one-shot policy. The public evidence supports code inspection, protocol review, aggregate-result verification, manuscript reconstruction, and independent replication under a newly reserved evaluation design.
