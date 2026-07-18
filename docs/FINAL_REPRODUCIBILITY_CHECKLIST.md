# Final Reproducibility Checklist

## Frozen data protocol

| Logical split | HF split | Center(s) | Rows | Role |
|---|---|---|---:|---|
| `train` | `train` | {0,3,4} | 302,436 | Source pool for model fitting |
| `id_val` | `validation` | {0,3,4} | 33,560 | In-distribution validation and threshold selection |
| `ood_val` | `validation` | {1} | 34,904 | Development-stage OOD assessment and calibration fitting |
| `ood_test` | `test` | {2} | 85,054 | Single reserved final evaluation, now complete |

Dataset identifier: `wltjr1007/Camelyon17-WILDS`.

## Frozen checkpoints and provenance

| Model | Role | Checkpoint path | File size | SHA256 |
|---|---|---|---:|---|
| 7F GroupDRO-by-center | Predeclared primary | `results/checkpoints/exp07f_groupdro_resnet18/best.pt` | 134,250,665 | `CE0DC65DC6106648F2ABB77C603746A1DB2B856F9672B01C2E8F540BEEAD8502` |
| 7F center-stratified ERM | Matched control | `results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt` | 134,250,601 | `8BF990BE517A41AA74D111E0A4F4111A05A9FE416919261D979F2FD65AE974FF` |

The hashes above are read from the accepted frozen manifests and inference sentinel. Milestone 9B does not load either checkpoint.

## Frozen temperatures

- GroupDRO: `2.974907`
- Center-stratified ERM: `3.496293`
- Calibration refit on `ood_test`: `false`
- Hard predictions changed by calibration: `false`

## Frozen operating points

- Source: `results/tables/exp08i_frozen_operating_points.csv`
- Fixed-specificity targets: 0.80, 0.85, 0.90, 0.95
- Fixed-sensitivity targets: 0.80, 0.90, 0.95
- Models: two
- Total thresholds: 14
- Test-set threshold selection: `false`
- Clinical status: candidate/non-clinical

## Authorization and one-shot run

- Authorization file: `configs/experiments/exp09_final_ood_test_authorization.json`
- Authorization timestamp: `2026-07-16T23:14:05.0366792+06:00`
- Authorization SHA256: `0B5977E3D92990B6FC59E1138AC2D398C737F4BDCBACCDD1025D0CE0E3B96C45`
- Protocol: `8D`
- Run limit: `1`
- Run start: `2026-07-16T17:16:25.455291+00:00`
- Test access start: `2026-07-16T17:16:31.227243+00:00`
- Dataset load complete: `2026-07-16T17:33:49.542237+00:00`
- Inference start: `2026-07-16T17:33:59.940980+00:00`
- Inference complete: `2026-07-16T17:35:17.051997+00:00`
- Summary complete: `2026-07-16T17:35:20.395673+00:00`
- Attempt counter: `1`
- Run counter: `1`
- Dataset instances: `1`
- Dataloader traversals: `1`
- Models evaluated per batch: `2`

## Final data validation

- HF split exactly `test`: PASS
- Center exactly `{2}`: PASS
- Rows/model exactly 85,054: PASS
- Labels binary: PASS
- Label counts: 42,527 non-tumor and 42,527 tumor
- Required metadata complete: PASS
- Probabilities finite and within [0,1]: PASS
- Probability sums within `1e-5`: PASS
- Raw/calibrated argmax identical: PASS

## Prohibited operations

- Training or fine-tuning: none
- Optimizer or backward pass: none
- Checkpoint save or weight update: none
- Calibration fitting on test: none
- Threshold selection on test: none
- Post-test model selection: none
- MC-dropout, ensemble, CORAL, or DANN: none
- Second inference attempt: none and not authorized

## Prediction outputs

- `results/predictions/exp09_final_ood_test/groupdro_ood_test_raw_predictions.csv`
- `results/predictions/exp09_final_ood_test/groupdro_ood_test_calibrated_predictions.csv`
- `results/predictions/exp09_final_ood_test/centerstrat_erm_ood_test_raw_predictions.csv`
- `results/predictions/exp09_final_ood_test/centerstrat_erm_ood_test_calibrated_predictions.csv`

## Metrics, tables, figures, and reports

- Metrics JSON: `results/metrics/exp09_final_ood_test/final_metrics.json`
- Final 9A tables: `results/tables/exp09_final_ood_test_*.csv`
- Final 9A figures: `results/figures/exp09_final_ood_test_*.png`
- Manuscript 9B tables: `results/tables/exp09b_*.csv`
- Manuscript 9B figures: `results/figures/exp09b_*.png`
- Inference report: `results/logs/exp09_final_ood_test_inference_report.txt`
- Final test report: `results/logs/exp09_final_ood_test_report.txt`
- Manuscript integration report: `results/logs/exp09b_final_manuscript_integration_report.txt`

## Durable state and sentinels

- Run state: `results/logs/exp09_final_ood_test_run_state.json`
- Inference-complete sentinel: `results/logs/exp09_final_ood_test_completed_run.json`
- Summary-complete sentinel: `results/logs/exp09_final_ood_test_summary_completed.json`

## Software and hardware environment

- OS: Windows
- Python: 3.12.10
- PyTorch: 2.9.0+cu128
- CUDA runtime: 12.8
- GPU: NVIDIA GeForce RTX 3060
- `datasets`: 5.0.0
- Matplotlib used for paper-facing figures

## Why a second run is prohibited

The reserved hospital was intended to provide one unbiased final evaluation after the model pair, checkpoints, temperatures, thresholds, metrics, and output policy were frozen. Once center 2 was read, another inference attempt could enable test-driven debugging or selection and would break the one-shot interpretation. The durable `RUN_STARTED`/attempt record and completed sentinels therefore remain permanent. All later manuscript and reporting work must use the saved exp09 artifacts.
