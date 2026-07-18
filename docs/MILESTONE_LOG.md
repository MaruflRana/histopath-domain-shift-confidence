# MILESTONE LOG

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in
Histopathology: Calibration, Uncertainty, and Selective Prediction

Chronological record of completed, accepted milestones. Each entry lists goal, what was
built, how it was verified, and the accepted outcome. Development milestones remain
development evidence; Milestone 9A is the completed locked final center-2 evaluation.

**Current state:** the latest completed milestone is **Milestone 9E**. The single explicitly
authorized `ood_test` inference and CSV-only summary completed with attempt count 1 in Milestone
9A; Milestones 9B through 9E did not rerun `ood_test`. No second `ood_test` inference is
authorized. Earlier entries preserve
their then-current “latest milestone” and “next step” wording as historical record only; those
statements are not active instructions. Current next action is final JPI portal review and manual
submission; model development remains closed.

---

## Milestone 1 — Foundation + dataset verification ✅ PASSED

- **Goal:** project scaffolding, environment check, and verification that the locked HF
  split mapping holds.
- **Scripts:** `00_check_env.py`, `01_verify_hf_dataset.py`, `02_dataset_sanity.py`.
- **Verified:** HF splits/fields/center distributions; per-split counts; label/center
  breakdowns; a sample patch grid.
- **Artifacts:** `results/tables/hf_split_verification.csv`,
  `dataset_split_summary.csv`, `dataset_label_counts.csv`, `dataset_center_counts.csv`;
  `results/figures/sample_patch_grid.png`; `results/logs/env_check.txt`.
- **Outcome:** PASS. Locked split mapping confirmed against the dataset.

## Milestone 2A — DataLoader smoke test ✅ PASSED

- **Goal:** PyTorch dataset + dataloaders producing correct tensors for all four splits.
- **Built:** `src/data/transforms.py` (RGB→float32→ImageNet norm, 96×96 guard),
  `torch_dataset.py` (`Camelyon17HFDataset`), `dataloaders.py` (`build_dataloader`,
  shuffle only for train).
- **Script:** `03_dataloader_smoke_test.py` (max_samples=128, batch_size=16,
  num_workers=0).
- **Verified:** every split yields `(16,3,96,96)` float32 images, int64 labels; correct
  center separation (train/id_val⊂{0,3,4}, ood_val={1}, ood_test={2}).
- **Outcome:** OVERALL PASS. `results/logs/dataloader_smoke_test.txt`.

## Milestone 2B — Model forward-pass smoke test ✅ PASSED

- **Goal:** minimal backbone + classifier producing logits `[B,2]`.
- **Built:** `src/models/backbones.py` (`create_backbone` resnet18/50, `fc`→`Identity`,
  returns `(backbone, feature_dim)`), `classifier.py` (`CamelyonClassifier`,
  `forward → [B,2]`).
- **Script:** `04_model_forward_smoke_test.py` (max_samples=32, batch_size=16).
- **Verified:** one batch per split → logits `(16,2)`, float32, on `cuda:0`, no NaN/Inf;
  input `(16,3,96,96)`.
- **Outcome:** OVERALL PASS. `results/logs/model_forward_smoke_test.txt`.

## Milestone 3A — Training mechanics smoke test ✅ PASSED

- **Goal:** verify backward/update loop on one tiny batch.
- **Built:** `src/training/losses.py` (`create_loss` cross-entropy),
  `optimizers.py` (`create_optimizer` AdamW), `steps.py`
  (`train_step`/`eval_step` → `{loss, accuracy, batch_size}`).
- **Script:** `05_train_mechanics_smoke_test.py` (one batch, 20 steps, lr 1e-4).
- **Verified:** loss fell 0.5497 → 0.0056 over 20 steps; all losses finite; a
  parameter changed after `optimizer.step()`; id_val no-grad sanity finite.
- **Outcome:** OVERALL PASS. `results/logs/train_mechanics_smoke_test.txt`.
- **Finding:** first `train` batch was single-class (label 1) — motivated 3B/3C.

## Milestone 3B — Short balanced training ✅ PASSED

- **Goal:** short multi-batch loop on a balanced `train` subset, eval on balanced
  `id_val`; no OOD, no checkpointing.
- **Built:** `src/data/balanced_subset.py` (`collect_balanced_examples`,
  `BalancedSubsetDataset`), `src/training/loop.py`
  (`train_one_epoch`/`evaluate`),
  `configs/experiments/exp03b_short_train_resnet18.yaml`,
  `scripts/06_train_short_run.py`.
- **Verified:** train `{0:128,1:128}`, id_val `{0:64,1:64}`; 3 epochs; finite losses;
  CSV + text logs saved.
- **Outcome:** OVERALL PASS.
- **Finding:** collecting balanced `id_val` required scanning **14,102** streamed
  records to reach label 0 → the stream is label-ordered → motivated 3C.

## Milestone 3C — Dataset access hardening + balanced cache ✅ PASSED

- **Goal:** make the data layer robust to the label-ordered stream before any longer
  training.
- **Built / changed:**
  - `src/data/balanced_subset.py`: added `max_scan`, `progress_every`, `seed`,
    `shuffle_collected`; **lazy image decode** (only kept images decoded, not the
    ~128k scanned-and-skipped); **disk-cache layer**
    `load_or_build_balanced_examples` (uint8-array serialization, instant reload).
  - `scripts/07_label_order_diagnostic.py`: quantifies label ordering (labels only).
  - `scripts/08_hf_cache_status.py`: read-only HF cache/disk status.
  - `configs/experiments/exp03b_short_train_resnet18.yaml`: `max_scan_train: 200000`,
    `max_scan_id_val: 30000`, `progress_every: 5000`, `shuffle_collected: true`,
    `use_cache: true`.
  - `scripts/06_train_short_run.py`: uses the cache function + new knobs.
- **Diagnostic result:** train label-0 not seen in first 5,000; id_val first label-0 at
  index 14,033; first 1,000 of both splits single-class (WARN emitted).
- **Design decision (user-approved):** cache the balanced subset to disk — one heavy
  scan ever, then instant reuse. See `docs/DATASET_DECISIONS.md`.
- **Build:** one-time run streamed to `scanned=128,310` to reach train balance
  (label-0 block starts ~128,182), cached both subsets. Re-run hit cache in seconds
  with identical metrics.
- **Interim issue (resolved):** an edit briefly dropped the `BalancedSubsetDataset`
  class declaration → `ImportError`; caught in the log, fixed, re-verified.
- **Latest accepted results (ResNet18, 3 epochs):**

  | epoch | train_loss | train_acc | id_val_loss | id_val_acc |
  |-------|-----------:|----------:|------------:|-----------:|
  | 1 | 0.471775 | 0.8086 | 1.484480 | 0.5000 |
  | 2 | 0.187952 | 0.9531 | 0.619032 | 0.6953 |
  | 3 | 0.087263 | 0.9805 | 0.272071 | 0.8672 |

- **Outcome:** OVERALL PASS.

---

## Milestone 4A — Checkpointed capped development training ✅ PASSED

- **Goal:** a larger-but-still-capped **development** training run using `train` +
  `id_val` only, adding checkpoint save/restore and best-model selection by
  `id_val_loss`. Still a development run, **not** a scientific result. Authorized by
  explicit user go-ahead.
- **Built:**
  - `src/training/checkpointing.py` — `save_checkpoint(path, model, optimizer, epoch,
    metrics, config)` and `load_checkpoint(path, model, optimizer=None,
    map_location="cpu")`; payload bundles `model_state_dict`, `optimizer_state_dict`,
    `epoch`, `metrics`, `config`. Writes `best.pt` (on `id_val_loss` improvement) and
    `last.pt` (every epoch).
  - `src/training/history.py` — `append_history_csv(path, row)`; incremental per-epoch
    CSV with columns `epoch, train_loss, train_accuracy, id_val_loss, id_val_accuracy,
    train_n, id_val_n, is_best` (header written once).
  - `configs/experiments/exp04a_dev_train_resnet18.yaml` — `train_per_class: 1000`,
    `id_val_per_class: 250`, `batch_size: 32`, `epochs: 5`, AdamW lr 1e-4 / wd 1e-4,
    `selection: {metric: id_val_loss, mode: min}`, checkpoint dir under
    `results/checkpoints/`.
  - `scripts/09_train_dev_checkpointed.py` — the checkpointed run: builds/loads cached
    balanced subsets, trains 5 epochs, evaluates `id_val` each epoch, saves
    `last.pt`/`best.pt`, then reloads `best.pt` into a fresh model and re-evaluates
    `id_val` to verify the checkpoint round-trips.
- **Data:** both balanced caches were **CREATED** on this run (first build at the larger
  `per_class`):
  - train `{0:1000, 1:1000}` — scanned to `scanned=129,283` to reach the label-0 block
    → `results/cache/balanced_train_pc1000_seed1337_shuf.pt` (~92.4 MB).
  - id_val `{0:250, 1:250}` — scanned to `scanned=14,308`
    → `results/cache/balanced_id_val_pc250_seed1337_shuf.pt` (~23.0 MB).
- **Results (ResNet18, 5 epochs, best by `id_val_loss`):**

  | epoch | train_loss | train_acc | id_val_loss | id_val_acc | is_best |
  |-------|-----------:|----------:|------------:|-----------:|:-------:|
  | 1 | 0.261794 | 0.8955 | 0.162842 | 0.9440 | ✓ |
  | 2 | 0.098321 | 0.9665 | 0.344737 | 0.8020 |   |
  | 3 | 0.040005 | 0.9865 | 0.173314 | 0.9380 |   |
  | 4 | 0.035312 | 0.9885 | 0.117608 | 0.9620 | ✓ |
  | 5 | 0.026127 | 0.9900 | 0.112202 | 0.9640 | ✓ |

- **Best:** epoch **5**, `id_val_loss` **0.112202**, `id_val_acc` 0.9640.
- **Checkpoints:** `results/checkpoints/exp04a_dev_train_resnet18/{best.pt,last.pt}`
  (~128 MB each).
- **Reload verification:** `best.pt` reloaded into a fresh `CamelyonClassifier` and
  re-evaluated on `id_val` reproduced `id_val_loss` 0.112202 / `id_val_acc` 0.9640
  exactly (match within 1e-5) → **PASS**.
- **Assertions (all passed):** `train_n==2000`; `id_val_n==500`; both labels present in
  train and id_val; all losses finite; `best.pt` and `last.pt` exist; loaded best
  checkpoint runs an `id_val` evaluation; no `ood_val` / `ood_test` usage.
- **Outcome:** OVERALL PASS. `results/logs/exp04a_dev_train_resnet18_report.txt`,
  `exp04a_dev_train_resnet18_log.csv`.
- **Note:** development numbers on a 2000-sample balanced subset — NOT a scientific
  result and NOT final performance. Warning seen: benign HF "unauthenticated requests"
  notice (ignored, as per RUNBOOK).

---

## Milestone 4B — Prediction export + classification metrics scaffold ✅ PASSED

- **Goal:** a reusable evaluation pipeline that loads the accepted best checkpoint, runs
  inference on the **cached** `train` + `id_val` balanced subsets, exports per-example
  prediction CSVs, and computes classification metrics — preparing for later
  calibration / OOD work. No training; no OOD splits. Authorized by explicit user
  go-ahead. Not a scientific result.
- **Built:**
  - `src/evaluation/__init__.py` — package exports.
  - `src/evaluation/metrics_classification.py` —
    `compute_binary_classification_metrics(y_true, prob_1, threshold=0.5)` with
    **tumor = 1** as positive; returns `n, accuracy, balanced_accuracy, auroc, auprc,
    sensitivity, specificity, precision, f1, tn, fp, fn, tp` (+ `threshold`,
    `positive_class`, `warnings`). AUROC/AUPRC return `None` with a warning when
    `y_true` is single-class; zero-denominator ratios return `None`. Uses sklearn for
    AUROC/AUPRC.
  - `src/evaluation/predict.py` — `predict_to_dataframe(model, dataloader, device,
    split_name)`: `model.eval()` + `torch.no_grad()` + softmax; returns a pandas
    DataFrame with `split, index, label, center, image_id, patient, node, x_coord,
    y_coord, slide, logit_0, logit_1, prob_0, prob_1, pred_label, correct`, preserving
    metadata.
  - `scripts/10_eval_cached_checkpoint.py` — loads
    `results/checkpoints/exp04a_dev_train_resnet18/best.pt`, rebuilds
    `CamelyonClassifier` from the checkpoint's stored config, loads the two cached
    subsets (cache HIT, no streaming), runs inference (`shuffle=False`, batch_size 64),
    writes prediction CSVs + metrics JSON/CSV + a text report.
- **Results (tumor = positive, threshold 0.5):**

  | split | n | accuracy | balanced_acc | auroc | auprc | sensitivity | specificity | precision | f1 | tn/fp/fn/tp |
  |-------|--:|---------:|-------------:|------:|------:|------------:|------------:|----------:|---:|:-----------:|
  | train  | 2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1000/0/0/1000 |
  | id_val |  500 | 0.9640 | 0.9640 | 0.9933 | 0.9943 | 0.9440 | 0.9840 | 0.9833 | 0.9633 | 246/4/14/236 |

- **Consistency:** `id_val` accuracy 0.9640 equals the checkpoint's stored
  `id_val_accuracy` exactly (the reload+eval path agrees with training-time evaluation).
- **Artifacts:**
  `results/predictions/exp04a_dev_train_resnet18/{train,id_val}_cached_predictions.csv`;
  `results/metrics/exp04a_dev_train_resnet18/classification_metrics.json`;
  `results/tables/exp04a_dev_train_resnet18_classification_metrics.csv`;
  `results/logs/exp04a_eval_cached_checkpoint_report.txt`.
- **Assertions (all passed):** checkpoint exists; both cache files exist;
  `train n==2000`; `id_val n==500`; prediction CSVs exist; metrics files exist;
  probabilities finite and in [0,1]; `prob_0+prob_1≈1` (1e-5); no `ood_val` / `ood_test`
  usage.
- **Outcome:** OVERALL PASS. No errors/warnings (no HF network access this run).
- **Note:** development evaluation on cached balanced subsets — NOT final performance.
  `train` metrics of 1.0 reflect evaluation on the model's own training data, not
  generalization.

---

## Milestone 5A — OOD-val calibration (temperature scaling) ✅ PASSED

- **Goal:** the **first explicitly authorized use of `ood_val`** — export raw `ood_val`
  predictions from the accepted 4A checkpoint, fit temperature scaling on the `ood_val`
  logits, export calibrated predictions, and compare classification + calibration
  metrics before vs after. No weight changes; `ood_test` stays LOCKED. Authorized by
  explicit user go-ahead. Not a final result.
- **Authorization recorded:** `ood_val` authorized 2026-07-08 for OOD decisions only
  (see `docs/SESSION_STATE.md` §5). `CLAUDE.md` §5 still needs the matching edit when in
  scope.
- **Built:**
  - `src/evaluation/metrics_calibration.py` — `compute_ece(y_true, prob_1, n_bins=15)`
    (equal-width confidence bins), `compute_brier_score`, `compute_nll_from_probs`
    (clip only for log stability), `compute_calibration_metrics` → `{n, ece, brier, nll,
    n_bins}`. numpy only; invalid probabilities raise clearly.
  - `src/calibration/__init__.py`, `src/calibration/temperature_scaling.py` —
    `TemperatureScaler(nn.Module)` with a stable `log_temperature` parameter
    (`T=exp(log_T)`, clamped to `[min,max]`), `get_temperature`, `forward = logits/T`;
    `fit_temperature(...)` (Adam on NLL, model weights untouched, returns fitted T +
    before/after NLL) and `apply_temperature_to_logits`.
  - `configs/experiments/exp05a_calibrate_ood_val_resnet18.yaml` — `split: ood_val`,
    `per_class: 1000`, `max_scan: 40000`, `shuffle_collected: false`; calibration
    `max_iter 200`, `lr 0.01`, `T_init 1.0`, bounds `[0.05, 10.0]`.
  - `scripts/11_calibrate_ood_val.py` — loads best.pt, builds/loads the `ood_val` cache,
    raw inference, temperature fit, calibrated export, metrics + temperature JSON/CSV,
    text report.
- **Data:** `ood_val` cache **CREATED** this run (center-1 scan to `scanned=18,452`;
  order = label-0 block then label-1) →
  `results/cache/balanced_ood_val_pc1000_seed1337_noshuf.pt` (~84.6 MB), `{0:1000,1:1000}`.
- **Result:** fitted **T = 3.1224** (NLL 0.915158 → 0.608814; `converged=false` at the
  200-iter cap, loss still improving slowly — acceptable).

  | variant | accuracy | balanced_acc | auroc | auprc | ECE | Brier | NLL | tn/fp/fn/tp |
  |---------|---------:|-------------:|------:|------:|----:|------:|----:|:-----------:|
  | raw        | 0.6195 | 0.6195 | 0.7696 | 0.6793 | 0.2330 | 0.2759 | 0.9152 | 834/166/595/405 |
  | calibrated | 0.6195 | 0.6195 | 0.7696 | 0.6793 | 0.0918 | 0.2160 | 0.6088 | 834/166/595/405 |

  Classification metrics identical raw vs calibrated (temperature scaling is
  argmax-invariant); calibration improved ECE ↓61%, NLL ↓33%, Brier ↓22%.
- **Artifacts:**
  `results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_{raw,calibrated}_predictions.csv`;
  `results/metrics/exp05a_calibrate_ood_val_resnet18/{calibration_metrics.json,temperature.json}`;
  `results/tables/exp05a_calibrate_ood_val_resnet18_calibration_metrics.csv`;
  `results/logs/exp05a_calibrate_ood_val_report.txt`.
- **Assertions (all passed):** `ood_val n==2000`; both labels present; checkpoint exists;
  **no `ood_test` usage**; raw + calibrated CSVs exist; metrics JSON/CSV + temperature
  JSON exist; raw and calibrated probabilities finite and in [0,1]; `prob_0+prob_1≈1`;
  temperature finite and within `[0.05, 10.0]`; calibrated NLL finite; calibrated NLL ≤
  raw NLL (+1e-6).
- **Outcome:** OVERALL PASS. One benign transient HF error
  (`peer closed connection ... validation-00002-of-00003.parquet`) was auto-retried by
  `datasets` (expected per CLAUDE.md §6) and did not affect the run.
- **Note:** first OOD result — the ID-trained model is over-confident and only
  moderately accurate under center-1 shift (acc 0.62, AUROC 0.77). DEVELOPMENT result on
  a 2000-sample balanced `ood_val` subset — NOT final performance.

---

## Milestone 5B — Deterministic uncertainty metrics from saved predictions ✅ PASSED

- **Goal:** compute uncertainty / error-detection metrics **purely from existing saved
  prediction CSVs** — no inference, no training, no MC-dropout, no ensembling, no
  selective-prediction thresholds, and `ood_test` never read. Authorized by explicit user
  go-ahead. Not a final result.
- **Inputs (unmodified):**
  - `results/predictions/exp04a_dev_train_resnet18/id_val_cached_predictions.csv` (n=500),
  - `results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_raw_predictions.csv` (n=2000),
  - `results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_calibrated_predictions.csv` (n=2000).
- **Built:**
  - `src/evaluation/metrics_uncertainty.py` — vectorised score functions
    (`max_confidence`, `uncertainty_from_confidence`, `predictive_entropy`,
    `probability_margin`, `uncertainty_from_margin`, `energy_score_from_logits`) and
    evaluation functions (`compute_error_detection_metrics` — error = positive class;
    `compute_high_confidence_error_summary`; `compute_high_confidence_false_negative_summary`;
    `compute_risk_coverage_curve` + `compute_aurc`). Every uncertainty score is oriented
    higher = more uncertain; undefined metrics return `None` with a warning. numpy/pandas
    only; sklearn only for AUROC.
  - `scripts/12_uncertainty_metrics_from_predictions.py` — loads each CSV, validates
    required columns + probabilities (finite, in [0,1], `prob_0+prob_1≈1`), appends score
    columns, computes all metrics, writes enriched CSVs + metrics JSON + flat tables +
    report.
- **Results (development analysis, positive = tumor = 1; error = `correct==False`):**

  | variant | n | errors | error_rate | FN (label1,pred0) | err-detect AUROC (entropy) | entropy AURC |
  |---------|--:|-------:|-----------:|------------------:|---------------------------:|-------------:|
  | id_val              |  500 |  18 | 0.0360 |  14 | 0.9267 | 0.003558 |
  | ood_val_raw         | 2000 | 761 | 0.3805 | 595 | 0.6883 | 0.219861 |
  | ood_val_calibrated  | 2000 | 761 | 0.3805 | 595 | 0.6883 | 0.219861 |

  - Error-detection AUROC is **identical across all four scores** within a variant
    (confidence/entropy/margin are monotone transforms of each other in the 2-class case;
    energy differs slightly: id_val 0.9259, ood_val 0.6743/0.6583). Uncertainty separates
    errors well ID (0.93) but only moderately under center-1 shift (0.69).
  - **Raw vs calibrated ood_val give identical AUROC and AURC** — temperature scaling is
    monotonic, so it preserves ranking (a correctness sanity check). It changes confidence
    *magnitudes*: high-confidence errors at `confidence ≥ 0.90` fall from n_high=1049
    (278 errors, 222 FNs) raw → n_high=256 (6 errors, **0 FNs**) calibrated. Calibration
    removes the confidently-wrong tumor misses in this subset.
  - **High-confidence false negatives (missed tumors the model is sure about):** id_val
    6/3/2 at t=0.90/0.95/0.99; ood_val_raw 222/133/33; ood_val_calibrated 0/0/0.
- **Artifacts:**
  `results/predictions/exp05b_uncertainty/{id_val,ood_val_raw,ood_val_calibrated}_uncertainty_scores.csv`;
  `results/metrics/exp05b_uncertainty/uncertainty_metrics.json`;
  `results/tables/exp05b_uncertainty_metrics.csv`, `exp05b_risk_coverage.csv`,
  `exp05b_high_confidence_errors.csv`;
  `results/logs/exp05b_uncertainty_metrics_report.txt`.
- **Assertions (all passed):** id_val n==500; ood_val raw/calibrated n==2000; all required
  input files exist; probabilities finite, in [0,1], `prob_0+prob_1≈1`; enriched CSVs +
  metrics JSON/CSV + risk-coverage CSV + high-confidence CSV + report saved;
  **no `ood_test` usage**. Input CSVs were read-only (not modified).
- **Outcome:** OVERALL PASS. One fix during the run: `np.trapz` was removed in NumPy 2.x →
  switched AURC to `np.trapezoid` with a NumPy-1.x fallback.
- **Note:** DEVELOPMENT analysis on capped balanced subsets — NOT final performance.
  `ood_val` used for OOD *analysis only* (authorized); `ood_test` untouched. No thresholds
  were selected or committed (risk–coverage is descriptive only — selective prediction is
  the next milestone).

---

## Milestone 5C — Selective prediction / abstention from saved uncertainty CSVs ✅ PASSED

- **Goal:** convert the Milestone 5B uncertainty scores into **selective-prediction /
  abstention policy tables** — coverage, risk, retained accuracy, and how false negatives
  (missed tumors) redistribute between retained and abstained sets — computed **purely
  from saved CSVs**. No inference, no training, no MC-dropout, no ensembling; `ood_test`
  never read; `ood_val` used for development-stage selective analysis only; no operating
  threshold committed. Authorized by explicit user go-ahead. Not a final result.
- **Inputs (unmodified):**
  `results/predictions/exp05b_uncertainty/{id_val,ood_val_raw,ood_val_calibrated}_uncertainty_scores.csv`
  (n = 500 / 2000 / 2000).
- **Built:**
  - `src/evaluation/metrics_selective.py` — `selective_metrics_at_coverages`,
    `threshold_for_target_coverage`, `retained_subset_by_coverage`,
    `summarize_selective_prediction`, `summarize_false_negatives_under_abstention`.
    Retention is rank-based (retain the k = round(coverage·n) lowest-uncertainty rows;
    stable sort), so actual coverage is exact up to rounding. Reuses the 5B error / FN
    masks. Undefined rates (zero denominator) return `None`.
  - `scripts/13_selective_prediction_from_uncertainty.py` — validates required columns,
    computes selective tables for all four uncertainty scores (primary = `entropy`),
    computes the FN-under-abstention breakdown, writes flat tables + metrics JSON + report.
- **Results (development analysis; error = `correct==False`; FN = `label==1 & pred==0`):**

  Entropy selective prediction, coverage → retained risk / retained accuracy / retained FN:

  | coverage | id_val risk | id_val ret_FN | ood_val risk | ood_val ret_FN |
  |---------:|------------:|--------------:|-------------:|---------------:|
  | 1.00 | 0.0360 | 14 | 0.3805 | 595 |
  | 0.95 | 0.0189 |  6 | 0.3758 | 562 |
  | 0.90 | 0.0111 |  3 | 0.3633 | 517 |
  | 0.80 | 0.0075 |  2 | 0.3431 | 435 |
  | 0.70 | 0.0029 |  1 | 0.3243 | 356 |
  | 0.50 | 0.0000 |  0 | 0.2500 | 198 |

  - **id_val:** abstaining on the least-confident 50% drives retained risk to 0 and
    abstains **all 14** false negatives (fraction_of_fn_abstained → 1.0). Even at 90%
    coverage, risk falls 0.036 → 0.011 and 11/14 FNs are abstained.
  - **ood_val:** selective prediction helps but far less under shift — at 50% coverage
    retained risk only falls 0.380 → 0.250 and **198 of 595 FNs remain in the retained
    set** (only 67% of FNs abstained). Confidence-based abstention cannot rescue a model
    that is confidently wrong under domain shift.
  - **Raw vs calibrated ood_val give identical risk / coverage / FN rows** — temperature
    scaling is monotone, so it preserves the entropy ranking and thus the selective
    tables. Only the entropy **thresholds** differ (e.g. at coverage 0.50: raw 0.2944 vs
    calibrated 0.6267), because calibration rescales entropy magnitudes. (Calibration's
    real benefit — relocating high-confidence FNs — was already quantified in 5B.)
- **Artifacts:**
  `results/tables/exp05c_selective_prediction_all_scores.csv` (72 rows = 3 variants × 4
  scores × 6 coverages), `exp05c_selective_prediction_entropy_summary.csv` (18),
  `exp05c_false_negative_abstention_summary.csv` (18);
  `results/metrics/exp05c_selective_prediction/selective_prediction_metrics.json`;
  `results/logs/exp05c_selective_prediction_report.txt`.
- **Assertions (all passed):** id_val n==500; ood_val raw/cal n==2000; all input files
  exist; all required columns present; all uncertainty scores finite; coverage table +
  FN table + metrics JSON + report saved; **no `ood_test` usage**. Input CSVs read-only.
- **Outcome:** OVERALL PASS.
- **Note:** DEVELOPMENT-stage selective-prediction analysis on capped balanced subsets —
  NOT final clinical performance. No abstention threshold was selected for deployment.

---

## Milestone 6A — Reliability reporting package from saved results ✅ PASSED

- **Goal:** consolidate the accepted outputs of Milestones 4B / 5A / 5B / 5C into a small
  set of publication-style **report tables + figures + a text summary** — computed
  **purely from saved CSV/JSON artifacts**. No inference, no training, no MC-dropout, no
  ensembling; input files unmodified; `ood_test` never read. Authorized by explicit user
  go-ahead. Not a final result.
- **Inputs (read-only):**
  `results/tables/exp04a_dev_train_resnet18_classification_metrics.csv` (id_val),
  `results/metrics/exp05a_calibrate_ood_val_resnet18/calibration_metrics.json` (ood_val
  classification + calibration + temperature),
  `results/tables/exp05b_uncertainty_metrics.csv`, `exp05b_risk_coverage.csv`,
  `exp05b_high_confidence_errors.csv`,
  `results/tables/exp05c_selective_prediction_entropy_summary.csv`,
  `exp05c_false_negative_abstention_summary.csv`.
- **Built:** `scripts/14_make_reliability_report_package.py` (matplotlib `Agg` backend;
  pandas/json only). Note: `ood_val` classification metrics were sourced from the 5A JSON
  (`classification.raw`), since the 4B classification CSV holds only `train` + `id_val`.
- **Tables generated (4):**
  - `results/tables/report_table_classification_shift.csv` — id_val vs ood_val_raw
    (accuracy, AUROC, AUPRC, sensitivity, specificity, F1, FN).
  - `results/tables/report_table_calibration.csv` — ood_val raw vs calibrated
    (temperature, ECE, Brier, NLL).
  - `results/tables/report_table_uncertainty.csv` — id_val / ood_val_raw /
    ood_val_calibrated (entropy error-detection AUROC, entropy AURC, error count, FN count).
  - `results/tables/report_table_selective_prediction.csv` — entropy selective prediction
    at coverage 1.00/0.90/0.70/0.50 for id_val and ood_val_calibrated (retained risk,
    retained FN, abstained FN, fraction of FN abstained).
- **Figures generated (5, matplotlib only):**
  `report_fig_domain_shift_bar.png`, `report_fig_calibration_metrics.png`,
  `report_fig_uncertainty_error_detection.png`, `report_fig_risk_coverage_entropy.png`,
  `report_fig_false_negative_abstention.png` (all under `results/figures/`).
- **Text summary:** `results/logs/report_reliability_package_summary.txt` — ID→OOD
  degradation, calibration improvement, high-confidence FN reduction, selective-prediction
  finding, and an explicit development-only / `ood_test`-locked scope statement.
- **Key consolidated numbers (development):**
  - ID→OOD degradation: accuracy 0.9640→0.6195, AUROC 0.9933→0.7696, sensitivity
    0.9440→0.4050.
  - Calibration (T=3.1224): ECE 0.2330→0.0918 (−61%), Brier 0.2759→0.2160 (−22%),
    NLL 0.9152→0.6088 (−33%); classification unchanged (argmax-invariant).
  - High-confidence (≥0.90) false negatives on ood_val: 222 (raw) → 0 (calibrated).
  - Entropy error-detection AUROC: id_val 0.9267 / ood_val 0.6883 (raw = calibrated).
  - Selective prediction @ 50% coverage: id_val risk 0.036→0.000 (all FNs abstained);
    ood_val_calibrated risk 0.3805→0.2500 with 198/595 FNs still retained (66.7% abstained).
- **Assertions (all passed):** every input exists; no `ood_test` path referenced;
  exactly one `id_val` classification row; all 4 tables + 5 figures exist and are
  non-empty. Input files were read-only (not modified).
- **Outcome:** OVERALL PASS. No warnings or errors.
- **Note:** consolidates DEVELOPMENT results on capped balanced subsets — NOT final
  clinical/paper performance. `ood_val` used for authorized OOD *analysis* only; `ood_test`
  remains LOCKED and untouched.

---

## Milestone 6B — Manuscript / readiness audit package (documentation-only) ✅ PASSED

- **Goal:** produce a paper-facing audit and manuscript-planning package from the completed
  development results through 6A. **Documentation and result-interpretation only** — no
  training, no inference, no new methods (no MC-dropout / ensemble), result CSV/JSON/figure
  inputs unmodified, `ood_test` never read, no final-performance claims. Authorized by
  explicit user go-ahead.
- **Inputs (read-only):** context docs (`CLAUDE.md`, `AGENTS.md`, `PROJECT_PROTOCOL.md`,
  `docs/SESSION_STATE.md`, `docs/MILESTONE_LOG.md`, `docs/NEXT_ACTIONS.md`) and the 6A
  reporting artifacts (`results/tables/report_table_*.csv`,
  `results/logs/report_reliability_package_summary.txt`).
- **Created:**
  - `docs/MANUSCRIPT_PLAN.md` — title options, abstract skeleton, intro argument, methods /
    experiments / results / discussion outlines, limitations, future work, target-journal
    positioning, explicit locked-`ood_test` note.
  - `docs/RESULTS_INTERPRETATION.md` — ID→OOD degradation, calibration effect,
    high-confidence FN reduction, uncertainty/error-detection behavior, selective-prediction
    behavior, why calibration changes confidence but not argmax, why OOD risk stays high
    after abstention, and a supports / does-not-support section.
  - `docs/CLAIMS_AND_LIMITATIONS.md` — two-column allowed vs forbidden claim map (includes
    all mandated allowed and forbidden claims), plus conditional claims and universal
    caveats.
  - `docs/FIGURE_TABLE_MANIFEST.md` — all 6A tables + figures, what each shows, manuscript
    placement, and main-vs-supplementary designation.
  - `docs/REVIEWER_RISK_AUDIT.md` — 9 anticipated reviewer risks (capped subsets, model
    strength, no DG baseline, no full/test eval, no WSI, no external data, same-subset
    calibration, deterministic-only uncertainty, class-order/cache sampling bias) each with
    severity, mitigation, and fix-now-vs-later disposition.
  - `results/tables/report_claims_audit.csv` — 21 claim rows
    (`claim, status, supporting_artifacts, caveat, manuscript_location`);
    9 allowed / 5 conditional / 7 forbidden.
- **Verification:** claims-audit CSV written and re-read (well-formed, 21 rows; status tally
  9/5/7). All numbers quoted in the docs trace to the 6A report tables / summary. Every
  document carries the development / capped-subset scope tag and the `ood_test`-locked note.
- **Outcome:** OVERALL PASS (documentation package). No code executed beyond writing the CSV;
  no inference/training; `ood_test` untouched; input result files unmodified.
- **Note:** manuscript *planning* and claim-governance only — establishes what may and may
  not be claimed. It does not add or change any experimental result.

---

## Milestone 7A — Stain/color augmentation ERM baseline ✅ PASSED

- **Goal:** train a **stain-robustness / domain-generalization baseline** — an ERM ResNet18
  with moderate stain/color augmentation — and compare it against the plain-ERM 4A
  development checkpoint on `id_val` + `ood_val`, to address reviewer risk **R3** (no
  stain-robustness / DG baseline). Development-stage analysis only, NOT final performance.
  Authorized by explicit user go-ahead.
- **Strict scope:** `train` (augmented) for training, `id_val` (eval transform) for model
  selection by `id_val_loss` ONLY. `ood_val` NEVER used for training or selection — read only
  AFTER training (script 16) for development-stage comparison. `ood_test` never touched. No
  MC-dropout, no ensemble, no new architecture, no locked-mapping/WILDS change, no
  calibration for this baseline yet.
- **Built:**
  - `src/data/stain_transforms.py` — train-only augmentation: `EnsurePILRGB` (coerces PIL/
    tensor input to size-guarded PIL RGB), `StainColorAugment` (torchvision `Compose`:
    `RandomHorizontalFlip`→`RandomVerticalFlip`→`RandomRotation`→`ColorJitter`→`ToTensor`→
    ImageNet `Normalize`, output float32 `[3,96,96]` matching the eval pipeline), and
    `build_stain_train_transform(aug_cfg)`. Moderate colour jitter only.
  - `configs/experiments/exp07a_stainaug_resnet18.yaml` — cache paths for train/id_val/
    ood_val, `augmentation` block (hflip/vflip 0.5, rotation 15°, brightness/contrast/
    saturation 0.20, hue 0.05), AdamW lr 1e-4 / wd 1e-4, 5 epochs, selection
    `id_val_loss`/min, output paths.
  - `scripts/15_train_stainaug_checkpointed.py` — loads the cached balanced `train`/`id_val`
    subsets (cache HIT, no streaming), applies stain augmentation to train only, trains 5
    epochs, saves `best.pt`/`last.pt` + CSV log + report, reloads `best.pt` and re-verifies
    the `id_val` evaluation. Does not read `ood_val`/`ood_test`.
  - `scripts/16_eval_stainaug_id_ood.py` — loads `best.pt`, runs eval-transform inference on
    cached `id_val` + `ood_val`, writes prediction CSVs + metrics JSON/CSV + report, and
    prints the plain-ERM-vs-stain-aug comparison. Does not read `ood_test`.
- **Data:** all three balanced caches were **cache HITs** (no HF streaming): train
  `{0:1000,1:1000}` (n=2000), id_val `{0:250,1:250}` (n=500), ood_val `{0:1000,1:1000}`
  (n=2000).
- **Training results (best by `id_val_loss`):**

  | epoch | train_loss | train_acc | id_val_loss | id_val_acc | is_best |
  |-------|-----------:|----------:|------------:|-----------:|:-------:|
  | 1 | 0.326070 | 0.8645 | 0.182616 | 0.9280 | ✓ |
  | 2 | 0.244792 | 0.9000 | 0.163943 | 0.9400 | ✓ |
  | 3 | 0.222708 | 0.9090 | 0.200999 | 0.9220 |   |
  | 4 | 0.193224 | 0.9195 | 0.177804 | 0.9380 |   |
  | 5 | 0.198658 | 0.9190 | 0.122964 | 0.9480 | ✓ |

  Best epoch **5**, `id_val_loss` **0.122964**, `id_val_acc` 0.9480. Reload verification
  reproduced id_val_loss/acc exactly (1e-5) → **PASS**.
- **Evaluation + comparison (tumor = positive, threshold 0.5):**

  | split | model | accuracy | auroc | sensitivity | specificity | FN | tn/fp/fn/tp |
  |-------|-------|---------:|------:|------------:|------------:|---:|:-----------:|
  | id_val  | plain ERM (4A)     | 0.9640 | 0.9933 | 0.9440 | 0.9840 |  14 | 246/4/14/236 |
  | id_val  | stain-aug (7A)     | 0.9480 | 0.9938 | 0.9080 | 0.9880 |  23 | 247/3/23/227 |
  | ood_val | plain ERM (5A raw) | 0.6195 | 0.7696 | 0.4050 | 0.8340 | 595 | 834/166/595/405 |
  | ood_val | stain-aug (7A)     | 0.5290 | 0.7492 | 0.0960 | 0.9620 | 904 | 962/38/904/96 |

- **Finding (honest negative result):** moderate stain/color augmentation did **not** improve
  OOD reliability on this capped balanced subset. ood_val accuracy fell 0.6195→0.5290
  (Δ−0.0905) and AUROC 0.7696→0.7492 (Δ−0.0204); ood_val tumor **sensitivity collapsed
  0.405→0.096** (FN 595→904) while specificity rose 0.834→0.962 — the augmented model became
  more conservative and misses ~90% of center-1 tumors. id_val was essentially preserved
  (accuracy 0.964→0.948, AUROC ~unchanged). So a *naive/moderate* stain-augmentation ERM
  baseline is insufficient here; it motivates stronger/tuned stain-robustness or a dedicated
  DG method rather than licensing any superiority claim.
- **Assertions (all passed):** train_n==2000; id_val_n==500; both labels present in train and
  id_val; losses finite; `best.pt`/`last.pt` exist; checkpoint reload matches; id_val
  predictions n==500; ood_val predictions n==2000; probabilities finite, in [0,1],
  `prob_0+prob_1≈1`; metrics JSON/CSV saved; **no `ood_test` usage**.
- **Outcome:** OVERALL PASS (both scripts). One benign fix mid-run: the script-16 comparison
  print used non-ASCII `Δ`/`—`/`−` characters that the Windows cp1252 console could not
  encode (the metrics/CSVs had already been computed and written correctly); replaced with
  ASCII and re-ran to a clean PASS with all assertions.
- **Note:** DEVELOPMENT baseline on capped balanced subsets — NOT final performance; no
  calibration/uncertainty run on this baseline yet; `ood_val` used post-training for
  authorized development comparison only; `ood_test` remains LOCKED and untouched.

---

## Milestone 7B — Threshold-normalized plain-ERM vs stain-aug comparison ✅ PASSED

- **Goal:** diagnose *why* the 7A stain/color augmentation baseline looked worse on
  `ood_val`. Distinguish (A) a genuine worsening of OOD score **ranking** from (B) a poorly
  placed default **0.5 threshold** that became overly conservative for the stain-aug model.
  **CSV-only** — no training, no inference, no calibration, no MC-dropout, no ensemble;
  `ood_test` never read; input prediction CSVs unmodified. Authorized by explicit user
  go-ahead. Not a final result.
- **Strict scope:** `id_val` used ONLY to select operating thresholds; `ood_val` used ONLY
  for development-stage comparison after thresholds are fixed. Positive class = tumor (1),
  `y_score = prob_1`, `pred = 1 if prob_1 >= t`.
- **Inputs (read-only):**
  - plain ERM: `results/predictions/exp04a_dev_train_resnet18/id_val_cached_predictions.csv`
    (n=500), `results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_raw_predictions.csv`
    (n=2000).
  - stain-aug: `results/predictions/exp07a_stainaug_resnet18/{id_val,ood_val}_predictions.csv`
    (n=500 / 2000).
- **Built:**
  - `src/evaluation/metrics_operating_points.py` — `compute_metrics_at_threshold`,
    `threshold_for_target_specificity` (smallest threshold with id_val spec ≥ target),
    `threshold_for_target_sensitivity` (largest threshold with id_val sens ≥ target),
    `make_threshold_sweep` (0.00–1.00 step 0.01), `apply_operating_points` (select on
    `id_val`, apply same threshold to both splits). Metrics per point: threshold, accuracy,
    balanced_accuracy, sensitivity, specificity, precision, f1, tn/fp/fn/tp,
    false_negative_rate_over_tumors, predicted_positive_rate. Zero-denominator ratios and
    unachievable/undefined targets return `None` explicitly.
  - `scripts/17_compare_plain_vs_stainaug_operating_points.py` — loads the four CSVs,
    validates columns/counts/probabilities, computes (A) default-0.5 metrics, (B) AUROC/AUPRC,
    (C) id_val-selected fixed-specificity operating points on `ood_val`, (D) id_val-selected
    fixed-sensitivity operating points on `ood_val`, (E) threshold sweeps; writes 5 tables +
    metrics JSON + report + 4 matplotlib figures.
- **Default threshold (0.5) — key ood_val numbers (tumor = positive):**

  | model | ood_val sens | ood_val spec | ood_val FN | ood_val acc |
  |-------|-------------:|-------------:|-----------:|------------:|
  | plain ERM (4A/5A)  | 0.4050 | 0.8340 | 595 | 0.6195 |
  | stain-aug (7A)     | 0.0960 | 0.9620 | 904 | 0.5290 |

- **Threshold-free ranking (AUROC / AUPRC):**

  | split | plain AUROC | stain AUROC | ΔAUROC | plain AUPRC | stain AUPRC | ΔAUPRC |
  |-------|------------:|------------:|-------:|------------:|------------:|-------:|
  | id_val  | 0.9933 | 0.9938 | +0.0005 | 0.9943 | 0.9940 | −0.0003 |
  | ood_val | 0.7696 | 0.7492 | −0.0204 | 0.6793 | 0.6881 | +0.0088 |

- **id_val-selected fixed-specificity operating points on ood_val (recovers sensitivity):**

  | target spec | model | thr (id_val) | ood_val sens | ood_val spec | ood_val FN |
  |------------:|-------|-------------:|-------------:|-------------:|-----------:|
  | 0.90 | plain ERM   | 0.0674 | 0.8330 | 0.6080 | 167 |
  | 0.90 | stain-aug   | 0.1383 | 0.4570 | 0.7960 | 543 |
  | 0.95 | plain ERM   | 0.1220 | 0.7570 | 0.6720 | 243 |
  | 0.95 | stain-aug   | 0.2347 | 0.2700 | 0.9090 | 730 |

  Stain-aug ood_val sensitivity rises from 0.096 (at 0.5) to 0.457 (spec≥0.90) / 0.677
  (spec≥0.80) once the operating point is matched on `id_val`.
- **Finding:** ood_val **ranking is largely preserved** (|ΔAUROC| 0.0204 < 0.03; AUPRC even
  slightly better for stain-aug), so the augmentation did **not** meaningfully worsen score
  ordering. The large default-0.5 sensitivity gap is driven by the stain-aug model scoring
  center-1 tumors lower in absolute terms — the **0.5 threshold is mis-placed for it**.
  Matching an id_val-selected operating point recovers most of the lost sensitivity →
  **the 7A "collapse" is mostly a threshold artifact, not a ranking failure.**
- **Recommendation:** **TUNE / RETAIN** the stain-aug model with a calibrated operating point
  rather than rejecting it on the default-threshold result. Natural next step: calibration +
  operating-point selection (development-stage), not abandonment.
- **Assertions (all passed):** all four inputs exist; required columns present; row counts
  500/2000/500/2000; probabilities finite, in [0,1], `prob_0+prob_1≈1`; input CSVs unmodified
  (mtime + size unchanged pre/post); 5 tables + 4 figures + metrics JSON + report saved;
  **no `ood_test` usage**.
- **Outcome:** OVERALL PASS. No warnings or errors.
- **Note:** DEVELOPMENT-stage diagnosis on capped balanced subsets — NOT final performance
  and NOT a superiority claim. `ood_val` used for authorized development comparison only after
  `id_val` threshold selection; `ood_test` remains LOCKED and untouched.

---

## Milestone 7C — Stain-aug calibration + candidate operating-point audit ✅ PASSED

- **Goal:** extend 7B by (a) fitting post-hoc temperature scaling on the **stain-aug 7A**
  `ood_val` logits (development-stage) and reporting raw-vs-calibrated confidence reliability
  vs the plain-ERM 5A values, (b) auditing high-confidence false negatives before/after
  calibration, and (c) auditing **candidate** `id_val`-selected operating points for both
  models on `ood_val`. **CSV / logit-only** — no training, no inference, no MC-dropout, no
  ensemble; `ood_test` never read; input prediction CSVs unmodified. Authorized by explicit
  user go-ahead. Not a final result; **no operating point committed.**
- **Strict scope:** `ood_val` used for development-stage calibration / comparison ONLY (never
  training, never final test reporting); `id_val` used ONLY to select candidate thresholds.
  Candidate thresholds are NOT final clinical operating points. Temperature scaling is
  argmax-invariant (hard predictions at 0.5 unchanged). Positive class = tumor (1).
- **Inputs (read-only):** plain ERM
  `results/predictions/exp04a_dev_train_resnet18/id_val_cached_predictions.csv` (n=500),
  `results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_{raw,calibrated}_predictions.csv`
  (n=2000 each); stain-aug
  `results/predictions/exp07a_stainaug_resnet18/{id_val,ood_val}_predictions.csv` (n=500 /
  2000); plain-ERM 5A calibration metrics JSON; the three 7B operating-point tables (context).
- **Built:** `scripts/18_calibrate_stainaug_and_operating_audit.py` (reuses
  `src/calibration/temperature_scaling.py`, `src/evaluation/metrics_calibration.py`,
  `metrics_classification.py`, `metrics_operating_points.py`, `metrics_uncertainty.py`). No new
  `src/` modules.
- **Stain-aug temperature (fit on ood_val logits, same settings as 5A: Adam lr 0.01, 200 iters,
  bounds [0.05, 10.0]):** **T = 3.210174** (NLL 1.106019 → 0.680251; `converged=false` at the
  200-iter cap — acceptable). Plain-ERM 5A T = 3.122416 for reference.
- **Calibration reliability on ood_val (development):**

  | model | variant | ECE | Brier | NLL |
  |-------|---------|----:|------:|----:|
  | plain ERM     | raw        | 0.2330 | 0.2759 | 0.9152 |
  | plain ERM     | calibrated | 0.0918 | 0.2160 | 0.6088 |
  | stain-aug ERM | raw        | 0.3560 | 0.3571 | 1.1060 |
  | stain-aug ERM | calibrated | 0.1908 | 0.2474 | 0.6803 |

  Stain-aug calibration improved ECE 0.3560→0.1908 (Δ−0.1651), Brier −0.1097, NLL −0.4257.
  Stain-aug classification **unchanged** raw→calibrated (argmax-invariant): acc 0.5290, sens
  0.0960, FN 904 — calibration does NOT move the 0.5 hard prediction.
- **High-confidence false-negative (missed-tumor) audit on ood_val** (count of `label==1,
  pred==0, confidence≥t`):

  | model | variant | t=0.90 | t=0.95 | t=0.99 |
  |-------|---------|-------:|-------:|-------:|
  | plain ERM     | raw        | 222 | 133 | 33 |
  | plain ERM     | calibrated |   0 |   0 |  0 |
  | stain-aug ERM | raw        | 399 | 181 |  4 |
  | stain-aug ERM | calibrated |   0 |   0 |  0 |

  Calibration removes the confidently-missed tumors in this subset for **both** models
  (stain-aug 399→0 at conf≥0.90) — it fixes over-confident magnitudes even though it cannot
  move the argmax.
- **Candidate operating-point audit (thresholds selected on id_val, raw prob_1) — ood_val at
  fixed specificity:**

  | target spec | model | thr (id_val) | ood_val sens | ood_val spec | ood_val FN | ood_val FP |
  |------------:|-------|-------------:|-------------:|-------------:|-----------:|-----------:|
  | 0.80 | plain ERM   | 0.0222 | 0.9340 | 0.4990 |  66 | 501 |
  | 0.80 | stain-aug   | 0.0819 | 0.6770 | 0.6740 | 323 | 326 |
  | 0.90 | plain ERM   | 0.0674 | 0.8330 | 0.6080 | 167 | 392 |
  | 0.90 | stain-aug   | 0.1383 | 0.4570 | 0.7960 | 543 | 204 |
  | 0.95 | plain ERM   | 0.1220 | 0.7570 | 0.6720 | 243 | 328 |
  | 0.95 | stain-aug   | 0.2347 | 0.2700 | 0.9090 | 730 |  91 |

  **Plain ERM has higher ood_val sensitivity than stain-aug at every fixed-specificity
  candidate target** (0.80/0.85/0.90/0.95). Candidate thresholds partly recover stain-aug
  ood_val sensitivity above the 0.5 collapse (0.096) but at a higher false-positive cost, and
  do not overtake plain ERM.
- **Findings:** (1) calibration **improved** stain-aug confidence reliability (ECE ↓46%; NLL
  ↓38%) and removed high-confidence missed tumors — but is **argmax-invariant**, so it does
  **not** by itself fix the 0.5 sensitivity collapse; (2) the 7A collapse is mostly threshold
  behavior (per 7B), and candidate id_val thresholds recover some ood_val sensitivity at higher
  FP; (3) **plain ERM remains the stronger deterministic baseline** on this capped subset.
- **Recommendation:** **RETAIN** the stain-aug baseline for the reliability audit with a
  calibrated confidence output and **candidate (not committed)** operating points; do not
  reject it on the default-0.5 result and do not claim it beats plain ERM.
- **Artifacts:**
  `results/predictions/exp07c_stainaug_calibration/ood_val_calibrated_predictions.csv`;
  `results/metrics/exp07c_stainaug_calibration/{temperature.json,calibration_metrics.json}`;
  `results/tables/exp07c_{calibration_comparison,high_confidence_fn_comparison,candidate_operating_point_audit}.csv`;
  `results/figures/exp07c_{calibration_raw_vs_calibrated,high_confidence_fn_comparison,fixed_specificity_sensitivity_comparison}.png`;
  `results/logs/exp07c_stainaug_calibration_operating_audit_report.txt`.
- **Assertions (all passed):** no `ood_test` usage; all inputs exist; stain id_val n==500 /
  ood_val n==2000; probabilities finite in [0,1]; calibrated probabilities finite in [0,1] and
  `calibrated_prob_0+calibrated_prob_1≈1`; argmax unchanged after scaling; input prediction CSVs
  unmodified (mtime + size); all tables/figures/JSON/report saved.
- **Outcome:** OVERALL PASS. No warnings or errors.
- **Note:** DEVELOPMENT-stage calibration + candidate-threshold audit on capped balanced
  subsets — NOT final performance and NOT a superiority claim. `ood_val` used for authorized
  development-stage calibration/comparison only; `ood_test` remains LOCKED and untouched; no
  operating point committed.

---

## Milestone 7D — Tuned stain/color augmentation v2 ERM baseline ✅ PASSED

- **Goal:** train a **stronger but controlled** pathology-relevant stain/color augmentation
  ERM baseline (v2) and compare it against (1) plain-ERM ResNet18 and (2) the 7A moderate
  stain-augmentation baseline, to address reviewer risk **R3** (no *effective* stain-robustness
  / DG baseline yet). Lowest-risk 7D variant: **tuned torchvision augmentation only** — NOT
  HED/Macenko, NOT GroupDRO/CORAL/DANN, NOT MC-dropout/ensemble, NOT calibration.
  Development-stage analysis only, NOT final performance. Authorized by explicit user go-ahead.
- **Strict scope:** `train` (v2-augmented) for training, `id_val` (eval transform) for model
  selection by `id_val_loss` ONLY. `ood_val` NEVER used for training or selection — read only
  AFTER training (script 20) for development-stage comparison and CANDIDATE (not committed)
  operating points. `ood_test` never touched. No calibration; no committed threshold.
- **Built:**
  - `src/data/stain_transforms.py` — added `StainColorAugmentV2` +
    `build_stain_color_aug_v2_transform(config)` (torchvision `Compose`: flips →
    `RandomRotation` 30° → `RandomAffine` translate (0.03,0.03) / scale (0.95,1.05) →
    `RandomApply([ColorJitter b0.15/c0.30/s0.15/h0.03], p=0.85)` → `RandomAutocontrast` p0.25
    → `ToTensor` → ImageNet `Normalize`, output float32 `[3,96,96]`). Stronger **contrast** +
    mild geometric jitter than 7A, with hue/saturation deliberately held LOW; module-level
    (picklable). The existing 7A `StainColorAugment` / `build_stain_train_transform` are
    unchanged.
  - `configs/experiments/exp07d_stainaug_v2_resnet18.yaml` — v2 augmentation block; same
    train/id_val/ood_val caches, AdamW lr 1e-4 / wd 1e-4, 5 epochs, selection `id_val_loss`/min.
  - `scripts/19_train_stainaug_v2_checkpointed.py` — checkpointed training (mirrors script 15);
    saves `best.pt`/`last.pt` + CSV log + report; reloads `best.pt` and re-verifies `id_val`.
    Does not read `ood_val`/`ood_test`.
  - `scripts/20_eval_stainaug_v2_id_ood.py` — eval-transform inference on cached `id_val` +
    `ood_val`; writes prediction CSVs + 7D metrics; runs a **3-model comparison** (plain ERM /
    7A / 7D): default-0.5 metrics, threshold-free AUROC/AUPRC, and id_val-selected
    fixed-specificity (0.80/0.85/0.90/0.95) candidate operating points on `ood_val`; writes 3
    comparison tables + 3 figures + comparison JSON + report. Verifies the plain/7A input CSVs
    are unmodified (mtime + size). Does not read `ood_test`.
- **Data:** all three balanced caches were **cache HITs** (no HF streaming): train
  `{0:1000,1:1000}` (n=2000), id_val `{0:250,1:250}` (n=500), ood_val `{0:1000,1:1000}` (n=2000).
- **Training results (best by `id_val_loss`):**

  | epoch | train_loss | train_acc | id_val_loss | id_val_acc | is_best |
  |-------|-----------:|----------:|------------:|-----------:|:-------:|
  | 1 | 0.311637 | 0.8705 | 0.195591 | 0.9380 | ✓ |
  | 2 | 0.242894 | 0.8945 | 0.136665 | 0.9440 | ✓ |
  | 3 | 0.202419 | 0.9125 | 0.166023 | 0.9380 |   |
  | 4 | 0.198134 | 0.9250 | 0.165097 | 0.9220 |   |
  | 5 | 0.190760 | 0.9200 | 0.123163 | 0.9440 | ✓ |

  Best epoch **5**, `id_val_loss` **0.123163**, `id_val_acc` 0.9440. Reload verification
  reproduced id_val_loss/acc exactly (1e-5) → **PASS**.
- **Evaluation + 3-model comparison (tumor = positive, threshold 0.5):**

  | split | model | accuracy | auroc | auprc | sensitivity | specificity | FN | tn/fp/fn/tp |
  |-------|-------|---------:|------:|------:|------------:|------------:|---:|:-----------:|
  | id_val  | plain ERM (4A)   | 0.9640 | 0.9933 | 0.9943 | 0.9440 | 0.9840 |  14 | 246/4/14/236 |
  | id_val  | stain-aug 7A     | 0.9480 | 0.9938 | 0.9940 | 0.9080 | 0.9880 |  23 | 247/3/23/227 |
  | id_val  | stain-aug v2 7D  | 0.9440 | 0.9925 | 0.9932 | 0.9040 | 0.9840 |  24 | 246/4/24/226 |
  | ood_val | plain ERM (5A)   | 0.6195 | 0.7696 | 0.6793 | 0.4050 | 0.8340 | 595 | 834/166/595/405 |
  | ood_val | stain-aug 7A     | 0.5290 | 0.7492 | 0.6881 | 0.0960 | 0.9620 | 904 | 962/38/904/96 |
  | ood_val | stain-aug v2 7D  | 0.5135 | 0.7397 | 0.6614 | 0.0650 | 0.9620 | 935 | 962/38/935/65 |

- **Candidate fixed-specificity operating points (id_val-selected, applied to ood_val) —
  ood_val sensitivity:**

  | target spec | plain ERM | stain-aug 7A | stain-aug v2 7D |
  |------------:|----------:|-------------:|----------------:|
  | 0.80 | 0.9340 | 0.6770 | 0.6100 |
  | 0.85 | 0.9010 | 0.5870 | 0.5310 |
  | 0.90 | 0.8330 | 0.4570 | 0.3800 |
  | 0.95 | 0.7570 | 0.2700 | 0.1780 |

- **Finding (development, honest negative result):** the tuned v2 augmentation did **not**
  improve OOD reliability on this capped balanced subset. ood_val AUROC **fell** 0.7696 (plain)
  / 0.7492 (7A) → **0.7397** (7D) — 7D − plain = **−0.0299**, 7D − 7A = **−0.0095**; ood_val
  AUPRC 0.6614 (below both). At the default 0.5 threshold ood_val sensitivity dropped further
  (7A 0.096 → 7D 0.065; FN 904 → 935). At **every** matched id_val-selected fixed-specificity
  target 7D sensitivity was **below both** plain ERM and 7A (7D ≥ 7A at 0/4; 7D ≥ plain at 0/4).
  id_val was essentially preserved (accuracy 0.964→0.944, AUROC ~0.9925). So a *stronger*
  torchvision-based stain/color augmentation was **not** an effective stain-robustness lever
  here — it slightly regressed OOD ranking vs both baselines. **Recommendation: MIXED / DO NOT
  PROMOTE** — retain only as a comparator; the next candidate is a stain-specific (HED/Macenko)
  or dedicated DG method (gated). Do **not** claim superiority over plain ERM.
- **Assertions (all passed):** train_n==2000; id_val_n==500; both labels present in train and
  id_val; losses finite; `best.pt`/`last.pt` exist; checkpoint reload matches (1e-5); id_val
  predictions n==500; ood_val predictions n==2000; probabilities finite, in [0,1],
  `prob_0+prob_1≈1`; all three comparison input CSVs exist and were unmodified (mtime + size);
  metrics JSON/CSV + 3 comparison tables + 3 figures + comparison JSON saved; **no `ood_test`
  usage**.
- **Outcome:** OVERALL PASS (both scripts). No warnings or errors.
- **Note:** DEVELOPMENT baseline on capped balanced subsets — NOT final performance and NOT a
  superiority claim. No calibration and no committed operating point in 7D. `ood_val` used
  post-training for authorized development comparison only; `ood_test` remains LOCKED and
  untouched.

---

## Milestone 7E-0 — Stain-space (OD/HED-style) augmentation smoke test ✅ PASSED

- **Goal:** implement and **visually + numerically validate** a pathology-specific
  optical-density (HED-style) stain augmentation transform **before any 7E training**. A
  transform-only smoke test — NO training, NO model inference, NO `ood_val`, NO `ood_test`, no
  existing result CSV/checkpoint modified. Authorized by explicit user go-ahead.
- **Strict scope:** reads ONLY the cached balanced `train` subset (disk cache — no HF
  streaming, no WILDS/CodaLab); writes ONLY transform-preview artifacts under `results/figures`
  + `results/logs`. No locked-mapping change.
- **Built:**
  - `src/data/stain_transforms.py` — added `StainSpaceAugment` +
    `build_stain_space_aug_transform(config)`. Perturbs in **optical-density space**
    (Beer–Lambert): `OD = -log((RGB+eps)/255)` → per-channel `OD' = OD*scale_c + bias_c`
    (`scale_c ~ U[0.90,1.10]`, `bias_c ~ U[-0.03,0.03]`, independent per channel), `OD'`
    lower-clipped at 0 so RGB never exceeds 255 → `RGB' = 255*exp(-OD') - eps` clipped to
    `[0,255]` (uint8, never NaN/Inf) → torchvision spatial (hflip/vflip 0.5, `RandomRotation`
    15°) → `ToTensor` + ImageNet `Normalize` (same tail as `eval_transform`, output float32
    `[3,96,96]`). No hue/saturation jitter. Helper split `perturb_rgb` (uint8 RGB, pre-norm) /
    `normalize_rgb` / `__call__`. Module-level (picklable). The existing 7A `StainColorAugment`
    and 7D `StainColorAugmentV2` builders are **unchanged**.
  - `scripts/21_stain_space_transform_smoke_test.py` — loads the cached `train` subset (cache
    HIT), selects 8 label-0 + 8 label-1 patches, augments each 3× (independent), saves a 16×4
    preview grid (original + 3 augmentations) + a pixel/stat report, and asserts numerical
    safety. Does not read `ood_val`/`ood_test`; no model/inference.
- **Data:** cache HIT on `results/cache/balanced_train_pc1000_seed1337_shuf.pt` (no streaming).
- **Results (transform-only, seed 1337):** selected 8 label-0 + 8 label-1 (16 patches × 3 augs
  = 48 augmentations). Output tensor shape **(3,96,96)**; augmented pre-normalization RGB range
  **[0, 254]** (all within [0,255]); **no NaN/Inf**. Normalized-tensor stats before vs after
  augmentation: before min/max/mean/std −1.9482/2.6400/0.9797/0.9368, after
  −2.1179/2.6226/0.8053/1.1365 (mild, expected shift). **Visual check:** augmentations show
  realistic H&E stain variation (pink/purple balance + mild lightness shifts) with tissue
  morphology preserved — no black / washed-out / neon patches (black corners on some tiles are
  the standard `RandomRotation` fill, as in 7A/7D). **VISUAL SANITY: PASS.**
- **Assertions (all passed):** train cache exists; selected label-0 count == 8; selected
  label-1 count == 8; augmented tensor shape == [3,96,96]; no NaN/Inf; augmented RGB within
  [0,255]; preview figure saved (non-empty); report saved; **no `ood_val` usage**; **no
  `ood_test` usage**.
- **Artifacts:** `results/figures/exp07e0_stain_space_aug_preview.png`,
  `results/logs/exp07e0_stain_space_aug_smoke_report.txt`.
- **Outcome:** OVERALL PASS. No warnings or errors.
- **Note:** transform validation only — NOT a training result and NOT a performance claim. It
  establishes that the OD/HED-style stain-space augmentation is numerically safe and
  pathology-plausible, clearing it for a *gated* 7E training proposal. `ood_val` / `ood_test`
  untouched.

---

## Milestone 7E — Stain-space (OD/HED-style) augmentation ERM baseline ✅ PASSED

- **Goal:** **train** the first *pathology-specific* stain-robustness / domain-generalization
  baseline — a ResNet18 ERM with the validated 7E-0 optical-density (HED-style) `StainSpaceAugment`
  applied to `train` only — and compare it against (1) plain-ERM ResNet18, (2) the 7A moderate
  torchvision stain-augmentation baseline, and (3) the 7D tuned torchvision stain-augmentation v2
  baseline, to address reviewer risk **R3** (no *effective* stain-robustness / DG baseline yet).
  7A/7D established that torchvision colour-jitter augmentation (moderate and tuned) does not lift
  OOD ranking over plain ERM; 7E tests whether a physically-motivated OD-space transform does.
  Development-stage analysis only, NOT final performance. Authorized by explicit user go-ahead.
- **Strict scope:** `train` (stain-space-augmented) for training, `id_val` (eval transform) for
  model selection by `id_val_loss` ONLY. `ood_val` NEVER used for training or selection — read only
  AFTER training (script 23) for development-stage comparison and CANDIDATE (not committed)
  operating points. `ood_test` never touched. No GroupDRO/CORAL/DANN, no MC-dropout/ensemble, no
  calibration, no committed threshold. Reused the validated `StainSpaceAugment` (no new `src/`
  module; `src/data/stain_transforms.py` unchanged).
- **Built:**
  - `configs/experiments/exp07e_stainspace_resnet18.yaml` — stain-space augmentation block
    (OD scale [0.90,1.10] / bias [-0.03,0.03], flips 0.5, rotation 15°); same train/id_val/ood_val
    caches, AdamW lr 1e-4 / wd 1e-4, 5 epochs, selection `id_val_loss`/min.
  - `scripts/22_train_stainspace_checkpointed.py` — checkpointed training (mirrors script 19 but
    uses `build_stain_space_aug_transform`); saves `best.pt`/`last.pt` + CSV log + train report;
    reloads `best.pt` and re-verifies `id_val`. Does not read `ood_val`/`ood_test`.
  - `scripts/23_eval_stainspace_id_ood.py` — eval-transform inference on cached `id_val` +
    `ood_val`; writes prediction CSVs + 7E metrics; runs a **4-model comparison** (plain ERM / 7A /
    7D / 7E): default-0.5 metrics, threshold-free AUROC/AUPRC, and id_val-selected fixed-specificity
    (0.80/0.85/0.90/0.95) candidate operating points on `ood_val`; writes 3 comparison tables + 3
    figures + comparison JSON + report. Verifies the plain/7A/7D input CSVs are unmodified
    (mtime + size). Does not read `ood_test`.
- **Data:** all three balanced caches were **cache HITs** (no HF streaming): train
  `{0:1000,1:1000}` (n=2000), id_val `{0:250,1:250}` (n=500), ood_val `{0:1000,1:1000}` (n=2000).
- **Training results (best by `id_val_loss`):**

  | epoch | train_loss | train_acc | id_val_loss | id_val_acc | is_best |
  |-------|-----------:|----------:|------------:|-----------:|:-------:|
  | 1 | 0.295800 | 0.8735 | 0.188955 | 0.9380 | ✓ |
  | 2 | 0.206271 | 0.9175 | 0.134861 | 0.9480 | ✓ |
  | 3 | 0.187358 | 0.9270 | 0.102048 | 0.9580 | ✓ |
  | 4 | 0.167339 | 0.9350 | 0.137747 | 0.9380 |   |
  | 5 | 0.152006 | 0.9405 | 0.161615 | 0.9520 |   |

  Best epoch **3**, `id_val_loss` **0.102048**, `id_val_acc` 0.9580. Reload verification
  reproduced id_val_loss/acc exactly (1e-5) → **PASS**. (Best is epoch 3 — unlike 7A/7D best epoch
  5 — later epochs over-fit id_val loss.)
- **Evaluation + 4-model comparison (tumor = positive, threshold 0.5):**

  | split | model | accuracy | auroc | auprc | sensitivity | specificity | FN | tn/fp/fn/tp |
  |-------|-------|---------:|------:|------:|------------:|------------:|---:|:-----------:|
  | id_val  | plain ERM (4A)   | 0.9640 | 0.9933 | 0.9943 | 0.9440 | 0.9840 |  14 | 246/4/14/236 |
  | id_val  | stain-aug 7A     | 0.9480 | 0.9938 | 0.9940 | 0.9080 | 0.9880 |  23 | 247/3/23/227 |
  | id_val  | stain-aug v2 7D  | 0.9440 | 0.9925 | 0.9932 | 0.9040 | 0.9840 |  24 | 246/4/24/226 |
  | id_val  | stain-space 7E   | 0.9580 | 0.9933 | 0.9940 | 0.9400 | 0.9760 |  15 | 244/6/15/235 |
  | ood_val | plain ERM (5A)   | 0.6195 | 0.7696 | 0.6793 | 0.4050 | 0.8340 | 595 | 834/166/595/405 |
  | ood_val | stain-aug 7A     | 0.5290 | 0.7492 | 0.6881 | 0.0960 | 0.9620 | 904 | 962/38/904/96 |
  | ood_val | stain-aug v2 7D  | 0.5135 | 0.7397 | 0.6614 | 0.0650 | 0.9620 | 935 | 962/38/935/65 |
  | ood_val | stain-space 7E   | 0.5055 | 0.7317 | 0.6244 | 0.0840 | 0.9270 | 916 | 927/73/916/84 |

- **Candidate fixed-specificity operating points (id_val-selected, applied to ood_val) —
  ood_val sensitivity:**

  | target spec | plain ERM | stain-aug 7A | stain-aug v2 7D | stain-space 7E |
  |------------:|----------:|-------------:|----------------:|---------------:|
  | 0.80 | 0.9340 | 0.6770 | 0.6100 | 0.8970 |
  | 0.85 | 0.9010 | 0.5870 | 0.5310 | 0.7880 |
  | 0.90 | 0.8330 | 0.4570 | 0.3800 | 0.6530 |
  | 0.95 | 0.7570 | 0.2700 | 0.1780 | 0.3030 |

- **Finding (development, honest mixed/negative result):** the OD-space stain augmentation did
  **not** improve OOD *ranking* — ood_val AUROC **0.7317** is the lowest of the four (7E − plain
  **−0.0379**, 7E − 7A **−0.0175**, 7E − 7D **−0.0080**) and AUPRC 0.6244 is also lowest;
  default-0.5 ood_val sensitivity 0.0840 (FN 916). **However, at every matched id_val-selected
  fixed-specificity target 7E is the strongest stain-augmentation variant** — it beats **both** 7A
  and 7D at **4/4** targets (large operating-point recovery, e.g. spec≥0.90 ood_val sensitivity
  0.653 vs 7A 0.457 / 7D 0.380), while remaining **below plain ERM at 0/4**. id_val was preserved
  (accuracy 0.9580, AUROC 0.9933, sensitivity 0.9400, FN 15). So the OD-space transform improves
  the *operating-point behavior* of stain augmentation over both torchvision recipes but does not
  fix the underlying OOD ranking, which stays below plain ERM. **Recommendation: TUNE FURTHER / DO
  NOT PROMOTE** — retain 7E as the strongest pathology-specific stain-robust comparator; the next
  candidate is a dedicated DG objective (GroupDRO/CORAL/DANN) or OD-range tuning (gated). Do **not**
  claim superiority over plain ERM.
- **Assertions (all passed):** train_n==2000; id_val_n==500; both labels present in train and
  id_val; losses finite; `best.pt`/`last.pt` exist; checkpoint reload matches (1e-5); id_val
  predictions n==500; ood_val predictions n==2000; probabilities finite, in [0,1],
  `prob_0+prob_1≈1`; all six comparison input CSVs (plain/7A/7D) exist and were unmodified
  (mtime + size); metrics JSON/CSV + 3 comparison tables + 3 figures + comparison JSON saved; **no
  `ood_test` usage**.
- **Outcome:** OVERALL PASS (both scripts). No warnings or errors.
- **Note:** DEVELOPMENT baseline on capped balanced subsets — NOT final performance and NOT a
  superiority claim. No calibration and no committed operating point in 7E. `ood_val` used
  post-training for authorized development comparison only; `ood_test` remains LOCKED and
  untouched.

---

## Milestone 7F-0 — DG feasibility / group-label metadata audit ✅ PASSED

- **Goal:** before any dedicated DG training, verify whether the existing balanced caches preserve
  enough group metadata (center / patient / slide) for GroupDRO / CORAL / DANN, and whether `train`
  contains more than one *sound* source domain. **Metadata/cache audit only — NO training, NO model
  inference, NO `ood_test`.** Authorized by explicit user go-ahead.
- **Strict scope:** reads ONLY the three balanced caches (disk, read-only — no HF streaming, no
  WILDS/CodaLab); writes ONLY audit tables + a report under `results/tables` + `results/logs`. No
  cache / prediction / checkpoint / metrics file modified; no cache rebuilt. `ood_val` inspected
  for metadata availability only.
- **Built:** `scripts/24_dg_group_metadata_audit.py` (reuses `utils.io` only — no new `src/`
  module). Loads each cache payload, inspects object type / keys / n / label counts / image
  presence / metadata fields, quantifies label × center and label × slide counts, and derives DG
  group-label feasibility (incl. a soundness check for center↔label confounding and source-center
  coverage).
- **Inputs (read-only, unmodified):**
  `results/cache/balanced_train_pc1000_seed1337_shuf.pt` (n=2000),
  `results/cache/balanced_id_val_pc250_seed1337_shuf.pt` (n=500),
  `results/cache/balanced_ood_val_pc1000_seed1337_noshuf.pt` (n=2000).
- **Findings:**
  - **Metadata fully preserved.** Every payload is a `dict` with keys `version, n, images, labels,
    meta`; `meta` carries all seven integer fields — `center, patient, slide, node, image_id,
    x_coord, y_coord`. DG is **not** blocked by missing metadata.
  - **Distinct groups:** train center `{0,3}`; id_val center `{0,3}`; ood_val center `{1}` (target
    domain identifiable). train patient 8 / slide 11; ood_val patient 8 / slide 10.
  - **Critical confound (honest finding):** the capped `train` cache is **center↔label
    confounded** — center 0 tumor fraction 0.091 (label0=1000, label1=100), **center 3 tumor
    fraction 1.0** (label0=0, label1=900) — and locked source **center 4 is entirely absent**
    (`missing=[4]`), an artifact of filling per-class quotas over the label-ordered stream. So
    center-based GroupDRO / CORAL / DANN are **RUNNABLE (≥2 center groups + metadata) but NOT
    SOUND** (a center-grouped objective would learn *center == label*).

  | split | n | label0/label1 | distinct center | center tumor-fraction | DG-by-center |
  |-------|--:|---------------|-----------------|-----------------------|--------------|
  | train   | 2000 | 1000/1000 | {0,3} | c0=0.091, c3=1.000 | runnable, NOT sound |
  | id_val  |  500 |  250/250  | {0,3} | (source-domain val ok) | n/a (validation) |
  | ood_val | 2000 | 1000/1000 | {1}   | target domain | identifiable |

- **Recommendation A — center-STRATIFIED metadata-rich cache rebuild required** before any sound
  DG-by-center training. Minimal design (PROPOSAL ONLY, NOT run here): keep the locked split
  mapping + per_class balance; change only the **sampling policy** to fill per-`(center,label)`
  quotas across source centers `{0,3,4}`; keep serializing all `META_INT_FIELDS` (schema already
  sufficient); write to **new** cache filenames (no overwrite → preserve 4A/5A/7A–7E
  reproducibility). The rebuild is a separate gated milestone (requires HF streaming).
- **Artifacts:** `results/tables/exp07f0_cache_metadata_summary.csv`,
  `exp07f0_group_label_counts.csv`, `exp07f0_dg_method_feasibility.csv`;
  `results/logs/exp07f0_dg_group_metadata_audit_report.txt`, `exp07f0_run_console.txt`.
- **Assertions (all passed):** train/id_val/ood_val caches exist; `train_n==2000`; `id_val_n==500`;
  `ood_val_n==2000`; labels present and finite/binary; all three tables saved; report saved; **no
  `ood_test` usage**.
- **Outcome:** OVERALL PASS. No warnings or errors.
- **Note:** development-stage feasibility audit on capped balanced subsets — NOT a final result. No
  training, no inference, no cache rebuild; `ood_val` inspected for metadata only; `ood_test`
  remains LOCKED and untouched.

---

## Milestone 7F-1 — Metadata-rich center-stratified cache rebuild ✅ PASSED

- **Goal:** per the 7F-0 verdict (Recommendation A), rebuild balanced **development**
  caches for `train` and `id_val` so each locked source center {0, 3, 4} carries **both**
  labels — removing the center↔label confound and giving GroupDRO / CORAL / DANN sound
  domain groups. **NO training, NO model inference, NO `ood_test`.** Authorized by explicit
  user go-ahead.
- **Strict scope:** source = the locked HF mirror `wltjr1007/Camelyon17-WILDS` via the
  sanctioned streaming loader `iter_logical_split` (NO WILDS/CodaLab; NO HF test / center 2).
  Only the **sampling policy** changes vs `src/data/balanced_subset.py` — fill a quota per
  `(center, label)` cell instead of first-per-class in stream order. The serialized payload
  schema is byte-for-byte the existing one (reused `_serialize_examples` /
  `_deserialize_examples` / `META_INT_FIELDS` from `balanced_subset`; **`src/` unchanged**),
  so the new caches load through the same `BalancedSubsetDataset`. `ood_val` was **not** read
  for cache construction. New cache filenames only — **no overwrite** of existing caches.
- **Built:** `scripts/25_rebuild_center_stratified_caches.py` (imports the existing
  serializer helpers; no new `src/` module). Adaptive per-cell quota: preferred **300**
  (train) / **75** (id_val); if any cell falls short, use the minimum available across the
  6 required cells; if any required `(center,label)` cell is empty, do **not** write a broken
  cache for that split (mark DG-by-center infeasible).
- **Source scan (streaming, label-ordered; early-stopped once all 6 cells filled):**
  - `train` (HF `train`, centers {0,3,4}): scanned **221,387** records; available per cell
    c0_l0=26,686 / c0_l1=20,584 / c3_l0=58,436 / c3_l1=49,253 / c4_l0=300 / c4_l1=66,128 —
    every required cell reached the preferred **300** quota (center 4 non-tumor was the
    binding cell, exactly 300 seen before early-stop).
  - `id_val` (HF `validation`, centers {0,3,4}; **center 1 excluded** by the loader):
    scanned **24,466** records; available per cell c0_l0=3,032 / c0_l1=2,280 / c3_l0=6,483 /
    c3_l1=5,363 / c4_l0=75 / c4_l1=7,233 — every required cell reached the preferred **75**.
- **New caches (both center×label balanced by construction):**

  | split  | file | n | label 0/1 | per center | per (center,label) cell | reload |
  |--------|------|--:|-----------|------------|-------------------------|:------:|
  | train  | `center_strat_train_c034_percell300_seed1337_shuf.pt` | 1800 | 900/900 | 0:600, 3:600, 4:600 | all six = 300 | ✓ |
  | id_val | `center_strat_id_val_c034_percell75_seed1337_shuf.pt`  |  450 | 225/225 | 0:150, 3:150, 4:150 | all six = 75  | ✓ |

  Metadata preserved in both (all seven `META_INT_FIELDS`): `center, image_id, node,
  patient, slide, x_coord, y_coord`.
- **DG feasibility after rebuild:** `train` DG-by-center **SOUND** (centers {0,3,4} present,
  every required cell carries both labels) → new caches **support GroupDRO by center** and
  **support CORAL/DANN by center**. The **old** confounded caches
  (`balanced_train_pc1000_seed1337_shuf.pt`, center 0 ~all non-tumor / center 3 all tumor /
  center 4 absent) should be **avoided for DG training** (kept for 4A/5A/7A–7E reproducibility).
- **Artifacts:** `results/cache/center_strat_train_c034_percell300_seed1337_shuf.pt`,
  `results/cache/center_strat_id_val_c034_percell75_seed1337_shuf.pt`;
  `results/tables/exp07f1_center_stratified_cache_availability.csv`,
  `exp07f1_center_stratified_cache_summary.csv`,
  `exp07f1_center_stratified_label_center_counts.csv`;
  `results/logs/exp07f1_center_stratified_cache_rebuild_report.txt`,
  `exp07f1_run_console.txt`.
- **Assertions (all 22 passed):** no `ood_test` usage; **existing caches present and
  unmodified** (size + mtime fingerprint match, verified pre/post); both splits feasible;
  new train + id_val caches exist; each cache has `images`/`labels`/`meta`; all seven
  metadata fields preserved; labels binary/finite; each required center present; each
  required `(center,label)` cell ≥ 1; label×center **balanced by construction** (all cells
  equal); all three tables + report saved.
- **Outcome:** OVERALL PASS. **WARNINGS: none.** Benign transient HF read-timeouts during
  the train stream were auto-retried by `datasets` (expected per CLAUDE.md §6) and did not
  affect the result.
- **Note:** development-stage cache rebuild on capped balanced subsets — NOT a final result.
  No training, no inference; `ood_test` never referenced; `ood_val` not read for cache
  construction. **GroupDRO/CORAL/DANN not started** — the gated Milestone 7F training step
  is the next candidate and still requires an explicit go-ahead.

---

## Milestone 7F — Matched center-stratified ERM + GroupDRO baseline ✅ PASSED

- **Goal:** the first **dedicated domain-generalization** baseline (reviewer risk R3). Train TWO
  ResNet18 models on the SAME new center-stratified `train` cache (Milestone 7F-1) — a **matched
  center-stratified ERM control** and a **GroupDRO-by-center** model — and compare them fairly on
  `id_val` + `ood_val`. The matched ERM control is REQUIRED because the prior plain-ERM 4A checkpoint
  was trained on the OLD confounded cache; a fair GroupDRO comparison needs an ERM trained on the
  SAME center-stratified data. Development-stage analysis only, NOT final performance. Authorized by
  explicit user go-ahead.
- **Strict scope:** `train` + `id_val` (center-stratified caches) ONLY for training and model
  selection (best by `id_val_loss`). `ood_val` NEVER used for training/selection — read only AFTER
  training (script 27) for development comparison and CANDIDATE (not committed) operating points.
  `ood_test` never touched. Normal train/eval transforms (NO stain augmentation). NO CORAL/DANN, NO
  MC-dropout/ensemble, NO calibration, NO committed threshold.
- **Built:**
  - `src/training/groupdro.py` — GroupDRO utilities: `compute_group_losses(logits, labels, groups,
    group_ids)` (per-group mean CE, present-mask, per-group counts), `GroupDROLoss` (persistent `q`
    buffer over centers {0,3,4}, uniform init; per batch computes per-group mean CE, updates `q_g ←
    q_g·exp(step_size·detached_L_g)` on **present** groups only then renormalizes, returns robust
    loss `sum_g q_g·L_g` differentiable in the model), `groups_from_batch`, `summarize_group_counts`.
    CE base loss. Absent-group weights are carried forward unchanged (not updated) that batch.
  - `configs/experiments/exp07f_groupdro_resnet18.yaml` — both run names, the center-stratified
    caches, `group_field: center`, `group_ids: [0,3,4]`, AdamW lr 1e-4 / wd 1e-4, 5 epochs, GroupDRO
    `step_size 0.01` / `normalize_group_weights true`, selection `id_val_loss`/min.
  - `scripts/26_train_centerstrat_erm_groupdro.py` — trains BOTH models (matched seeds/init), logs
    per-epoch train/id_val loss+acc, worst-center id_val loss/acc, and (GroupDRO) `q` weights +
    per-group losses; saves `best.pt`/`last.pt` + per-model CSV logs + a shared train report; reloads
    each `best.pt` and re-verifies `id_val`. Does not read `ood_val`/`ood_test`.
  - `scripts/27_eval_centerstrat_erm_groupdro_id_ood.py` — eval-transform inference on the
    center-stratified `id_val` + balanced `ood_val`; per-example prediction CSVs; overall +
    per-center id_val classification metrics; id_val-selected fixed-specificity (0.80/0.85/0.90/0.95)
    candidate operating points on `ood_val`; FAIR comparison (7F ERM vs 7F GroupDRO) plus an
    old-cache REFERENCE comparison (plain/7A/7D/7E, clearly labeled not-matched); 4 tables + 4
    figures + comparison JSON + report. Verifies the reference input CSVs unmodified (mtime + size).
- **Data:** all caches were disk loads (no HF streaming): train `center_strat_...percell300` (n=1800,
  centers 0/3/4 × labels 0/1 = 300 each), id_val `center_strat_...percell75` (n=450, 75/cell), ood_val
  `balanced_ood_val_pc1000` (n=2000).
- **Training (best by `id_val_loss`; both reload-verified 1e-5):**

  | model | best epoch | id_val_loss | id_val_acc |
  |-------|:----------:|------------:|-----------:|
  | center-strat ERM (7F) | 2 | 0.248137 | 0.9089 |
  | GroupDRO (7F)         | 3 | 0.259698 | 0.8978 |

  GroupDRO `q` drifted toward the worst center (c4: 0.333 → 0.381 over 5 epochs), as expected.
- **Evaluation + fair comparison (tumor = positive, threshold 0.5):**

  | split | model | accuracy | auroc | auprc | sensitivity | specificity | FN |
  |-------|-------|---------:|------:|------:|------------:|------------:|---:|
  | id_val  | center-strat ERM (7F) | 0.9089 | 0.9662 | 0.9717 | 0.8933 | 0.9244 |  24 |
  | id_val  | GroupDRO (7F)         | 0.8978 | 0.9666 | 0.9703 | 0.8578 | 0.9378 |  32 |
  | ood_val | center-strat ERM (7F) | 0.4765 | 0.6180 | 0.5214 | 0.0100 | 0.9430 | 990 |
  | ood_val | GroupDRO (7F)         | 0.5065 | 0.7129 | 0.6287 | 0.0700 | 0.9430 | 930 |

  Per-center id_val: ERM c0/c3/c4 acc 0.907/0.933/0.887 (worst c3 loss 0.268); GroupDRO
  0.913/0.920/0.860 (worst c4 loss 0.337). Candidate fixed-specificity ood_val sensitivity
  (spec 0.80/0.85/0.90/0.95): ERM 0.021/0.015/0.013/0.008, **GroupDRO 0.306/0.241/0.148/0.057**.
- **Finding (development, honest positive-for-DG result):** on the matched center-stratified data
  **GroupDRO beat its matched ERM control on OOD** — ood_val AUROC **0.7129 vs 0.6180 (+0.0949)**,
  AUPRC 0.6287 vs 0.5214 (+0.1073), default-0.5 sensitivity 0.0700 vs 0.0100, and matched
  fixed-specificity ood_val sensitivity **4/4** targets. This is the **first DG lever to beat its
  matched control on OOD ranking** (stain augmentation 7A/7D/7E all stayed below plain ERM). GroupDRO
  did **not** improve worst-center id_val behavior (it up-weighted center 4, whose id_val loss stayed
  highest). **Reference caveat:** old plain ERM (ood_val AUROC 0.7696) / 7A / 7D / 7E were trained on
  the OLD confounded cache and a **different** id_val subset — old-vs-new numbers are **context only,
  NOT a controlled comparison**; the controlled comparison is 7F ERM vs 7F GroupDRO.
- **Recommendation:** **RETAIN GroupDRO** — next candidate is calibration + candidate operating-point
  audit of the 7F models (development-stage, gated). Do not claim final performance.
- **Assertions (all passed):** train_n==1800; id_val_n==450; train/id_val centers exactly {0,3,4};
  every center×label cell present in both splits; labels binary; losses finite; ERM + GroupDRO
  `best.pt`/`last.pt` exist; both reload-verify (1e-5); each 7F model id_val predictions n==450 and
  ood_val n==2000; probabilities finite, in [0,1], `prob_0+prob_1≈1`; center metadata present on
  id_val; reference input CSVs unmodified (mtime + size); all metrics JSONs + 4 tables + 4 figures
  saved; **no `ood_test` usage**.
- **Outcome:** OVERALL PASS (both scripts). No warnings or errors.
- **Note:** DEVELOPMENT baseline on capped balanced subsets — NOT final performance and NOT a
  clinical/superiority claim. No calibration and no committed operating point in 7F. `ood_val` used
  post-training for authorized development comparison only; `ood_test` remains LOCKED and untouched.

---

## Milestone 7G — 7F ERM vs GroupDRO calibration + candidate operating-point audit ✅ PASSED

- **Goal:** the development follow-up to 7F (reviewer risk R7 — calibration fitted/evaluated on the
  same capped `ood_val`). Take the *already-saved* 7F predictions for the two matched
  center-stratified models (center-stratified ERM control + GroupDRO-by-center) and, per model,
  (a) fit temperature scaling on its `ood_val` logits, (b) report raw-vs-calibrated reliability
  (ECE / Brier / NLL), (c) audit high-confidence false negatives before/after calibration, and
  (d) audit candidate `id_val`-selected operating points (fixed specificity + fixed sensitivity) on
  `ood_val` — then decide whether GroupDRO remains better than its matched ERM after the audit.
  **CSV / logit-only** — no training, no inference, no MC-dropout, no ensemble, no CORAL/DANN;
  `ood_test` never read; input prediction CSVs unmodified; checkpoints untouched. Authorized by
  explicit user go-ahead. Not a final result; **no operating point committed.**
- **Strict scope:** `ood_val` used for development-stage calibration / comparison ONLY (never
  training, never final test reporting); `id_val` used ONLY to select candidate thresholds.
  Candidate thresholds are NOT final clinical operating points. Temperature scaling is
  argmax-invariant (hard predictions at 0.5 unchanged). Positive class = tumor (1). The FAIR
  comparison is 7F ERM vs 7F GroupDRO (same 7F-1 cache); old plain ERM / 7A / 7D / 7E are
  context-only, NOT controlled, and were NOT re-derived here.
- **Inputs (read-only):** center-strat ERM
  `results/predictions/exp07f_centerstrat_erm_resnet18/{id_val,ood_val}_predictions.csv` (n=450 /
  2000); GroupDRO `results/predictions/exp07f_groupdro_resnet18/{id_val,ood_val}_predictions.csv`
  (n=450 / 2000); the four 7F comparison tables (context).
- **Built:** `scripts/28_calibrate_7f_groupdro_operating_audit.py` (reuses
  `src/calibration/temperature_scaling.py`, `src/evaluation/metrics_calibration.py`,
  `metrics_classification.py`, `metrics_operating_points.py`, `metrics_uncertainty.py`). No new
  `src/` modules.
- **Temperatures (fit on ood_val logits; Adam lr 0.01, 200 iters, bounds [0.05,10.0]):**
  center-strat ERM **T = 3.923949** (NLL 3.485933 → 1.067191); GroupDRO **T = 3.657000** (NLL
  1.913863 → 0.790820). Both `converged=false` at the 200-iter cap — acceptable.
- **Calibration reliability on ood_val (development):**

  | model | variant | ECE | Brier | NLL |
  |-------|---------|----:|------:|----:|
  | center-strat ERM (7F) | raw        | 0.4984 | 0.5059 | 3.4859 |
  | center-strat ERM (7F) | calibrated | 0.3698 | 0.3759 | 1.0672 |
  | GroupDRO (7F)         | raw        | 0.4258 | 0.4278 | 1.9139 |
  | GroupDRO (7F)         | calibrated | 0.2529 | 0.2896 | 0.7908 |

  Calibration **improved all three** metrics for **both** models (ERM ECE −0.1286; GroupDRO ECE
  −0.1729). Classification unchanged raw→calibrated (argmax-invariant): ERM acc 0.4765 / sens
  0.0100 / FN 990; GroupDRO acc 0.5065 / sens 0.0700 / FN 930. GroupDRO is **better-calibrated than
  its matched ERM both raw and calibrated** (lower ECE/Brier/NLL at each variant).
- **High-confidence false-negative (missed-tumor) audit on ood_val** (count of `label==1,
  pred==0, confidence≥t`):

  | model | variant | t=0.90 | t=0.95 | t=0.99 |
  |-------|---------|-------:|-------:|-------:|
  | center-strat ERM (7F) | raw        | 950 | 925 | 784 |
  | center-strat ERM (7F) | calibrated | 249 |  40 |   0 |
  | GroupDRO (7F)         | raw        | 688 | 571 | 333 |
  | GroupDRO (7F)         | calibrated |  25 |   1 |   0 |

  Calibration sharply reduces confidently-missed tumors for both (both reach 0 at conf≥0.99);
  GroupDRO has fewer high-confidence FNs than ERM at every threshold and variant.
- **Candidate operating-point audit (thresholds selected on id_val ONLY, raw prob_1) — ood_val
  sensitivity at fixed specificity:**

  | target spec | center-strat ERM (7F) | GroupDRO (7F) |
  |------------:|----------------------:|--------------:|
  | 0.80 | 0.0210 | **0.3060** |
  | 0.85 | 0.0150 | **0.2410** |
  | 0.90 | 0.0130 | **0.1480** |
  | 0.95 | 0.0080 | **0.0570** |

  Fixed-sensitivity candidates (0.80/0.90/0.95) — at each matched target sensitivity GroupDRO keeps
  **higher ood_val specificity** than ERM (e.g. tgt sens 0.90: GroupDRO ood_val sens 0.169 / spec
  0.900 vs ERM 0.010 / 0.940 — GroupDRO recovers far more ood_val tumor sensitivity at a comparable
  operating region). GroupDRO wins **4/4** fixed-specificity targets and dominates the
  fixed-sensitivity operating points.
- **Finding (development):** (1) calibration **improved confidence reliability for both** 7F models
  (ECE/Brier/NLL all down) and removed high-confidence missed tumors — but is **argmax-invariant**,
  so it does not change default sensitivity/specificity; (2) **GroupDRO remains better than its
  matched center-stratified ERM control after the candidate operating-point audit** — higher ood_val
  AUROC (0.7129 vs 0.6180) and GroupDRO ood_val sensitivity ≥ ERM at all 4 fixed-specificity targets
  (strictly greater at 4/4); (3) GroupDRO is also the better-calibrated model both raw and calibrated.
- **Recommendation:** **RETAIN GroupDRO** with its calibrated confidence output; report **candidate
  (not committed)** operating points only. Do not commit a clinical/paper operating threshold; do not
  claim final performance.
- **Artifacts:**
  `results/predictions/exp07g_7f_calibration/{centerstrat_erm,groupdro}_ood_val_calibrated_predictions.csv`;
  `results/metrics/exp07g_7f_calibration/{temperature.json,calibration_metrics.json}`;
  `results/tables/exp07g_{calibration_comparison,high_confidence_fn_comparison,candidate_operating_point_audit}.csv`;
  `results/figures/exp07g_{calibration_raw_vs_calibrated,high_confidence_fn_comparison,fixed_specificity_sensitivity_comparison}.png`;
  `results/logs/exp07g_7f_calibration_operating_audit_report.txt`.
- **Assertions (all passed):** no `ood_test` usage; all inputs exist; ERM/GroupDRO id_val n==450 and
  ood_val n==2000; id_val centers ⊆ {0,3,4}; probabilities finite in [0,1] and `prob_0+prob_1≈1`;
  calibrated probabilities finite in [0,1] and `calibrated_prob_0+calibrated_prob_1≈1`; argmax
  unchanged after scaling for both models; input prediction CSVs unmodified (mtime + size); both
  calibrated CSVs + 3 tables + 3 figures + temperature JSON + metrics JSON + report saved.
- **Outcome:** OVERALL PASS. No warnings or errors.
- **Note:** DEVELOPMENT-stage calibration + candidate-threshold audit on capped balanced subsets —
  NOT final performance and NOT a superiority/clinical claim. `ood_val` used for authorized
  development-stage calibration/comparison only; `ood_test` remains LOCKED and untouched; no
  operating point committed.

---

## Milestone 7H — Consolidated DG reporting package (reporting-only) ✅ PASSED

- **Goal:** consolidate the accepted development results through Milestone 7G into clean,
  paper-facing domain-generalization tables, figures, and documentation. **Reporting only** —
  no training, no model inference, no calibration fitting, no MC-dropout, no ensemble, no
  CORAL/DANN, no `ood_test`, no committed operating threshold, no final-performance claim.
  Authorized by explicit user go-ahead.
- **Strict scope:** reads existing saved CSV/JSON result artifacts read-only (fingerprint
  verified unmodified) and writes new consolidated artifacts. Every number is copied/derived
  from saved files; no images/logits/forward passes. `ood_val` referenced for development-stage
  reporting only; `ood_test` never read.
- **Built:** `scripts/29_make_dg_reporting_package.py` (pandas + matplotlib Agg; reuses no
  model code). Generates the two DG docs from computed values and verifies the three
  hand-updated docs carry a 7H marker.
- **Inputs (read-only, unmodified):** the 6A report tables + summary; the 7B/7D/7E stain
  comparison tables; the 7F-0 feasibility + 7F-1 cache tables; the 7F comparison tables
  (`exp07f_ood_val_metrics_comparison.csv` etc.); the 7G calibration / high-confidence-FN /
  candidate-operating-point tables and `exp07g_7f_calibration/{calibration_metrics,temperature}.json`.
- **Generated tables (6, `results/tables/`):** `exp07h_consolidated_model_comparison.csv` (6
  models: 4 context-only + 2 controlled), `exp07h_controlled_dg_comparison.csv` (14 metrics, 7F
  ERM vs 7F GroupDRO with deltas), `exp07h_context_vs_controlled_warning_table.csv` (3
  comparisons, validity + allowed/forbidden interpretation), `exp07h_groupdro_calibration_summary.csv`
  (2 models, raw→cal ECE/Brier/NLL + deltas), `exp07h_groupdro_operating_point_summary.csv` (8
  rows, fixed-spec candidates), `exp07h_reviewer_risk_update.csv` (R1/R3/R4/R7/R8/R9).
- **Generated figures (5, `results/figures/`):** `exp07h_ood_auroc_auprc_model_comparison.png`
  (controlled bars hatched), `exp07h_controlled_groupdro_gain.png`,
  `exp07h_calibration_ece_brier_nll.png`, `exp07h_high_confidence_fn_reduction.png`,
  `exp07h_fixed_specificity_sensitivity.png`.
- **Generated docs (2, `docs/`):** `DG_RESULTS_INTERPRETATION.md`, `DG_MANUSCRIPT_UPDATE.md`.
- **Updated docs (3):** `CLAIMS_AND_LIMITATIONS.md` (allowed A5–A7 / forbidden F6–F10 DG claims),
  `REVIEWER_RISK_AUDIT.md` (R3 reduced HIGH→partly-mitigated; R9 partly mitigated; R1/R4/R7 open),
  `FIGURE_TABLE_MANIFEST.md` (all 7H tables/figures + placement).
- **Report:** `results/logs/exp07h_dg_reporting_package_summary.txt`.
- **Key results consolidated (development):** controlled 7F ERM vs 7F GroupDRO — ood_val AUROC
  0.6180 → **0.7129** (+0.0949), AUPRC 0.5214 → **0.6287** (+0.1073), default sensitivity
  0.0100 → 0.0700, default FN 990 → 930; GroupDRO wins **4/4** fixed-specificity candidate targets.
  Calibration (7G): both improved (argmax-invariant); high-confidence (conf≥0.90) missed tumors
  ERM 950→249, GroupDRO 688→25; GroupDRO better-calibrated. **Context-only:** old plain ERM ood_val
  AUROC 0.7696 / 7A 0.7492 / 7D 0.7397 / 7E 0.7317 — **reference gap, not a controlled loss**;
  GroupDRO does NOT beat old plain ERM in a controlled comparison.
- **Assertions (all passed):** no `ood_test` in any path; input artifacts unmodified
  (size+mtime fingerprint); all 6 tables + 5 figures + 2 new docs saved (non-empty); 3 existing
  docs carry a 7H marker; controlled GroupDRO > ERM on AUROC/AUPRC and ≥ ERM on 4/4
  fixed-specificity sensitivity; calibration reduced ECE for both; table row-count sanity.
- **Outcome:** OVERALL PASS. No warnings or errors. (Console shows benign `�` glyphs where
  Windows cp1252 renders em-dash/arrow in stdout; files are UTF-8 and correct.)
- **Note:** reporting-only consolidation of DEVELOPMENT results on capped balanced subsets — NOT
  final performance and NOT a superiority claim over context-only models. No operating point
  committed; `ood_test` remains LOCKED and untouched.

---

## Milestone 7I — Pre-final-evaluation decision audit ✅ PASSED

- **Status:** ✅ **COMPLETE.** Audit / documentation only.
- **Goal:** create a decision / readiness audit before any larger/full development evaluation or
  the final locked `ood_test` evaluation — decide what is frozen, what is still development-only,
  what the primary controlled candidate model is, and what exact next steps are allowed before
  `ood_test`. Authorized by explicit user go-ahead.
- **Type:** **docs / tables / logs only.** **No training, no inference, no calibration fitting,
  no MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final-performance claim, no
  committed clinical threshold.** Reads the saved Milestone 7H consolidated DG artifacts
  read-only (fingerprint-verified unmodified) purely for a few provenance numbers; every other
  value is a copied/derived decision record.
- **Built:** `scripts/30_make_pre_final_decision_audit.py` (pandas + csv; reuses no model code).
  Generates the 3 decision tables + the report and verifies the narrative + context docs carry a
  7I marker.
- **Inputs (read-only, unmodified):** `results/tables/exp07h_consolidated_model_comparison.csv`,
  `exp07h_controlled_dg_comparison.csv`, `exp07h_context_vs_controlled_warning_table.csv`,
  `exp07h_groupdro_calibration_summary.csv`, `exp07h_groupdro_operating_point_summary.csv`,
  `exp07h_reviewer_risk_update.csv`; `results/logs/exp07h_dg_reporting_package_summary.txt`.
- **Decisions recorded:**
  - **Primary controlled candidate:** **7F GroupDRO with temperature-scaled confidence**
    (`exp07f_groupdro_resnet18`; GroupDRO T=3.6570).
  - **Matched control:** **7F center-stratified ERM** (`exp07f_centerstrat_erm_resnet18`).
  - **Context-only baselines:** **old plain ERM, 7A, 7D, 7E** (old confounded cache / different
    id_val subset — never controlled comparators for GroupDRO).
  - **Frozen:** locked split mapping; `ood_test` locked; `ood_val` development-only; old
    confounded caches barred from DG training; center-stratified caches required for DG; fair DG
    comparison = 7F ERM vs 7F GroupDRO; old plain ERM / 7A / 7D / 7E context-only.
  - **Historical not-committed decisions at 7I:** final-test pair; final operating threshold; final
    calibration policy for the paper-level test; whether to run a full `ood_val`/full development
    evaluation; whether to add CORAL/DANN; whether to add model-based uncertainty.
  - **Full `ood_val`/full development evaluation before `ood_test`:** recommended **if
    compute/time allows** (Path A); otherwise manuscript skeleton (Path E).
  - **Model-based uncertainty:** **not necessary before `ood_test`; remains DEFERRED / optional**;
    not the default next step.
  - **Additional DG methods (CORAL/DANN):** not necessary before `ood_test`; optional for
    reviewer-risk breadth (Path C).
- **Recommended next path:** **Path A (full development-scale evaluation planning) if compute/time
  allows, otherwise Path E (manuscript skeleton)**; Path B (freeze candidate + write final-eval
  protocol) alongside; Path C (CORAL/DANN) optional; Path D (model-based uncertainty)
  deferred/optional and NOT the default. The default next action is a **decision**, not automatic
  model-based uncertainty.
- **Artifacts:** `docs/PRE_FINAL_EVALUATION_AUDIT.md`, `docs/FROZEN_DECISIONS.md`,
  `docs/NEXT_EXPERIMENT_DECISION_TREE.md`;
  `results/tables/exp07i_pre_final_decision_matrix.csv` (10 rows),
  `exp07i_model_readiness_table.csv` (6 rows),
  `exp07i_forbidden_claims_before_test.csv` (8 rows);
  `results/logs/exp07i_pre_final_evaluation_audit_report.txt`.
- **Assertions (all passed):** no `ood_test` in any path; input artifacts unmodified (size+mtime
  fingerprint); all 3 tables saved non-empty; row-count sanity (10/6/8); controlled GroupDRO
  ood_val AUROC > matched ERM; context old plain ERM AUROC > GroupDRO (reference gap); all
  narrative + context docs present and carrying a 7I marker.
- **Outcome:** OVERALL PASS. No warnings or errors.
- **Note:** decision/documentation audit on the existing development-stage evidence — NOT a final
  result and NOT a performance claim. `ood_val` referenced for development-stage decision recording
  only; `ood_test` remains LOCKED and untouched.

---

## Milestone 8A — Full development-scale evaluation planning package ✅ PASSED

- **Status at that historical handoff:** ✅ **COMPLETE** — then the latest completed milestone.
  **Planning / protocol only.**
- **Goal:** create a planning/protocol package for a larger/full development-scale evaluation
  (Path A from Milestone 7I) — decide exactly what would be evaluated, on which development
  splits, with which frozen checkpoints, what outputs would be produced, and what compute/storage
  risks and stop rules exist — **before** any final locked `ood_test` evaluation. Authorized by
  explicit user go-ahead.
- **Type:** **docs / tables / logs only.** **No training, no model inference, no calibration
  fitting, no MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final-performance claim,
  no committed clinical threshold.** The script does **not** import torch and does not load model
  weights or images; it checks the frozen checkpoints/predictions read-only (existence only).
  Planning (8A) is explicitly separated from execution (8B).
- **Built:** `scripts/31_make_full_development_eval_plan.py` (csv/os only; reuses no model code).
  Generates the four `exp08a_*` tables + the planning report and runs 23 assertions.
- **Inputs (read-only, unmodified):** frozen checkpoints
  `results/checkpoints/exp07f_groupdro_resnet18/best.pt`,
  `results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt` (existence checked only); the 7F
  prediction CSVs (reference); the Milestone 7H/7I docs and tables for grounding.
- **Delivered docs:** `docs/FULL_DEVELOPMENT_EVALUATION_PROTOCOL.md` (purpose; why before
  `ood_test`; exact models — 7F GroupDRO primary + 7F center-stratified ERM control required,
  old plain ERM/7A/7D/7E optional context-only; exact checkpoints; development splits full/larger
  `id_val` {0,3,4} + `ood_val` {1}, no `ood_test`; threshold policy id_val-selected candidate only;
  calibration development-only; required metrics; explicit `ood_test` locked statement),
  `..._RESOURCE_PLAN.md` (verified split sizes train 302,436 / id_val 33,560 / ood_val 34,904 /
  ood_test 85,054 LOCKED-unused; RTX 3060 runtime band; CSV disk estimate; batch size 64;
  cache-predictions-not-images; failure/retry; checkpoint loading; id_val-before-ood_val;
  ood_test-not-included warning), `..._OUTPUT_SCHEMA.md` (prediction CSV required columns incl.
  `confidence`; metrics/calibration/operating-point/per-center/high-conf-FN/report schemas),
  `..._STOP_RULES.md` (nine stop rules S-1…S-9).
- **Delivered tables:** `results/tables/exp08a_evaluation_scope_matrix.csv` (10 rows),
  `exp08a_model_checkpoint_freeze_table.csv` (6 rows), `exp08a_metric_output_plan.csv` (9 rows),
  `exp08a_compute_storage_risk_table.csv` (9 rows);
  report `results/logs/exp08a_full_development_evaluation_planning_report.txt`.
- **Frozen for 8B (required, controlled):** primary candidate **7F GroupDRO** + matched control
  **7F center-stratified ERM** (`frozen_for_next_execution=yes`, `include=yes`). **Optional
  context-only (`include=optional`, not required):** old plain ERM, 7A, 7D, 7E.
- **Recommended execution order (8B):** load frozen checkpoints → evaluate full/larger `id_val`
  first (reload sanity + candidate threshold selection + per-center) → evaluate full/larger
  `ood_val` (apply id_val-selected candidate thresholds) → optional development-only calibration +
  high-confidence FN audit + optional uncertainty/selective → optional context-only models →
  never touch `ood_test`.
- **Assertions (all 23 passed):** frozen GroupDRO + center-strat ERM checkpoints present
  (read-only); 7F predictions present (reference); all four tables saved non-empty; row counts
  10/6/9/9; every scope row `uses_ood_test==no`; final `ood_test` row present and excluded; no
  generated table asserts `ood_test` data usage; both 7F models frozen/include=yes; all four
  context-only models include=optional; all four planning docs present; torch NOT imported.
- **Outcome:** OVERALL PASS. WARNINGS 0, ERRORS 0.
- **Note:** planning only — NOT a result and NOT a performance claim; no inference/training/
  calibration performed; no checkpoint/prediction/cache modified. **Next: gated Milestone 8B —
  execute the full development-scale evaluation plan (inference only, no training); `ood_test`
  remains LOCKED and untouched.**

---

## Milestone 8B — Full development-scale evaluation (inference-only execution) ✅ PASSED

- **Goal:** execute the Milestone 8A plan — run **inference-only** evaluation of the two FROZEN 7F
  checkpoints (primary GroupDRO candidate + matched center-stratified ERM control) on the
  **full/larger development splits** (`id_val` centers {0,3,4}, `ood_val` center {1}), to obtain a
  more trustworthy development-stage OOD estimate than the 2000-sample capped `ood_val` subset
  (reviewer risks R1/R4/R9). Development-stage evaluation only, NOT final performance. Authorized by
  explicit user go-ahead.
- **Strict scope:** NO training, NO weight update, NO checkpoint modification, NO calibration
  fitting, NO MC-dropout, NO ensemble, NO CORAL/DANN. `ood_test` (HF test / center 2) NEVER read.
  `ood_val` used for development-stage OOD analysis only. Candidate operating thresholds selected on
  `id_val` only and NOT committed. New run names only under `exp08b_full_dev_eval` (no overwrite of
  any prior-milestone artifact).
- **Built:**
  - `scripts/32_run_full_development_eval.py` — loads the two frozen checkpoints (exact-path
    verified; `results/checkpoints/exp07f_groupdro_resnet18/best.pt`,
    `.../exp07f_centerstrat_erm_resnet18/best.pt`), rebuilds `CamelyonClassifier` from each stored
    config, and runs eval-transform inference on the FULL `id_val` and `ood_val` splits via
    `Camelyon17HFDataset(split, max_samples=None)` (the sanctioned locked-mapping HF loader; only
    the `validation` split is loaded and center-filtered — center 2 / test never referenced). Writes
    four prediction CSVs with the extended schema (adds `confidence`). Reuses
    `evaluation/predict.py`, `models/classifier.py`, `training/checkpointing.py`,
    `data/torch_dataset.py`, `data/transforms.py` — no `src/` module changed.
  - `scripts/33_summarize_full_development_eval.py` — CSV-only: overall metrics, per-center `id_val`
    metrics, candidate operating points (id_val-selected → ood_val; fixed spec 0.80/0.85/0.90/0.95 +
    fixed sens 0.80/0.90/0.95), raw high-confidence FN audit (conf 0.90/0.95/0.99). Writes JSON + 4
    tables + report + 5 matplotlib figures. Reuses `evaluation/metrics_classification.py` and
    `metrics_operating_points.py` — no `src/` module changed.
  - One benign fix mid-run: a non-ASCII `∩` in a script-32 console print could not be encoded by the
    Windows cp1252 console; replaced with ASCII and re-ran to a clean PASS (no artifact affected).
- **Data / provenance note:** the HF `datasets` builder printed "Generating train/validation/**test**
  split" on first materialization — that is the library building its local arrow cache for the whole
  repo, **not** the pipeline reading the test split. Our loader only calls
  `load_dataset(split="validation")` and filters to the allowed centers; only centers {0,3,4} (id_val)
  and {1} (ood_val) appear in the outputs. **`ood_test` / center 2 was never loaded into inference.**
- **Row counts & centers (exact):** `id_val` n=**33,560** (label0=16,952 / label1=16,608), centers
  **{0,3,4}**; `ood_val` n=**34,904** (label0=label1=17,452), center **{1}**. Both models identical
  counts. No center 1 in id_val; no center 2 anywhere.
- **Overall classification metrics (tumor = positive, threshold 0.5):**

  | split | model | accuracy | auroc | auprc | sensitivity | specificity | precision | f1 | tn/fp/fn/tp |
  |-------|-------|---------:|------:|------:|------------:|------------:|----------:|---:|:-----------:|
  | id_val  | 7F GroupDRO (primary)      | 0.8416 | 0.9323 | 0.9322 | 0.7572 | 0.9243 | 0.9074 | 0.8255 | 15668/1284/4033/12575 |
  | id_val  | 7F center-strat ERM (ctrl) | 0.7987 | 0.9085 | 0.9131 | 0.6821 | 0.9129 | 0.8847 | 0.7703 | 15475/1477/5280/11328 |
  | ood_val | 7F GroupDRO (primary)      | 0.7847 | 0.8956 | 0.8958 | 0.6407 | 0.9288 | 0.9000 | 0.7485 | 16209/1243/6271/11181 |
  | ood_val | 7F center-strat ERM (ctrl) | 0.7675 | 0.8673 | 0.8757 | 0.6074 | 0.9276 | 0.8935 | 0.7232 | 16188/1264/6852/10600 |

- **Per-center `id_val` metrics (threshold 0.5):**

  | model | center | n | acc | auroc | sensitivity | specificity | loss |
  |-------|:------:|--:|----:|------:|------------:|------------:|-----:|
  | GroupDRO | 0 | 6011 | 0.8553 | 0.9338 | 0.7667 | 0.9423 | 0.4341 |
  | GroupDRO | 3 | 12879 | 0.8790 | 0.9603 | 0.7852 | 0.9716 | 0.3629 |
  | GroupDRO | 4 | 14670 | 0.8031 | 0.8980 | 0.7285 | 0.8756 | 0.4812 |
  | center-strat ERM | 0 | 6011 | 0.8486 | 0.9149 | 0.7808 | 0.9152 | 0.4645 |
  | center-strat ERM | 3 | 12879 | 0.8990 | 0.9611 | 0.8385 | 0.9587 | 0.3457 |
  | center-strat ERM | 4 | 14670 | 0.6901 | 0.8323 | 0.5031 | 0.8720 | 0.6121 |

  Worst center is **center 4** for both. **On the full id_val, GroupDRO markedly improved the
  worst-center (center 4) behavior** — acc 0.6901→0.8031, AUROC 0.8323→0.8980, loss 0.6121→0.4812 —
  a stronger positive result than the capped 7F id_val (n=450), where GroupDRO did not improve
  worst-center behavior. (ERM is slightly better on centers 0 and 3.)
- **Candidate id_val-selected fixed-specificity operating points → ood_val sensitivity
  (CANDIDATE only; no committed threshold):**

  | target spec | GroupDRO thr | GroupDRO ood_sens | ERM thr | ERM ood_sens |
  |------------:|-------------:|------------------:|--------:|-------------:|
  | 0.80 | 0.1014 | **0.7743** | 0.2030 | 0.6582 |
  | 0.85 | 0.1836 | **0.7335** | 0.2973 | 0.6394 |
  | 0.90 | 0.3584 | **0.6753** | 0.4505 | 0.6143 |
  | 0.95 | 0.6946 | **0.5936** | 0.6825 | 0.5745 |

  GroupDRO ood_val sensitivity > ERM at **4/4** fixed-specificity targets. Fixed-sensitivity points
  (0.80/0.90/0.95) likewise keep GroupDRO ≥ ERM on ood_val specificity at matched sensitivity.
- **High-confidence false-negative audit (RAW; `label==1 & pred==0 & confidence≥t`), ood_val:**
  GroupDRO 3922/3131/1679 vs ERM 5447/4976/3794 at t=0.90/0.95/0.99 (total FN GroupDRO 6271 vs ERM
  6852). GroupDRO has **fewer** high-confidence missed tumors at every threshold on both splits
  (id_val too). These are RAW (uncalibrated) counts — no calibration was fit in 8B.
- **Controlled verdict:** **DOES GROUPDRO STILL BEAT THE MATCHED ERM AT FULL/LARGER DEVELOPMENT
  SCALE? → YES.** ood_val AUROC **0.8956 vs 0.8673 (+0.0283)**, AUPRC 0.8958 vs 0.8757 (+0.0201),
  default-0.5 sensitivity 0.6407 vs 0.6074 (+0.0333), matched fixed-specificity sensitivity **4/4**,
  and (new at full scale) improved worst-center id_val behavior. **Scale finding (honest):** the
  full/larger splits give **much higher** OOD AUROC than the 2000-sample capped `ood_val`
  (GroupDRO 0.7129→0.8956; ERM 0.6180→0.8673) — the small balanced subset **understated** both
  models; the GroupDRO-over-ERM advantage persists and the ranking direction is preserved.
- **Assertions (all passed, both scripts):** frozen checkpoint paths exactly match; id_val centers
  ⊆ {0,3,4} (no center 1/2); ood_val center == {1} (no center 2); id_val n==33,560; ood_val
  n==34,904; probabilities finite, in [0,1], `prob_0+prob_1≈1`; labels binary; all required
  prediction columns present (incl. `confidence`); all metadata columns present; outputs under the
  `exp08b_full_dev_eval` run name only (no overwrite of prior milestones); all tables/figures/JSON
  saved non-empty; **no `ood_test` usage**.
- **Outcome:** OVERALL PASS (both scripts; no errors). Benign: HF "unauthenticated requests" notice
  and Windows symlink-cache warning (both expected/ignored per RUNBOOK).
- **Artifacts:**
  - predictions: `results/predictions/exp08b_full_dev_eval/{groupdro,centerstrat_erm}_{id_val,ood_val}_predictions.csv`;
  - metrics: `results/metrics/exp08b_full_dev_eval/full_dev_metrics.json`;
  - tables: `results/tables/exp08b_{overall_metrics,id_val_per_center_metrics,candidate_operating_points,high_confidence_false_negatives}.csv`;
  - figures: `results/figures/exp08b_{ood_val_roc_groupdro_vs_erm,ood_val_pr_groupdro_vs_erm,candidate_fixed_specificity_sensitivity,id_val_per_center_accuracy,high_confidence_fn_comparison}.png`;
  - logs: `results/logs/exp08b_full_dev_eval_inference_report.txt`,
    `exp08b_full_development_eval_report.txt`.
- **Note:** DEVELOPMENT-stage evaluation on full/larger development splits — NOT final performance and
  NOT a superiority/clinical claim. No calibration fitted, no MC-dropout/ensemble/CORAL/DANN, no
  committed operating point. `ood_val` used for authorized development-stage OOD analysis only;
  `ood_test` remains LOCKED and untouched. **Historical handoff status: Milestone 8B was then the
  latest completed milestone.**

**Milestones 7A (stain/color augmentation ERM baseline), 7B (threshold-normalized
comparison), 7C (stain-aug calibration + candidate operating-point audit), 7D (tuned
stain/color augmentation v2 ERM baseline), 7E-0 (stain-space OD/HED augmentation smoke
test), and 7E (stain-space OD/HED augmentation ERM baseline) are now COMPLETE** — see their
entries above. 7B refined 7A's finding: the OOD
sensitivity collapse at the default 0.5 threshold is **mostly a threshold artifact** (ood_val
ranking largely preserved, |ΔAUROC| 0.020; an id_val-selected spec≥0.90 threshold recovers
stain-aug ood_val sensitivity 0.096→0.457). 7C then calibrated the stain-aug model (T=3.2102;
ECE 0.356→0.191, high-confidence FNs 399→0) and audited candidate operating points, confirming
calibration is argmax-invariant (does not fix the 0.5 collapse) and that **plain ERM remains
stronger than stain-aug at every matched fixed-specificity candidate point**. 7D then tried a
**stronger but controlled torchvision v2 augmentation** and found it **did not help** — ood_val
AUROC 0.7397 (below plain 0.7696 and 7A 0.7492), and 7D sensitivity was below both baselines at
every matched fixed-specificity target — so **naive→tuned torchvision stain/color augmentation is
an insufficient stain-robustness lever here**. **7E-0** then implemented and validated a
pathology-specific **optical-density (HED-style) stain-space** augmentation (transform-only smoke
test, PASS: numerically safe, valid RGB, pathology-plausible previews). **7E** then **trained** an
ERM with that stain-space augmentation and ran a 4-model comparison: the OD-space transform did
**not** lift OOD ranking (ood_val AUROC 0.7317, lowest of plain/7A/7D/7E) but is the **strongest
stain-augmentation variant at every matched fixed-specificity operating point** (beats 7A and 7D
at 4/4, still below plain ERM at 0/4). **Net across 7A→7E: no stain-augmentation lever (torchvision
colour jitter or OD-space) beats plain-ERM OOD ranking on this capped subset.** All items below are
**gated behind an explicit written go-ahead; none is started.**

**DG feasibility / group-label audit (Milestone 7F-0) — ✅ COMPLETE (see entry above).** The
no-training metadata/cache audit ran and found: full group metadata IS preserved in all caches
(`center`/`patient`/`slide`/`node`/`image_id`/`x_coord`/`y_coord`), but the capped `train` cache is
**center↔label confounded** (center 0 ~9% tumor, center 3 100% tumor; source center 4 absent), so
center-based DG is **runnable but not sound**. **Verdict: Recommendation A — a center-stratified
cache rebuild is required before sound DG-by-center training.**

**Center-stratified metadata-rich cache rebuild (Milestone 7F-1) — ✅ COMPLETE (see entry
above).** The gated, no-training rebuild ran: `scripts/25_rebuild_center_stratified_caches.py`
built new center×label-balanced caches by filling per-`(center,label)` quotas across source
centers `{0,3,4}` (train 300/cell → n=1800, 900/900, 600 per center; id_val 75/cell → n=450,
225/225, 150 per center), preserving all seven metadata fields and the existing serializer
schema, writing to **new** filenames (`center_strat_train_c034_percell300_seed1337_shuf.pt`,
`center_strat_id_val_c034_percell75_seed1337_shuf.pt`) **without overwriting** the existing
caches. **Verdict: DG-by-center is now SOUND** — the new caches support GroupDRO / CORAL / DANN
by center; the old confounded caches must be avoided for DG training. OVERALL PASS (22/22
assertions; no warnings). `ood_test` untouched.

**Dedicated domain-generalization method (Milestone 7F) — ✅ COMPLETE (see entry above).** The
gated DG training step ran: `src/training/groupdro.py` + `scripts/26` + `scripts/27` trained a
matched center-stratified ERM control and a GroupDRO-by-center model on the new sound
center-stratified caches (`id_val_loss` selection only; `ood_val` read only after training). **Verdict:
GroupDRO beat its matched ERM control on OOD** — ood_val AUROC 0.7129 vs 0.6180 (+0.0949); AUPRC
0.6287 vs 0.5214; default-0.5 sensitivity 0.0700 vs 0.0100; matched fixed-specificity ood_val
sensitivity 4/4 targets — the first DG lever to beat its matched control on OOD ranking. It did not
improve worst-center id_val behavior. Old plain/7A/7D/7E are reference-only (old confounded cache,
not matched). OVERALL PASS; no calibration; no committed threshold; `ood_test` untouched.
**Historical recommendation (subsequently completed): RETAIN GroupDRO; proceed to calibration +
candidate operating-point audit of the 7F models.**

**Calibration + candidate operating-point audit of the 7F models (Milestone 7G) — ✅ COMPLETE.**
See the "Milestone 7G" entry above and the just-completed summary block. GroupDRO stays better than
its matched ERM control after the audit and is the better-calibrated model.

**Calibration + candidate operating-point audit of the 7F models (Milestone 7G) — ✅ COMPLETE
(see entry above).** The gated CSV/logit-only audit ran: `scripts/28` temperature-scaled both matched
7F models on `ood_val` logits (center-strat ERM T=3.9239, GroupDRO T=3.6570) and audited candidate
operating points. **Calibration improved confidence reliability for both** (ERM ECE 0.4984→0.3698,
GroupDRO 0.4258→0.2529; high-confidence missed tumors reduced sharply, both →0 at conf≥0.99) but is
**argmax-invariant**. **GroupDRO remains better than its matched ERM after the audit** — higher
ood_val AUROC (0.7129 vs 0.6180) and ood_val sensitivity ≥ ERM at all 4 fixed-specificity targets
(4/4 strictly greater), and it is the better-calibrated model both raw and calibrated. **Recommendation:
RETAIN GroupDRO with calibrated confidence; candidate (not committed) operating points only.** OVERALL
PASS; no committed threshold; `ood_test` untouched.

**Consolidated DG reporting package (Milestone 7H) — ✅ COMPLETE (see the "Milestone 7H" entry
above).** Reporting-only: `scripts/29_make_dg_reporting_package.py` consolidated the saved results
through 7G into 6 DG tables, 5 figures, 2 new docs, and a report, with 3 existing docs updated. No
training, no inference, no calibration fitting, no MC-dropout, no ensemble, no CORAL/DANN, no
`ood_test`, no committed threshold, no final-performance claim. The fair controlled comparison (7F
center-stratified ERM vs 7F GroupDRO) is kept separate from context-only old plain ERM / 7A / 7D / 7E;
R3 reduced HIGH→partly-mitigated; R1/R4/R7 remain open. OVERALL PASS.

**Pre-final-evaluation decision audit (Milestone 7I — ✅ COMPLETE; docs/tables/logs only)** —
see the "Milestone 7I" entry above. Recorded what is frozen before any larger/full development
evaluation or final `ood_test`; named the primary controlled candidate (7F GroupDRO with
temperature-scaled confidence), the matched control (7F center-stratified ERM), and the context-only
baselines (old plain ERM, 7A, 7D, 7E); recommended a full `ood_val`/full development evaluation before
`ood_test` if compute/time allows (Path A); kept model-based uncertainty DEFERRED / optional; and
produced a decision tree for next experiments. No training, no inference, no calibration fitting, no
MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final-performance claim, no committed clinical
threshold. OVERALL PASS. Recommended next action: Path A (full development-scale evaluation planning).

**Full development-scale evaluation planning package (Milestone 8A — ✅ COMPLETE; planning-only)** —
see the "Milestone 8A" entry above. `scripts/31_make_full_development_eval_plan.py` produced the
`docs/FULL_DEVELOPMENT_EVALUATION_*` docs, the four `results/tables/exp08a_*` tables, and the planning
report; froze the required 7F GroupDRO candidate + 7F center-stratified ERM control checkpoints (old
plain ERM/7A/7D/7E optional context-only); defined development splits (full/larger `id_val` {0,3,4} +
`ood_val` {1}; `ood_test` excluded), threshold/calibration policy, metrics/output schema, compute/storage
risks, and nine stop rules. No training, no inference, no calibration fitting, no `ood_test`. OVERALL PASS
(23/23 assertions). **Historical handoff status: Milestone 8A was then the latest completed milestone.**

**Execute the full development-scale evaluation (Milestone 8B — ✅ COMPLETE; INFERENCE ONLY, no
training)** — done; see the "Milestone 8B" entry above. Evaluated the frozen 7F GroupDRO candidate
+ 7F center-stratified ERM control on full `id_val` (33,560) + `ood_val` (34,904); `ood_test` never
touched. **GroupDRO still beat its matched ERM at full/larger development scale** (ood_val AUROC
0.8956 vs 0.8673; matched fixed-specificity sensitivity 4/4; improved worst-center id_val). OVERALL
PASS. **Historical handoff status:** Milestone 8D was treated as the current readiness point;
the then-next decision was protocol review or a manuscript skeleton. Both were subsequently
addressed.
See the "Milestone 8C" completed entry and the "Milestone 8D" completed entry below.

**Milestone 8C — Full-development calibration + reliability audit ✅ COMPLETE / ACCEPTED
(CSV / logit-only; NO training, NO inference, NO weight update, NO MC-dropout, NO ensemble,
NO CORAL/DANN, NO `ood_test`).** `scripts/34_calibrate_full_development_eval.py` reused the saved
8B full-split prediction CSVs (`results/predictions/exp08b_full_dev_eval/…`) plus
`src/calibration/` + `src/evaluation/` to temperature-scale the two frozen 7F models on the FULL
`ood_val` logits (n=34,904) as **development-stage** calibration (weights unchanged; checkpoints
untouched; `ood_test` never read).
- **Temperature scaling:** 7F GroupDRO (primary candidate) **T=2.974907** (NLL
  **0.7591→0.4451** at four decimal places; one source summary displays 0.7592 because of
  source-display rounding);
  7F center-stratified ERM (matched control) **T=3.496293** (NLL 1.1911→0.5321). Both
  `converged=false` at the 200-iteration cap, but NLL/ECE improved substantially — **not a
  failure**.
- **Raw → calibrated reliability (full `ood_val`):** GroupDRO ECE **0.1560→0.0394**,
  Brier 0.1782→0.1458, NLL 0.7591→0.4451; ERM ECE **0.1829→0.0695**, Brier 0.2070→0.1727,
  NLL 1.1911→0.5321. Calibration improved all three metrics for **both** models.
- **High-confidence false negatives (missed tumors), raw → calibrated:** GroupDRO conf≥0.90
  3922→533, ≥0.95 3131→82, ≥0.99 **1679→0**; ERM conf≥0.90 5447→1693, ≥0.95 4976→539,
  ≥0.99 3794→5.
- **Reliability summary (argmax-invariant — ranking/hard predictions unchanged raw vs
  calibrated):** GroupDRO acc 0.7847, AUROC **0.8956**, AUPRC 0.8958, sensitivity 0.6407,
  specificity 0.9288, FN 6271; ERM acc 0.7675, AUROC **0.8673**, AUPRC 0.8757, sensitivity
  0.6074, specificity 0.9276, FN 6852.
- **Verdict:** **GroupDRO REMAINS BETTER than its matched ERM after full-development
  calibration** — higher `ood_val` AUROC/AUPRC, ≤ high-confidence FN at every threshold/variant,
  and better-calibrated both raw and calibrated (lower ECE). Calibration is **development-stage**
  (fit + evaluated on the same `ood_val`) — a reliability audit, **NOT** an independent-test
  calibration claim (reviewer risk **R7 remains open**). **No committed clinical/paper
  threshold.** OVERALL PASS; all assertions; `ood_test` usage NONE.
- **Artifacts:** `scripts/34_calibrate_full_development_eval.py`;
  `results/predictions/exp08c_full_dev_calibration/{groupdro,centerstrat_erm}_ood_val_calibrated_predictions.csv`;
  `results/metrics/exp08c_full_dev_calibration/{temperature,calibration_metrics}.json`;
  `results/tables/exp08c_{calibration_comparison,high_confidence_fn_comparison,reliability_summary,candidate_operating_points_with_calibration_note}.csv`;
  `results/figures/exp08c_{calibration_raw_vs_calibrated,high_confidence_fn_reduction,ece_brier_nll_groupdro_vs_erm}.png`;
  `results/logs/exp08c_full_development_calibration_report.txt`.

**Milestone 8D — Final protocol freeze / final-evaluation readiness package (✅ COMPLETE; gated;
docs / tables / logs only).** Done — see the "Milestone 8D — Final protocol freeze" completed entry
above. **Docs / tables / logs only — no training, no inference, no calibration fitting, no
MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final test run.** Froze the checkpoint
artifacts/paths; proposed the final-test primary/control pair, calibration policy, and candidate
operating-point policy; and defined the single-shot
`ood_test` (center 2) evaluation protocol, output schema, stop rules, claims-unlock map, and
authorization template. The checkpoint artifacts remain frozen and unchanged; the proposed pair,
final threshold, final calibration policy, and final `ood_test` result remained uncommitted and
unauthorized. **Historical 8D recommendation (subsequently completed):** review the protocol or
prepare a manuscript skeleton. `ood_val` used for authorized development-stage analysis only; `ood_test` remains fully
LOCKED and unread (single final locked evaluation only, after the completed authorization template).

**Milestone 8D — Final protocol freeze / final-evaluation readiness package ✅ COMPLETE /
ACCEPTED (docs / tables / logs only; NO training, NO inference, NO calibration fitting, NO
MC-dropout, NO ensemble, NO CORAL/DANN, NO `ood_test`).**
`scripts/35_make_final_eval_readiness_package.py` (**does not import torch**; reads only saved
8B/8C artifacts read-only and fingerprint-verifies them unmodified) produced a **readiness /
protocol-freeze package — NOT a final evaluation**. It consolidated the 8B/8C development evidence,
`docs/FROZEN_DECISIONS.md`, `docs/PRE_FINAL_EVALUATION_AUDIT.md`, and the
`docs/FULL_DEVELOPMENT_EVALUATION_*` protocol into a final-evaluation readiness package and a draft
single-shot `ood_test` protocol.
- **Frozen checkpoint artifacts / proposed final-test pair:** 7F GroupDRO
  (`results/checkpoints/exp07f_groupdro_resnet18/best.pt`) proposed as primary; 7F
  center-stratified ERM (`results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt`) proposed as
  matched control. The artifacts and paths are frozen and unchanged. The candidate operating-point
  family and **8C development temperatures** (GroupDRO **T=2.974907**, ERM **T=3.496293**) remain as
  proposed final-test policies, applied never refit if later authorized.
- **Not committed/authorized:** the proposed final-test pair, final operating threshold, final
  calibration policy, and final `ood_test` result. Explicit template completion is required; after
  authorization, the exact unchanged checkpoint artifacts become the committed final-test pair.
- **Recommended policies:** calibration = report **both** raw and calibrated (frozen 8C
  temperatures; fitting on `ood_test` FORBIDDEN); threshold = **no single final clinical threshold**
  (id_val-selected candidate families allowed; choosing after test FORBIDDEN — test leakage).
- **Historical readiness verdict:** READY to seek authorization. A **manuscript skeleton** on
  development evidence was an allowed next documentation step and was subsequently completed. The
  single `ood_test` (center 2, expected n=85,054) run stays **fully LOCKED** and may proceed **only**
  after the user completes `docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md` (exact phrase + all 7
  checkboxes). 8D does **not** authorize `ood_test`.
- **Assertions (all passed):** torch NOT imported; no `ood_test`/test/center-2 path referenced;
  read-only 8B/8C inputs fingerprint-unchanged; all 5 docs + 6 tables + report written non-empty;
  CLAUDE.md < 30,000 chars; AGENTS.md < 20,000 chars. **OVERALL PASS.**
- **Artifacts:** `scripts/35_make_final_eval_readiness_package.py`;
  `docs/{FINAL_EVALUATION_READINESS_PACKAGE,FROZEN_FINAL_EVAL_PROTOCOL_DRAFT,OOD_TEST_AUTHORIZATION_TEMPLATE,FINAL_TEST_OUTPUT_SCHEMA,FINAL_TEST_STOP_RULES}.md`;
  `results/tables/exp08d_{freeze_decision_table,final_eval_model_plan,final_eval_metric_plan,threshold_policy_options,calibration_policy_options,claims_unlock_table}.csv`;
  `results/logs/exp08d_final_eval_readiness_report.txt`.
- **Note:** Readiness / protocol package — NOT a final evaluation and NOT a final-performance claim.
  `ood_test` remains LOCKED and unread. **Historical 8D handoff:** the then-next decision was
  protocol review or a manuscript skeleton; both were subsequently addressed.

**Milestone 8E — Manuscript skeleton on development evidence ✅ COMPLETE / ACCEPTED
(docs / tables / logs only; NO training, NO inference, NO calibration fitting, NO MC-dropout,
NO ensemble, NO CORAL/DANN, NO `ood_test`, NO final-performance claim; no script required; no
result file modified).** Authored a full manuscript skeleton and paper-facing drafts built **only**
from completed development-stage evidence (6A/7H/8B/8C/8D). Authorized by explicit user go-ahead.
- **Goal:** produce a manuscript skeleton + development-stage narrative so a paper can be drafted
  now, keeping the reserved `ood_test` number a marked placeholder.
- **Built (docs):** `docs/MANUSCRIPT_SKELETON_DEV_EVIDENCE.md` (title options, abstract outline,
  intro/methods/experiments/results/discussion/limitations/conclusion section lists, and a marked
  **LOCKED / NOT-RUN** final-test placeholder section with no numbers); `docs/MANUSCRIPT_METHODS_
  DRAFT_DEV_ONLY.md` (M1–M9); `docs/MANUSCRIPT_RESULTS_DRAFT_DEV_ONLY.md` (R1–R7, development numbers
  only); `docs/MANUSCRIPT_DISCUSSION_POINTS.md` (D1–D8).
- **Built (tables + report):** `results/tables/exp08e_manuscript_table_manifest.csv` (21 rows incl.
  a LOCKED reserved row); `results/tables/exp08e_manuscript_claims_map.csv` (16 claim rows with
  allowed-now / before-test / after-test-if-supported wording + risk level);
  `results/logs/exp08e_manuscript_skeleton_report.txt`.
- **Key development-stage numbers cited (all scope-tagged):** 8B full `ood_val` GroupDRO AUROC
  **0.8956** vs matched ERM **0.8673** (+0.0283), fixed-specificity sensitivity 4/4, worst-center
  (center 4) id_val acc 0.6901→0.8031; 8C temperature scaling GroupDRO ECE 0.1560→0.0394 / ERM
  0.1829→0.0695 (argmax-invariant), high-confidence missed tumors GroupDRO conf≥0.90 3922→533 /
  conf≥0.99 1679→0.
- **R7 caveat preserved:** 8C calibration is fit + evaluated on the same `ood_val` → development-stage
  reliability audit, NOT an independent-test calibration claim; stated in the skeleton, methods (M5),
  results (R5), discussion (D6), and the claims map.
- **Allowed claims (scope-tagged):** GroupDRO > matched ERM in development; worst-center improvement;
  temperature scaling improves `ood_val` calibration (argmax-invariant); high-confidence missed tumors
  reduced after calibration; hospital shift degrades reliability. **Forbidden before `ood_test`:** final
  generalization performance; independent-test calibration validity; superiority over untested DG
  methods; GroupDRO-beats-old-plain-ERM as a controlled claim; clinical readiness; committed clinical
  threshold; all-hospital robustness; MC-dropout/ensemble findings; any `ood_test` conclusion.
- **Assertions (all passed):** no `ood_test` usage; no training/inference; no calibration fitting; no
  result file modified; all 4 docs + 2 tables + report written non-empty; context docs updated;
  CLAUDE.md < 30,000 chars; AGENTS.md < 20,000 chars. **OVERALL PASS** (no warnings/errors).
- **Note:** Writing/planning package — NOT a submission and NOT a final-performance claim. `ood_test`
  remains fully LOCKED and unread. **Historical 8E handoff:** review the manuscript/protocol package
  before any authorization; the subsequent 8F–8H audits/reconciliation completed that documentation
  path.

## Milestone 8F — Final protocol/manuscript consistency audit ✅ PASSED

- **Goal:** as the final safety/readiness check before any `ood_test` authorization, audit the
  consistency of the 8D final-evaluation protocol package against the 8E manuscript skeleton, and
  produce a pre-authorization checklist + remaining-risk register.
- **Scope:** audit / documentation only — **no training, no inference, no calibration fitting, no
  MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final test run, no final-performance
  claim.** Read only existing docs/tables (read-only, fingerprint-verified unmodified); wrote only new
  `exp08f_*` artifacts + two new docs + context-doc updates. Does not import torch.
- **Script:** `scripts/37_make_final_pre_authorization_audit.py`.
- **Consistency verdict: CONSISTENT — READY FOR MANUAL REVIEW, no blocking issues** (16/16 items
  MATCH): same primary model (7F GroupDRO), matched control (7F center-stratified ERM), checkpoint
  paths, calibration temperatures (GroupDRO T=2.974907 / ERM T=3.496293), operating-point family
  (id_val-selected spec 0.80/0.85/0.90/0.95 + sens 0.80/0.90/0.95), `ood_val`/`ood_test` distinction;
  no final-test numbers in the manuscript; no clinical-readiness / independent-test-calibration /
  GroupDRO-vs-old-plain-ERM controlled claim; final-test placeholder marked LOCKED / NOT RUN.
- **Pre-authorization checklist (11 items):** 8 `needs_user_decision` (the 7 authorization boxes +
  checkpoint-path approval — final model / control / checkpoints / temperatures / metric plan /
  threshold plan / no-retraining / no-post-test-selection) + 3 pass (final-test script not run,
  authorization template not completed, one-shot rule understood); **0 fail**.
- **Remaining-risk register (8 rows):** R1 (capped subsets, reduced/superseded by 8B/8C), R4 (no final
  `ood_test`, HIGH), R7 (same-split development calibration, HIGH), lack of CORAL/DANN (R3 residual),
  model-based uncertainty deferred (R8), no clinical threshold, no external non-Camelyon validation
  (R6), one-shot `ood_test` irreversibility (HIGH) — each with before/after mitigation + manuscript
  disclosure.
- **Non-blocking follow-up:** legacy capped-subset wording in `docs/CLAIMS_AND_LIMITATIONS.md` (header
  + universal caveats) and `docs/FIGURE_TABLE_MANIFEST.md` (caption tag) predates the 8B/8C full-dev
  evidence the manuscript now cites — a wording refresh only; changes no result and unlocks no
  `ood_test` access.
- **Authorization:** unchanged — `docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md` (paste the verbatim phrase +
  check all 7 boxes, in a future prompt). 8F does **not** authorize `ood_test`.
- **Assertions (14/14 passed):** no `ood_test` / center-2 artifact exists; authorization template still
  inert; all read-only inputs present; no input file modified by the audit; consistency table has no
  blocking mismatch (16 rows); checklist has no fail (11 rows); risk register (8 rows); 3 tables + 2
  audit docs written. **OVERALL PASS** (no warnings/errors); CLAUDE.md < 30,000 chars; AGENTS.md <
  20,000 chars.
- **Artifacts:** `scripts/37_make_final_pre_authorization_audit.py`;
  `docs/{FINAL_PRE_AUTHORIZATION_AUDIT,MANUSCRIPT_PROTOCOL_CONSISTENCY_CHECK}.md`;
  `results/tables/exp08f_{protocol_manuscript_consistency,pre_authorization_checklist,remaining_risk_register}.csv`;
  `results/logs/exp08f_final_pre_authorization_audit_report.txt`.
- **Note:** Audit package only — NOT a submission and NOT a final-performance claim. `ood_test` remains
  fully LOCKED and unread. **Historical 8F handoff:** review the audit/manuscript/protocol package
  before authorization. Model-based uncertainty remained DEFERRED / optional.

---

## Milestone 8G — Manuscript-governance wording refresh ✅ PASSED

- **Goal:** resolve the single non-blocking documentation follow-up flagged by 8F — refresh the legacy
  capped-subset wording in the manuscript-governance docs so they align with the full-development
  evidence (Milestones 8B/8C) that the manuscript now cites.
- **Scope:** documentation wording cleanup only — **no training, no inference, no calibration fitting,
  no MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final test run, no final-performance
  claim.** No result artifact (checkpoint / prediction CSV / metrics JSON / result table / figure) was
  modified; **no accepted metric value was changed**; no script was written or run.
- **Edited docs:** `docs/CLAIMS_AND_LIMITATIONS.md`, `docs/FIGURE_TABLE_MANIFEST.md` (the two flagged
  docs), plus context docs `CLAUDE.md`, `AGENTS.md`, `docs/SESSION_STATE.md`, `docs/MILESTONE_LOG.md`,
  `docs/NEXT_ACTIONS.md`. The 8E manuscript drafts already reflected 8B/8C full-development evidence
  and were left unchanged.
- **Wording categories refreshed:**
  - `CLAIMS_AND_LIMITATIONS.md`: header now names two evidence eras — **early capped/subset
    development evidence (4B–7G; id_val n=500 / ood_val n=2000)** and **full-development evidence
    (8B/8C; id_val {0,3,4} n=33,560 / ood_val {1} n=34,904)**; the shift-degradation row and A1 tagged
    "early capped"; a new **full-development allowed-claims block** (A8–A11: GroupDRO > matched ERM in
    full-development eval; temperature scaling improves ood_val reliability; high-confidence FNs reduced
    after calibration; GroupDRO primary / ERM control) with matching forbidden claims (F11–F13); the
    universal caveats made precise (both eras named; R7 stated for the 8C full-ood_val fit).
  - `FIGURE_TABLE_MANIFEST.md`: header lists five artifact eras; the 6A and 7H sections tagged
    early-capped; **new 8B (full-development inference) and 8C (full-development calibration/reliability)
    artifact inventories** added (existing result paths only, no new numbers); 8D/8E/8F labeled
    docs-only; an explicit **reserved final-test = LOCKED / NOT RUN / not produced** section; era-specific
    caption tags.
- **Preserved (unchanged):** all forbidden claims (no final generalization, no clinical readiness, no
  independent-test calibration, no committed clinical threshold, no GroupDRO-vs-old-plain-ERM controlled
  claim, no all-hospital robustness, no `ood_test` conclusion); reviewer risk **R7** (8C calibration fit +
  evaluated on the same full `ood_val`); the 8D final-test protocol; the 8E manuscript skeleton.
- **Assertions (all passed):** no `ood_test` usage; no training/inference; no calibration fitting; no
  result artifact modified; only allowed docs modified; no accepted metric value changed; no
  final-performance / clinical-readiness / independent-test-calibration claim introduced; CLAUDE.md <
  30,000 chars; AGENTS.md < 20,000 chars. **OVERALL PASS** (no warnings/errors).
- **Note:** Wording refresh only — changes no result and unlocks no `ood_test` access. `ood_test` remains
  fully LOCKED and unread. **Historical 8G handoff:** 8G was then the latest milestone; the next
  decision was review of the audit/governance/manuscript/protocol package before authorization.

## Milestone 8H — Active-state documentation reconciliation ✅ COMPLETE / ACCEPTED

- **Goal:** remove stale or contradictory active-state instructions without changing scientific
  results, checkpoints, protocol substance, or authorization status.
- **Scope:** documentation only. Modified only `CLAUDE.md`, `AGENTS.md`,
  `docs/SESSION_STATE.md`, `docs/MILESTONE_LOG.md`, `docs/NEXT_ACTIONS.md`,
  `docs/FROZEN_FINAL_EVAL_PROTOCOL_DRAFT.md`, `docs/FINAL_PRE_AUTHORIZATION_AUDIT.md`, and
  `docs/MANUSCRIPT_PROTOCOL_CONSISTENCY_CHECK.md`.
- **Reconciled active state at the time:** latest completed milestone = 8H; historical 7I/8A/8D next-step
  statements are explicitly historical/superseded; current next action = manual authorization
  review, not an automatic test run.
- **Terminology:** the checkpoint artifacts and paths
  `results/checkpoints/exp07f_groupdro_resnet18/best.pt` and
  `results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt` are frozen and unchanged. Their use
  as the 7F GroupDRO-primary + 7F center-stratified-ERM-control final-test pair remains proposed,
  uncommitted, and unauthorized until explicit completion of
  `docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md`. After authorization, these exact artifacts become the
  committed final-test pair.
- **8G post-audit resolution:** preserved the original 8F finding and added the resolution note:
  legacy capped-subset wording was refreshed by 8G; no result changed and no `ood_test` access was
  unlocked.
- **NLL display precision:** GroupDRO raw NLL is reported as **0.7591** to four decimal places;
  0.7592 is retained only as an explicitly labeled source-display rounding difference. No JSON/CSV
  or accepted metric artifact changed.
- **Assertions:** no training, inference, calibration fitting, dataset loading, HF test split,
  center-2 data, `ood_test`, final-test authorization, or result recomputation; checkpoint paths and
  temperatures unchanged; no authorization phrase inserted and no checkbox changed; model-based
  uncertainty remains deferred/optional.
- **Outcome:** PASS. No scientific result or protocol decision changed. `ood_test` remained fully
  LOCKED and unread. **At that historical handoff, the latest completed milestone was Milestone 8H.**
- **Exact next recommendation:** manual authorization review of the 8F audit package, refreshed 8G
  governance docs, 8E manuscript skeleton, 8D protocol draft, and inert authorization template —
  not an automatic final-test run.

## Milestone 8I — Final-test implementation and offline preflight ✅ COMPLETE / ACCEPTED

- **Goal:** prepare the complete single-run final `ood_test` execution package so an explicitly
  authorized run can proceed once without further coding or scientific decisions.
- **Scope:** implementation and offline validation only — no training, inference, dataset loading,
  Hugging Face test split, center-2 read, calibration fitting, threshold selection on test,
  MC-dropout, ensemble, CORAL/DANN, final-test execution, or final-performance claim.
- **Frozen config:** `configs/experiments/exp09_final_ood_test_frozen.yaml` records status
  `awaiting_user_authorization`, authorization false, the exact dataset/split/center/row contract,
  proposed GroupDRO + matched-ERM pair, frozen temperatures, default threshold, all 14 accepted
  8B `id_val`-selected candidate operating points, one-shot policy, and fresh exp09 outputs.
- **Checkpoint manifest:** `results/tables/exp08i_frozen_checkpoint_manifest.csv`.
  GroupDRO: 134250665 bytes,
  SHA256 `CE0DC65DC6106648F2ABB77C603746A1DB2B856F9672B01C2E8F540BEEAD8502`;
  center-stratified ERM: 134250601 bytes,
  SHA256 `8BF990BE517A41AA74D111E0A4F4111A05A9FE416919261D979F2FD65AE974FF`.
- **Operating-point manifest:** `results/tables/exp08i_frozen_operating_points.csv` copies all 14
  required rows exactly from `results/tables/exp08b_candidate_operating_points.csv`; thresholds
  remain candidate-only and test selection is forbidden.
- **Executor:** `scripts/38_run_final_ood_test.py`. Offline mode validates frozen source hashes,
  checkpoint paths/hashes/sizes/temperatures, operating-point provenance, authorization-template
  integrity, static absence of training/calibration-fitting calls, and output freshness without
  importing the dataset stack. The later execution path requires valid authorization JSON and
  enforces all authorization fields, frozen effective authorized status, zero run counter, one-run
  maximum, absent outputs/run state/sentinel, FT-1 through FT-9, no overwrite, and frozen
  temperatures/thresholds before the dataset is imported.
- **CSV-only summary:** `scripts/39_summarize_final_ood_test.py` reads only saved exp09 predictions
  when invoked by the authorized executor; it performs no inference, calibration fitting, or
  test-set threshold tuning and creates the protocol-defined metrics, tables, figures, and report.
- **Operator docs/log:** `docs/FINAL_TEST_EXECUTION_CHECKLIST.md`;
  `results/logs/exp08i_final_test_offline_preflight_report.txt`.
- **Offline verification:** both scripts compiled successfully; permitted command
  `scripts/38_run_final_ood_test.py --preflight-only` passed, reporting
  `dataset_loaded=False`, `ood_test_accessed=False`, and `inference_run=False`.
- **Authorization/final-pair status:** unchanged. The checkpoint artifacts and paths are frozen;
  their use as the GroupDRO-primary + matched-ERM-control pair remains proposed and uncommitted
  until the user explicitly completes the unchanged authorization template. The final execution
  code exists but has never run.
- **Outcome:** PASS. No earlier result artifact or accepted value changed. `ood_test` remains fully
  LOCKED and unread. Model-based uncertainty remains deferred/optional.
- **Exact next recommendation:** explicit manual authorization review of the frozen exp09
  config/manifests, final execution checklist, offline preflight report, 8D protocol/stop
  rules/output schema, and unchanged inert authorization template — not an automatic test run. No
  further coding or scientific decision should be needed before the authorized run.

## Milestone 9A — Single explicitly authorized final locked `ood_test` evaluation ✅ COMPLETE / ACCEPTED

- **Authorization:** the user explicitly authorized one final inference attempt under the frozen
  8D protocol and approved the final model, matched control, temperatures, metric plan, threshold
  plan, no retraining, and no post-test selection. Runtime authorization file:
  `configs/experiments/exp09_final_ood_test_authorization.json`; user timestamp
  `2026-07-16T23:14:05.0366792+06:00`; SHA256
  `0B5977E3D92990B6FC59E1138AC2D398C737F4BDCBACCDD1025D0CE0E3B96C45`.
- **Execution:** preflight passed with `dataset_loaded=false`, `ood_test_accessed=false`, and
  `inference_run=false`; then `scripts/38_run_final_ood_test.py` was launched exactly once.
  Durable run start `2026-07-16T17:16:25.455291+00:00`; inference completed
  `2026-07-16T17:35:17.051997+00:00`; summary completed
  `2026-07-16T17:35:20.395673+00:00`; attempt counter/run counter = **1**.
- **Data contract:** HF split `test`; center exactly `{2}`; 85,054 rows/model; labels 42,527
  non-tumor and 42,527 tumor; required metadata complete; one dataset instance; one dataloader
  traversal; both frozen models evaluated per batch; no cap/sample/random split/truncation.
- **Frozen artifacts:** GroupDRO checkpoint SHA256
  `CE0DC65DC6106648F2ABB77C603746A1DB2B856F9672B01C2E8F540BEEAD8502`,
  T=2.974907; matched ERM checkpoint SHA256
  `8BF990BE517A41AA74D111E0A4F4111A05A9FE416919261D979F2FD65AE974FF`,
  T=3.496293. The default threshold was 0.5. All 14 candidate operating points were loaded
  unchanged from `results/tables/exp08i_frozen_operating_points.csv`.
- **Prohibitions satisfied:** no training, optimizer, backward pass, checkpoint save, weight
  update, calibration fitting, MC-dropout, ensemble, CORAL/DANN, threshold selection on test, or
  post-test model selection. Raw/calibrated hard predictions were identical.

### Final default-threshold and threshold-free metrics

| Model | Accuracy | Balanced accuracy | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GroupDRO | 0.5337197545089002 | 0.5337197545089002 | 0.6633704256200204 | 0.63641261236421 | 0.11056505278999224 | 0.9568744562278082 | 0.7194002447980417 | 0.19167193200578847 | 40693 | 1834 | 37825 | 4702 |
| Center-stratified ERM | 0.5711547957768006 | 0.5711547957768006 | 0.6984352121958427 | 0.6556283801091156 | 0.24107037881816257 | 0.9012392127354386 | 0.7093827843897038 | 0.3598518752522859 | 38327 | 4200 | 32275 | 10252 |

- **Honest final comparison (GroupDRO minus ERM):** AUROC
  `−0.0350647865758223`; AUPRC `−0.0192157677449056`; sensitivity
  `−0.13050532602817033`; FN `+5550`; specificity `+0.0556352434923696`; precision
  `+0.0100174604083379`. The matched ERM therefore outperformed the predeclared GroupDRO
  primary on ranking, accuracy, sensitivity, F1, and FN count. GroupDRO retained higher
  specificity and slightly higher precision.

### Final raw-versus-calibrated reliability

| Model | Variant | ECE | Brier | NLL |
|---|---|---:|---:|---:|
| GroupDRO | raw | 0.4006439581474795 | 0.41233859378344173 | 1.9441627836686814 |
| GroupDRO | calibrated | 0.2583399203510463 | 0.3096680541395694 | 0.8743126064429281 |
| Center-stratified ERM | raw | 0.3084182197890404 | 0.3446598909605201 | 1.4183276475805955 |
| Center-stratified ERM | calibrated | 0.1489841758680231 | 0.2555235818479269 | 0.7130964662716416 |

- Frozen temperature scaling improved ECE/Brier/NLL for both models without changing hard
  predictions or threshold-free ranking. Calibrated ERM remained better calibrated than
  calibrated GroupDRO.

### Frozen candidate operating-point transfer results

| Model | Family | Target | Frozen threshold | Test sensitivity | Test specificity | Test FN | Test FP |
|---|---|---:|---:|---:|---:|---:|---:|
| GroupDRO | specificity | 0.80 | 0.1013885140419006 | 0.2980694617537094 | 0.8497190020457591 | 29851 | 6391 |
| GroupDRO | specificity | 0.85 | 0.1835944801568985 | 0.2239988713052884 | 0.8957133115432548 | 33001 | 4435 |
| GroupDRO | specificity | 0.90 | 0.3583780527114868 | 0.1480000940578926 | 0.9384861382180733 | 36233 | 2616 |
| GroupDRO | specificity | 0.95 | 0.6946145296096802 | 0.0725656641662943 | 0.9747924847743786 | 39441 | 1072 |
| GroupDRO | sensitivity | 0.80 | 0.3734750747680664 | 0.1430385402215063 | 0.9410962447386366 | 36444 | 2505 |
| GroupDRO | sensitivity | 0.90 | 0.1095319464802742 | 0.2884285277588355 | 0.8564206268958544 | 30261 | 6106 |
| GroupDRO | sensitivity | 0.95 | 0.0295121539384126 | 0.4747807275377995 | 0.7285254073882474 | 22336 | 11545 |
| Center-stratified ERM | specificity | 0.80 | 0.2029989957809448 | 0.4281750417381899 | 0.7809156535847814 | 24318 | 9317 |
| Center-stratified ERM | specificity | 0.85 | 0.2972504794597626 | 0.3531873868365979 | 0.8334234721471065 | 27507 | 7084 |
| Center-stratified ERM | specificity | 0.90 | 0.4505151808261871 | 0.265055141439556 | 0.8879300209278811 | 31255 | 4766 |
| Center-stratified ERM | specificity | 0.95 | 0.6825078725814819 | 0.1581583464622475 | 0.943094974957086 | 35801 | 2420 |
| Center-stratified ERM | sensitivity | 0.80 | 0.2905814945697784 | 0.3573259341124462 | 0.8299903590660052 | 27331 | 7230 |
| Center-stratified ERM | sensitivity | 0.90 | 0.1177357584238052 | 0.5223269922637384 | 0.7059750276295059 | 20314 | 12504 |
| Center-stratified ERM | sensitivity | 0.95 | 0.0456100143492221 | 0.669198391610036 | 0.5912714275636655 | 14068 | 17382 |

- These are frozen `id_val`-selected **candidate/non-clinical** operating points. None was
  selected or optimized using `ood_test`; the center-2 values are observations, not new targets.

### High-confidence false-negative audit

| Model | Variant | confidence≥0.90 | confidence≥0.95 | confidence≥0.99 | Total FN |
|---|---|---:|---:|---:|---:|
| GroupDRO | raw | 29775 | 25620 | 15485 | 37825 |
| GroupDRO | calibrated | 5479 | 755 | 1 | 37825 |
| Center-stratified ERM | raw | 19142 | 14626 | 7354 | 32275 |
| Center-stratified ERM | calibrated | 1828 | 513 | 0 | 32275 |

- **Outputs:** four validated prediction CSVs under
  `results/predictions/exp09_final_ood_test/`; metrics JSON under
  `results/metrics/exp09_final_ood_test/final_metrics.json`; four final tables under
  `results/tables/exp09_final_ood_test_*.csv`; five figures under
  `results/figures/exp09_final_ood_test_*.png`; inference/final reports and run-state/sentinels
  under `results/logs/`.
- **Outcome:** PASS. `ood_test` is no longer unread. The single authorized inference attempt is
  complete, and no second attempt is authorized.
- **Exact next recommendation:** Milestone 9B final manuscript/results and submission-package
  integration, using only the saved exp09 artifacts. Do not run `ood_test` again and do not add an
  exploratory method before manuscript integration. Model-based uncertainty remains
  deferred/optional and was not added.

---

**Model-based uncertainty estimation (DEFERRED / optional; AFTER 8B; NOT the default next step)** —
MC-dropout / deep ensembles (requires inference), evaluated on `id_val` + `ood_val`. Only if
explicitly approved. The deterministic reliability story (5A/5B/5C) and its consolidated
reporting (6A) require no model-based uncertainty; it addresses a MEDIUM risk (R8).

**Abstention-threshold selection (optional; LATER)** — if a deployable operating point is
ever wanted, choose it on `ood_val` (may reuse the saved temperature and the 5C/6A tables).
No threshold has been committed.

**Final locked evaluation (COMPLETED in Milestone 9A)** — the single authorized `ood_test`
(center 2) inference was completed exactly once. Do not access or infer on `ood_test` again.

## Milestone 9B — Final manuscript/results and submission-package integration ✅ COMPLETE / ACCEPTED

- **Date:** 2026-07-17.
- **Scope:** saved-artifact-only manuscript integration. No dataset loading, HF test access,
  center-2 image read, inference, training, checkpoint access, calibration fitting, threshold
  tuning, model-pair change, second authorization, or exploratory method.
- **Source evidence:** accepted 8B/8C development tables and immutable 9A metrics, tables, reports,
  run state, and sentinels. The integration script verified the saved authorization, prediction,
  metric, sentinel, and existing exp09-figure hashes before and after generation.
- **Manuscript package:** created `docs/FINAL_MANUSCRIPT_DRAFT.md`,
  `docs/FINAL_TITLE_ABSTRACT_KEYWORDS.md`, `docs/FINAL_RESULTS_INTERPRETATION.md`,
  `docs/FINAL_DISCUSSION_AND_LIMITATIONS.md`, `docs/FINAL_CLAIMS_AND_LIMITATIONS.md`,
  `docs/LITERATURE_CITATION_GAPS.md`, `docs/SUBMISSION_PACKAGE_README.md`,
  `docs/FINAL_REPRODUCIBILITY_CHECKLIST.md`, and
  `docs/FINAL_MANUSCRIPT_CONSISTENCY_AUDIT.md`.
- **Tables:** created seven `results/tables/exp09b_*.csv` artifacts covering
  development-versus-final metrics, final model comparison, calibration, operating-point transfer,
  high-confidence false negatives, claims, and the manuscript table/figure manifest.
- **Figures:** created six new matplotlib-only `results/figures/exp09b_*.png` paper-facing figures.
  Visual QA found and corrected an initial operating-point panel that clipped low achieved
  sensitivities; the final figure uses a full 0–1 achieved-metric axis and displays the poor
  transfer honestly. Existing exp09 figures were not modified.
- **Scientific interpretation preserved:** GroupDRO was stronger on development center 1 AUROC
  (0.8956 vs 0.8673), but the matched ERM control outperformed the predeclared GroupDRO primary on
  final center-2 AUROC, AUPRC, accuracy, sensitivity, F1, and false-negative count. GroupDRO
  retained higher specificity and slightly higher precision. The development advantage did not
  generalize to center 2.
- **Calibration interpretation:** frozen temperatures improved held-out ECE, Brier score, and NLL
  for both models without changing hard predictions or total false negatives. Calibrated ERM
  remained better calibrated on center 2.
- **Operating-point and FN interpretation:** all 14 thresholds remained frozen
  `id_val`-selected candidate/non-clinical points and transferred poorly. Calibration sharply
  reduced high-confidence false negatives but did not reduce the underlying total misses.
- **Claims discipline:** no clinical-readiness, all-hospital-robustness, universal method,
  WSI/patient-level, clinically validated threshold, or untested-method superiority claim was
  introduced. No post-test model switching was implied.
- **Implementation:** `scripts/40_make_final_manuscript_package.py` imports only saved-artifact and
  plotting libraries; it imports no torch, torchvision, datasets, or project data/model/training
  module. The permitted command completed with `OVERALL=PASS`.
- **Immutability:** existing exp09 predictions, metrics, authorization JSON, run state, completed
  sentinels, and final-test figures remained unchanged. Attempt count remains 1; no second
  `ood_test` inference is authorized.
- **Outcome:** PASS. Latest completed milestone = **9B**.
- **Exact next recommendation:** literature/citation verification using
  `docs/LITERATURE_CITATION_GAPS.md`, then journal selection/template formatting and completion of
  author, ethics, funding, conflict, contribution, and data/code-availability metadata. Do not
  reopen model development or rerun `ood_test`. Model-based uncertainty remains deferred/optional
  and is not required before manuscript completion.

## Milestone 9C — Verified literature, citation integration, and target-journal shortlist ✅ COMPLETE / ACCEPTED

- **Date:** 2026-07-17.
- **Scope:** saved-manuscript-artifact and authoritative-web research only. No dataset loading, HF
  test access, center-2 image read, model/checkpoint access, inference, training, calibration
  fitting, threshold tuning, authorization/sentinel change, second final-test attempt, or model
  development.
- **Bibliography:** created `references/final_references.bib` with 46 verified records. Metadata
  was checked against DOI, PubMed/PMC, official proceedings, publisher pages, or authoritative
  dataset/project pages. Publication type and peer-review status are explicit.
- **Citation integration:** created `docs/FINAL_MANUSCRIPT_WITH_CITATIONS.md`; replaced all 22
  literature-gap markers; preserved every scientific number, GroupDRO-primary/ERM-control role,
  unfavorable development-to-test reversal, limitations, and forbidden claims.
- **Accuracy narrowing:** narrowed only three source-sensitive claims: human reliance on confident
  errors, conditional limitations of known group labels, and the distinction between established
  external-validation guidance and this study's stricter one-hospital lock recommendation.
- **Audit artifacts:** created `docs/CITATION_INTEGRATION_CHANGELOG.md`,
  `docs/VERIFIED_LITERATURE_REVIEW.md`, `docs/FINAL_CITATION_AUDIT.md`,
  `docs/FINAL_NOVELTY_STATEMENT.md`, and five `results/tables/exp09c_*.csv` reference, claim,
  unresolved-gap, journal, and novelty records.
- **Audit result:** PASS. BibTeX entries 46; direct manuscript citation keys 38; package citation
  keys 46; background/review-only records 8; unresolved gaps 0; duplicate DOI/title records 0.
  `scripts/41_verify_literature_and_citations.py` compiled and passed offline with
  `dataset_loaded=false`, `ood_test_accessed=false`, and `inference_run=false`.
- **Journal research:** official publisher pages were used to rank:
  1. Journal of Pathology Informatics,
  2. Journal of Medical Imaging,
  3. PLOS ONE,
  4. Scientific Reports,
  5. BMC Medical Informatics and Decision Making,
  6. Journal of Imaging Informatics in Medicine,
  7. Artificial Intelligence in Medicine.
- **Immutability:** accepted exp09 authorization, metrics, tables, run state, and sentinels retained
  their baseline hashes. No final evidence artifact changed; attempt count remains 1.
- **Outcome:** PASS. Latest completed milestone = **9C**.
- **Exact next recommendation:** after explicit approval, select the target venue and perform
  journal-specific formatting and final submission QA. Complete author, ethics, funding,
  conflicts, contributions, data/code release, and reporting-checklist metadata. Do not rerun
  `ood_test`, reopen model development, or add exploratory methods.

## Milestone 9D — Journal of Pathology Informatics formatting and submission QA ✅ COMPLETE / ACCEPTED

- **Date:** 2026-07-17.
- **Scope:** saved-artifact-only publication formatting and QA. No dataset/HF split/center-2 image,
  checkpoint, model, inference, training, calibration fitting, threshold tuning, authorization or
  sentinel access, second final-test attempt, scientific-result change, or scientific-figure
  generation/editing.
- **Journal and identity:** Journal of Pathology Informatics, Original Research Article. Sole and
  corresponding author: Jishan Islam Maruf (`jishanislammaruf62@gmail.com`), with the binding IUBAT
  affiliation/address and institutional telephone. No co-author or ORCID was added.
- **Submission package:** created `submission/jpi/` with identity-bearing title page,
  double-anonymized editable manuscript, PDF proof, cover letter, highlights, declaration of
  interest, author declarations, editable figure captions/tables, supplementary material, CLAIM
  2024 checklist, upload README/checklist, figure and code manifests, and a sanitized local code
  ZIP. No file was uploaded or submitted.
- **Manuscript format:** accepted title retained; abstract **221 words**; exactly **7** keywords;
  five highlights with lengths **68, 69, 63, 64, 69**; numbered JPI sections; five editable main
  tables; six accepted figures; 30 first-appearance AMA-style references; complete data/code,
  ethics/consent, funding, competing-interest, and generative-AI declarations.
- **Figures:** six exp09b figures copied losslessly to TIFF and PNG without resampling or pixel
  change, tagged at 300 dpi. Dimensions and visual-QA PASS status are recorded in
  `submission/jpi/JPI_Figure_Manifest.csv`. No graphical abstract or AI-generated image was made.
- **Anonymization:** PASS. Author name, email, affiliation, acknowledgement, named CRediT content,
  personal paths, comments, tracked changes, hidden text, and custom properties are absent from the
  anonymized DOCX/PDF; creator/last-modified-by/author metadata is generic `Anonymous`.
- **Code release:** 91-file reproducible local archive, credential/path scan PASS; excludes source
  images, caches, `.venv`, checkpoints, predictions, authorization records, sentinels, and JPI
  build helpers containing author metadata. ZIP SHA256
  `27FF9ED54AA4C4A1898216B8E098D6716DB0709FC81D6845E835FA6ABDE1ED2C`.
- **Scientific integrity:** PASS. The unfavorable center-2 GroupDRO result remains visible;
  GroupDRO/ERM roles are unchanged; calibration remains confidence rather than error correction;
  candidate operating points remain non-clinical; no universal or clinical-readiness claim was
  introduced. Accepted exp09 authorization, metric, table, and run/sentinel hashes remained
  unchanged; attempt count remains 1.
- **Environment note:** `python-docx` 1.2.0 and `lxml` 6.1.1 were installed in `.venv`. Because
  Word/LibreOffice/Pandoc was unavailable, the 11-page PDF proof used a document-only ReportLab
  fallback and was inspected page by page. Final Microsoft Word/JPI portal rendering remains a
  required human check.
- **Audits:** `submission/jpi/JPI_Anonymization_Audit.md`,
  `docs/JPI_FINAL_SUBMISSION_AUDIT.md`, and
  `results/logs/exp09d_jpi_submission_package_report.txt` all report PASS.
- **Outcome:** PASS. Latest completed milestone = **9D**.
- **Exact next recommendation:** human visual review in Microsoft Word, re-verification of current
  APC/waiver and portal fields, and manual submission through the JPI submission system. Do not
  rerun `ood_test`, reopen model development, or begin another experiment/manuscript rewrite.

## Milestone 9E — Sanitized GitHub publication ✅ COMPLETE / ACCEPTED

- **Date:** 2026-07-18.
- **Scope:** repository sanitation, portability/documentation edits, Git initialization,
  security/privacy/large-file audits, and GitHub publication only. No dataset/HF split/image,
  model/checkpoint, inference, training, calibration fitting, threshold tuning, or final-test
  action occurred; scripts 38 and 39 were not run.
- **Repository:** `https://github.com/MaruflRana/histopath-domain-shift-confidence`, public,
  default branch `main`. Initial publication commit:
  `9907d6efb546c529fa5226fd0182edac17878b86`.
- **Published content:** reproducible source, scripts, configurations, documentation, references,
  tests, aggregate metrics/tables, PNG figures, manuscript source, and sanitized JPI materials.
- **Excluded content:** datasets/source images, HF/local caches, Python environments, checkpoints,
  raw predictions/logits, authorization and one-shot run-state/sentinel files, credentials and
  certificates, local archives/code-package ZIP duplication, redundant TIFF copies, and
  identity-bearing documents containing private telephone information.
- **Audits:** staged security/credential, privacy/email, absolute-path, portability, file-size,
  ignored-file, protected-hash, remote-tree, and rendered-README audits passed. The initial commit
  contained 324 files; no committed file exceeded 10 MB, and the largest blob was 171,631 bytes.
- **Immutability:** all nine protected exp09 evidence/governance files retained their recorded
  SHA256 hashes. Authorization/run-state/sentinel files were not committed. Attempt count remains
  1 and `ood_test` was not accessed or rerun.
- **Git safety:** the target remote was empty before publication, no remote history was
  overwritten, and no force push was used.
- **Outcome:** PASS. Latest completed milestone = **9E**.
- **Exact next recommendation:** complete the human Microsoft Word/JPI portal review, re-check
  current APC/waiver and submission fields, and submit manually. Do not reopen model development,
  rerun `ood_test`, or begin another experiment/manuscript rewrite.
