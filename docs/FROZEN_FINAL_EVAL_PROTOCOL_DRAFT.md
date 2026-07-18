# FROZEN FINAL-EVALUATION PROTOCOL — DRAFT (Milestone 8D)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology.

> **DRAFT — NOT AUTHORIZED.** This is the proposed protocol for the single, one-time locked
> `ood_test` evaluation. Writing this draft does **not** authorize the evaluation and does **not**
> touch `ood_test`. The final test may run **only** after the user pastes the authorization phrase
> in `docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md`. Milestone 8D performs no training, no inference, no
> calibration fitting, and reads no `ood_test` / center-2 data.
>
> **Terminology:** the two checkpoint artifacts and their paths below are frozen and unchanged.
> Their use as the final-test primary/control pair is **proposed**, not committed or authorized.
> After the user explicitly completes the authorization template, these exact unchanged checkpoint
> artifacts become the committed final-test pair.

---

## 1. Frozen checkpoint artifacts; proposed final-test pair

| Role | Checkpoint (frozen; loaded read-only when later authorized) |
|------|-------------------------------------------------------------|
| **PROPOSED PRIMARY model** | `results/checkpoints/exp07f_groupdro_resnet18/best.pt` (7F GroupDRO-by-center) |
| **PROPOSED MATCHED control** | `results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt` (7F center-stratified ERM) |

- **Checkpoint artifacts and paths are FROZEN.** No retraining, fine-tuning, replacement, or weight
  update of any kind.
- **Final-test pair is PROPOSED, not yet committed.** Explicit completion of
  `docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md` commits these exact artifacts for the authorized run.
- **No model selection after seeing `ood_test`.** Once authorized, the primary model and control
  cannot be switched based on the test result.
- Context-only models (old plain ERM, 7A, 7D, 7E) are **excluded** from the final test unless
  separately and explicitly authorized later; they are not controlled comparators.

## 2. Final-test data (LOCKED — DO NOT ACCESS IN 8D)

| Logical split | HF split | Center | Expected n | Status |
|---------------|----------|--------|-----------:|--------|
| `ood_test` | `test` | {2} | 85054 | **LOCKED — not accessed in 8D; single one-time run only after authorization** |

- The locked split mapping (`PROJECT_PROTOCOL.md` §3 / `CLAUDE.md` §4) is unchanged.
- `ood_test` is read **exactly once**, only after written authorization. No `ood_test` access
  occurs in Milestone 8D.

## 3. Frozen prohibitions for the final test

- **No retraining** / no weight update.
- **No threshold tuning on `ood_test`.** Operating thresholds are selected on `id_val` only,
  **before** the test is read.
- **No calibration fitting on `ood_test`.** Calibration uses the pre-frozen 8C temperatures only.
- **No model selection after seeing `ood_test`** (no post-hoc model / epoch / checkpoint switch).
- **No more than one `ood_test` run** without a separate explicit override.

## 4. Outputs to report (raw AND calibrated)

Both **raw** and **calibrated** outputs must be reported for each model if the final test is later
authorized. The temperature values below are unchanged development artifacts; the policy to apply
them at final test remains proposed until authorization. Calibrated probabilities use these
**pre-frozen 8C development temperatures** (applied, not refit):

- **GroupDRO T = 2.974907**
- **ERM T = 3.496293**

Temperature scaling is argmax-invariant (it rescales confidence, not the decision), so raw and
calibrated share the same hard-prediction metrics.

## 5. Threshold-free metrics (required)

- **AUROC**
- **AUPRC**

(tumor = positive class = 1)

## 6. Default-threshold metrics (required; threshold = 0.5)

- accuracy
- sensitivity (recall, tumor)
- specificity
- precision
- F1
- **confusion matrix** (tn / fp / fn / tp)

## 7. Candidate operating points (candidate-only; NOT clinical thresholds)

- Thresholds are selected on **`id_val` only** (never on `ood_test`), reusing
  `src/evaluation/metrics_operating_points.py`, then applied to `ood_test`.
- **Target specificities:** 0.80, 0.85, 0.90, 0.95.
- **Target sensitivities:** 0.80, 0.90, 0.95.
- These are **CANDIDATE development-selected** operating points reported for context — **not**
  committed clinical thresholds. Committing a deployable operating point is a separate gated step.

## 8. High-confidence false-negative audit (required)

- `label==1 & pred==0 & confidence ≥ t`, at confidence thresholds **t = 0.90, 0.95, 0.99**.
- Reported for **both raw and calibrated** confidence.

## 9. Calibration metrics (only after authorization)

- **ECE, Brier, NLL** on the final test, raw vs calibrated (using the pre-frozen 8C temperatures;
  **no fitting on `ood_test`**). Computed **only** once the final test is authorized and run.

## 10. Per-center / metadata

- `ood_test` is a single center (center 2); report overall metrics plus any
  available metadata breakdown. Required metadata columns must be preserved (see
  `docs/FINAL_TEST_OUTPUT_SCHEMA.md`).

## 11. No final claims before `ood_test` is run

No final-performance, generalization, calibration-generalization, or clinical claim may be made
until the authorized `ood_test` run is complete. Until then, all evidence is development-stage.
This protocol is a **draft**; the proposed final-test pair and policies are not committed or
authorized, and the document does **not** authorize the test.
