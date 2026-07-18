# FULL DEVELOPMENT-SCALE EVALUATION PROTOCOL (Milestone 8A)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology:
Calibration, Uncertainty, and Selective Prediction

_Milestone 8A — **planning / protocol only**. This document specifies **what a later
execution milestone (8B) would do**; it does **not** run anything. **No training, no model
inference, no calibration fitting, no MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`,
no final-performance claim, and no committed clinical threshold** are performed in 8A. Every
number quoted here is copied from already-saved development-stage artifacts (Milestones
4B–7I). `ood_test` (center 2) is **LOCKED and unread**._

---

## 0. Planning vs execution boundary (read first)

- **8A (this milestone) = PLANNING.** It writes docs, tables, and a report describing the
  intended full development-scale evaluation. It performs no inference and touches no
  checkpoint, prediction CSV, or cache.
- **8B (a separate, gated milestone) = EXECUTION.** It would run the inference described
  below on larger/full development splits. 8B requires **inference only, no training**, and
  still **never** touches `ood_test`. 8B must not begin without an explicit written go-ahead.

Anything in this document phrased as "will", "would", or "the executor" refers to 8B, not 8A.

---

## 1. Purpose of a full development-scale evaluation

All out-of-distribution (OOD) conclusions to date rest on a **capped balanced `ood_val`
subset** (n=2000, 50/50) and a capped `id_val` subset (n=450 center-stratified / n=500 old).
These support only a development-stage reliability/DG audit (reviewer risks R1, R4, R9). The
full development-scale evaluation would:

1. Re-evaluate the **frozen 7F models** (GroupDRO primary candidate + matched center-stratified
   ERM control) on **larger / full development splits** — full or larger `id_val` (centers
   {0,3,4}) and full or larger `ood_val` (center 1) — using **inference only** (weights
   unchanged).
2. Produce a **more trustworthy development OOD estimate** than the 2000-sample balanced
   subset, directly attacking R1 (capped subsets) and partially R4 (no full eval) and R9
   (subsetting bias).
3. Keep the controlled comparison intact: the scientific claim remains **7F center-stratified
   ERM vs 7F GroupDRO**; context-only models are optional and never controlled comparators.

This is **development-stage** work: `ood_val` is authorized for development-stage OOD
analysis only. It is **not** the final locked evaluation and yields **no** final-performance
claim.

## 2. Why it must occur before the final `ood_test`

- The single `ood_test` (center 2) run is **one-time and irreversible**. Running it on a
  candidate whose OOD behavior is only known from a 2000-sample subset would risk "burning"
  the reserved test on an under-characterized configuration.
- A larger/full `ood_val` estimate **de-risks** the locked test: it strengthens the
  development evidence for the frozen candidate and its control, and surfaces any instability
  (e.g. sensitivity/specificity drift at scale) **before** the test is unlocked.
- The pre-final decision audit (Milestone 7I) explicitly recommends this as **Path A**
  ("full development-scale evaluation planning"), preferred **if compute/time allows**, ahead
  of any final test and ahead of the deferred model-based-uncertainty work.
- It changes nothing that is frozen: split mapping, `ood_test` lock, `ood_val` development-only
  status, and the "old confounded caches barred from DG" rules all remain in force.

## 3. Exact models to evaluate

### Required (controlled final story)

| Model | Role | Why required |
|---|---|---|
| **7F GroupDRO-by-center** (`exp07f_groupdro_resnet18`) | **PRIMARY controlled candidate** (with development-stage temperature-scaled confidence, 7G T=3.6570) | The only lever that beat its matched control on OOD ranking; the model whose full-scale development behavior we most need to characterize before any test. |
| **7F center-stratified ERM** (`exp07f_centerstrat_erm_resnet18`) | **MATCHED control** | The only fair baseline for the GroupDRO claim (same sound 7F-1 cache, matched init/seed/schedule, `id_val_loss` selection only). Required so the controlled comparison scales with the candidate. |

### Optional (context-only)

| Model | Role | Status |
|---|---|---|
| old plain ERM (4A/5A) | context-only baseline | optional |
| 7A moderate stain/color augmentation | context-only baseline | optional |
| 7D tuned stain/color augmentation v2 | context-only baseline | optional |
| 7E OD/HED-style stain-space augmentation | context-only baseline | optional |

**Context-only models are NOT required for the controlled final story.** They were trained on
the **old confounded** balanced cache (center↔label confound; center 4 absent) and/or a
different `id_val` subset, so they can **never** serve as controlled comparators for GroupDRO.
Re-evaluating them at full scale only situates GroupDRO among prior development levers (as
reference/context). If compute/time is limited, **skip them**; the controlled claim (7F ERM vs
7F GroupDRO) does not depend on them. If they are evaluated, every output must retain the
`context_only` label and the "reference gap, not a controlled loss" caveat.

## 4. Exact frozen checkpoints

The executor (8B) must load exactly these two checkpoints for the required models and must not
modify them:

- **Primary candidate:** `results/checkpoints/exp07f_groupdro_resnet18/best.pt`
- **Matched control:** `results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt`

Both exist (~134 MB each) and were reload-verified in Milestone 7F (id_val_loss reproduced to
1e-5). Each checkpoint bundles `model_state_dict`, `optimizer_state_dict`, `epoch`, `metrics`,
`config`; the classifier must be rebuilt from the checkpoint's stored `config` (as in
`scripts/10`/`27`). Optional context-only checkpoints, only if evaluated:
`results/checkpoints/exp04a_dev_train_resnet18/best.pt`,
`.../exp07a_stainaug_resnet18/best.pt`, `.../exp07d_stainaug_v2_resnet18/best.pt`,
`.../exp07e_stainspace_resnet18/best.pt`.

> Note: the 7F models were trained on the **center-stratified** caches; the context-only models
> were trained on the **old confounded** cache. This is exactly why they are not matched and
> why context-only outputs must never be framed as controlled comparisons.

## 5. Development splits (LOCKED mapping; `ood_test` excluded)

The locked split mapping (`PROJECT_PROTOCOL.md` §3 / `CLAUDE.md` §4) is unchanged. 8B evaluates
only development splits:

| Logical split | HF split | Centers | Verified total | Role in 8B |
|---|---|---|---|---|
| `id_val` | `validation` | {0,3,4} | 33,560 | full or larger in-distribution evaluation (threshold selection + per-center analysis) |
| `ood_val` | `validation` | {1} | 34,904 | full or larger out-of-distribution (center-1) development evaluation |
| `train` | `train` | {0,3,4} | 302,436 | **not evaluated** (training split; listed for context only) |
| `ood_test` | `test` | {2} | 85,054 | **LOCKED — NOT used in 8A or 8B** |

- **Full or larger `id_val`** (centers {0,3,4}): used for candidate threshold selection and
  per-center analysis.
- **Full or larger `ood_val`** (center 1): the development OOD estimate.
- **No `ood_test`** (center 2): excluded from this milestone and from 8B. It is reserved for
  the single final locked evaluation only, after separate written authorization.

"Full or larger" means the executor may run the complete `id_val`/`ood_val` splits, or a
substantially larger stratified evaluation set than the current caps, per the 8B resource
decision (see `FULL_DEVELOPMENT_EVALUATION_RESOURCE_PLAN.md`). Any capping that remains must be
disclosed in the 8B outputs (no silent truncation).

## 6. Threshold policy

- Candidate operating thresholds are selected **only on `id_val`** (matched fixed-specificity
  targets 0.80/0.85/0.90/0.95 and, optionally, matched fixed-sensitivity targets 0.80/0.90/0.95),
  reusing `src/evaluation/metrics_operating_points.py`.
- The **same** `id_val`-selected threshold is applied to `ood_val` for development comparison.
- These are **CANDIDATE** development operating points only. **No final/clinical threshold is
  committed** in 8A or 8B. Committing a deployable operating point is a separate, later, gated
  step (Not-Frozen decision N-2).

## 7. Calibration policy

- Any temperature scaling performed in 8B is **development-only** (reusing
  `src/calibration/temperature_scaling.py`), fit on `ood_val` logits exactly as in 5A/7C/7G.
  8B may also simply **reuse the already-saved 7G temperatures** (ERM T=3.9239, GroupDRO
  T=3.6570) rather than refitting.
- Calibration is **argmax-invariant**: it changes confidence magnitudes, not hard predictions.
- **No final independent-test calibration claim** is made. Fitting-and-evaluating on the same
  `ood_val` remains reviewer risk **R7 (open)**; any "calibration generalizes" claim requires a
  separate calibration/evaluation split and remains a **Not-Frozen** decision (N-3). 8A performs
  **no calibration fitting at all**.

## 8. Required metrics (produced by 8B)

For each required model, on `id_val` and `ood_val`:

- **Classification metrics** — accuracy, balanced accuracy, precision, F1 (tumor = 1 positive).
- **AUROC / AUPRC** — threshold-free ranking.
- **Sensitivity / specificity** — at the default 0.5 threshold and at candidate operating points.
- **Confusion matrix** — tn / fp / fn / tp.
- **Fixed-specificity candidate operating points** — `id_val`-selected, applied to `ood_val`
  (spec 0.80/0.85/0.90/0.95); optionally fixed-sensitivity points (0.80/0.90/0.95).
- **Calibration metrics** — ECE / Brier / NLL, raw vs (development-only) temperature-scaled.
- **High-confidence false negatives** — `label==1 & pred==0 & confidence≥t` at t=0.90/0.95/0.99,
  raw vs calibrated.
- **Per-center `id_val` analysis** — overall + per-center {0,3,4} metrics (worst-center behavior).
- **Optional:** deterministic uncertainty / error-detection metrics and risk–coverage /
  selective-prediction tables (reusing `src/evaluation/metrics_uncertainty.py` /
  `metrics_selective.py`). These are **optional** in 8B and **do not** include MC-dropout or
  ensembles (model-based uncertainty remains DEFERRED / optional, Path D).

All metrics are **development-stage** and carry the capped/large-development-subset scope tag.
None is a final or clinical performance claim.

## 9. Explicit `ood_test` locked statement

`ood_test` (center 2, HF `test` split, expected n=85,054) remains **FULLY LOCKED and unread**.
It is **not** used in Milestone 8A (planning) or Milestone 8B (execution). It may be used
**once**, for the single final locked evaluation, **only after** a final model, calibration
policy, and operating point are frozen **and** explicit written authorization is given. No 8A
artifact reads or references `ood_test` as data.
