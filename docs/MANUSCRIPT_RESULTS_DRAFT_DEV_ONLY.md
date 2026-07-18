# MANUSCRIPT RESULTS DRAFT — DEVELOPMENT ONLY (Milestone 8E)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology.

> **DEVELOPMENT-STAGE RESULTS ONLY.** Every number below is from completed development-stage
> milestones — full-development inference (8B) and development-stage calibration (8C) — on the
> development splits `id_val` (centers {0,3,4}, n=33,560) and `ood_val` (center 1, n=34,904).
> **These are NOT final performance and NOT a held-out generalization estimate.** The reserved test
> hospital (`ood_test`, center 2) is **LOCKED and unread**; no test-set numbers exist. Tumor = the
> positive class (1). Caption tag applies to every table.

---

## R1. Setup recap

The primary candidate is the **7F GroupDRO-by-center** ResNet-18; the matched control is the **7F
center-stratified ERM**, trained on the same sound center-stratified cache with matched
init/seed/schedule (Methods M3–M4). Both were selected by `id_val_loss` only; `ood_val` was read
only after training. All results are development-stage.

## R2. Controlled DG comparison on full development splits (Milestone 8B, inference only)

**Table R2 — Overall metrics (threshold 0.5; development-stage).**
_Source: `results/tables/exp08b_overall_metrics.csv`._

| split | model | acc | AUROC | AUPRC | sens | spec | FN | tn/fp/fn/tp |
|-------|-------|----:|------:|------:|-----:|-----:|---:|:-----------:|
| id_val  | 7F GroupDRO (primary candidate)      | 0.8416 | 0.9323 | 0.9322 | 0.7572 | 0.9243 | 4033 | 15668/1284/4033/12575 |
| id_val  | 7F center-stratified ERM (control)   | 0.7987 | 0.9085 | 0.9131 | 0.6821 | 0.9129 | 5280 | 15475/1477/5280/11328 |
| ood_val | 7F GroupDRO (primary candidate)      | 0.7847 | **0.8956** | 0.8958 | 0.6407 | 0.9288 | 6271 | 16209/1243/6271/11181 |
| ood_val | 7F center-stratified ERM (control)   | 0.7675 | **0.8673** | 0.8757 | 0.6074 | 0.9276 | 6852 | 16188/1264/6852/10600 |

**Reading (development-stage).** On the full development `ood_val`, GroupDRO beats its matched ERM
control on every axis: AUROC 0.8956 vs 0.8673 (+0.0283), AUPRC 0.8958 vs 0.8757 (+0.0201), default
sensitivity 0.6407 vs 0.6074, and fewer default false negatives (6271 vs 6852). The advantage is
consistent with — and larger than — the earlier capped-subset result (GroupDRO 0.7129 vs ERM 0.6180
on the 2000-sample `ood_val`); the capped subset understated both models. This remains a
**development-stage controlled comparison**, not a final or held-out result.

## R3. Per-center `id_val` behavior (Milestone 8B)

**Table R3 — Per-center `id_val` (threshold 0.5; development-stage).**
_Source: `results/tables/exp08b_id_val_per_center_metrics.csv`._

| model | center | n | acc | AUROC | sens | spec | loss |
|-------|:-----:|--:|----:|------:|-----:|-----:|-----:|
| GroupDRO | 0 | 6011 | 0.8553 | 0.9338 | 0.7667 | 0.9423 | 0.4341 |
| GroupDRO | 3 | 12879 | 0.8790 | 0.9603 | 0.7852 | 0.9716 | 0.3629 |
| GroupDRO | 4 | 14670 | 0.8031 | 0.8980 | 0.7285 | 0.8756 | 0.4812 |
| ERM (control) | 0 | 6011 | 0.8486 | 0.9149 | 0.7808 | 0.9152 | 0.4645 |
| ERM (control) | 3 | 12879 | 0.8990 | 0.9611 | 0.8385 | 0.9587 | 0.3457 |
| ERM (control) | 4 | 14670 | 0.6901 | 0.8323 | 0.5031 | 0.8720 | 0.6121 |

**Reading (development-stage).** The worst center for both models is center 4. GroupDRO markedly
improves worst-center behavior (center-4 accuracy 0.6901 → 0.8031, AUROC 0.8323 → 0.8980, loss
0.6121 → 0.4812), consistent with its worst-group up-weighting objective. ERM is marginally better on
centers 0 and 3; GroupDRO trades a small amount there for a large worst-center gain.

## R4. Candidate operating points (Milestone 8B; candidate-only, NOT committed)

Thresholds are selected on `id_val` and applied to `ood_val`. **These are candidate development
operating points, not committed clinical thresholds.**
_Source: `results/tables/exp08b_candidate_operating_points.csv`._

**Table R4a — Fixed-specificity (`ood_val` sensitivity).**

| target spec | GroupDRO | ERM (control) |
|------------:|---------:|--------------:|
| 0.80 | **0.7743** | 0.6582 |
| 0.85 | **0.7335** | 0.6394 |
| 0.90 | **0.6753** | 0.6143 |
| 0.95 | **0.5936** | 0.5745 |

GroupDRO `ood_val` sensitivity exceeds ERM at **4/4** fixed-specificity targets.

**Table R4b — Fixed-sensitivity (`ood_val` specificity).**

| target sens | GroupDRO `ood_val` sens / spec | ERM `ood_val` sens / spec |
|------------:|:------------------------------:|:--------------------------:|
| 0.80 | 0.6713 / 0.9140 | 0.6407 / 0.8978 |
| 0.90 | 0.7694 / 0.8525 | 0.6816 / 0.8524 |
| 0.95 | 0.8514 / 0.7644 | 0.7188 / 0.8040 |

## R5. Calibration: raw vs calibrated (Milestone 8C; development-stage, CSV/logit-only)

Temperature scaling on the full `ood_val` logits. **Fit and evaluated on the same `ood_val` split
(reviewer risk R7) — development-stage reliability audit, not an independent-test calibration claim.**
_Source: `results/tables/exp08c_calibration_comparison.csv`, `exp08c_reliability_summary.csv`._

**Table R5 — Reliability, raw → calibrated (`ood_val`; development-stage).**

| model | T | ECE raw→cal | Brier raw→cal | NLL raw→cal |
|-------|--:|:-----------:|:-------------:|:-----------:|
| 7F GroupDRO (primary candidate)    | 2.974907 | 0.1560 → **0.0394** | 0.1782 → 0.1458 | 0.7591 → 0.4451 |
| 7F center-stratified ERM (control) | 3.496293 | 0.1829 → **0.0695** | 0.2070 → 0.1727 | 1.1911 → 0.5321 |

**Reading (development-stage).** Temperature scaling improves ECE, Brier, and NLL for both models.
It is **argmax-invariant**: accuracy, AUROC, AUPRC, sensitivity, specificity, and FN are unchanged
raw vs calibrated (GroupDRO AUROC stays 0.8956; ERM 0.8673). GroupDRO is the better-calibrated model
both raw and calibrated.

## R6. High-confidence false-negative reduction (Milestone 8C)

Confidently missed tumors (`label==1 & pred==0 & confidence ≥ t`) on `ood_val`.
_Source: `results/tables/exp08c_high_confidence_fn_comparison.csv`._

**Table R6 — High-confidence FN counts, raw → calibrated (`ood_val`; development-stage).**

| model | conf ≥ 0.90 | conf ≥ 0.95 | conf ≥ 0.99 |
|-------|:-----------:|:-----------:|:-----------:|
| GroupDRO — raw        | 3922 | 3131 | 1679 |
| GroupDRO — calibrated | **533** | 82 | **0** |
| ERM — raw             | 5447 | 4976 | 3794 |
| ERM — calibrated      | 1693 | 539 | 5 |

**Reading (development-stage).** Calibration sharply reduces the number of *confidently* missed
tumors: for GroupDRO, conf ≥ 0.90 falls 3922 → 533 and conf ≥ 0.99 falls 1679 → 0. GroupDRO has fewer
high-confidence false negatives than ERM at every threshold and variant. Note the total FN count is
unchanged by calibration (argmax-invariance, R5); what changes is how many misses are stated with
high confidence — the clinically salient quantity. This is development-stage evidence on `ood_val`.

## R7. Reserved test — LOCKED (no numbers)

The single locked evaluation on the reserved unseen test hospital (`ood_test`, center 2, expected
n=85,054) has **not** been run and is unread. **No test-set result appears in this manuscript
draft.** A held-out generalization number will exist only after the authorized single-shot run under
the frozen protocol. Until then, all results above are explicitly development-stage.

---

## Universal caveat (attach to every result above)

Development-stage results on development splits (`id_val` centers {0,3,4}, n=33,560; `ood_val`
center 1, n=34,904); ResNet-18; short training. Single OOD-val hospital; reserved test hospital
(center 2) unread. Calibration fit and evaluated on the same `ood_val` split (R7). Deterministic +
temperature-scaled confidence only. Patch-level; no WSI aggregation; no committed operating
threshold. **Not final or clinical performance.**
