# FINAL-EVALUATION READINESS PACKAGE (Milestone 8D)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology:
Calibration, Uncertainty, and Selective Prediction

> **THIS IS A READINESS / PROTOCOL-FREEZE PACKAGE — NOT A FINAL EVALUATION.** Milestone 8D is
> **documentation / planning only**: no training, no inference, no calibration fitting, no
> MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final test run, and no
> final-performance claim. It decides what is ready, what remains unfrozen, what the final-test
> protocol should contain, and the exact written authorization required before `ood_test` may be
> touched. **`ood_test` remains fully LOCKED and has not been used.**

---

## 1. Current state after Milestone 8C

Milestones 1 → 8C are complete and accepted. The project has produced a **development-stage**
reliability + domain-generalization (DG) evaluation. Milestone 8B evaluated the two frozen 7F
checkpoints (inference only) on the **full** development splits (`id_val` centers {0,3,4}
n=33,560; `ood_val` center {1} n=34,904); Milestone 8C then temperature-scaled those saved
full-split logits on the full `ood_val` as **development-stage** calibration (CSV/logit-only).

Full-development headline (development-stage; NOT final performance):

| split | model | acc | AUROC | AUPRC | sens | spec | FN |
|-------|-------|----:|------:|------:|-----:|-----:|---:|
| ood_val | 7F GroupDRO (primary candidate)      | 0.7847 | **0.8956** | 0.8958 | 0.6407 | 0.9288 | 6271 |
| ood_val | 7F center-stratified ERM (control)   | 0.7675 | **0.8673** | 0.8757 | 0.6074 | 0.9276 | 6852 |
| id_val  | 7F GroupDRO (primary candidate)      | 0.8416 | 0.9323 | 0.9322 | 0.7572 | 0.9243 | 4033 |
| id_val  | 7F center-stratified ERM (control)   | 0.7987 | 0.9085 | 0.9131 | 0.6821 | 0.9129 | 5280 |

`ood_test` (center 2, HF `test`, expected n=85054) has **never** been read in any milestone.

## 2. Why GroupDRO is the primary candidate

7F GroupDRO-by-center is the **only** lever that, in a **controlled, matched** comparison (same
sound center-stratified 7F-1 cache, matched init/seed/schedule, `id_val_loss` selection only),
beat its matched control on OOD. On the full development `ood_val` (8B/8C) it beats the matched
ERM on **every** axis:

- **Ranking:** AUROC 0.8956 vs 0.8673 (+0.0283); AUPRC 0.8958 vs 0.8757 (+0.0201).
- **Default sensitivity / FN:** sens 0.6407 vs 0.6074; FN 6271 vs 6852.
- **Candidate operating points:** GroupDRO `ood_val` sensitivity > ERM at **4/4** id_val-selected
  fixed-specificity targets (0.7743/0.7335/0.6753/0.5936 vs 0.6582/0.6394/0.6143/0.5745 at
  spec 0.80/0.85/0.90/0.95).
- **Reliability:** better-calibrated both raw and calibrated (ECE 0.1560→0.0394 vs 0.1829→0.0695).
- **High-confidence missed tumors:** fewer at every threshold/variant (calibrated conf≥0.90
  533 vs 1693; conf≥0.99 0 vs 5).
- **Worst-center id_val:** at full scale GroupDRO improved center-4 acc 0.6901→0.8031.

No stain-augmentation lever (7A/7D/7E) ever beat plain-ERM OOD ranking; GroupDRO is the first
positive controlled DG lever.

## 3. Why 7F center-stratified ERM remains the matched control

The 7F center-stratified ERM is the **only fair baseline** for the GroupDRO claim: it was trained
on the **same** sound center-stratified 7F-1 cache with matched initialization, seed, and schedule,
selected by `id_val_loss` only, with `ood_val` read only after training. It isolates the GroupDRO
objective from cache/subset differences. Old plain ERM / 7A / 7D / 7E were trained on the **old
confounded** cache (center↔label confound; center 4 absent) and/or a different `id_val` subset —
they are **context-only** and can never serve as controlled comparators for GroupDRO.

## 4. Development evidence from 8B and 8C

- **8B (inference only, full splits):** GroupDRO beat the matched ERM at full/larger development
  scale (ood_val AUROC 0.8956 vs 0.8673; 4/4 fixed-specificity sensitivity; improved worst-center
  id_val). Full-split OOD AUROC is far higher than the capped 2000-sample `ood_val`
  (GroupDRO 0.7129→0.8956; ERM 0.6180→0.8673) — the capped subset understated both models.
- **8C (development-stage calibration, CSV/logit-only):** temperature scaling improved
  ECE/Brier/NLL for both models (GroupDRO ECE 0.1560→0.0394, Brier 0.1782→0.1458, NLL
  0.7591→0.4451; ERM ECE 0.1829→0.0695, Brier 0.2070→0.1727, NLL 1.1911→0.5321) and sharply cut
  high-confidence missed tumors (GroupDRO conf≥0.90 3922→533, conf≥0.99 1679→0; ERM conf≥0.90
  5447→1693, conf≥0.99 3794→5). Calibration is **argmax-invariant** — accuracy/AUROC/AUPRC/
  sensitivity/specificity/FN are unchanged raw vs calibrated.

**Required interpretation (binding):**
- GroupDRO **remains better** than the matched ERM on full-development `ood_val`
  AUROC/AUPRC/sensitivity/FN and reliability.
- Calibration **improves ECE/Brier/NLL and reduces high-confidence FNs**, but does not change
  hard predictions (argmax-invariant).
- 8C calibration is **development-stage**, fit **and** evaluated on the same `ood_val`, so
  **independent-test calibration validity is still unproven** (reviewer risk **R7 open**).
- **No final clinical threshold is committed.**

## 5. What is ready for a final-evaluation protocol

- **Primary candidate model** (7F GroupDRO) and **matched control** (7F center-stratified ERM)
  are identified, and their checkpoints are frozen and reload-verified (7F/8B).
- **Pre-frozen development temperatures** exist for a report-both policy: GroupDRO T=2.974907,
  ERM T=3.496293 (8C, development-stage).
- **A metric plan** (threshold-free + default-threshold + confusion matrix + candidate operating
  points + calibration + high-confidence FN + per-center) is specified and reuses existing
  `src/evaluation/` and `src/calibration/` modules — no new methods.
- **A candidate operating-point family** (id_val-selected, fixed spec 0.80/0.85/0.90/0.95 and
  fixed sens 0.80/0.90/0.95) is defined as candidate-only.
- **A draft final-test protocol, output schema, and stop rules** are written (see the companion
  8D docs). The single-shot `ood_test` protocol is fully specified **but not authorized**.

## 6. What is NOT yet a final result

- No frozen **final model** for `ood_test` (GroupDRO is the primary *candidate*, not committed).
- No committed **final operating threshold** (only candidate id_val-selected points).
- No **validated calibration policy** — 8C calibration is fit/eval on the same `ood_val` (R7).
- No **held-out generalization estimate** — `ood_test` unread.
- No breadth of DG methods — only GroupDRO (no CORAL/DANN); no model-based uncertainty.
- Everything to date is **development-stage on development splits**, not final performance.

## 7. Claims that remain FORBIDDEN before `ood_test`

- Final generalization / performance for an unseen hospital.
- Independent-test calibration validity ("calibration generalizes").
- A final controlled GroupDRO-vs-ERM verdict as final performance.
- Clinical deployment / triage readiness.
- Fixed-threshold clinical use.
- Robustness to all hospital shifts.
- Any conclusion about `ood_test` / center 2.
- Paper-level final metrics quoted from development splits as if final.

(See `docs/CLAIMS_AND_LIMITATIONS.md` and `results/tables/exp08d_claims_unlock_table.csv`.)

## 8. Can a manuscript skeleton begin before `ood_test`?

**Yes.** A manuscript skeleton around the current **development-stage** evidence may begin now,
provided every result carries the development-scope tag and none of the forbidden claims above
appear. The final `ood_test` number is left as a reserved, clearly-marked placeholder ("to be
completed under the frozen 8D protocol after written authorization"). Drafting the skeleton does
**not** unlock or require `ood_test`.

## 9. Explicit `ood_test` locked statement

`ood_test` (center 2, HF `test` split, expected n=85054) remains **FULLY
LOCKED and UNREAD.** Milestone 8D did not access it, did not load the HF test split, and did not
read any center-2 data. It may be used **once**, for the single final locked evaluation, **only
after** a final model, calibration policy, and operating-point policy are frozen **and** explicit
written authorization is given using `docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md`. Milestone 8D does
**not** authorize `ood_test`.
