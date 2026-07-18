# PRE-FINAL-EVALUATION AUDIT (Milestone 7I)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology:
Calibration, Uncertainty, and Selective Prediction

_Milestone 7I — pre-final-evaluation decision audit. **Audit / documentation only**: no
training, no model inference, no calibration fitting, no MC-dropout, no ensemble, no
CORAL/DANN, no `ood_test`, no final-performance claim, no committed clinical threshold. Every
number below is copied/derived from already-saved development-stage artifacts (Milestones
4B–7H). `ood_test` (center 2) is **LOCKED and unread**._

---

## 1. Current project status after Milestone 7H

Milestones 1 → 7H are complete and accepted. The project has produced a **development-stage
reliability + domain-generalization (DG) audit** on **capped balanced subsets** (id_val
n=450/500, ood_val n=2000, center 1), with ResNet-18 and short training. The deterministic
reliability story (5A calibration → 5B uncertainty → 5C selective prediction → 6A reporting →
6B manuscript audit) is complete. The DG story (7A/7D/7E stain augmentation → 7F-0 feasibility
→ 7F-1 center-stratified cache rebuild → 7F matched ERM + GroupDRO → 7G calibration/operating
audit → 7H consolidated reporting) is complete.

**Headline development findings (NOT final performance):**
- Hospital-domain shift severely degrades reliability in the capped development setting
  (plain-ERM context: id_val→ood_val accuracy 0.964→0.620, AUROC 0.993→0.770, sensitivity
  0.944→0.405).
- In the **fair controlled DG comparison** (same sound center-stratified 7F-1 cache),
  **GroupDRO-by-center beat its matched center-stratified ERM control** on `ood_val` (AUROC
  0.7129 vs 0.6180, +0.0949; AUPRC 0.6287 vs 0.5214, +0.1073; 4/4 fixed-specificity sensitivity
  targets). This is the **first DG lever to beat its matched control on OOD ranking**.
- No stain-augmentation lever (7A moderate torchvision, 7D tuned torchvision v2, 7E OD/HED
  stain-space) beat plain-ERM OOD ranking on the old confounded cache.

`ood_test` was never touched in any milestone.

---

## 2. Primary controlled method candidate

**7F GroupDRO-by-center with temperature-scaled confidence** (`exp07f_groupdro_resnet18`;
GroupDRO temperature T=3.6570 from Milestone 7G).

Rationale: it is the only lever that, in a **controlled, matched** comparison, improved OOD
ranking, precision-recall, default sensitivity, and all four candidate fixed-specificity
operating points over its matched control — and it is the **better-calibrated** of the two 7F
models (raw and calibrated). Its confidence output should be reported with the 7G temperature
scaling (development-stage calibration), stated as argmax-invariant.

---

## 3. Matched control

**7F center-stratified ERM** (`exp07f_centerstrat_erm_resnet18`) — trained on the **same** sound
center-stratified 7F-1 cache with matched init/seed/schedule, `id_val_loss` selection only,
`ood_val` read only after training. This is the **only** fair baseline for the GroupDRO claim,
because it isolates the GroupDRO objective from cache/subset differences.

---

## 4. Context-only baselines

These were trained on the **old confounded** balanced cache (center↔label confound; center 4
absent) and/or a different id_val subset. They are **reference/context only** relative to 7F and
must **never** be used as controlled comparators for GroupDRO:

- **old plain ERM** (4A/5A) — ood_val AUROC 0.7696 (highest ranking, but context-only).
- **7A moderate stain augmentation** — ood_val AUROC 0.7492.
- **7D stain augmentation v2 (tuned torchvision)** — ood_val AUROC 0.7397.
- **7E stain-space (OD/HED) augmentation** — ood_val AUROC 0.7317 (strongest stain variant at
  fixed-specificity operating points; lowest AUROC).

---

## 5. What evidence SUPPORTS GroupDRO

All development-stage, on the capped balanced `ood_val` subset, in the controlled 7F setup:

- **Higher OOD ranking:** ood_val AUROC 0.7129 vs matched ERM 0.6180 (+0.0949).
- **Higher OOD precision-recall:** ood_val AUPRC 0.6287 vs 0.5214 (+0.1073).
- **More usable operating points:** GroupDRO ood_val sensitivity ≥ matched ERM at **4/4**
  id_val-selected fixed-specificity candidate targets (0.306/0.241/0.148/0.057 vs
  0.021/0.015/0.013/0.008 at spec 0.80/0.85/0.90/0.95).
- **Fewer missed tumors at 0.5:** default FN 930 vs 990; default sensitivity 0.0700 vs 0.0100.
- **Better calibrated:** lower ECE/Brier/NLL than matched ERM both raw and after temperature
  scaling (ECE raw 0.4258 vs 0.4984; calibrated 0.2529 vs 0.3698).
- **Fewer high-confidence missed tumors:** conf≥0.90 FN 688→25 (calibrated) vs ERM 950→249.

---

## 6. What evidence does NOT support GroupDRO

- **No controlled win over old plain ERM.** Old plain ERM's higher ood_val AUROC (0.7696) is a
  **reference gap from a different (confounded) cache and id_val subset — NOT a controlled
  loss**, and equally **NOT a controlled win** for GroupDRO. The only controlled comparison is
  7F ERM vs 7F GroupDRO.
- **Low absolute default operating behavior.** At the fixed 0.5 argmax, GroupDRO ood_val
  sensitivity is only 0.07 (930 tumors missed of 1000) — the improvement is in ranking and
  candidate operating points, not in a deployable default.
- **No worst-center id_val improvement.** GroupDRO up-weighted center 4, whose id_val loss
  stayed highest; it did not improve worst-center id_val behavior.
- **Capped subset only.** ood_val n=2000, 50/50 balanced — not the full center-1 distribution or
  prevalence.
- **No held-out generalization estimate.** `ood_test` (center 2) is unread; no final number
  exists.

---

## 7. What remains development-only

- All classification / AUROC / AUPRC / sensitivity / specificity numbers (capped balanced
  subsets).
- The GroupDRO-vs-ERM controlled result (development-stage, capped).
- All calibration results (7G temperatures fitted **and** evaluated on the same `ood_val`
  subset — optimistic; reviewer risk R7 open).
- All candidate operating points (id_val-selected; **none committed**).
- The context-only stain-augmentation comparisons.

---

## 8. What is NOT ready for final `ood_test`

- **No frozen final model** — GroupDRO is the primary *candidate*, not a committed final model.
- **No committed operating threshold** — only candidate id_val-selected points exist.
- **No validated calibration policy** — calibration is fit/eval on the same `ood_val` subset;
  it is not validated on an independent split (R7).
- **No full-scale evidence** — only capped balanced subsets (R1, R4 open).
- **No breadth of DG methods** — only GroupDRO; no CORAL/DANN (R3 residual).

`ood_test` must not be touched until a final model, calibration policy, and operating point are
frozen and explicitly authorized in writing.

---

## 9. Is a full `ood_val` / full development evaluation recommended before `ood_test`?

**Yes — recommended before the final test if compute/time allows (Path A).** The current OOD
conclusions rest on a 2000-sample balanced subset (R1, R4). Evaluating the frozen candidate
(7F GroupDRO) and its matched control on **larger / full `id_val` and full `ood_val`** (center 1)
would substantially strengthen the development evidence **without touching `ood_test`**, and
would de-risk the single locked test run. This is a development-stage evaluation only; `ood_val`
remains authorized for development use. If compute/time does not allow, proceed to manuscript
skeleton (Path E) around the current controlled evidence, keeping the test locked.

---

## 10. Is model-based uncertainty necessary before `ood_test`?

**No — not necessary; remains DEFERRED / optional.** The deterministic reliability story
(entropy/confidence + temperature scaling, 5A–5C/6A) is self-contained and addresses a MEDIUM
risk (R8). MC-dropout / deep ensembles would strengthen the uncertainty narrative but are **not
required** to freeze the primary candidate or to run the final test. They require inference (and,
for ensembles, additional training) and should be run only on explicit approval, and only after
this 7I decision.

---

## 11. Are additional DG methods (CORAL/DANN) necessary before `ood_test`?

**No — not necessary before the test; optional for reviewer-risk breadth (Path C).** GroupDRO
already provides the first positive controlled DG lever (R3 reduced HIGH→partly-mitigated).
CORAL/DANN would broaden the DG comparison and further reduce R3, but they require training and
add scope. They are optional and gated; not required to freeze the candidate or run the final
test. If added, they must use the sound center-stratified 7F-1 caches (never the old confounded
caches).

---

## 12. Recommended next path

**Path A (full development-scale evaluation planning) is preferred if compute/time allows;
otherwise Path E (manuscript skeleton).** Neither touches `ood_test`. See
`docs/NEXT_EXPERIMENT_DECISION_TREE.md` for the full path analysis. The default next action is a
**decision**, not automatic model-based uncertainty.

---

## 13. Explicit `ood_test` locked statement

`ood_test` (center 2, HF `test` split, expected n=85,054) remains **FULLY LOCKED and unread**.
It may be used **once**, for the single final locked evaluation, **only after explicit written
authorization**, and only after a final model, calibration policy, and operating point are
frozen. No 7I artifact references or reads `ood_test`. No result to date is a final or clinical
performance claim.
