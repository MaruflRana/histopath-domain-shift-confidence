# REVIEWER RISK AUDIT

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in
Histopathology: Calibration, Uncertainty, and Selective Prediction

Anticipated reviewer criticisms of a manuscript built on the **development-stage, capped
balanced subset** results through Milestone 6A, with `ood_test` (center 2) still **LOCKED**.
Each risk has a severity (low / medium / high), a mitigation, and a fix-now-or-later
disposition. "Fix later" items are gated behind explicit authorization and are **not**
started in this milestone.

Severity reflects likelihood × impact on acceptance if unaddressed.

---

## R1 — Capped balanced subsets are not a real evaluation
- **Severity:** HIGH.
- **Why raised:** id_val n=500, ood_val n=2000, and 50/50 class balance do not reflect the
  full center distributions or prevalence; reviewers will not accept these as performance.
- **Mitigation:** label every artifact "development / capped subset"; frame the paper as an
  audit/protocol contribution; run full `ood_val` and the reserved `ood_test` before any
  performance claim.
- **Fix:** LATER (requires full-split evaluation — gated).

## R2 — ID-trained model may not be strong enough
- **Severity:** MEDIUM–HIGH.
- **Why raised:** ResNet-18 with short capped training may be under-fit; degradation could be
  partly weakness, not just shift. A reviewer will ask whether a stronger model shifts less.
- **Mitigation:** disclose the development configuration; note ID performance is high
  (id_val acc 0.964) so the model is not trivially weak; plan a full-data/tuned model.
- **Fix:** LATER (stronger/full training — gated).

## R3 — No domain-generalization baseline
- **Severity:** HIGH → **PARTLY MITIGATED / REDUCED** (residual MEDIUM) after Milestones 7F/7G/7H.
- **Why raised:** the field expects comparison to DG methods (GroupDRO, CORAL, DANN, etc.);
  without one, "shift hurts" is known and superiority cannot be claimed.
- **Update (7F/7G/7H):** a dedicated DG method — **GroupDRO-by-center** — was implemented
  (`src/training/groupdro.py`) and, in the **controlled 7F setup** (matched center-stratified ERM
  vs GroupDRO on the same sound 7F-1 cache), **GroupDRO beat its matched ERM control on `ood_val`**
  (AUROC 0.7129 vs 0.6180; AUPRC 0.6287 vs 0.5214; 4/4 fixed-specificity sensitivity targets) and
  is the better-calibrated model (7G). Milestone 7H consolidated this into paper-facing tables/figures
  and the controlled-vs-context framing.
- **Residual issue (NOT fully closed):** only **GroupDRO** was tested — **no CORAL/DANN**; the
  comparison is development-stage on capped balanced subsets; and there is **no full/final `ood_test`
  evaluation**. So R3 is reduced, not closed.
- **Mitigation:** report GroupDRO as the first positive DG lever in the controlled setup; keep the
  forbidden-claim guard (no superiority over untested DG methods; GroupDRO does not beat old plain ERM
  in a controlled comparison — that is context-only).
- **Fix:** PARTIAL NOW (GroupDRO done, consolidated in 7H); remainder LATER (CORAL/DANN + full/final
  evaluation — gated).

## R4 — No full ood_val / no test evaluation yet
- **Severity:** HIGH.
- **Why raised:** conclusions rest on a 2000-sample ood_val subset; the reserved test is
  unread, so there is no held-out generalization estimate.
- **Mitigation:** the reserved-test discipline is a *feature* (honest protocol); commit to a
  single final `ood_test` run for the camera-ready; report full `ood_val` too.
- **Fix:** LATER (final locked evaluation — gated, one-time).

## R5 — No WSI-level aggregation
- **Severity:** MEDIUM.
- **Why raised:** clinical relevance is at slide/patient level; patch metrics do not
  translate directly to diagnosis.
- **Mitigation:** state patch-level scope explicitly; avoid clinical claims; plan slide-level
  aggregation as future work (metadata `slide`/`patient` already present).
- **Fix:** LATER.

## R6 — No external dataset
- **Severity:** MEDIUM.
- **Why raised:** single-dataset (Camelyon17-WILDS) results may not transfer; external
  validation strengthens generalization claims.
- **Mitigation:** scope claims to Camelyon17-WILDS; note center-2 already acts as an unseen
  domain; list external cohorts as future work.
- **Fix:** LATER.

## R7 — Calibration fitted and evaluated on the same ood_val subset
- **Severity:** HIGH.
- **Why raised:** temperature was fit on `ood_val` logits and calibration metrics reported on
  the same subset → optimistic, potentially leaky calibration estimate.
- **Mitigation:** disclose prominently in Limitations; split `ood_val` into calibration vs
  evaluation folds (or fit on one hospital, evaluate on another) before claiming calibration
  *generalizes*; keep current claim narrow ("improves calibration on this subset").
- **Fix:** PARTIAL NOW (documented as a limitation in `CLAIMS_AND_LIMITATIONS.md`), method
  fix LATER (gated).

## R8 — Deterministic uncertainty only
- **Severity:** MEDIUM.
- **Why raised:** single-forward-pass entropy/confidence is a weak uncertainty proxy;
  reviewers expect MC-dropout / ensembles for an uncertainty paper.
- **Mitigation:** scope the claim to *deterministic* uncertainty; present the ID-vs-OOD gap
  as motivation for model-based uncertainty (explicitly deferred/optional).
- **Fix:** LATER (deferred by protocol — gated).

## R9 — Possible class-order / cache sampling bias
- **Severity:** MEDIUM.
- **Why raised:** the HF stream is label-ordered and balanced subsets are built by scanning
  to a fixed cap with a seed; a reviewer may worry the cached subset is a biased draw.
- **Mitigation:** document the label-ordered-stream handling and seeded balanced caching
  (already in `DATASET_DECISIONS.md` / `SESSION_STATE.md`); report seed and cache provenance;
  plan a full-split evaluation that removes subsetting entirely.
- **Fix:** PARTIAL NOW (provenance already documented), full removal LATER via full-split
  evaluation.

---

## Severity roll-up

| Severity | Risks |
|---|---|
| HIGH | R1 (capped subsets), R4 (no test/full eval), R7 (calibration same-subset) |
| MEDIUM | R2 (model strength — med/high), R3 (no DG baseline — **reduced from HIGH after 7F/7G/7H: GroupDRO done, CORAL/DANN + final test remain**), R5 (no WSI), R6 (no external), R8 (deterministic only), R9 (sampling bias — partly mitigated by 7F-1) |
| LOW | — (none currently rated low; the honest-scope framing is what keeps them from being fatal) |

## Fix-now vs fix-later summary

- **Fix now (documentation / scoping — done in this milestone):** disclose R1, R2, R7, R8,
  R9 as explicit limitations; enforce forbidden-claim guards (`CLAIMS_AND_LIMITATIONS.md`,
  `report_claims_audit.csv`); apply the development-scope caption to every artifact.
- **Fix later (gated experiments — not started, need authorization):** R1/R4 full-split and
  final `ood_test` evaluation; R2 stronger/full training; **R3 remaining DG breadth (CORAL/DANN)
  and full/final evaluation** (GroupDRO itself is done and consolidated in Milestone 7H); R5 WSI
  aggregation; R6 external cohort; R7 separate calibration split; R8 model-based uncertainty.

**Milestone 7H update:** R3 is **reduced from HIGH to partly-mitigated (residual MEDIUM)** because a
GroupDRO DG baseline was implemented and beat its matched center-stratified ERM control in the
controlled 7F setup; R9 is partly mitigated by the 7F-1 center-stratified rebuild. R1, R4, and R7
**remain open** (capped subsets; no full/final `ood_test`; calibration fitted/evaluated on the same
`ood_val` subset). The machine-readable companion is `results/tables/exp07h_reviewer_risk_update.csv`.

**Overall:** the current results are defensible **only** as a development-stage reliability/DG
audit with a reserved test hospital. Submission as a performance/generalization paper is
premature until at least R4 (final locked evaluation) and the remaining R3 breadth are addressed.
