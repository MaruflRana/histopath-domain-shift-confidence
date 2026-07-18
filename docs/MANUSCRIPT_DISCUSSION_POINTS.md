# MANUSCRIPT DISCUSSION POINTS — DEVELOPMENT ONLY (Milestone 8E)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology.

> Discussion bullets for the eventual manuscript. **All interpretation is development-stage.** The
> reserved test hospital (`ood_test`, center 2) is **LOCKED and unread**; no discussion point may
> assert final generalization or clinical readiness. Numbers are from milestones 8B/8C on the
> development splits.

---

## D1. Main interpretation

- Hospital-domain shift (train centers {0,3,4} → OOD center 1) degrades classifier *reliability*,
  not just raw accuracy: confidence becomes miscalibrated and the model produces high-confidence
  false negatives (confidently missed tumors).
- In a **controlled, matched** development comparison, a domain-generalization objective
  (GroupDRO-by-center) improves OOD-val ranking, sensitivity, and worst-center behavior over its
  matched ERM control, and is better calibrated.
- Temperature scaling improves confidence reliability (ECE/Brier/NLL) and sharply reduces
  high-confidence missed tumors, but does **not** change hard decisions (argmax-invariant).
- Net message: reliability under shift can be *audited and partially improved in development*, but a
  held-out estimate on an unseen hospital is still owed — hence the reserved test.

## D2. Why GroupDRO helps

- GroupDRO minimizes a weighted loss that up-weights the worst-performing center group, so training
  pressure concentrates where generalization is hardest.
- The development evidence matches this mechanism: GroupDRO's largest gain is on the **worst center
  (center 4)** — `id_val` center-4 accuracy 0.6901 → 0.8031, AUROC 0.8323 → 0.8980 — and it carries
  over to higher OOD-val AUROC/AUPRC and sensitivity vs the matched ERM.
- Framing caveat: the *only* controlled comparison is 7F GroupDRO vs 7F center-stratified ERM (same
  sound cache, matched schedule). Old plain ERM / 7A / 7D / 7E used a different (confounded) cache
  and are **context-only**; their numbers must never be reported as GroupDRO winning or losing a
  controlled comparison.

## D3. Why calibration matters

- Under shift the raw model is over-confident (GroupDRO ECE 0.156; ERM 0.183 on `ood_val`).
  Temperature scaling reduces ECE to 0.039 (GroupDRO) and 0.070 (ERM), with matching Brier/NLL gains.
- Calibration makes the *confidence number* trustworthy for downstream use (thresholding,
  abstention, triage prioritization) without altering which patches are called positive.
- Because it is argmax-invariant, calibration is a reliability improvement, **not** an accuracy or
  sensitivity improvement — this must be stated to avoid over-claiming.

## D4. Why high-confidence false negatives matter clinically (in principle)

- A *confidently* missed tumor is the worst failure mode: it is unlikely to be flagged for review.
- Calibration cuts GroupDRO's conf ≥ 0.90 missed tumors 3922 → 533 and conf ≥ 0.99 to 0 on `ood_val`
  — i.e. after calibration essentially no tumor is missed with ≥0.99 stated confidence in this
  development split.
- Clinical-salience caveat: this is a **patch-level, development-stage** observation. It motivates
  confidence-aware review workflows but does **not** establish clinical safety; total misses are
  unchanged, and there is no WSI/patient-level or held-out validation.

## D5. Why `ood_val` is development-only

- `ood_val` (center 1) is used for OOD *decisions* during development — comparison, calibration,
  operating-point exploration — so it cannot also serve as an unbiased held-out estimate.
- Keeping a *separate* reserved hospital (center 2) unread is what makes a future generalization
  claim credible. Reusing `ood_val` for the final number would be leakage.

## D6. Reviewer risks and mitigation (development framing)

- **R1 / R4 (capped/dev splits; no held-out test):** framed as an honest audit with a reserved
  test; the single final evaluation is the planned unlock. Mitigation now: development-scope tags on
  every artifact.
- **R3 (no DG baseline):** reduced by the controlled GroupDRO result; residual = no CORAL/DANN and
  no final test. Do not claim superiority over untested DG methods.
- **R7 (calibration fit+eval on same `ood_val`):** disclosed prominently; the correct fix is a
  separate calibration split (or fit-on-one-hospital/eval-on-another), deferred. Until then only the
  narrow claim "improves calibration on this subset" is allowed.
- **R8 (deterministic + temperature only):** MC-dropout / ensembles are deferred/optional and framed
  as motivated future work, not a gap in the current claim.

## D7. Remaining limitations before `ood_test`

- No held-out generalization estimate (test hospital unread).
- No independent-test calibration validity (R7).
- Only GroupDRO tested; no CORAL/DANN; no model-based uncertainty.
- Patch-level only; single dataset; no external cohort; no committed operating threshold.
- ResNet-18, short training — not tuned/full-data-optimal.

## D8. What the reserved `ood_test` run can and cannot unlock

**Can unlock (if authorized, run once, and supported):**
- A single held-out generalization estimate on one unseen hospital (center 2).
- A controlled final GroupDRO-vs-matched-ERM comparison on that reserved hospital.
- Behavior of the pre-frozen temperatures and candidate operating points on unseen data.

**Cannot unlock (even after the test):**
- Multi-hospital / population generalization from a single test center.
- Independent-split calibration *generalization* proof (still needs a separate calibration split).
- Clinical deployment / triage readiness (out of scope: patch-level, single dataset, no WSI, no
  clinical endpoint).
- A validated/committed clinical operating threshold.

(See `results/tables/exp08d_claims_unlock_table.csv` and
`results/tables/exp08e_manuscript_claims_map.csv` for the exact allowed wording.)
