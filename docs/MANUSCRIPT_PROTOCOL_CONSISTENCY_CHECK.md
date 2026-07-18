# MANUSCRIPT ↔ PROTOCOL CONSISTENCY CHECK (Milestone 8F)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology.

> **AUDIT / DOCUMENTATION ONLY.** No training, no inference, no calibration fitting, no `ood_test`.
> This document verifies that the 8D final-evaluation protocol package and the 8E manuscript skeleton
> tell the same story, and that the manuscript contains no premature/forbidden claim. The
> machine-readable companion is `results/tables/exp08f_protocol_manuscript_consistency.csv`.

---

## Sources compared

**Protocol side (8D):** `docs/FROZEN_FINAL_EVAL_PROTOCOL_DRAFT.md`, `docs/FINAL_TEST_OUTPUT_SCHEMA.md`,
`docs/FINAL_TEST_STOP_RULES.md`, `docs/FINAL_EVALUATION_READINESS_PACKAGE.md`,
`docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md`, and the `results/tables/exp08d_*` tables.

**Manuscript side (8E):** `docs/MANUSCRIPT_SKELETON_DEV_EVIDENCE.md`,
`docs/MANUSCRIPT_METHODS_DRAFT_DEV_ONLY.md`, `docs/MANUSCRIPT_RESULTS_DRAFT_DEV_ONLY.md`,
`docs/MANUSCRIPT_DISCUSSION_POINTS.md`, and `results/tables/exp08e_manuscript_claims_map.csv`.

**Governance:** `docs/CLAIMS_AND_LIMITATIONS.md`, `docs/REVIEWER_RISK_AUDIT.md`,
`docs/FROZEN_DECISIONS.md`, `docs/FIGURE_TABLE_MANIFEST.md`.

> **Post-audit resolution — Milestone 8G:** Resolved by Milestone 8G: legacy capped-subset wording
> in `CLAIMS_AND_LIMITATIONS.md` and `FIGURE_TABLE_MANIFEST.md` was refreshed. No result changed and
> no `ood_test` access was unlocked. The original 8F follow-up remains recorded below as history.

---

## 1. Same primary model

- **Protocol:** 7F GroupDRO-by-center is the PROPOSED PRIMARY model (§1;
  `exp08d_final_eval_model_plan.csv`).
- **Manuscript:** primary candidate is the 7F GroupDRO-by-center ResNet-18 (Methods M3–M4; Results
  R1/R2).
- **Status: MATCH.**

## 2. Same matched control

- **Protocol:** 7F center-stratified ERM is the PROPOSED MATCHED control (§1).
- **Manuscript:** matched control is the 7F center-stratified ERM, "the sole fair baseline" (Methods
  M3; Results R1/R2).
- **Status: MATCH.**

## 3. Same checkpoint paths

- **Protocol:** `results/checkpoints/exp07f_groupdro_resnet18/best.pt` (primary);
  `results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt` (control) (§1;
  `exp08d_final_eval_model_plan.csv`; stop rule FT-1).
- **Manuscript:** Methods M8 defers to the 8D protocol draft (`docs/FROZEN_FINAL_EVAL_PROTOCOL_DRAFT.md`)
  for the frozen weights; it introduces no alternative checkpoint. Same paths by reference.
- **Authorization status:** the checkpoint artifacts and paths are frozen and unchanged, while their
  use as the final-test pair remains proposed until the authorization template is explicitly
  completed.
- **Status: MATCH.**

## 4. Same calibration temperatures

- **Protocol:** GroupDRO **T=2.974907**; ERM **T=3.496293** (§4;
  `exp08d_calibration_policy_options.csv`; `FINAL_TEST_OUTPUT_SCHEMA.md`).
- **Manuscript:** Results R5 table and Methods M5 both state GroupDRO T=2.974907, ERM T=3.496293
  (pre-frozen, applied not refit).
- **Status: MATCH.**

## 5. Same operating-point family

- **Protocol:** id_val-selected; fixed specificity 0.80/0.85/0.90/0.95; fixed sensitivity
  0.80/0.90/0.95 (§7; `exp08d_threshold_policy_options.csv`; `exp08d_final_eval_metric_plan.csv`).
- **Manuscript:** Methods M6 and Results R4 use the identical id_val-selected fixed-specificity
  (0.80/0.85/0.90/0.95) and fixed-sensitivity (0.80/0.90/0.95) families, candidate-only.
- **Status: MATCH.**

## 6. Same `ood_val` / `ood_test` distinction

- **Protocol:** `ood_val` (center 1) is development-only; `ood_test` (center 2, n=85,054) is the
  reserved single-shot locked evaluation, read once, only after authorization (§2; stop rules
  FT-2/FT-3).
- **Manuscript:** Methods M1 fixes the same locked split mapping; `ood_val` development-only,
  `ood_test` reserved/unread (skeleton §10; Results R7).
- **Status: MATCH.**

## 7. No final-test numbers inserted into the manuscript

- The results draft R7 states "No test-set result appears in this manuscript draft"; skeleton §10 is a
  LOCKED / NOT-RUN placeholder with explicit "contains NO numbers and NO results."
- Grep-level check of the manuscript drafts finds **no** center-2 / `ood_test` metric value anywhere.
- **Status: MATCH (no final-test numbers present).**

## 8. No clinical-readiness claim

- Forbidden by `CLAIMS_AND_LIMITATIONS.md` F1 and `exp08e_manuscript_claims_map.csv` (row: clinical
  deployment readiness — allowed_now = no). Discussion D4/D8 explicitly disclaim clinical safety.
- No clinical-readiness sentence appears in any manuscript draft.
- **Status: MATCH (absent, as required).**

## 9. No independent-test calibration claim

- Reviewer risk **R7** (calibration fit + evaluated on the same `ood_val`) is disclosed in Methods
  M5, Limitations §8, and Discussion D6. The claims map marks "independent-test calibration validity"
  as forbidden now.
- Only the narrow wording "improves calibration on the `ood_val` subset (fit+eval on same split)"
  is used.
- **Status: MATCH (only the narrow, allowed claim appears).**

## 10. No GroupDRO-vs-old-ERM controlled claim

- Forbidden by `CLAIMS_AND_LIMITATIONS.md` F6 and `FROZEN_DECISIONS.md` F-7; the only controlled
  comparison is 7F GroupDRO vs 7F center-stratified ERM.
- Manuscript Discussion D2 explicitly flags old plain ERM / 7A / 7D / 7E as **context-only** and
  bars framing them as GroupDRO's controlled comparator; Results R1/R2 compare only the matched 7F
  pair.
- **Status: MATCH (context-only framing preserved).**

## 11. Final-test placeholder clearly marked LOCKED / NOT RUN

- Skeleton §10 header: "This section is a reserved placeholder. It contains NO numbers and NO
  results." Results R7: reserved, unread, "No test-set result appears." Table manifest row: "Reserved
  final-test table/figure — LOCKED / NOT RUN."
- **Status: MATCH.**

---

## Summary

| # | Item | Status |
|---|------|--------|
| 1 | Same primary model | MATCH |
| 2 | Same matched control | MATCH |
| 3 | Same checkpoint paths | MATCH |
| 4 | Same calibration temperatures | MATCH |
| 5 | Same operating-point family | MATCH |
| 6 | Same `ood_val` / `ood_test` distinction | MATCH |
| 7 | No final-test numbers in manuscript | MATCH |
| 8 | No clinical-readiness claim | MATCH |
| 9 | No independent-test calibration claim | MATCH |
| 10 | No GroupDRO-vs-old-ERM controlled claim | MATCH |
| 11 | Final-test placeholder LOCKED / NOT RUN | MATCH |

**Consistency verdict: FULLY CONSISTENT — no mismatches, no missing items, no blocking issues.**

**Non-blocking follow-up (documentation only):** `CLAIMS_AND_LIMITATIONS.md` (header "through
Milestone 6A" and universal caveats citing capped n=500 / n=2000) and `FIGURE_TABLE_MANIFEST.md`
(caption tag citing the capped subsets) predate the full-development (8B/8C) evidence the manuscript
now cites. Their allow/deny **logic** is current (the 7H DG-claims section and per-artifact caveats
are up to date), so this is a wording refresh, not a claim conflict. It changes no result and unlocks
no `ood_test` access; it may be done during manuscript finalization.

**Post-audit status — resolved by Milestone 8G:** the legacy capped-subset wording in both documents
was refreshed. No result changed and no `ood_test` access was unlocked.
