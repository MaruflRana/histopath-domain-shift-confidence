# NEXT EXPERIMENT DECISION TREE (Milestone 7I)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology:
Calibration, Uncertainty, and Selective Prediction

_Milestone 7I — pre-final-evaluation decision audit. Documentation only. This is a decision
aid, not an authorization. **Every path below is gated behind an explicit written go-ahead.**
No path may touch `ood_test` except the final locked evaluation, which is itself separately
authorized. All non-test work is development-stage on capped or larger `ood_val` (center 1)
only. The recommended default next action is a **decision**, not automatic model-based
uncertainty._

---

## Decision context

- **Primary controlled candidate:** 7F GroupDRO + temperature-scaled confidence.
- **Matched control:** 7F center-stratified ERM.
- **Context-only baselines:** old plain ERM, 7A, 7D, 7E.
- **Open decisions:** final model, final threshold, final calibration policy, full-eval scope,
  CORAL/DANN, model-based uncertainty (see `FROZEN_DECISIONS.md`).

---

## Top-level decision tree

```
Is ood_test authorized in writing AND is a final model/threshold/calibration frozen?
├── NO  (current state)
│    ├── Do you have compute/time for a larger/full development evaluation?
│    │     ├── YES ─► PATH A  (full development-scale evaluation planning)   [RECOMMENDED]
│    │     └── NO  ─► PATH E  (manuscript skeleton around current evidence)  [RECOMMENDED fallback]
│    ├── Want broader DG reviewer-risk coverage?      ─► PATH C  (CORAL/DANN)            [optional]
│    └── Want a stronger uncertainty narrative?       ─► PATH D  (model-based uncertainty)[deferred]
└── YES ─► FINAL LOCKED ood_test EVALUATION  (single, one-time; separate authorization)
```

**Preferred:** **Path A if compute/time allows, otherwise Path E.** Path C and Path D are
optional and gated; Path D is explicitly deferred and not the default.

---

## Path A — Full development-scale evaluation planning  ·  Recommendation: HIGH

- **Purpose:** Decide whether to evaluate the frozen candidate (7F GroupDRO) and its matched
  control on larger / full `id_val` and full `ood_val` (center 1), to strengthen development
  evidence before any final test. **No `ood_test`.**
- **Pros:** Directly attacks R1/R4 (capped-subset fragility); de-risks the single locked test;
  larger `ood_val` gives a more trustworthy development OOD estimate; reuses existing models
  (no retraining required for evaluation).
- **Cons:** Larger inference cost; requires building/streaming larger `ood_val` (and `id_val`)
  evaluation sets; still development-stage (not the final test).
- **Required compute:** Moderate — model **inference** only over larger evaluation sets (no
  training). (Note: this planning milestone 7I does not run it; execution is a separate gated
  step.)
- **Risk reduced:** R1 (capped subsets), partially R4 (no full eval), R9 (subsetting bias).
- **Touches `ood_val`?** Yes — development-stage evaluation (authorized).
- **Touches `ood_test`?** No.

## Path B — Freeze GroupDRO as primary candidate + prepare final-eval protocol  ·  Recommendation: MEDIUM-HIGH

- **Purpose:** No new method. Freeze 7F GroupDRO (+ calibration) as the primary candidate and
  write the exact final-evaluation checklist (model, calibration policy, operating-point rule,
  metrics, one-time test procedure). `ood_test` stays LOCKED until explicit final authorization.
- **Pros:** Cheapest path to a submission-ready protocol; forces the open decisions (N-1..N-3)
  to be made explicitly; makes the eventual test run a mechanical, pre-registered step.
- **Cons:** Adds no new evidence; still rests on capped subsets unless combined with Path A.
- **Required compute:** None (documentation only).
- **Risk reduced:** Process risk (uncommitted final decisions); supports R4 discipline.
- **Touches `ood_val`?** No (planning only).
- **Touches `ood_test`?** No.

## Path C — Add another DG method (CORAL or DANN)  ·  Recommendation: MEDIUM (optional)

- **Purpose:** Broaden the DG comparison beyond GroupDRO to further reduce R3 breadth.
- **Pros:** Stronger DG story; addresses "only one DG method tested"; uses the sound
  center-stratified 7F-1 caches.
- **Cons:** Requires **training** new models; adds scope; only worth it if the reviewer-risk
  reduction justifies the added compute; must remain development-stage.
- **Required compute:** High — training + evaluation of one or two additional DG models.
- **Risk reduced:** R3 residual (DG breadth).
- **Touches `ood_val`?** Yes — development-stage comparison only (after training/selection on
  train/id_val).
- **Touches `ood_test`?** No.

## Path D — Model-based uncertainty (MC-dropout or ensemble)  ·  Recommendation: LOW (deferred/optional)

- **Purpose:** Add model-based predictive uncertainty (MC-dropout or a small deep ensemble) on
  `id_val` + `ood_val` to strengthen the uncertainty narrative.
- **Pros:** Addresses R8 (deterministic-only uncertainty); expected by an uncertainty-focused
  reviewer.
- **Cons:** **Not required** to freeze the candidate or run the final test; requires inference
  (MC-dropout) or training (ensemble); explicitly deferred; not the default next step.
- **Required compute:** Medium (MC-dropout inference) to High (ensemble training).
- **Risk reduced:** R8 (deterministic-only).
- **Touches `ood_val`?** Yes — development-stage uncertainty analysis only.
- **Touches `ood_test`?** No.

## Path E — Manuscript skeleton  ·  Recommendation: HIGH (fallback if compute/time is limited)

- **Purpose:** No new experiments. Start drafting methods/results around the current controlled
  evidence (7F ERM vs 7F GroupDRO + 7G calibration + 5A–5C/6A reliability), reusing the 6A/7H
  tables and figures and the `DG_MANUSCRIPT_UPDATE.md` guidance.
- **Pros:** Zero compute; captures the controlled DG result and reliability story now; surfaces
  gaps that later experiments should fill; enforces the allowed/forbidden claim boundary.
- **Cons:** Adds no new evidence; still development-stage; submission remains gated on the final
  locked test.
- **Required compute:** None (documentation only).
- **Risk reduced:** Process/framing risk; consolidates claim governance.
- **Touches `ood_val`?** No (writing only).
- **Touches `ood_test`?** No.

---

## Final locked `ood_test` evaluation (separate, one-time, fully gated)

- **Preconditions (all must be frozen first):** final model (N-1), final calibration policy
  (N-3), final operating threshold (N-2), and explicit written authorization.
- The single, one-time `ood_test` (center 2) run for reporting. Touched exactly once, at the
  very end. **Not part of any path above.**

---

## Recommendation summary

1. **Path A** (full development-scale evaluation planning) — **preferred if compute/time
   allows.**
2. **Path E** (manuscript skeleton) — **preferred fallback** if compute/time is limited; can run
   in parallel with Path B.
3. **Path B** (freeze candidate + write final-eval protocol) — low-cost, do alongside A or E.
4. **Path C** (CORAL/DANN) — optional, only if DG breadth is worth the added training scope.
5. **Path D** (model-based uncertainty) — deferred/optional; **not** the default next step.

`ood_test` remains **LOCKED** throughout; no path reads it.
