# FROZEN DECISIONS (Milestone 7I)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology:
Calibration, Uncertainty, and Selective Prediction

_Milestone 7I — pre-final-evaluation decision audit. Documentation only. This file records
what is **FROZEN** (locked; must not change without explicit re-authorization) versus **NOT
FROZEN** (open decisions that require an explicit go-ahead before any experiment). All current
evidence is development-stage on capped balanced subsets; `ood_test` is LOCKED and unread._

---

## FROZEN (locked — do not change without explicit re-authorization)

| # | Frozen decision | Basis |
|---|---|---|
| F-1 | **Locked split mapping remains unchanged** — train {0,3,4}, id_val {0,3,4}, ood_val {1}, ood_test {2}. | `PROJECT_PROTOCOL.md` §3; `CLAUDE.md` §4. |
| F-2 | **`ood_test` remains LOCKED** — used once, at the very end, only after explicit written authorization. | `CLAUDE.md` §5 rule 3; `PROJECT_PROTOCOL.md` §4. |
| F-3 | **`ood_val` is development-only** — authorized for development-stage comparison / calibration / uncertainty / selective analysis only; never training; never final test reporting. | `CLAUDE.md` §5 rule 2 (authorized 2026-07-08). |
| F-4 | **Old confounded caches must NOT be used for DG training** — the pc1000 balanced caches are center↔label confounded (center 4 absent) and unsound for center-grouped objectives. | Milestone 7F-0 finding. |
| F-5 | **Center-stratified caches are the required DG caches** — `center_strat_train_c034_percell300_seed1337_shuf.pt` (n=1800) and `center_strat_id_val_c034_percell75_seed1337_shuf.pt` (n=450) are the sound DG caches for any center-grouped method. | Milestone 7F-1 rebuild. |
| F-6 | **Fair DG comparison = 7F center-stratified ERM vs 7F GroupDRO** — the only controlled DG comparison (same sound cache, matched init/seed/schedule, id_val selection only). | Milestone 7F/7H; `exp07h_context_vs_controlled_warning_table.csv`. |
| F-7 | **Old plain ERM / 7A / 7D / 7E are CONTEXT-ONLY for the 7F comparison** — different (confounded) cache and/or id_val subset; never a controlled comparator for GroupDRO. | Milestone 7H framing; `DG_RESULTS_INTERPRETATION.md`. |
| F-8 | **Reserved-test discipline** — no milestone (4B–7I) touched `ood_test`; this discipline is maintained. | Milestone log; reviewer risk R4. |

---

## NOT FROZEN (open — require an explicit go-ahead before any action)

| # | Open decision | Current stance / default |
|---|---|---|
| N-1 | **Final model for `ood_test`** | Primary *candidate* is 7F GroupDRO (+ temperature-scaled confidence). NOT yet frozen as the final model. |
| N-2 | **Final operating threshold** | Only candidate id_val-selected points exist; **none committed**. To be chosen on `ood_val` later (optional, gated). |
| N-3 | **Final calibration policy for paper-level test** | 7G temperatures exist but are fit/eval on the same `ood_val` subset (R7). A validated policy needs a separate calibration split. |
| N-4 | **Whether to run a full `ood_val` / full development evaluation** | Recommended before final test if compute allows (Path A). Not yet decided/authorized. |
| N-5 | **Whether to add CORAL/DANN** | Optional; reduces R3 breadth; requires training. Not decided. |
| N-6 | **Whether to add model-based uncertainty (MC-dropout / ensemble)** | DEFERRED / optional; requires inference (and training for ensembles). Not the default next step. |

---

## Interpretation guardrails (carried from 7H, still binding)

- The controlled DG claim is **development-stage** and **capped-subset**; not final performance.
- GroupDRO's advantage is a **controlled win over its matched ERM control only** — never framed
  as a controlled win or loss versus old plain ERM.
- Candidate operating points are **not** committed clinical thresholds.
- Calibration claims stay narrow ("improves calibration on this `ood_val` subset"); no
  "calibration generalizes" claim without a separate calibration/eval split.
