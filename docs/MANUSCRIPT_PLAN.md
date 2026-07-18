# MANUSCRIPT PLAN

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in
Histopathology: Calibration, Uncertainty, and Selective Prediction

**Status of this document:** manuscript *planning* only (Milestone 6B). It is built from
**development-stage, capped-balanced-subset** results through Milestone 6A. **No numbers in
this plan are final performance.** `ood_test` (center 2) remains **LOCKED** and unread; the
final locked evaluation has not occurred and is a prerequisite for any headline performance
claim in a submitted paper.

---

## 0. One-paragraph framing

We study whether a tumor/non-tumor patch classifier trained on a subset of Camelyon17-WILDS
hospitals stays *trustworthy* — not just accurate — when evaluated on an unseen hospital
(domain shift by acquisition center). "Trustworthy" is operationalised as calibration
(temperature scaling), deterministic predictive uncertainty (entropy/confidence/margin/energy
error detection), and selective prediction (abstention). The current pipeline is a fully
reproducible, script-based, streaming HF workflow with a locked split protocol; all reported
numbers so far are **development mechanics** on capped balanced subsets, staged behind an
untouched `ood_test`.

---

## 1. Proposed title (options)

- **Primary:** "Confidence-Aware Tumor Detection Under Hospital Domain Shift: A Calibration,
  Uncertainty, and Selective-Prediction Audit on Camelyon17-WILDS."
- Alt A (methods-neutral, audit-forward): "When Accuracy Is Not Enough: Reliability
  Degradation of Histopathology Classifiers Across Hospitals."
- Alt B (selective-prediction-forward): "Can Abstention Rescue Out-of-Distribution
  Histopathology Classification? A Reliability Study Under Hospital Shift."

Title must not assert clinical readiness or SOTA. "Audit"/"study" framing is deliberate and
matches the development scope.

---

## 2. Abstract skeleton

1. **Context:** deployment of histopathology classifiers is limited by hospital-to-hospital
   domain shift; accuracy alone hides reliability failures.
2. **Gap:** calibration, uncertainty, and selective prediction under center shift are under-
   reported relative to raw accuracy, especially with an explicit *held-out* test hospital.
3. **Approach:** a locked-split, streaming, reproducible pipeline on Camelyon17-WILDS with
   ID centers {0,3,4}, an OOD validation hospital (center 1), and a **reserved** OOD test
   hospital (center 2, untouched). Temperature scaling on `ood_val`; deterministic
   uncertainty and selective prediction from saved predictions.
4. **Findings (development):** severe ID→OOD degradation; temperature scaling substantially
   improves calibration metrics without changing decisions; deterministic uncertainty
   separates errors well ID but only moderately OOD; abstention helps far less under shift.
5. **Scope statement (mandatory in abstract):** results are development-stage on capped
   balanced subsets; the final held-out hospital test is reserved and not reported here.
6. **Takeaway:** reliability, not accuracy, is the binding constraint under hospital shift;
   we provide a reproducible protocol and an honest audit rather than a deployment claim.

> The abstract must contain an explicit "development / not final" sentence and must not quote
> the capped-subset numbers as generalization performance.

---

## 3. Introduction argument (logical spine)

1. Digital pathology models achieve high in-distribution accuracy but must operate across
   hospitals with different scanners/staining → covariate shift.
2. High accuracy can coexist with miscalibration and confidently-wrong errors; in a clinical
   safety framing, **confident false negatives (missed tumors)** are the worst failure mode.
3. Confidence-awareness (calibration + uncertainty + selective prediction) is the right lens,
   but is usually evaluated ID or with leaky splits.
4. We commit to a strict protocol: locked center-based splits, `ood_val` used only for OOD
   development decisions, and a **single reserved** `ood_test` hospital.
5. Contribution list (below), each phrased as method/protocol/audit — **not** as a
   performance record.

**Stated contributions:**
- A reproducible, streaming, script-based reliability pipeline with a locked split protocol
  and hardened data access (label-ordered stream handling, balanced caching).
- A staged reliability audit — calibration → deterministic uncertainty → selective
  prediction → consolidated reporting — with `ood_test` held out.
- An honest characterization of *how much* abstention and calibration can and cannot recover
  under hospital shift, in the development regime.

---

## 4. Methods section outline

1. **Dataset & splits (LOCKED).** Camelyon17-WILDS via Hugging Face `wltjr1007/Camelyon17-WILDS`;
   center-based mapping: `train`/`id_val` = {0,3,4}, `ood_val` = {1}, `ood_test` = {2}.
   RGBA→RGB; label 0/1 = non-tumor/tumor. State the reserved-test discipline explicitly.
2. **Model.** ResNet-18 backbone + 2-logit classifier; ImageNet-normalized 96×96 patches.
3. **Training (development).** Capped balanced subsets; AdamW; best-by-`id_val_loss`
   selection; checkpointing. Emphasise this is a *development* configuration.
4. **Calibration.** Temperature scaling: single scalar T fit by NLL on `ood_val` logits;
   argmax-invariant; weights frozen.
5. **Deterministic uncertainty.** Confidence, predictive entropy, margin, energy — all
   oriented higher = more uncertain; error-detection AUROC and risk–coverage/AURC.
6. **Selective prediction.** Rank-based retention of most-confident rows; coverage, retained
   risk, retained/abstained false negatives.
7. **Reproducibility.** Global seeding, config-in-`configs/`, artifacts-in-`results/`,
   numbered scripts, CSV/JSON-only reporting stages.

---

## 5. Experiments section outline

- **E1 — ID vs OOD classification** (development): center-held-out degradation.
- **E2 — Calibration on OOD-val:** raw vs temperature-scaled ECE/Brier/NLL.
- **E3 — Uncertainty / error detection:** entropy (and confidence/margin/energy) AUROC and
  AURC, ID vs OOD, raw vs calibrated.
- **E4 — Selective prediction:** risk–coverage and false-negative redistribution under
  abstention, ID vs OOD.
- **E5 (reserved, NOT in this submission unless authorized):** final `ood_test` locked
  evaluation; optionally model-based uncertainty (MC-dropout / ensembles) — currently
  **deferred**.

Each experiment table/figure must carry a "development / capped subset" caption tag.

---

## 6. Results section outline

1. **Domain shift degrades accuracy and sensitivity sharply** (Table: classification_shift;
   Fig: domain_shift_bar). Accuracy 0.964→0.620, AUROC 0.993→0.770, sensitivity 0.944→0.405
   (development).
2. **Temperature scaling fixes calibration without touching decisions** (Table: calibration;
   Fig: calibration_metrics). ECE −61%, Brier −22%, NLL −33%; confusion matrix unchanged.
3. **Calibration relocates high-confidence false negatives** (from 5B high-confidence table):
   ≥0.90-confidence FNs 222→0 on the `ood_val` subset.
4. **Uncertainty separates errors well ID, only moderately OOD** (Table: uncertainty; Fig:
   uncertainty_error_detection). Entropy error-detection AUROC 0.927 ID vs 0.688 OOD.
5. **Abstention helps far less under shift** (Table: selective_prediction; Figs:
   risk_coverage_entropy, false_negative_abstention). @50% coverage: ID risk→0.000 (all FNs
   abstained) vs OOD-calibrated risk 0.380→0.250 with 198/595 FNs retained.

---

## 7. Discussion section outline

- Accuracy is not a sufficient release criterion under hospital shift; calibration and
  selective prediction expose failure modes accuracy hides.
- Temperature scaling is cheap and effective for *calibration* but is monotone → it cannot
  improve *ranking-based* error detection or selective-prediction risk ordering; its safety
  value is in confidence magnitude (removing confidently-wrong tumor misses at fixed
  thresholds), not in decision changes.
- Confidence-based abstention has a hard ceiling under shift: a model that is *confidently
  wrong* cannot be rescued by confidence thresholds; this motivates (future) model-based
  uncertainty and domain-generalization training.
- Protocol contribution: an explicitly reserved test hospital is the honest way to report
  OOD generalization; we deliberately do not spend it in the development phase.

---

## 8. Limitations section (must be prominent)

- **Development regime only:** capped balanced subsets (train n=2000, id_val n=500,
  ood_val n=2000), ResNet-18, short training — not a tuned or full-data model.
- **Single OOD-val hospital; test hospital reserved and unread** — no generalization number.
- **Calibration fit and evaluated on the same `ood_val` subset** (no separate OOD calibration
  split) → optimistic calibration estimate.
- **Deterministic (single-forward-pass) uncertainty only**; MC-dropout / ensembles deferred.
- **Patch-level only:** no WSI/slide-level aggregation, no clinical endpoint.
- **No domain-generalization baseline** (e.g. GroupDRO/CORAL/DANN) yet.
- **No external dataset** beyond Camelyon17-WILDS.
- Full details in `docs/CLAIMS_AND_LIMITATIONS.md` and `docs/REVIEWER_RISK_AUDIT.md`.

---

## 9. Future work

1. Final locked `ood_test` evaluation (touched once) — the headline generalization number.
2. Full-data / stronger training and a proper OOD calibration/validation split separation.
3. Model-based uncertainty (MC-dropout, deep ensembles) — currently deferred/optional.
4. Domain-generalization baselines for comparison.
5. WSI-level aggregation and external-cohort validation.
6. Abstention-threshold selection for a stated operating point (none committed yet).

---

## 10. Target journal / venue positioning

- **Primary fit:** medical-imaging + ML reliability venues — e.g. *MICCAI* / *MICCAI UNSURE
  workshop*, *Medical Image Analysis*, *IEEE TMI*, or *Journal of Medical Imaging* for the
  full study; a reliability/uncertainty workshop for an interim protocol paper.
- **Positioning:** an **honest reliability audit + reproducible protocol** under hospital
  shift, not a leaderboard/SOTA claim. The reserved-test discipline and the calibration-vs-
  abstention ceiling analysis are the differentiators.
- Submission-readiness is **gated** on the final `ood_test` evaluation and at least one
  domain-generalization or model-based-uncertainty comparison; until then this is a
  work-in-progress / protocol contribution.

---

## 11. Locked-data note (mandatory)

`ood_test` (center 2, n=85,054) is **LOCKED and untouched**. It will be evaluated exactly
once, at the very end, only under explicit written authorization. No table, figure, or
sentence in the eventual manuscript may present a capped development subset number as final
generalization or clinical performance.
