# MANUSCRIPT SKELETON — DEVELOPMENT EVIDENCE (Milestone 8E)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology:
Calibration, Uncertainty, and Selective Prediction

> **DEVELOPMENT-STAGE WRITING/PLANNING PACKAGE — NOT A SUBMISSION.** Milestone 8E is
> documentation only: no training, no inference, no calibration fitting, no MC-dropout, no
> ensemble, no CORAL/DANN, no `ood_test`, no final-performance claim. Every quantitative result
> cited in this skeleton is **development-stage evidence** on development splits (`id_val` centers
> {0,3,4}; `ood_val` center 1). The reserved test hospital (`ood_test`, center 2) is **fully
> LOCKED and has never been read.** No sentence in a draft built from this skeleton may claim
> final generalization performance or clinical readiness.

---

## 0. How to use this skeleton

This is the **structural blueprint** for the eventual manuscript. It fixes the section order,
what each section must contain, and — crucially — which claims are allowed *now* (before
`ood_test`) versus reserved for after an authorized final evaluation. The prose drafts live in
the companion docs:

- `docs/MANUSCRIPT_METHODS_DRAFT_DEV_ONLY.md` — Methods prose.
- `docs/MANUSCRIPT_RESULTS_DRAFT_DEV_ONLY.md` — Results prose (development numbers only).
- `docs/MANUSCRIPT_DISCUSSION_POINTS.md` — Discussion / interpretation bullets.
- `results/tables/exp08e_manuscript_table_manifest.csv` — figure/table placement.
- `results/tables/exp08e_manuscript_claims_map.csv` — allowed/forbidden claim wording.

The authoritative claim allow/deny list remains `docs/CLAIMS_AND_LIMITATIONS.md`; when any draft
sentence conflicts with it, that list wins.

---

## 1. Title options

Working titles (development framing — all avoid final-performance / clinical wording):

1. **"Confidence-Aware Tumor Detection Under Hospital Domain Shift: A Calibration, Uncertainty,
   and Selective-Prediction Audit on Camelyon17-WILDS"**
2. "When the Hospital Changes: A Reliability Audit of Domain Generalization and Confidence
   Calibration for Histopathology Patch Classification"
3. "Reliability Before Deployment: Group-DRO, Temperature Scaling, and High-Confidence
   False-Negative Reduction Under Hospital Shift"
4. "A Reserved-Test Protocol for Confidence-Aware Tumor Detection Under Domain Shift"

Positioning note: the paper is framed as a **methodological / reliability-audit contribution
with a reserved test hospital**, not as a state-of-the-art performance claim. Title 1 is the
recommended primary.

---

## 2. Abstract outline (structure — NOT a final abstract)

Draft the abstract along these beats; keep every quantitative statement scoped as
development-stage:

- **Problem.** Hospital-to-hospital (domain) shift degrades the reliability of patch-level
  tumor classifiers; confidence estimates become untrustworthy exactly where they matter.
- **Gap.** Most reports quote accuracy under shift but not *reliability* (calibration,
  high-confidence errors, controlled domain-generalization comparison, reserved-test discipline).
- **What we do.** A reproducible, locked-split, streaming protocol on Camelyon17-WILDS with a
  reserved unseen test hospital; a controlled matched comparison of a GroupDRO-by-center model
  against a matched center-stratified ERM control; development-stage temperature-scaling
  calibration; candidate operating-point and high-confidence false-negative audits.
- **Development-stage findings (scope-tagged).** GroupDRO improves OOD-val ranking and reliability
  over its matched control; temperature scaling improves ECE/Brier/NLL and sharply reduces
  high-confidence missed tumors (argmax-invariant); calibration is fit and evaluated on the same
  OOD-val split (a stated limitation).
- **Reserved.** The single unseen test hospital (center 2) is held out for one final locked
  evaluation and is not reported here.
- **Placeholder sentence (marked):** "[Final held-out result on the reserved center-2 test set to
  be completed under the frozen evaluation protocol after written authorization.]"

---

## 3. Introduction (bullet structure)

- **Clinical/technical motivation.** Deep models for tumor detection are trained at a few
  hospitals but deployed at others; staining, scanners, and populations differ.
- **The reliability problem, not just accuracy.** Under shift a model can stay moderately accurate
  yet be badly *calibrated* and produce **high-confidence false negatives** (confidently missed
  tumors) — the most dangerous failure mode.
- **What is usually missing.** (i) a *controlled* domain-generalization comparison (matched cache,
  matched schedule) rather than cross-cache anecdotes; (ii) an explicit reserved test hospital;
  (iii) a calibration + selective-prediction reliability audit.
- **Our protocol contribution.** Locked HF-streaming split mapping with a reserved center; sound
  center-stratified caches (fixing a center↔label confound); a matched ERM-vs-GroupDRO design;
  development-stage calibration and operating-point/high-confidence-FN audits.
- **Scope statement (must appear early).** All results here are development-stage on development
  splits; the reserved test hospital is unread; no clinical or final-performance claim is made.
- **Contributions list** (3–4 bullets): protocol + reserved-test discipline; controlled DG result
  (development-stage); reliability/calibration + high-confidence-FN audit; an honest
  claims-governance / limitations framework.

---

## 4. Methods (section list — prose in `MANUSCRIPT_METHODS_DRAFT_DEV_ONLY.md`)

- **4.1 Dataset and locked split mapping.** Camelyon17-WILDS via Hugging Face streaming; centers →
  logical splits; reserved test center.
- **4.2 Center-stratified cache construction.** Label-ordered stream, the center↔label confound in
  the naive cache, per-`(center,label)` quota rebuild.
- **4.3 Backbone and ERM baseline.** ResNet-18 classifier; matched center-stratified ERM control.
- **4.4 GroupDRO-by-center.** Group objective over centers {0,3,4}; exponentiated-gradient weight
  update; matched training setup; `id_val_loss` selection only.
- **4.5 Calibration / temperature scaling.** Single-temperature NLL fit on OOD-val logits
  (development-stage); argmax-invariance.
- **4.6 Candidate operating points.** id_val-selected fixed-specificity and fixed-sensitivity
  thresholds applied to OOD-val (candidate only, not committed).
- **4.7 High-confidence false-negative audit.** Confidently missed tumors at conf ≥ 0.90/0.95/0.99.
- **4.8 Reserved `ood_test` protocol.** The frozen single-shot final-evaluation protocol (drafted,
  not executed); LOCKED.
- **4.9 Reproducibility.** Seeds, config, streaming/SSL handling, caching.

---

## 5. Experiments (section list)

- **5.1 Development evaluation scope.** Full development splits (`id_val` n=33,560; `ood_val`
  n=34,904); positive class = tumor.
- **5.2 Models compared.** Primary candidate 7F GroupDRO vs matched control 7F center-stratified
  ERM; context-only baselines (old plain ERM, 7A/7D/7E stain augmentation) explicitly flagged as
  non-controlled.
- **5.3 Metrics.** Threshold-free (AUROC/AUPRC); default-threshold confusion metrics; per-center;
  calibration (ECE/Brier/NLL); candidate operating points; high-confidence FN.
- **5.4 What is deliberately NOT run here.** `ood_test`; MC-dropout/ensembles; CORAL/DANN;
  committed clinical threshold.

---

## 6. Results (section list — prose in `MANUSCRIPT_RESULTS_DRAFT_DEV_ONLY.md`)

- **6.1 ID→OOD reliability degradation** (motivating result).
- **6.2 Controlled DG comparison (development-stage): GroupDRO vs matched ERM** on full
  development splits (8B): AUROC/AUPRC/sensitivity/specificity/FN.
- **6.3 Per-center `id_val` behavior**, incl. worst-center (center 4) improvement.
- **6.4 Candidate operating points** (fixed-specificity / fixed-sensitivity), candidate-only.
- **6.5 Calibration (development-stage, 8C):** raw vs calibrated ECE/Brier/NLL; argmax-invariance.
- **6.6 High-confidence false-negative reduction** after calibration.
- **6.7 Reserved-test placeholder** (LOCKED — no numbers).

Every subsection carries the development-scope caption tag.

---

## 7. Discussion (section list — bullets in `MANUSCRIPT_DISCUSSION_POINTS.md`)

- Main interpretation: shift degrades reliability; a controlled DG objective helps in development;
  calibration fixes confidence magnitude but not decisions.
- Why GroupDRO helps (worst-group up-weighting under center shift).
- Why calibration matters and why it is argmax-invariant.
- Why high-confidence false negatives are the clinically salient failure and why their reduction
  is meaningful even in development.
- Why OOD-val is development-only and why the test hospital is reserved.
- Reviewer risks and how the protocol pre-empts them.
- What the reserved `ood_test` run can and cannot unlock.

---

## 8. Limitations (must be prominent)

- Development-stage; capped/dev splits; ResNet-18; short training — not tuned/full-data-optimal.
- Single OOD-val hospital (center 1); reserved test hospital (center 2) unread.
- **R7 (calibration):** temperature is fit and evaluated on the *same* `ood_val` split →
  optimistic; independent-test calibration validity is unproven.
- Only GroupDRO tested — no CORAL/DANN; deterministic + temperature-scaled confidence only (no
  MC-dropout/ensemble).
- Patch-level only; no WSI aggregation; single dataset; no external cohort; no clinical endpoint.
- No committed operating threshold.

---

## 9. Conclusion (structure)

- Restate the protocol + reserved-test contribution.
- Restate the development-stage controlled DG and reliability findings, scope-tagged.
- State the reserved final evaluation as the explicit next step; no clinical/final claim.

---

## 10. FINAL-TEST PLACEHOLDER SECTION — **LOCKED / NOT RUN**

> **This section is a reserved placeholder. It contains NO numbers and NO results.**
>
> The single, one-time locked evaluation on the reserved unseen test hospital (`ood_test`,
> center 2, HF `test` split, expected n=85,054) has **not** been run. It remains **fully LOCKED
> and unread.** It may proceed **only** after a final model, calibration policy, and
> operating-point policy are frozen **and** explicit written authorization is provided via
> `docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md` (exact phrase + all 7 checkboxes), under the frozen
> protocol in `docs/FROZEN_FINAL_EVAL_PROTOCOL_DRAFT.md`.
>
> When (and only when) authorized and executed, this section will report — for both the primary
> candidate and the matched control — threshold-free metrics (AUROC/AUPRC), default-threshold
> confusion metrics, candidate operating points, calibration (ECE/Brier/NLL using the pre-frozen
> 8C temperatures, never refit on the test), and the high-confidence false-negative audit.
>
> Placeholder text for the manuscript body:
> _"[RESERVED] Held-out results on the unseen center-2 test hospital will be reported here after
> the single authorized locked evaluation under the frozen protocol. No test-set numbers exist at
> submission of this development-stage manuscript skeleton."_

---

## 11. Development-scope caption tag (apply to every table/figure)

> "Development-stage results on development splits (`id_val` centers {0,3,4}, n=33,560; `ood_val`
> center 1, n=34,904); ResNet-18; the reserved test hospital (`ood_test`, center 2) is held out
> and unread. Not final or clinical performance."
