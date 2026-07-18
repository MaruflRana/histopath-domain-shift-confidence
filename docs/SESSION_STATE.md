# SESSION STATE

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in
Histopathology: Calibration, Uncertainty, and Selective Prediction

_As of: 2026-07-17. This document captures the exact, current, accepted state of the
project so a fresh session can resume with zero ambiguity._

---

## 1. One-line status

**Milestones 1 → 9D are complete and accepted; the latest completed milestone is
Milestone 9D.** The explicitly authorized single final locked `ood_test` inference completed once,
followed by the CSV-only final summary. Run start:
**2026-07-16T17:16:25.455291+00:00**; inference complete:
**2026-07-16T17:35:17.051997+00:00**; summary complete:
**2026-07-16T17:35:20.395673+00:00**. Attempt count = **1**. `ood_test` is no longer unread, and no
second access or inference attempt is authorized.

Milestone 9B used saved 8B/8C/9A artifacts only to create the final venue-neutral manuscript,
title/abstract package, results interpretation, discussion/limitations, final claims map,
literature-citation checklist, submission-package guide, reproducibility checklist, consistency
audit, seven manuscript tables, and six new paper-facing matplotlib figures. No dataset, HF split,
center-2 image, model, checkpoint, or inference path was accessed; no calibration was fit and no
threshold was selected. Existing exp09 predictions, metrics, authorization, run-state, sentinels,
and figures remained unchanged.

Milestone 9C then verified and integrated **46 references**, replaced all 22 manuscript
`[CITATION NEEDED: ...]` markers, created `references/final_references.bib` and
`docs/FINAL_MANUSCRIPT_WITH_CITATIONS.md`, recorded **0 unresolved citation gaps**, and completed a
claim-level citation audit. The ranked submission order is **Journal of Pathology Informatics**,
**Journal of Medical Imaging**, then **PLOS ONE** as the safer fallback. Milestone 9C used local
saved manuscript artifacts and authoritative literature/journal web pages only. It loaded no
dataset, accessed no HF split, center-2 image, model, checkpoint, prediction, authorization, or
sentinel, and performed no inference, training, calibration fitting, threshold tuning, or
post-test selection. Existing exp09 evidence remained unchanged.

Milestone 9D formatted the accepted 9C manuscript for **Journal of Pathology Informatics** as an
Original Research Article. Sole author and corresponding author: **Jishan Islam Maruf**
(`jishanislammaruf62@gmail.com`). The complete package is under `submission/jpi/` and includes an
identity-bearing title page, double-anonymized editable manuscript, 11-page PDF proof, five
highlights, cover letter, declarations, five editable main tables, six lossless 300-dpi TIFF/PNG
figure pairs, supplementary material, completed CLAIM 2024 checklist, and a sanitized local code
archive. The abstract is **221 words**, the keyword count is **7**, and highlight lengths are
**68, 69, 63, 64, and 69**. Double-anonymization and final scientific-integrity audits both
returned **PASS**. Standard ethics, funding, competing-interest, consent, CRediT, data/code, and
AI-use declarations are complete. No dataset, image data, model, checkpoint, inference, training,
calibration fitting, threshold tuning, authorization/sentinel change, or second `ood_test` attempt
occurred; accepted exp09 hashes remained unchanged.

The exact frozen pair was used unchanged: 7F GroupDRO primary checkpoint
`results/checkpoints/exp07f_groupdro_resnet18/best.pt`
(SHA256 `CE0DC65DC6106648F2ABB77C603746A1DB2B856F9672B01C2E8F540BEEAD8502`) and 7F
center-stratified ERM matched control
`results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt`
(SHA256 `8BF990BE517A41AA74D111E0A4F4111A05A9FE416919261D979F2FD65AE974FF`). Frozen
temperatures were GroupDRO **2.974907** and ERM **3.496293**. All 14 candidate operating-point
thresholds came from `results/tables/exp08i_frozen_operating_points.csv`; none was selected or tuned
on `ood_test`.

Dataset validation passed: HF split **`test`**, center exactly **{2}**, **85,054 rows per model**,
labels **42,527 non-tumor / 42,527 tumor**, complete required metadata, finite normalized
probabilities, and identical raw/calibrated argmax predictions. Both models were evaluated in one
dataloader traversal. No training, optimizer, backward pass, checkpoint save, weight update,
calibration fitting, test-set threshold tuning, or post-test model selection occurred.

### Final default-threshold and threshold-free results

| Model | Accuracy | Balanced accuracy | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GroupDRO | 0.5337197545089002 | 0.5337197545089002 | 0.6633704256200204 | 0.63641261236421 | 0.11056505278999224 | 0.9568744562278082 | 0.7194002447980417 | 0.19167193200578847 | 40693 | 1834 | 37825 | 4702 |
| Center-stratified ERM | 0.5711547957768006 | 0.5711547957768006 | 0.6984352121958427 | 0.6556283801091156 | 0.24107037881816257 | 0.9012392127354386 | 0.7093827843897038 | 0.3598518752522859 | 38327 | 4200 | 32275 | 10252 |

Raw and calibrated rows have identical classification metrics at threshold 0.5 because frozen
temperature scaling is argmax-invariant. The honest final comparison is unfavorable to the
predeclared primary: GroupDRO minus matched ERM = **−0.0350647865758223 AUROC**,
**−0.0192157677449056 AUPRC**, **−0.13050532602817033 sensitivity**, and **+5,550 FN**. GroupDRO
retained higher specificity (**+0.0556352434923696**) and slightly higher precision
(**+0.0100174604083379**).

### Final reliability results

| Model | Variant | ECE | Brier | NLL |
|---|---|---:|---:|---:|
| GroupDRO | raw | 0.4006439581474795 | 0.41233859378344173 | 1.9441627836686814 |
| GroupDRO | calibrated | 0.2583399203510463 | 0.3096680541395694 | 0.8743126064429281 |
| Center-stratified ERM | raw | 0.3084182197890404 | 0.3446598909605201 | 1.4183276475805955 |
| Center-stratified ERM | calibrated | 0.1489841758680231 | 0.2555235818479269 | 0.7130964662716416 |

Frozen temperature scaling improved all three reliability metrics for both models, but calibrated
ERM remained better calibrated than calibrated GroupDRO. Hard predictions and threshold-free
ranking were unchanged.

### High-confidence false negatives

| Model | Variant | confidence≥0.90 | confidence≥0.95 | confidence≥0.99 | Total FN |
|---|---|---:|---:|---:|---:|
| GroupDRO | raw | 29775 | 25620 | 15485 | 37825 |
| GroupDRO | calibrated | 5479 | 755 | 1 | 37825 |
| Center-stratified ERM | raw | 19142 | 14626 | 7354 | 32275 |
| Center-stratified ERM | calibrated | 1828 | 513 | 0 | 32275 |

The 14 frozen `id_val`-selected operating-point results are in
`results/tables/exp09_final_ood_test_frozen_operating_points.csv`. They remain explicitly
**candidate/non-clinical** and transferred poorly to center 2; they must not be reframed as achieved
clinical targets.

### Milestone 9B final manuscript package

Primary artifacts:

- `docs/FINAL_MANUSCRIPT_DRAFT.md`
- `docs/FINAL_TITLE_ABSTRACT_KEYWORDS.md`
- `docs/FINAL_RESULTS_INTERPRETATION.md`
- `docs/FINAL_DISCUSSION_AND_LIMITATIONS.md`
- `docs/FINAL_CLAIMS_AND_LIMITATIONS.md`
- `docs/LITERATURE_CITATION_GAPS.md`
- `docs/SUBMISSION_PACKAGE_README.md`
- `docs/FINAL_REPRODUCIBILITY_CHECKLIST.md`
- `docs/FINAL_MANUSCRIPT_CONSISTENCY_AUDIT.md`
- `results/tables/exp09b_*.csv`
- `results/figures/exp09b_*.png`
- `results/logs/exp09b_final_manuscript_integration_report.txt`

### Milestone 9C verified citation and journal package

Primary artifacts:

- `docs/FINAL_MANUSCRIPT_WITH_CITATIONS.md`
- `references/final_references.bib`
- `docs/CITATION_INTEGRATION_CHANGELOG.md`
- `docs/VERIFIED_LITERATURE_REVIEW.md`
- `docs/FINAL_CITATION_AUDIT.md`
- `docs/FINAL_NOVELTY_STATEMENT.md`
- `docs/TARGET_JOURNAL_SHORTLIST.md`
- `docs/RECOMMENDED_SUBMISSION_STRATEGY.md`
- `results/tables/exp09c_verified_references.csv`
- `results/tables/exp09c_citation_to_claim_map.csv`
- `results/tables/exp09c_unresolved_citation_gaps.csv`
- `results/tables/exp09c_journal_comparison.csv`
- `results/tables/exp09c_literature_novelty_comparison.csv`
- `scripts/41_verify_literature_and_citations.py`

Offline citation audit: **PASS**. BibTeX entries 46; manuscript citation keys 38; package citation
keys 46; background/review-only entries 8; unresolved gaps 0; duplicate DOI/title records 0.
Scientific numeric tokens remained identical after citation markup was removed. No journal
template had been applied at Milestone 9C; Milestone 9D subsequently completed JPI formatting.

### Milestone 9D JPI submission package

Primary artifacts:

- `submission/jpi/JPI_Submission_Package_README.md`
- `submission/jpi/JPI_Title_Page.docx`
- `submission/jpi/JPI_Anonymized_Manuscript.docx`
- `submission/jpi/JPI_Anonymized_Manuscript.pdf`
- `submission/jpi/JPI_Cover_Letter.docx`
- `submission/jpi/JPI_Supplementary_Material.docx`
- `submission/jpi/JPI_CLAIM_Checklist.docx`
- `submission/jpi/JPI_Anonymization_Audit.md`
- `submission/jpi/JPI_Reproducibility_Code_Package.zip`
- `docs/JPI_FINAL_SUBMISSION_AUDIT.md`
- `results/logs/exp09d_jpi_submission_package_report.txt`

**Exact next recommendation:** perform a human visual review in Microsoft Word, check the JPI
submission preview and current APC/waiver information, then upload and submit manually. Do not
rerun `ood_test`, reopen model development, add exploratory methods, or revise the scientific
result in response to the held-out outcome.

The manuscript preserves the negative final model result and the positive protocol contribution:
the development GroupDRO advantage did not generalize to center 2, while the reserved-hospital
design prevented `ood_val` performance from being misreported as final generalization.

**Historical Milestone 9B handoff (superseded by 9C):** literature/citation verification was the
then-next action. Milestone 9C completed that work. Model-based uncertainty remains
deferred/optional and was not added.

### Historical pre-authorization state through Milestone 8I (superseded by 9A)

Prior context (Milestone 8F): the final protocol/manuscript consistency audit — docs/tables/logs only;
no training, no inference, no calibration fitting, no MC-dropout, no ensemble, no CORAL/DANN, no
`ood_test`, no final test run, no final-performance claim; no earlier result file modified — inputs
read-only, fingerprint-verified. **Milestone 8F** (`scripts/37_make_final_pre_authorization_audit.py`; does not
import torch) audited the 8D final-test protocol package against the 8E manuscript skeleton as the
final safety/readiness check before any `ood_test` authorization. **Verdict: CONSISTENT — READY FOR
MANUAL REVIEW, no blocking issues** (16/16 items MATCH: same primary model, matched control,
checkpoint paths, calibration temperatures GroupDRO T=2.974907 / ERM T=3.496293, operating-point
family, `ood_val`/`ood_test` distinction; no final-test numbers in the manuscript; no
clinical/independent-test-calibration/GroupDRO-vs-old-plain-ERM controlled claim; final-test
placeholder marked LOCKED / NOT RUN). Pre-authorization checklist: **8 `needs_user_decision`** (the 7
authorization boxes + checkpoint approval) + **3 pass** (test script not run, template not completed,
one-shot rule understood), **0 fail**. Remaining-risk register: 8 rows (R1 reduced; R4/R7/one-shot HIGH
open; CORAL/DANN, model-based uncertainty, clinical threshold, external validation open/deferred).
The original 8F **non-blocking** documentation follow-up was to refresh legacy capped-subset wording
in `docs/CLAIMS_AND_LIMITATIONS.md` / `docs/FIGURE_TABLE_MANIFEST.md`; **Milestone 8G resolved it.
No result changed and no `ood_test` access was unlocked.**
OVERALL PASS (14/14 assertions; no warnings/errors). Artifacts:
`docs/{FINAL_PRE_AUTHORIZATION_AUDIT,MANUSCRIPT_PROTOCOL_CONSISTENCY_CHECK}.md`;
`results/tables/exp08f_{protocol_manuscript_consistency,pre_authorization_checklist,remaining_risk_register}.csv`;
`results/logs/exp08f_final_pre_authorization_audit_report.txt`. **Historical next action after 8I,
subsequently completed by 9A:** explicit manual authorization review, not an automatic test run.
At that handoff `ood_test` remained fully locked and unread; model-based uncertainty
(MC-dropout / ensembles) remained deferred/optional.

The prior milestone (context): **Milestone 8E** (manuscript skeleton on development evidence —
docs/tables/logs only; no training, no inference, no calibration fitting, no `ood_test`, no
final-performance claim; no result file modified). **Milestone 8E** authored a full paper skeleton
plus methods/results/discussion drafts built **only** from completed development-stage evidence
(6A/7H/8B/8C/8D), a table/figure manifest, and a claims map — every quantitative statement
scope-tagged development-stage, the R7 caveat preserved (8C calibration fit + evaluated on the same
`ood_val`, not an independent-test claim), and the final-test section left as a marked LOCKED /
NOT-RUN placeholder with no numbers. Artifacts:
`docs/{MANUSCRIPT_SKELETON_DEV_EVIDENCE,MANUSCRIPT_METHODS_DRAFT_DEV_ONLY,MANUSCRIPT_RESULTS_DRAFT_DEV_ONLY,MANUSCRIPT_DISCUSSION_POINTS}.md`;
`results/tables/exp08e_{manuscript_table_manifest,manuscript_claims_map}.csv`;
`results/logs/exp08e_manuscript_skeleton_report.txt`. **Historical 8E handoff recommendation
(superseded by 8F–8I):** manual review of the manuscript and protocol before any authorization.
`ood_test` remains fully LOCKED and unread; model-based uncertainty (MC-dropout / ensembles) remains
DEFERRED / optional and is not the default next step.

The prior milestone (context): **Milestone 8D** (final protocol freeze / final-evaluation readiness
package — docs/tables/logs only; no training,
no inference, no calibration fitting, no MC-dropout, no ensemble, no CORAL/DANN, no
`ood_test`). **Milestone 8D** (`scripts/35_make_final_eval_readiness_package.py`; does not import
torch; reads only saved 8B/8C artifacts read-only, fingerprint-verified unmodified) produced a
**readiness / protocol-freeze package — NOT a final evaluation**: it froze the two unchanged
checkpoint artifacts/paths, proposed 7F GroupDRO as final-test primary and 7F center-stratified ERM
as matched control, and specified the candidate operating-point family (id_val-selected; fixed spec
0.80/0.85/0.90/0.95 + fixed sens 0.80/0.90/0.95) and the 8C development temperatures (GroupDRO
**T=2.974907**, ERM **T=3.496293**); drafted the
single-shot `ood_test` protocol (`docs/FROZEN_FINAL_EVAL_PROTOCOL_DRAFT.md`), output schema
(`docs/FINAL_TEST_OUTPUT_SCHEMA.md`), stop rules (`docs/FINAL_TEST_STOP_RULES.md`), a claims-unlock
map, and the `ood_test` authorization template (`docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md`). The
proposed pair, final threshold, and final calibration policy remain **not committed or authorized**
until the user explicitly completes the template. After authorization, the exact unchanged
checkpoint artifacts become the committed final-test pair. `ood_test` access remains locked.

The prior milestone (context): **Milestone 8C** (`scripts/34_calibrate_full_development_eval.py`)
reused the saved 8B full-split
prediction CSVs to temperature-scale the two frozen 7F models on the **full** `ood_val`
(n=34,904) as development-stage calibration: GroupDRO **T=2.974907** (NLL **0.7591→0.4451**
at four decimal places; one source summary displays 0.7592 due to source-display rounding),
center-stratified ERM **T=3.496293** (NLL 1.1911→0.5321) — both `converged=false` at the
200-iter cap but NLL/ECE improved (not a failure). **Calibration improved reliability for both**
(GroupDRO ECE **0.1560→0.0394**, Brier 0.1782→0.1458, NLL 0.7591→0.4451; ERM ECE **0.1829→0.0695**,
Brier 0.2070→0.1727, NLL 1.1911→0.5321) and sharply reduced high-confidence missed tumors
(GroupDRO conf≥0.90 3922→533, ≥0.99 1679→**0**; ERM conf≥0.90 5447→1693, ≥0.99 3794→5) — but is
**argmax-invariant** (acc/AUROC/AUPRC/sensitivity/specificity/FN unchanged). **GroupDRO REMAINS
BETTER than its matched ERM after full-development calibration** — GroupDRO acc 0.7847, AUROC
**0.8956**, AUPRC 0.8958, sens 0.6407, spec 0.9288, FN 6271 vs ERM acc 0.7675, AUROC **0.8673**,
AUPRC 0.8757, sens 0.6074, spec 0.9276, FN 6852; ≤ high-confidence FN at every threshold/variant;
better-calibrated both raw and calibrated. Calibration is **development-stage** (fit + evaluated on
the same `ood_val`) — a reliability audit, **NOT** an independent-test calibration claim (reviewer
risk **R7 open**). **No committed clinical/paper threshold; `ood_test` never read (OVERALL PASS).**
8C's finding was consolidated by the 8D readiness/protocol-freeze package described above.

The earlier arc (context): Through 4B the pipeline was development-only.
**5A** was the first authorized `ood_val` use — temperature scaling (T≈3.12). **5B**
computed deterministic uncertainty / error-detection metrics from the saved 4B/5A
prediction CSVs. **5C** converted the 5B uncertainty scores into **selective-prediction /
abstention tables**. **6A** consolidated 4B/5A/5B/5C into a **reliability reporting package**
(4 report tables + 5 matplotlib figures + a text summary). **6B** is a **documentation-only
manuscript/readiness audit package**. **7A** trained a moderate **stain/color augmentation ERM
baseline** (did NOT improve OOD at the default 0.5 threshold). **7B** (CSV-only) showed that
collapse is **mostly a threshold artifact** (ood_val ranking largely preserved). **7C**
(CSV/logit-only) calibrated the stain-aug model (T=3.2102; ECE 0.3560→0.1908; high-confidence
missed tumors 399→0) but calibration is **argmax-invariant** and **plain ERM stayed stronger at
every matched fixed-specificity candidate point**. **7D** trained a **tuned stain/color
augmentation v2 ERM baseline** (torchvision only — stronger contrast + mild affine, hue/sat held
low; NOT HED/Macenko, NOT GroupDRO/CORAL/DANN, NOT MC-dropout/ensemble, NOT calibration) and
compared plain ERM vs 7A vs 7D. **7D finding (honest negative):** the tuned v2 augmentation did
**not** improve OOD — ood_val AUROC **0.7397** (below plain **0.7696** and 7A **0.7492**; 7D −
plain = −0.0299, 7D − 7A = −0.0095), AUPRC 0.6614 (below both); default-0.5 ood_val sensitivity
0.0650 (7A 0.096, plain 0.405; FN 935); and at every matched id_val-selected fixed-specificity
target 7D sensitivity was **below both** plain ERM and 7A (7D ≥ 7A at 0/4; 7D ≥ plain at 0/4).
id_val preserved (accuracy 0.964→0.944, AUROC ~0.9925). **Recommendation:** MIXED / DO NOT
PROMOTE — naive→tuned **torchvision** stain/color augmentation is an insufficient
stain-robustness lever here; the next candidate is a **stain-specific (HED/Macenko) or dedicated
DG method**, gated. Do not claim superiority over plain ERM. **7E-0** (transform-only smoke
test, no training/inference) then implemented and **validated** a pathology-specific
optical-density (HED-style) **stain-space** augmentation (`StainSpaceAugment`): on 16 cached
`train` patches × 3 augmentations it produced numerically safe output (shape [3,96,96], RGB in
[0,254], no NaN/Inf) and pathology-plausible previews (H&E stain variation, morphology preserved;
no black/washed-out/neon) — VISUAL SANITY PASS — clearing the transform for a *gated* 7E training
proposal. **7E** (gated training, authorized) then **trained** a ResNet18 ERM with the validated
`StainSpaceAugment` (train-only OD/HED-style stain-space augmentation; id_val selection by
`id_val_loss` only; ood_val read only after training) and ran a **4-model** development comparison
(plain ERM / 7A / 7D / 7E). **7E finding (honest mixed/negative):** stain-space augmentation did
**not** lift OOD *ranking* — ood_val AUROC **0.7317** is the **lowest** of the four (plain 0.7696,
7A 0.7492, 7D 0.7397; 7E − plain −0.0379, 7E − 7A −0.0175, 7E − 7D −0.0080) and ood_val AUPRC
0.6244 is also lowest; default-0.5 ood_val sensitivity 0.0840 (FN 916). **But** at matched
id_val-selected fixed-specificity operating points 7E is the **strongest stain-augmentation
variant** — it beats **both** 7A and 7D at **4/4** targets (7E ood_val sensitivity
0.897/0.788/0.653/0.303 vs 7A 0.677/0.587/0.457/0.270 and 7D 0.610/0.531/0.380/0.178 at
spec 0.80/0.85/0.90/0.95), though it remains **below plain ERM at 0/4**. id_val was **preserved**
(accuracy 0.958, AUROC 0.9933, sensitivity 0.940, FN 15). **Recommendation: TUNE FURTHER / DO NOT
PROMOTE** — the OD-space transform improves operating-point behavior over the torchvision stain
baselines but does not beat plain ERM on ranking; a dedicated DG objective (GroupDRO/CORAL/DANN) or
OD-range tuning is the next candidate. Prior deterministic headline
(unchanged): ID→OOD accuracy 0.964→0.620 / AUROC 0.993→0.770 / sensitivity 0.944→0.405;
plain-ERM calibration ECE 0.233→0.092 (−61%); entropy error-detection AUROC id_val 0.927 vs
ood_val 0.688; @50% coverage id_val risk→0.000 but ood_val_calibrated risk only 0.380→0.250.
**`ood_test` remains LOCKED and untouched.** Not a final/paper result; no operating point
committed; no calibration in 7D/7E; no training/inference in 7E-0. **Milestone 7F-0 (DG
feasibility / group-label audit — no training, no inference, no `ood_test`) is complete:** all
three balanced caches (`train`/`id_val`/`ood_val`) DO preserve full group metadata (`center`,
`patient`, `slide`, `node`, `image_id`, `x_coord`, `y_coord`), so 7F is **not** blocked by missing
metadata. **However**, the capped `train` cache is **center↔label confounded** — center 0 is ~9%
tumor (100/1100) and center 3 is 100% tumor (900/900), and locked source **center 4 is absent** —
so center-based **GroupDRO/CORAL/DANN are RUNNABLE but NOT SOUND** on the current cache (a model
would learn center==label). **Recommendation A: a center-STRATIFIED metadata-rich cache rebuild
(new filenames; no overwrite; proposal only) is required before any sound DG-by-center training.**
**Milestone 7F-1 (that center-stratified cache rebuild — gated, NO training, NO inference, NO
`ood_test`) is now complete:** new center×label-balanced caches were built by per-`(center,label)`
quota sampling across source centers `{0,3,4}` (train 300/cell → n=1800, 900/900, 600 per center;
id_val 75/cell → n=450, 225/225, 150 per center), preserving all seven metadata fields and the
existing serializer schema, written to **new** filenames
(`center_strat_train_c034_percell300_seed1337_shuf.pt`,
`center_strat_id_val_c034_percell75_seed1337_shuf.pt`) with **no overwrite** of any existing cache.
**DG-by-center is now SOUND** — the new caches support GroupDRO / CORAL / DANN by center; the old
confounded caches are kept only for 4A/5A/7A–7E reproducibility and must be avoided for DG training.
**Milestone 7F (the gated DG training step) is now also complete:** `scripts/26` trained TWO
ResNet18 models on the SAME new center-stratified `train` cache — a **matched center-stratified ERM
control** and a **GroupDRO-by-center** model (`src/training/groupdro.py`, exponentiated-gradient `q`
update over centers {0,3,4}) — both selected by center-stratified `id_val_loss` only (ERM best epoch
2, id_val_loss 0.2481; GroupDRO best epoch 3, id_val_loss 0.2597; both reload-verified). `scripts/27`
read `ood_val` **only after training** for the fair comparison. **Finding: GroupDRO beat its matched
ERM control on OOD** — ood_val AUROC **0.7129 vs 0.6180 (+0.0949)**, AUPRC 0.6287 vs 0.5214, default
sensitivity 0.0700 vs 0.0100, matched fixed-specificity ood_val sensitivity **4/4** targets — the
first DG lever to beat its matched control on OOD ranking. GroupDRO did **not** improve worst-center
id_val behavior (it up-weighted center 4, whose id_val loss stayed highest). Old plain/7A/7D/7E are
**reference-only** (old confounded cache, different id_val subset — not a controlled comparison).
**Milestone 7G (the gated CSV/logit-only calibration + candidate operating-point audit of the two 7F
models) is now also complete:** `scripts/28` temperature-scaled each matched 7F model on its `ood_val`
logits (center-strat ERM T=3.9239, GroupDRO T=3.6570; weights unchanged), reported raw-vs-calibrated
reliability, audited high-confidence false negatives, and audited candidate `id_val`-selected operating
points. **Calibration improved confidence reliability for BOTH models** (ERM ECE 0.4984→0.3698,
GroupDRO ECE 0.4258→0.2529; both Brier & NLL down; high-confidence missed tumors reduced sharply, both
→0 at conf≥0.99) but is **argmax-invariant** (default hard predictions unchanged). **GroupDRO remains
better than its matched ERM after the audit** — higher ood_val AUROC (0.7129 vs 0.6180) and ood_val
sensitivity ≥ ERM at all 4 fixed-specificity targets (4/4 strictly greater), and it is the
**better-calibrated** model both raw and calibrated. **Recommendation: RETAIN GroupDRO with its
calibrated confidence output; candidate (not committed) operating points only.** **Milestone 7H
(the reporting-only consolidated DG reporting package) is now also COMPLETE:**
`scripts/29_make_dg_reporting_package.py` consolidated the saved deterministic-reliability,
stain-augmentation, center-stratified-cache, GroupDRO, calibration, and operating-point-audit results
into paper-facing DG tables (6), figures (5), two new docs (`docs/DG_RESULTS_INTERPRETATION.md`,
`docs/DG_MANUSCRIPT_UPDATE.md`), and a text report — clearly separating the **fair controlled
comparison (7F center-stratified ERM vs 7F GroupDRO)** from the **context-only** old plain ERM / 7A /
7D / 7E (older/different confounded caches), and updating the claims / reviewer-risk / figure-table
docs (R3 no-DG-baseline reduced HIGH→partly-mitigated; R1/R4/R7 remain open). It performed **no
training, no inference, no calibration fitting, no MC-dropout, no ensemble, no CORAL/DANN**, read only
saved artifacts (verified unmodified), touched **no `ood_test`**, committed **no operating threshold**,
and made **no final-performance claim** (OVERALL PASS; all assertions; no warnings/errors).
**Historical sequence note:** Milestone 7I then became the completed handoff point. **Milestone 7I (the docs/tables/logs-only
pre-final-evaluation decision audit) is now also COMPLETE** (`scripts/30_make_pre_final_decision_audit.py`):
it recorded frozen vs not-frozen decisions before any larger/full development evaluation or final
`ood_test`, named the primary controlled candidate (**7F GroupDRO with temperature-scaled confidence**),
the matched control (**7F center-stratified ERM**), and the context-only baselines (**old plain ERM, 7A,
7D, 7E**), and produced a decision tree (`docs/PRE_FINAL_EVALUATION_AUDIT.md`, `docs/FROZEN_DECISIONS.md`,
`docs/NEXT_EXPERIMENT_DECISION_TREE.md`; tables `results/tables/exp07i_*`; report
`results/logs/exp07i_pre_final_evaluation_audit_report.txt`) — with **no training, no inference, no
calibration fitting, no MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final-performance claim,
and no committed clinical threshold** (inputs verified unmodified; OVERALL PASS). 7I's recommended next
action was Path A (full development-scale evaluation planning). **Milestone 8A (the planning-only full
development-scale evaluation planning package) is now also COMPLETE**
(`scripts/31_make_full_development_eval_plan.py`): it produced a full development-scale evaluation
planning/protocol package — four docs (`docs/FULL_DEVELOPMENT_EVALUATION_{PROTOCOL,RESOURCE_PLAN,
OUTPUT_SCHEMA,STOP_RULES}.md`), four tables (`results/tables/exp08a_{evaluation_scope_matrix,
model_checkpoint_freeze_table,metric_output_plan,compute_storage_risk_table}.csv`), and a report
(`results/logs/exp08a_full_development_evaluation_planning_report.txt`) — that names the exact models
(frozen **7F GroupDRO** primary candidate + **7F center-stratified ERM** matched control; old plain
ERM/7A/7D/7E optional context-only), the frozen checkpoints, the development splits (full/larger `id_val`
{0,3,4} + full/larger `ood_val` {1}; **no `ood_test`**), the threshold/calibration policy, the required
metrics/output schema, the compute/storage risks, and nine stop rules — with **no training, no inference,
no calibration fitting, no MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final-performance
claim, and no committed clinical threshold** (frozen inputs read-only; 23/23 assertions; OVERALL PASS).
**Historical sequence note:** Milestone 8B then became the completed handoff point. **Milestone 8B (the inference-only execution
of the full development-scale evaluation plan) is now also COMPLETE**
(`scripts/32_run_full_development_eval.py` + `scripts/33_summarize_full_development_eval.py`): it
loaded the two FROZEN 7F checkpoints (exact-path verified; weights unchanged) and ran eval-transform
**inference** on the FULL development splits — `id_val` centers {0,3,4} (n=33,560) and `ood_val`
center {1} (n=34,904) — writing four prediction CSVs under `results/predictions/exp08b_full_dev_eval/`
and a CSV-only metrics/summary package. **No training, no weight update, no calibration fitting, no
MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`** (both scripts OVERALL PASS; all assertions).
**On the full/larger development splits GroupDRO still beat its matched center-stratified ERM
control** — ood_val AUROC **0.8956 vs 0.8673 (+0.0283)**, AUPRC 0.8958 vs 0.8757, default-0.5
sensitivity 0.6407 vs 0.6074, matched fixed-specificity ood_val sensitivity **4/4** targets, and (new
at full scale) GroupDRO improved worst-center (center 4) id_val behavior (acc 0.6901→0.8031). **Scale
finding:** full-split OOD AUROC is far higher than the 2000-sample capped `ood_val` (GroupDRO
0.7129→0.8956; ERM 0.6180→0.8673) — the capped subset understated both models; the GroupDRO advantage
persists. Candidate operating points are id_val-selected only; **no clinical threshold committed**;
high-confidence FN audit is RAW (no calibration fit in 8B). **Milestone 8C (full-development
calibration + reliability audit) is now also COMPLETE and ACCEPTED** — see the one-line status
above: it temperature-scaled the two frozen 7F models on the full `ood_val` (GroupDRO T=2.9749,
ERM T=3.4963), improved ECE/Brier/NLL for both, sharply cut high-confidence missed tumors, and
confirmed **GroupDRO remains better than its matched ERM** (ood_val AUROC 0.8956 vs 0.8673);
argmax-invariant; development-stage only; no committed threshold; `ood_test` never read.
**Milestone 8D (final protocol freeze / final-evaluation readiness package — docs/tables/logs
only) is now also COMPLETE and ACCEPTED** — see the one-line status above: it produced a
readiness/protocol-freeze package (froze the checkpoint artifacts/paths; proposed the 7F GroupDRO
primary + 7F center-stratified ERM matched-control final-test pair; specified the candidate
operating-point family + 8C temperatures;
drafted the single-shot `ood_test` protocol, output schema, stop rules, claims-unlock map, and the
`ood_test` authorization template), committing **no final model, no final threshold, no final
calibration policy, and no `ood_test` access**. **Historical 8D handoff recommendation
(subsequently completed by 8E–8H):** review the protocol and prepare the development-evidence
manuscript before any authorization. A later abstention operating-point
commitment remains a separate gated step. Model-based
uncertainty (MC-dropout / ensembles) remains **DEFERRED / optional** (only if explicitly approved;
not the default next step); the final `ood_test` evaluation remains fully LOCKED and may proceed
**only** after the user completes `docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md`. Later gated steps
are an abstention operating-point commitment and the single final `ood_test` evaluation.

---

## 2. Environment (validated)

- Windows · Python **3.12.10** in `.venv`
- PyTorch **2.9.0+cu128** · CUDA available **True** · GPU **NVIDIA GeForce RTX 3060**
- torchvision 0.24.0+cu128 · `datasets` **5.0.0**
- Interpreter: `.venv/Scripts/python.exe`; run scripts from project root.
- The original HF cache paths were machine-local and are intentionally redacted in the public
  repository (partial cache — streaming shards only). The project drive had **71.87 GB free**.

## 3. Dataset (locked)

- Hugging Face **`wltjr1007/Camelyon17-WILDS`** via streaming. No WILDS/CodaLab.
- `image` may be RGBA → convert to RGB. `label`: 0 = non-tumor, 1 = tumor.

## 4. Locked split mapping

| Logical split | HF split     | Centers | Expected total |
|---------------|--------------|---------|----------------|
| `train`       | `train`      | {0,3,4} | 302,436        |
| `id_val`      | `validation` | {0,3,4} | 33,560         |
| `ood_val`     | `validation` | {1}     | 34,904         |
| `ood_test`    | `test`       | {2}     | 85,054         |

## 5. No-go rules (active)

- No random splits.
- **`ood_val`** — **AUTHORIZED as of 2026-07-08** (Milestone 5A) for OOD decisions ONLY
  (calibration / temperature scaling; later: selective-prediction thresholds, model
  selection under shift). **Never** for training. First use: temperature scaling in
  `scripts/11_calibrate_ood_val.py`.
- **`ood_test`** — the single authorized final evaluation is complete. Do not access it again,
  rerun inference, reset the one-shot state, or replace the exp09 final result.
- No WILDS/CodaLab download.
- Smoke/development runs are **not** final results; Milestone 9A exp09 is the locked final
  center-2 evaluation.
- Pipeline stays script-based; notebooks for inspection only.
- No new longer/full training or later milestones without an explicit go-ahead.

> **Note:** `CLAUDE.md` §5 rule 2 has been **reconciled** (Milestone 7H documentation update)
> to match this active authorization — `ood_val` is authorized for development-stage
> comparison / calibration / uncertainty / selective analysis only, never training, never
> final test reporting, and is **no longer described as "off-limits"** except for uses outside
> those authorized development purposes.

## 6. What exists in code (by module)

- `src/utils/`: `seed.py`, `io.py`, `ssl_setup.py`.
- `src/data/`: `hf_camelyon17.py` (locked loader), `inspect_dataset.py`,
  `transforms.py` (RGB→float32→ImageNet norm; 96×96 guard), `torch_dataset.py`
  (`Camelyon17HFDataset`), `dataloaders.py` (`build_dataloader`; shuffle only for
  train), `balanced_subset.py` (balanced collection + **disk cache**),
  `stain_transforms.py` (Milestone 7A — train-only moderate stain/color augmentation:
  `EnsurePILRGB`, `StainColorAugment`, `build_stain_train_transform`; **Milestone 7D** added
  the tuned v2 recipe `StainColorAugmentV2` + `build_stain_color_aug_v2_transform` — stronger
  contrast + mild affine, hue/sat held low; **Milestone 7E-0** added the optical-density
  (HED-style) **stain-space** recipe `StainSpaceAugment` + `build_stain_space_aug_transform`
  — per-channel OD scale/bias perturbation, no hue/sat jitter; torchvision).
- `src/models/`: `backbones.py` (`create_backbone` resnet18/50, fc→Identity),
  `classifier.py` (`CamelyonClassifier` → logits `[B,2]`).
- `src/training/`: `losses.py` (`create_loss` CE), `optimizers.py`
  (`create_optimizer` AdamW), `steps.py` (`train_step`/`eval_step`), `loop.py`
  (`train_one_epoch`/`evaluate`), `checkpointing.py` (`save_checkpoint`/`load_checkpoint`
  → best.pt/last.pt), `history.py` (`append_history_csv`), `groupdro.py` (Milestone 7F —
  `compute_group_losses`, `GroupDROLoss` with the exponentiated-gradient `q_g ←
  q_g·exp(step·detached_L_g)` update + renormalize, absent groups not updated; `groups_from_batch`,
  `summarize_group_counts`; groups = centers {0,3,4}; CE base loss).
- `src/evaluation/`: `metrics_classification.py`
  (`compute_binary_classification_metrics`, tumor=1 positive),
  `predict.py` (`predict_to_dataframe` → per-example logits/probs/pred DataFrame),
  `metrics_calibration.py` (`compute_ece`/`compute_brier_score`/
  `compute_nll_from_probs`/`compute_calibration_metrics`),
  `metrics_uncertainty.py` (score fns `max_confidence`/`uncertainty_from_confidence`/
  `predictive_entropy`/`probability_margin`/`uncertainty_from_margin`/
  `energy_score_from_logits`; eval fns `compute_error_detection_metrics`/
  `compute_high_confidence_error_summary`/`compute_high_confidence_false_negative_summary`/
  `compute_risk_coverage_curve`/`compute_aurc` — Milestone 5B, CSV-only),
  `metrics_selective.py` (`selective_metrics_at_coverages`/`threshold_for_target_coverage`/
  `retained_subset_by_coverage`/`summarize_selective_prediction`/
  `summarize_false_negatives_under_abstention` — Milestone 5C, CSV-only,
  rank-based retention),
  `metrics_operating_points.py` (`compute_metrics_at_threshold`/
  `threshold_for_target_specificity`/`threshold_for_target_sensitivity`/
  `make_threshold_sweep`/`apply_operating_points` — Milestone 7B, CSV-only threshold /
  operating-point analysis; tumor=1 positive; id_val selects thresholds, ood_val compares).
- `src/calibration/`: `temperature_scaling.py` (`TemperatureScaler`,
  `fit_temperature`, `apply_temperature_to_logits`).
- `scripts/`: `00`–`25` (see `docs/RUNBOOK.md`), `_bootstrap.py`. Milestone 7C reused
  existing `src/` modules only (calibration + evaluation) — no new `src/` module; delivered
  `scripts/18_calibrate_stainaug_and_operating_audit.py`. Milestone 7D added the v2 transform
  to `src/data/stain_transforms.py` and delivered
  `scripts/19_train_stainaug_v2_checkpointed.py` (training) +
  `scripts/20_eval_stainaug_v2_id_ood.py` (eval + 3-model comparison; reuses
  `metrics_classification` / `metrics_operating_points` / `predict`). Milestone 7E-0 added the
  stain-space transform to `src/data/stain_transforms.py` and delivered
  `scripts/21_stain_space_transform_smoke_test.py` (transform-only preview + numerical sanity;
  no model, no inference). Milestone 7E reused the validated `StainSpaceAugment` (no new `src/`
  module) and delivered `scripts/22_train_stainspace_checkpointed.py` (training) +
  `scripts/23_eval_stainspace_id_ood.py` (eval + **4-model** comparison plain/7A/7D/7E; reuses
  `metrics_classification` / `metrics_operating_points` / `predict`). Milestone 7F-0 reused
  existing `src/` modules only (no new `src/` module) and delivered
  `scripts/24_dg_group_metadata_audit.py` (metadata/cache audit only — reads the three balanced
  caches read-only, quantifies label × center/slide counts and DG group-label feasibility; no
  training, no inference, no `ood_test`). Milestone 7F-1 reused the existing `balanced_subset`
  serializer helpers only (no new `src/` module; `src/` unchanged) and delivered
  `scripts/25_rebuild_center_stratified_caches.py` (gated no-training center-stratified cache
  rebuild — HF streaming, per-`(center,label)` quota sampling across centers {0,3,4}, new cache
  filenames, no overwrite; no inference, no `ood_test`). Milestone 7F added `src/training/groupdro.py`
  and delivered `configs/experiments/exp07f_groupdro_resnet18.yaml`,
  `scripts/26_train_centerstrat_erm_groupdro.py` (trains a matched center-stratified ERM control +
  a GroupDRO-by-center model on the new center-stratified caches; `id_val_loss` selection only;
  per-center id_val loss/acc, q weights, and per-group losses logged) +
  `scripts/27_eval_centerstrat_erm_groupdro_id_ood.py` (eval on center-strat id_val + balanced
  ood_val; overall + per-center metrics; id_val-selected fixed-specificity operating points; fair
  7F-ERM-vs-7F-GroupDRO comparison + old plain/7A/7D/7E reference; reuses `metrics_classification` /
  `metrics_operating_points` / `predict`). Milestone 7G reused existing `src/` modules only
  (`src/calibration/temperature_scaling.py` + `src/evaluation/*`; no new `src/` module) and delivered
  `scripts/28_calibrate_7f_groupdro_operating_audit.py` (CSV/logit-only calibration + candidate
  operating-point audit of the two matched 7F models — temperature-scales each on `ood_val` logits,
  raw-vs-calibrated ECE/Brier/NLL, high-confidence-FN audit, id_val-selected candidate operating
  points; no training, no inference, no `ood_test`). Milestone 8B reused existing `src/` modules
  only (`evaluation/predict.py`, `evaluation/metrics_classification.py`,
  `evaluation/metrics_operating_points.py`, `models/classifier.py`, `training/checkpointing.py`,
  `data/torch_dataset.py`, `data/transforms.py`; no new `src/` module — `src/` unchanged) and
  delivered `scripts/32_run_full_development_eval.py` (inference-only full-split eval of the two
  frozen 7F checkpoints on FULL `id_val` {0,3,4} + `ood_val` {1}; writes four prediction CSVs with
  a `confidence` column; no training, no calibration, no `ood_test`) +
  `scripts/33_summarize_full_development_eval.py` (CSV-only overall + per-center + candidate
  operating-point + raw high-confidence-FN summary, tables/figures/JSON/report; no inference, no
  `ood_test`).

## 7. Known issues & fixes (carry forward)

- **CodaLab/WILDS download unreliable** → HF streaming only.
- **Norton TLS interception** → fixed by `src/utils/ssl_setup.py` (combined CA bundle;
  runs from `scripts/_bootstrap.py` before HF import).
- **HF stream is label-ordered.** train label-0 (non-tumor) block begins at scan
  **~128,182** (build reached balance at `scanned=128,310`). id_val first label-0 at
  index **14,033**. First 1,000 samples of both splits are single-class.
- **Balanced subset cache required.** Built once, then instant cache HIT:
  - `results/cache/balanced_train_pc128_seed1337_shuf.pt` (~12.3 MB) — Milestone 3C
  - `results/cache/balanced_id_val_pc64_seed1337_shuf.pt` (~6.1 MB) — Milestone 3C
  - `results/cache/balanced_train_pc1000_seed1337_shuf.pt` (~92.4 MB) — Milestone 4A
    (train build scanned to `scanned=129,283` to reach the label-0 block)
  - `results/cache/balanced_id_val_pc250_seed1337_shuf.pt` (~23.0 MB) — Milestone 4A
    (id_val build scanned to `scanned=14,308`)
  - `results/cache/balanced_ood_val_pc1000_seed1337_noshuf.pt` (~84.6 MB) — Milestone 5A
    (ood_val center-1 build scanned to `scanned=18,452`; label-0 block first, then
    label-1; `shuffle_collected=false`)
  - `results/cache/center_strat_train_c034_percell300_seed1337_shuf.pt` (~90 MB) —
    Milestone 7F-1 (center-stratified: per-`(center,label)` quota 300 across centers
    {0,3,4}; n=1800, 900/900, 600 per center; train stream scanned to 221,387)
  - `results/cache/center_strat_id_val_c034_percell75_seed1337_shuf.pt` (~22 MB) —
    Milestone 7F-1 (center-stratified: per-`(center,label)` quota 75 across centers
    {0,3,4}; n=450, 225/225, 150 per center; validation stream scanned to 24,466)

## 8. Latest accepted results

### Milestone 9A — Single explicitly authorized final locked `ood_test` evaluation

**COMPLETE / ACCEPTED.** One inference attempt only; `RUN_STARTED`, inference-complete, and
summary-complete sentinels exist; attempt counter = 1. The run used HF split `test`, center `{2}`,
85,054 rows/model, the exact frozen checkpoint hashes and temperatures, and all 14 frozen
`id_val`-selected candidate thresholds. Both models ran in one dataloader traversal. No training,
calibration fitting, test-set threshold selection, or post-test model selection occurred.

The matched ERM outperformed the predeclared GroupDRO primary on final AUROC, AUPRC, accuracy,
sensitivity, F1, and FN count. GroupDRO retained higher specificity and slightly higher precision.
Frozen temperature scaling improved ECE/Brier/NLL for both while leaving hard predictions and
ranking unchanged. Complete exact values are recorded in Section 1 and in
`results/metrics/exp09_final_ood_test/final_metrics.json`; the complete final report is
`results/logs/exp09_final_ood_test_report.txt`.

### Milestone 8G — Manuscript-governance wording refresh (docs only)

A **documentation wording cleanup — NOT an experiment and NOT a final evaluation.** No script; no
`results/` artifact touched; **no accepted metric value changed.** **No training, no inference, no
calibration fitting, no MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final test run, no
final-performance claim.** 8G resolved the single non-blocking follow-up flagged by 8F.

- **Edited:** `docs/CLAIMS_AND_LIMITATIONS.md` and `docs/FIGURE_TABLE_MANIFEST.md` (the two flagged
  docs), plus context docs (`CLAUDE.md`, `AGENTS.md`, `docs/SESSION_STATE.md`, `docs/MILESTONE_LOG.md`,
  `docs/NEXT_ACTIONS.md`).
- **What changed (wording only):** both governance docs now distinguish **early capped/subset
  development evidence (4B–7G; id_val n=500 / ood_val n=2000)** from **full-development evidence
  (8B/8C; id_val {0,3,4} n=33,560 / ood_val {1} n=34,904)**. `CLAIMS_AND_LIMITATIONS.md` gained a
  full-development allowed-claims block (A8–A11 / F11–F13) and precise universal caveats;
  `FIGURE_TABLE_MANIFEST.md` gained 8B/8C artifact inventories (existing result paths only), era tags,
  8D/8E/8F docs-only labels, an explicit reserved final-test **LOCKED / NOT RUN / not produced** note,
  and era-specific caption tags.
- **Preserved:** all forbidden claims (no final generalization / clinical readiness /
  independent-test calibration / committed clinical threshold / GroupDRO-vs-old-plain-ERM controlled
  claim / all-hospital robustness / `ood_test` conclusion); reviewer risk **R7**; the 8D final-test
  protocol; the 8E manuscript skeleton (unchanged).
- **Assertions (all passed):** no `ood_test` usage; no training/inference; no calibration fitting; no
  result artifact modified; only allowed docs modified; no accepted metric value changed; no
  final-performance / clinical-readiness / independent-test-calibration claim introduced; CLAUDE.md <
  30,000 chars; AGENTS.md < 20,000 chars. **OVERALL PASS.** `ood_test` remains fully LOCKED and unread;
  8G does not authorize it.

### Milestone 8F — Final protocol/manuscript consistency audit (docs/tables/logs only)

An **audit / documentation package — NOT a final evaluation.** `scripts/37_make_final_pre_authorization_audit.py`
(**does not import torch; no training, no inference, no calibration fitting, no MC-dropout, no
ensemble, no CORAL/DANN, no `ood_test`, no final test run, no final-performance claim**; reads only
existing docs/tables read-only, fingerprint-verified unmodified) audited the 8D final-test protocol
package against the 8E manuscript skeleton as the final safety/readiness check before any `ood_test`
authorization. OVERALL PASS (14/14 assertions; no warnings/errors).

- **Consistency verdict:** CONSISTENT — READY FOR MANUAL REVIEW, **no blocking issues** (16/16
  items MATCH). Verified: same primary model (7F GroupDRO), matched control (7F center-stratified
  ERM), checkpoint paths, calibration temperatures (GroupDRO **T=2.974907** / ERM **T=3.496293**),
  operating-point family (id_val-selected spec 0.80/0.85/0.90/0.95 + sens 0.80/0.90/0.95),
  `ood_val`/`ood_test` distinction; **no** final-test numbers in the manuscript; **no** clinical /
  independent-test-calibration / GroupDRO-vs-old-plain-ERM controlled claim; final-test placeholder
  marked LOCKED / NOT RUN.
- **Pre-authorization checklist (11 items):** 8 `needs_user_decision` (the 7 authorization boxes +
  checkpoint-path approval) + 3 pass (final-test script not run, authorization template not completed,
  one-shot rule understood); **0 fail**. All 8 decisions map to the authorization template — model /
  threshold / calibration policy stay NOT frozen until the user checks them.
- **Remaining-risk register (8 rows):** R1 (capped subsets, reduced/superseded by 8B/8C), R4 (no final
  `ood_test`, HIGH), R7 (same-split development calibration, HIGH), lack of CORAL/DANN (R3 residual),
  model-based uncertainty deferred (R8), no clinical threshold, no external non-Camelyon validation
  (R6), one-shot `ood_test` irreversibility (HIGH).
- **Non-blocking follow-up:** refresh legacy capped-subset wording in `docs/CLAIMS_AND_LIMITATIONS.md`
  (header + universal caveats) and `docs/FIGURE_TABLE_MANIFEST.md` (caption tag), which predate the
  8B/8C full-dev evidence the manuscript now cites — wording only; changes no result and unlocks no
  `ood_test` access.
- **8F artifacts:** `scripts/37_make_final_pre_authorization_audit.py`;
  `docs/{FINAL_PRE_AUTHORIZATION_AUDIT,MANUSCRIPT_PROTOCOL_CONSISTENCY_CHECK}.md`;
  `results/tables/exp08f_{protocol_manuscript_consistency,pre_authorization_checklist,remaining_risk_register}.csv`;
  `results/logs/exp08f_final_pre_authorization_audit_report.txt`. `ood_test` remains fully LOCKED and
  unread; 8F does **not** authorize it.

### Milestone 8E — Manuscript skeleton on development evidence (docs/tables/logs only)

A **writing / planning package — NOT a submission and NOT a final evaluation.** No script was
required (docs authored directly). **No training, no inference, no calibration fitting, no
MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final-performance claim; no result file
modified.** Built **only** from completed development-stage evidence (6A/7H/8B/8C/8D), it produced a
full paper skeleton (title options, abstract outline, section lists, and a marked LOCKED / NOT-RUN
final-test placeholder), methods/results/discussion prose drafts (development numbers only), a
table/figure manifest, and a claims map. Every quantitative statement is scope-tagged
development-stage; the R7 caveat is preserved throughout (8C calibration fit + evaluated on the same
`ood_val` → development-stage reliability audit, not an independent-test calibration claim). Allowed
claims (scope-tagged): GroupDRO > matched ERM in development; worst-center (center 4) improvement;
temperature scaling improves calibration on `ood_val` (argmax-invariant); high-confidence missed
tumors reduced after calibration; hospital shift degrades reliability. Forbidden before `ood_test`:
final generalization, independent-test calibration validity, superiority over untested DG methods,
GroupDRO-beats-old-plain-ERM as a controlled claim, clinical readiness, committed clinical
threshold, all-hospital robustness, MC-dropout/ensemble findings, any `ood_test` conclusion.
**OVERALL PASS** (all assertions; no warnings/errors). `ood_test` remains fully LOCKED and unread.
Artifacts: `docs/{MANUSCRIPT_SKELETON_DEV_EVIDENCE,MANUSCRIPT_METHODS_DRAFT_DEV_ONLY,MANUSCRIPT_RESULTS_DRAFT_DEV_ONLY,MANUSCRIPT_DISCUSSION_POINTS}.md`;
`results/tables/exp08e_{manuscript_table_manifest,manuscript_claims_map}.csv`;
`results/logs/exp08e_manuscript_skeleton_report.txt`.

### Milestone 8D — Final protocol freeze / final-evaluation readiness package (docs/tables/logs only)

A **readiness / protocol-freeze package — NOT a final evaluation.** `scripts/35_make_final_eval_
readiness_package.py` (**does not import torch; no training, no inference, no calibration fitting,
no MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`; reads only saved 8B/8C artifacts
read-only, fingerprint-verified unmodified**) consolidated the 8B/8C development evidence and the
frozen-decision record into a final-evaluation readiness package. It froze the checkpoint artifacts
and exact paths (`results/checkpoints/exp07f_groupdro_resnet18/best.pt` and
`results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt`), proposed 7F GroupDRO as final-test
primary and 7F center-stratified ERM as matched control, specified the candidate operating-point
family (id_val-selected; fixed spec 0.80/0.85/0.90/0.95 + fixed sens 0.80/0.90/0.95), and recorded
the **8C development temperatures** (GroupDRO **T=2.974907**, ERM **T=3.496293**);
drafted the single-shot `ood_test` protocol, output schema, stop rules (FT-1…FT-9), a claims-unlock
map, and the `ood_test` authorization template. The proposed pair, final threshold, final calibration
policy, and `ood_test` access remain uncommitted/unauthorized until explicit template completion.
After authorization, the exact unchanged checkpoint artifacts become the committed final-test pair.
Recommended calibration policy = report **both** raw and calibrated (frozen
8C temperatures applied, never refit on `ood_test`); recommended threshold policy = **no single
final clinical threshold** (id_val-selected candidate families allowed; choosing after test
FORBIDDEN). **OVERALL PASS** (all assertions; `ood_test` NOT touched). Artifacts:
`scripts/35_make_final_eval_readiness_package.py`;
`docs/{FINAL_EVALUATION_READINESS_PACKAGE,FROZEN_FINAL_EVAL_PROTOCOL_DRAFT,OOD_TEST_AUTHORIZATION_TEMPLATE,FINAL_TEST_OUTPUT_SCHEMA,FINAL_TEST_STOP_RULES}.md`;
`results/tables/exp08d_{freeze_decision_table,final_eval_model_plan,final_eval_metric_plan,threshold_policy_options,calibration_policy_options,claims_unlock_table}.csv`;
`results/logs/exp08d_final_eval_readiness_report.txt`. Readiness/protocol package — NOT a final
evaluation; `ood_test` remains LOCKED and unread.

### Milestone 8B — Full development-scale evaluation (inference-only execution)

An **inference-only** execution of the Milestone 8A plan. **No training, no weight update, no
checkpoint modification, no calibration fitting, no MC-dropout, no ensemble, no CORAL/DANN; `ood_test`
never read; new run names only (no overwrite of prior milestones).** `scripts/32` loaded the two
FROZEN 7F checkpoints (exact-path verified) and ran eval-transform inference on the FULL development
splits; `scripts/33` computed the CSV-only summary. Positive class = tumor (1).

- **Splits (exact):** `id_val` n=**33,560** (l0=16,952 / l1=16,608), centers **{0,3,4}**; `ood_val`
  n=**34,904** (l0=l1=17,452), center **{1}**. No center 1 in id_val; no center 2 anywhere.
  (The HF builder's one-time "Generating test split" console line is the library caching the whole
  repo, NOT the pipeline reading test — only `validation` is loaded and center-filtered.)
- **Overall metrics (tumor = positive, threshold 0.5):**

  | split | model | acc | auroc | auprc | sens | spec | FN | tn/fp/fn/tp |
  |-------|-------|----:|------:|------:|-----:|-----:|---:|:-----------:|
  | id_val  | 7F GroupDRO (primary)      | 0.8416 | 0.9323 | 0.9322 | 0.7572 | 0.9243 | 4033 | 15668/1284/4033/12575 |
  | id_val  | 7F center-strat ERM (ctrl) | 0.7987 | 0.9085 | 0.9131 | 0.6821 | 0.9129 | 5280 | 15475/1477/5280/11328 |
  | ood_val | 7F GroupDRO (primary)      | 0.7847 | 0.8956 | 0.8958 | 0.6407 | 0.9288 | 6271 | 16209/1243/6271/11181 |
  | ood_val | 7F center-strat ERM (ctrl) | 0.7675 | 0.8673 | 0.8757 | 0.6074 | 0.9276 | 6852 | 16188/1264/6852/10600 |

- **Per-center id_val:** worst center = **center 4** for both. GroupDRO improved center-4 acc
  0.6901→0.8031 / AUROC 0.8323→0.8980 / loss 0.6121→0.4812 (ERM slightly better on centers 0 & 3).
- **Candidate fixed-specificity ood_val sensitivity (id_val-selected; NOT committed):** GroupDRO
  0.7743/0.7335/0.6753/0.5936 vs ERM 0.6582/0.6394/0.6143/0.5745 at spec 0.80/0.85/0.90/0.95 →
  GroupDRO > ERM at **4/4**.
- **High-confidence FN audit (RAW), ood_val:** GroupDRO 3922/3131/1679 vs ERM 5447/4976/3794 at
  conf 0.90/0.95/0.99 (GroupDRO fewer high-confidence missed tumors at every threshold).
- **Controlled verdict: GroupDRO STILL beats the matched ERM at full/larger development scale — YES**
  (ood_val AUROC +0.0283, AUPRC +0.0201, default sens +0.0333, fixed-spec 4/4, improved worst-center
  id_val). **Scale finding:** full splits give much higher OOD AUROC than the capped 2000-sample
  `ood_val` (GroupDRO 0.7129→0.8956; ERM 0.6180→0.8673) — the capped subset understated both models.
- **OVERALL: PASS** (both scripts; all assertions; no errors). DEVELOPMENT-stage evaluation on
  full/larger development splits — NOT final performance; no committed operating point; no calibration
  fit; `ood_test` untouched. Artifacts:
  `results/predictions/exp08b_full_dev_eval/{groupdro,centerstrat_erm}_{id_val,ood_val}_predictions.csv`;
  `results/metrics/exp08b_full_dev_eval/full_dev_metrics.json`;
  `results/tables/exp08b_{overall_metrics,id_val_per_center_metrics,candidate_operating_points,high_confidence_false_negatives}.csv`;
  `results/figures/exp08b_{ood_val_roc_groupdro_vs_erm,ood_val_pr_groupdro_vs_erm,candidate_fixed_specificity_sensitivity,id_val_per_center_accuracy,high_confidence_fn_comparison}.png`;
  `results/logs/exp08b_full_dev_eval_inference_report.txt`, `exp08b_full_development_eval_report.txt`.

### Milestone 8A — Full development-scale evaluation PLANNING package (planning-only)

A **planning / protocol only** milestone. **No training, no model inference, no calibration fitting, no
MC-dropout, no ensemble, no CORAL/DANN; `ood_test` never referenced as data; no checkpoint / prediction
CSV / cache modified; no final-performance claim; no committed clinical threshold.**
`scripts/31_make_full_development_eval_plan.py` (does **not** import torch; frozen inputs checked
read-only, existence only) produced a package specifying **what a later gated Milestone 8B would do**:

- **Docs:** `docs/FULL_DEVELOPMENT_EVALUATION_PROTOCOL.md` (models, checkpoints, splits, threshold +
  calibration policy, required metrics, ood_test-locked statement), `..._RESOURCE_PLAN.md` (verified split
  sizes train 302,436 / id_val 33,560 / ood_val 34,904 / ood_test 85,054 LOCKED-unused; RTX 3060 runtime
  band; CSV disk estimate; batch size; cache-predictions-not-images; retry/checkpoint/order plan),
  `..._OUTPUT_SCHEMA.md` (prediction CSV required columns incl. `confidence`; metrics/calibration/
  operating-point/per-center/report schemas), `..._STOP_RULES.md` (nine stop rules S-1…S-9).
- **Tables:** `results/tables/exp08a_evaluation_scope_matrix.csv` (10 rows),
  `exp08a_model_checkpoint_freeze_table.csv` (6 rows), `exp08a_metric_output_plan.csv` (9 rows),
  `exp08a_compute_storage_risk_table.csv` (9 rows).
- **Report:** `results/logs/exp08a_full_development_evaluation_planning_report.txt`.

**Frozen for 8B (required, controlled):** primary candidate **7F GroupDRO**
(`results/checkpoints/exp07f_groupdro_resnet18/best.pt`) + matched control **7F center-stratified ERM**
(`results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt`); both present (~134 MB) and reload-verified
in 7F. **Optional context-only (NOT required):** old plain ERM, 7A, 7D, 7E. **Development splits:**
full/larger `id_val` (centers {0,3,4}) + full/larger `ood_val` (center 1); **`ood_test` excluded**.
**Recommended 8B order:** id_val first (reload sanity + candidate threshold selection + per-center), then
ood_val (apply id_val-selected candidate thresholds), then optional dev-only calibration / high-confidence
FN audit / optional uncertainty-selective, then optional context-only models; never touch `ood_test`.
**OVERALL PASS** (23/23 assertions; no warnings/errors). Planning only — NOT a result and NOT a
performance claim; `ood_test` untouched. **Next: gated Milestone 8B (execute the plan — inference only, no
training).**

### Milestone 7H — Consolidated DG reporting package (reporting-only)

A reporting-only consolidation of the accepted development results through 7G into paper-facing DG
artifacts. **No training, no inference, no calibration fitting, no MC-dropout, no ensemble, no
CORAL/DANN; `ood_test` never read; input CSV/JSON verified unmodified; no committed threshold; no
final-performance claim.** `scripts/29_make_dg_reporting_package.py` reads saved artifacts read-only
and emits 6 tables, 5 figures, 2 new docs, and a report. The **fair controlled comparison is 7F
center-stratified ERM vs 7F GroupDRO**; **old plain ERM / 7A / 7D / 7E are context-only** (older
confounded caches + different id_val subset).

Controlled result (7F ERM vs 7F GroupDRO, same sound 7F-1 cache): ood_val AUROC **0.6180 →
0.7129** (+0.0949), AUPRC 0.5214 → 0.6287 (+0.1073), default sensitivity 0.0100 → 0.0700, default FN
990 → 930; GroupDRO wins **4/4** id_val-selected fixed-specificity candidate targets (ood_val
sensitivity 0.306/0.241/0.148/0.057 vs ERM 0.021/0.015/0.013/0.008). Calibration (7G): both models
improved on ECE/Brier/NLL (argmax-invariant); high-confidence (conf≥0.90) missed tumors ERM 950→249,
GroupDRO 688→25; GroupDRO better-calibrated raw and calibrated. Context-only: old plain ERM ood_val
AUROC 0.7696 is a **reference gap, not a controlled loss**.

Reviewer-risk update (`results/tables/exp07h_reviewer_risk_update.csv`): R3 (no DG baseline) reduced
**HIGH → partly mitigated** (GroupDRO implemented + beat matched ERM; residual MEDIUM — no CORAL/DANN,
no full/final test); R9 partly mitigated (7F-1 rebuild); **R1, R4, R7 remain open**. **OVERALL PASS**
(all assertions; no warnings/errors). Development analysis on capped balanced subsets — NOT final
performance; `ood_test` untouched. Artifacts:
`results/tables/exp07h_{consolidated_model_comparison,controlled_dg_comparison,context_vs_controlled_warning_table,groupdro_calibration_summary,groupdro_operating_point_summary,reviewer_risk_update}.csv`;
`results/figures/exp07h_{ood_auroc_auprc_model_comparison,controlled_groupdro_gain,calibration_ece_brier_nll,high_confidence_fn_reduction,fixed_specificity_sensitivity}.png`;
`docs/DG_RESULTS_INTERPRETATION.md`, `docs/DG_MANUSCRIPT_UPDATE.md`;
`results/logs/exp07h_dg_reporting_package_summary.txt`.

### Milestone 7G — 7F ERM vs GroupDRO calibration + candidate operating-point audit (CSV/logit-only)

A CSV/logit-only development follow-up to 7F (reviewer risk R7 — calibration fitted/evaluated on the
same capped `ood_val`). **No training, no inference, no MC-dropout/ensemble, no CORAL/DANN; `ood_test`
never read; input prediction CSVs unmodified; checkpoints untouched.** `scripts/28` reused
`src/calibration/temperature_scaling.py` + `src/evaluation/*` (no new `src/` module) to, for each of
the two matched 7F center-stratified models: temperature-scale its `ood_val` logits, report
raw-vs-calibrated reliability, audit high-confidence false negatives, and audit candidate
`id_val`-selected operating points. The FAIR comparison is 7F ERM vs 7F GroupDRO; old plain/7A/7D/7E
are context-only and were NOT re-derived.

Temperatures (fit on ood_val logits; Adam lr 0.01, 200 iters, bounds [0.05,10.0]; both
`converged=false` at cap, acceptable): center-strat ERM **T=3.923949** (NLL 3.4859→1.0672); GroupDRO
**T=3.657000** (NLL 1.9139→0.7908).

Calibration reliability on ood_val (development):

| model | variant | ECE | Brier | NLL |
|-------|---------|----:|------:|----:|
| center-strat ERM (7F) | raw        | 0.4984 | 0.5059 | 3.4859 |
| center-strat ERM (7F) | calibrated | 0.3698 | 0.3759 | 1.0672 |
| GroupDRO (7F)         | raw        | 0.4258 | 0.4278 | 1.9139 |
| GroupDRO (7F)         | calibrated | 0.2529 | 0.2896 | 0.7908 |

Calibration improved **all three** metrics for **both** models; classification unchanged raw→cal
(argmax-invariant): ERM acc 0.4765 / sens 0.0100 / FN 990; GroupDRO acc 0.5065 / sens 0.0700 / FN 930.

High-confidence missed tumors on ood_val (`label==1 & pred==0 & conf≥t`): ERM raw 950/925/784 →
calibrated 249/40/0; GroupDRO raw 688/571/333 → calibrated 25/1/0 (t=0.90/0.95/0.99). Both reach 0 at
conf≥0.99; GroupDRO has fewer high-confidence FNs at every threshold/variant.

Candidate id_val-selected fixed-specificity operating points — ood_val sensitivity:

| target spec | center-strat ERM (7F) | GroupDRO (7F) |
|------------:|----------------------:|--------------:|
| 0.80 | 0.0210 | **0.3060** |
| 0.85 | 0.0150 | **0.2410** |
| 0.90 | 0.0130 | **0.1480** |
| 0.95 | 0.0080 | **0.0570** |

Fixed-sensitivity candidates (0.80/0.90/0.95): at each matched target sensitivity GroupDRO keeps more
usable ood_val behavior (e.g. tgt sens 0.90 → GroupDRO ood_val sens 0.169 / spec 0.900 vs ERM 0.010 /
0.940).

**Finding (development):** (1) calibration **improved confidence reliability for both** 7F models
(ECE/Brier/NLL all down) and removed high-confidence missed tumors, but is **argmax-invariant** (does
not change default hard predictions); (2) **GroupDRO remains better than its matched center-stratified
ERM control after the candidate operating-point audit** — higher ood_val AUROC (0.7129 vs 0.6180) and
ood_val sensitivity ≥ ERM at all 4 fixed-specificity targets (strictly greater at 4/4); (3) GroupDRO is
the **better-calibrated** model both raw and calibrated. **Recommendation: RETAIN GroupDRO with its
calibrated confidence output; report candidate (not committed) operating points only.** **OVERALL:
PASS** (all assertions; no warnings/errors). Development analysis on capped balanced subsets — NOT final
performance; no committed operating point; `ood_test` untouched. Artifacts:
`results/predictions/exp07g_7f_calibration/{centerstrat_erm,groupdro}_ood_val_calibrated_predictions.csv`;
`results/metrics/exp07g_7f_calibration/{temperature,calibration_metrics}.json`;
`results/tables/exp07g_{calibration_comparison,high_confidence_fn_comparison,candidate_operating_point_audit}.csv`;
`results/figures/exp07g_{calibration_raw_vs_calibrated,high_confidence_fn_comparison,fixed_specificity_sensitivity_comparison}.png`;
`results/logs/exp07g_7f_calibration_operating_audit_report.txt`.

### Milestone 7F — Matched center-stratified ERM + GroupDRO baseline (development)

The first **dedicated domain-generalization** baseline (reviewer risk R3), and the first DG lever to
beat its matched control on OOD. TWO ResNet18 models were **trained** on the SAME new center-
stratified `train` cache (Milestone 7F-1, per-`(center,label)` balanced across centers {0,3,4}):
a **matched center-stratified ERM control** (`exp07f_centerstrat_erm_resnet18`) and a
**GroupDRO-by-center** model (`exp07f_groupdro_resnet18`). New `src/training/groupdro.py`
(exponentiated-gradient `q_g ← q_g·exp(step·detached_L_g)` update, renormalized; absent groups not
updated; groups = centers {0,3,4}; CE base loss), `configs/experiments/exp07f_groupdro_resnet18.yaml`,
`scripts/26` (train), `scripts/27` (eval). `id_val` selection by center-stratified `id_val_loss`
ONLY; `ood_val` read only AFTER training (script 27) for development comparison. `ood_test` never
touched. **No CORAL/DANN, no MC-dropout/ensemble, no calibration, no committed operating point.** All
caches were disk loads (no HF streaming).

Training (best by `id_val_loss`; both reload-verified 1e-5):

| model | best epoch | id_val_loss | id_val_acc | notes |
|-------|:----------:|------------:|-----------:|-------|
| center-strat ERM (7F) | 2 | 0.248137 | 0.9089 | plain CE |
| GroupDRO (7F)         | 3 | 0.259698 | 0.8978 | q drifted to worst center (c4: 0.333→0.381) |

Evaluation + fair comparison (tumor = positive, threshold 0.5):

| split | model | accuracy | auroc | auprc | sensitivity | specificity | FN | tn/fp/fn/tp |
|-------|-------|---------:|------:|------:|------------:|------------:|---:|:-----------:|
| id_val  | center-strat ERM (7F) | 0.9089 | 0.9662 | 0.9717 | 0.8933 | 0.9244 |  24 | 208/17/24/201 |
| id_val  | GroupDRO (7F)         | 0.8978 | 0.9666 | 0.9703 | 0.8578 | 0.9378 |  32 | 211/14/32/193 |
| ood_val | center-strat ERM (7F) | 0.4765 | 0.6180 | 0.5214 | 0.0100 | 0.9430 | 990 | 943/57/990/10 |
| ood_val | GroupDRO (7F)         | 0.5065 | 0.7129 | 0.6287 | 0.0700 | 0.9430 | 930 | 943/57/930/70 |

Per-center id_val (threshold 0.5): ERM c0 acc 0.907 / c3 0.933 / c4 0.887 (worst = c3, loss 0.268);
GroupDRO c0 0.913 / c3 0.920 / c4 0.860 (worst = c4, loss 0.337).

Candidate id_val-selected fixed-specificity operating points — ood_val sensitivity:

| target spec | center-strat ERM (7F) | GroupDRO (7F) | old plain ERM (ref) |
|------------:|----------------------:|--------------:|--------------------:|
| 0.80 | 0.0210 | **0.3060** | 0.9340 |
| 0.85 | 0.0150 | **0.2410** | 0.9010 |
| 0.90 | 0.0130 | **0.1480** | 0.8330 |
| 0.95 | 0.0080 | **0.0570** | 0.7570 |

**Finding (development, honest positive-for-DG result): GroupDRO beat its matched ERM control on
OOD** — ood_val AUROC **0.7129 vs 0.6180 (+0.0949)**, AUPRC 0.6287 vs 0.5214 (+0.1073), default-0.5
sensitivity 0.0700 vs 0.0100, and matched fixed-specificity ood_val sensitivity **4/4** targets. This
is the **first DG lever to beat its matched control on OOD ranking** (stain augmentation 7A/7D/7E all
stayed below plain ERM). GroupDRO did **not** improve worst-center id_val behavior (it up-weighted
center 4, whose id_val loss stayed highest). **Reference caveat:** old plain ERM (ood_val AUROC
0.7696) / 7A / 7D / 7E were trained on the OLD confounded cache and a **different** id_val subset, so
old-vs-new numbers are **context only, NOT a controlled comparison** — the controlled comparison is
7F ERM vs 7F GroupDRO. **Recommendation: RETAIN GroupDRO** — next candidate is calibration +
candidate operating-point audit of the 7F models (development-stage, gated). **OVERALL: PASS** (both
scripts; all assertions; no warnings/errors). Development analysis on capped balanced subsets — NOT
final performance; no calibration and no committed operating point in 7F; `ood_test` untouched.
Artifacts: `results/checkpoints/{exp07f_centerstrat_erm_resnet18,exp07f_groupdro_resnet18}/{best,last}.pt`;
`results/predictions/{exp07f_centerstrat_erm_resnet18,exp07f_groupdro_resnet18}/{id_val,ood_val}_predictions.csv`;
`results/metrics/exp07f_centerstrat_erm_resnet18/classification_metrics.json`,
`results/metrics/exp07f_groupdro_resnet18/{classification_metrics,comparison_metrics}.json`;
`results/tables/exp07f_{overall_metrics_comparison,id_val_per_center_metrics,ood_val_metrics_comparison,fixed_specificity_comparison}.csv`;
`results/figures/exp07f_{roc_centerstrat_erm_vs_groupdro_ood_val,pr_centerstrat_erm_vs_groupdro_ood_val,fixed_specificity_sensitivity_comparison,id_val_per_center_accuracy}.png`;
`results/logs/{exp07f_centerstrat_erm_groupdro_train_report,exp07f_centerstrat_erm_groupdro_eval_report}.txt`,
`results/logs/{exp07f_centerstrat_erm_resnet18_log,exp07f_groupdro_resnet18_log}.csv`.

### Milestone 7F-1 — Metadata-rich center-stratified cache rebuild (no training)

A **gated, no-training** cache rebuild — **NO model inference, NO `ood_test`**; existing caches
**not** overwritten or modified. `scripts/25_rebuild_center_stratified_caches.py` streamed the
locked HF mirror `wltjr1007/Camelyon17-WILDS` via `iter_logical_split` and filled per-
`(center,label)` quotas across source centers `{0,3,4}` — the only change vs the old collector is
the **sampling policy** (the old first-per-class-in-stream-order policy is what confounded center
with label). It reused the existing `balanced_subset` serializer helpers (`_serialize_examples` /
`_deserialize_examples` / `META_INT_FIELDS`) so the new caches are schema-identical and load through
the same `BalancedSubsetDataset` — **`src/` unchanged**. `ood_val` was **not** read for cache
construction. **All 22 assertions PASS; no warnings.**

Adaptive per-cell quota: preferred **300** (train) / **75** (id_val); both reached in full (train
scanned 221,387; id_val scanned 24,466; early-stopped once all 6 cells filled). New caches, both
center×label balanced by construction:

| split  | new cache file | n | label 0/1 | per center (0/3/4) | per (center,label) cell | reload |
|--------|----------------|--:|-----------|--------------------|-------------------------|:------:|
| train  | `center_strat_train_c034_percell300_seed1337_shuf.pt` | 1800 | 900/900 | 600/600/600 | all six = 300 | ✓ |
| id_val | `center_strat_id_val_c034_percell75_seed1337_shuf.pt`  |  450 | 225/225 | 150/150/150 | all six = 75  | ✓ |

Both preserve all seven `META_INT_FIELDS` (`center, image_id, node, patient, slide, x_coord,
y_coord`). **DG feasibility after rebuild: SOUND** — `train` has centers {0,3,4} each carrying both
labels, so the new caches **support GroupDRO by center** and **support CORAL/DANN by center**; the
**old** confounded caches (center 0 ~all non-tumor, center 3 all tumor, center 4 absent) must be
**avoided for DG training** and are retained only for 4A/5A/7A–7E reproducibility. Existing caches
verified unmodified (size + mtime fingerprint pre/post). Benign transient HF read-timeouts during
the train stream were auto-retried (expected). Artifacts:
`results/cache/center_strat_train_c034_percell300_seed1337_shuf.pt`,
`center_strat_id_val_c034_percell75_seed1337_shuf.pt`;
`results/tables/exp07f1_center_stratified_cache_availability.csv`,
`exp07f1_center_stratified_cache_summary.csv`,
`exp07f1_center_stratified_label_center_counts.csv`;
`results/logs/exp07f1_center_stratified_cache_rebuild_report.txt`, `exp07f1_run_console.txt`.
Development-stage rebuild — NOT a final result; `ood_test` untouched; **GroupDRO/CORAL/DANN not
started.**

### Milestone 7F-0 — DG feasibility / group-label metadata audit (no training)

A **metadata/cache audit only** — **NO training, NO inference, NO `ood_test`**; no cache /
prediction / checkpoint / metrics file modified. `scripts/24_dg_group_metadata_audit.py` loads the
three balanced caches read-only and inspects payload structure, label counts, image presence, and
metadata fields, then derives DG group-label feasibility. **All assertions PASS; no warnings/errors.**

Findings:
- **Metadata is fully preserved** in all three caches. Each payload is a `dict` with keys
  `version, n, images, labels, meta`; `meta` carries **all seven** integer fields — `center`,
  `patient`, `slide`, `node`, `image_id`, `x_coord`, `y_coord`. So DG is **not** blocked by missing
  metadata. Counts verified: train n=2000 (1000/1000), id_val n=500 (250/250), ood_val n=2000
  (1000/1000); labels finite/binary.
- **Distinct groups:** train center `{0,3}`, id_val center `{0,3}`, ood_val center `{1}` (target
  domain identifiable); train patient 8 / slide 11; ood_val patient 8 / slide 10.
- **Critical confound (honest finding):** the capped `train` cache is **center↔label confounded** —
  center 0 is ~9.1% tumor (label0=1000, label1=100) and **center 3 is 100% tumor** (label0=0,
  label1=900), and locked source **center 4 is entirely absent** (`missing=[4]`). This is an
  artifact of building the balanced subset by scanning the label-ordered stream to fill per-class
  quotas. Consequence: center-based **GroupDRO / CORAL / DANN are RUNNABLE (≥2 center groups +
  metadata present) but NOT SOUND** — a center-grouped objective would learn *center == label*
  rather than a domain-invariant tumor signal.
- **Recommendation A — center-STRATIFIED metadata-rich cache rebuild required** (before any sound
  DG-by-center training). Minimal design (proposal only, NOT run here): keep the locked split
  mapping and per_class balance; change only the **sampling policy** to fill per-`(center,label)`
  quotas across source centers `{0,3,4}` so each center carries both labels; keep serializing all
  `META_INT_FIELDS` (schema already sufficient); write to **new** cache filenames (no overwrite, to
  preserve 4A/5A/7A–7E reproducibility). The rebuild is itself a separate gated milestone (requires
  HF streaming) and is **not** performed in 7F-0.

Artifacts: `results/tables/exp07f0_cache_metadata_summary.csv`,
`exp07f0_group_label_counts.csv`, `exp07f0_dg_method_feasibility.csv`;
`results/logs/exp07f0_dg_group_metadata_audit_report.txt`, `exp07f0_run_console.txt`.
Development-stage feasibility audit — NOT a final result; `ood_val` inspected for metadata
availability only; `ood_test` untouched.

### Milestone 7E — Stain-space (OD/HED-style) augmentation ERM baseline (development)

The first **pathology-specific** stain-robustness / DG baseline (reviewer risk R3): a ResNet18
ERM **trained** with the validated 7E-0 optical-density (HED-style) `StainSpaceAugment` applied to
`train` only (per-channel OD scale ~U[0.90,1.10] + bias ~U[-0.03,0.03], flips, 15° rotation, no
hue/sat jitter); `id_val` on the deterministic eval transform, best by `id_val_loss` ONLY.
`ood_val` read only AFTER training (script 23) for a **4-model** development comparison
(plain ERM / 7A / 7D / 7E). `ood_test` never touched. **No GroupDRO/CORAL/DANN, no
MC-dropout/ensemble, no calibration, no committed operating point.** Reused `StainSpaceAugment`
(no new `src/` module); `configs/experiments/exp07e_stainspace_resnet18.yaml`, `scripts/22`,
`scripts/23`. All three balanced caches were **cache HITs** (no HF streaming).

Training (best by `id_val_loss`):

| epoch | train_loss | train_acc | id_val_loss | id_val_acc | is_best |
|-------|-----------:|----------:|------------:|-----------:|:-------:|
| 1 | 0.295800 | 0.8735 | 0.188955 | 0.9380 | ✓ |
| 2 | 0.206271 | 0.9175 | 0.134861 | 0.9480 | ✓ |
| 3 | 0.187358 | 0.9270 | 0.102048 | 0.9580 | ✓ |
| 4 | 0.167339 | 0.9350 | 0.137747 | 0.9380 |   |
| 5 | 0.152006 | 0.9405 | 0.161615 | 0.9520 |   |

Best epoch **3**, `id_val_loss` **0.102048**, `id_val_acc` 0.9580. Reload verification
reproduced id_val_loss/acc exactly (1e-5) → **PASS**. (Note: unlike 7A/7D — best epoch 5 — 7E's
best is epoch 3; later epochs over-fit id_val loss.)

Evaluation + 4-model comparison (tumor = positive, threshold 0.5):

| split | model | accuracy | auroc | auprc | sensitivity | specificity | FN | tn/fp/fn/tp |
|-------|-------|---------:|------:|------:|------------:|------------:|---:|:-----------:|
| id_val  | plain ERM (4A)   | 0.9640 | 0.9933 | 0.9943 | 0.9440 | 0.9840 |  14 | 246/4/14/236 |
| id_val  | stain-aug 7A     | 0.9480 | 0.9938 | 0.9940 | 0.9080 | 0.9880 |  23 | 247/3/23/227 |
| id_val  | stain-aug v2 7D  | 0.9440 | 0.9925 | 0.9932 | 0.9040 | 0.9840 |  24 | 246/4/24/226 |
| id_val  | stain-space 7E   | 0.9580 | 0.9933 | 0.9940 | 0.9400 | 0.9760 |  15 | 244/6/15/235 |
| ood_val | plain ERM (5A)   | 0.6195 | 0.7696 | 0.6793 | 0.4050 | 0.8340 | 595 | 834/166/595/405 |
| ood_val | stain-aug 7A     | 0.5290 | 0.7492 | 0.6881 | 0.0960 | 0.9620 | 904 | 962/38/904/96 |
| ood_val | stain-aug v2 7D  | 0.5135 | 0.7397 | 0.6614 | 0.0650 | 0.9620 | 935 | 962/38/935/65 |
| ood_val | stain-space 7E   | 0.5055 | 0.7317 | 0.6244 | 0.0840 | 0.9270 | 916 | 927/73/916/84 |

Candidate id_val-selected fixed-specificity operating points — ood_val sensitivity:

| target spec | plain ERM | stain-aug 7A | stain-aug v2 7D | stain-space 7E |
|------------:|----------:|-------------:|----------------:|---------------:|
| 0.80 | 0.9340 | 0.6770 | 0.6100 | **0.8970** |
| 0.85 | 0.9010 | 0.5870 | 0.5310 | **0.7880** |
| 0.90 | 0.8330 | 0.4570 | 0.3800 | **0.6530** |
| 0.95 | 0.7570 | 0.2700 | 0.1780 | **0.3030** |

**Finding (development, honest mixed/negative result):** the OD-space stain augmentation did
**not** improve OOD *ranking* — ood_val AUROC **0.7317** is the lowest of the four (7E − plain
−0.0379, 7E − 7A −0.0175, 7E − 7D −0.0080) and AUPRC 0.6244 is also lowest; default-0.5 ood_val
sensitivity 0.0840 (FN 916). **However, at every matched id_val-selected fixed-specificity target
7E is the strongest stain-augmentation variant** — it beats **both** 7A and 7D at **4/4** targets
(large operating-point recovery: e.g. spec≥0.90 ood_val sensitivity 0.653 vs 7A 0.457 / 7D 0.380),
though still **below plain ERM at 0/4**. id_val was preserved (accuracy 0.958, AUROC 0.9933,
sensitivity 0.940). **Recommendation: TUNE FURTHER / DO NOT PROMOTE** — the OD-space transform
gives markedly better fixed-specificity operating-point behavior than the torchvision stain
baselines but does not beat plain ERM on ranking; the next candidate is a dedicated DG objective
(GroupDRO/CORAL/DANN) or OD-range tuning (gated). Do **not** claim superiority over plain ERM.
**OVERALL: PASS** (both scripts; all assertions; no warnings/errors). Development analysis on
capped balanced subsets — NOT final performance; no calibration and no committed operating point
in 7E; `ood_test` untouched.

### Milestone 7E-0 — Stain-space (OD/HED-style) augmentation smoke test (transform-only)

A **transform-only** visual + numerical sanity check of a new pathology-specific
optical-density (HED-style) stain augmentation — **NO training, NO model inference, NO
`ood_val`, NO `ood_test`**, no existing result CSV/checkpoint modified. Added
`StainSpaceAugment` + `build_stain_space_aug_transform` to `src/data/stain_transforms.py`
(OD `= -log((RGB+eps)/255)` → per-channel `OD' = OD*scale_c + bias_c`, `scale_c ~ U[0.90,1.10]`,
`bias_c ~ U[-0.03,0.03]`, OD lower-clipped at 0 → back to RGB clipped [0,255]; then
flips/rotation 15° + ImageNet norm; no hue/sat jitter). `scripts/21` loads the cached `train`
subset (cache HIT), selects 8 label-0 + 8 label-1 patches, augments each 3×, and writes a
preview grid + stat report.

Results (seed 1337): 16 patches × 3 augs = 48 augmentations; output tensor shape **(3,96,96)**;
augmented pre-normalization RGB range **[0, 254]** (all within [0,255]); **no NaN/Inf**.
Normalized-tensor stats before→after: mean 0.9797→0.8053, std 0.9368→1.1365 (mild). **Visual
check PASS** — realistic H&E stain variation, tissue morphology preserved, no black/washed-out/
neon patches. **OVERALL: PASS** (all assertions; no warnings/errors). Transform validation only
— NOT a training result and NOT a performance claim; clears the stain-space transform for a
*gated* 7E training proposal. `ood_val` / `ood_test` untouched.

### Milestone 7D — Tuned stain/color augmentation v2 ERM baseline (development)

A **stronger but controlled torchvision** stain/color augmentation ERM baseline for reviewer
risk R3 (no *effective* stain-robustness / DG baseline yet). Trained ResNet18 for 5 epochs on
the SAME cached balanced `train` subset (v2 augmentation: flips → rotation 30° → affine
translate (0.03,0.03)/scale (0.95,1.05) → `RandomApply([ColorJitter b0.15/c0.30/s0.15/h0.03],
p=0.85)` → `RandomAutocontrast` p0.25; hue/sat deliberately low), `id_val` on the deterministic
eval transform, best by `id_val_loss` ONLY (epoch 5, `id_val_loss` 0.123163; reload verified).
`ood_val` read only AFTER training (script 20) for development comparison. `ood_test` never
touched. **No HED/Macenko, no GroupDRO/CORAL/DANN, no MC-dropout/ensemble, no calibration, no
committed operating point.** New: `StainColorAugmentV2` in `src/data/stain_transforms.py`,
`configs/experiments/exp07d_stainaug_v2_resnet18.yaml`, `scripts/19`, `scripts/20`.

Evaluation + 3-model comparison (tumor = positive, threshold 0.5):

| split | model | accuracy | auroc | auprc | sensitivity | specificity | FN |
|-------|-------|---------:|------:|------:|------------:|------------:|---:|
| id_val  | plain ERM (4A)  | 0.9640 | 0.9933 | 0.9943 | 0.9440 | 0.9840 |  14 |
| id_val  | stain-aug 7A    | 0.9480 | 0.9938 | 0.9940 | 0.9080 | 0.9880 |  23 |
| id_val  | stain-aug v2 7D | 0.9440 | 0.9925 | 0.9932 | 0.9040 | 0.9840 |  24 |
| ood_val | plain ERM (5A)  | 0.6195 | 0.7696 | 0.6793 | 0.4050 | 0.8340 | 595 |
| ood_val | stain-aug 7A    | 0.5290 | 0.7492 | 0.6881 | 0.0960 | 0.9620 | 904 |
| ood_val | stain-aug v2 7D | 0.5135 | 0.7397 | 0.6614 | 0.0650 | 0.9620 | 935 |

Candidate id_val-selected fixed-specificity operating points — ood_val sensitivity: plain
0.934/0.901/0.833/0.757, 7A 0.677/0.587/0.457/0.270, **7D 0.610/0.531/0.380/0.178** at
spec 0.80/0.85/0.90/0.95.

**Finding (honest negative):** the tuned v2 augmentation did **not** improve OOD. ood_val AUROC
**0.7397** is below plain (0.7696) and 7A (0.7492) — 7D − plain = **−0.0299**, 7D − 7A =
**−0.0095**; AUPRC 0.6614 below both; default-0.5 ood_val sensitivity 0.065 (FN 935) below 7A;
and 7D sensitivity is **below both baselines at every matched fixed-specificity target** (7D ≥
7A at 0/4; 7D ≥ plain at 0/4). id_val preserved (accuracy 0.964→0.944, AUROC ~0.9925).
**Recommendation: MIXED / DO NOT PROMOTE** — naive→tuned torchvision stain/color augmentation is
an insufficient stain-robustness lever here; the next candidate is a stain-specific
(HED/Macenko) or dedicated DG method (gated). Do not claim superiority over plain ERM.
**OVERALL: PASS** (both scripts; all assertions). Development analysis on capped balanced
subsets — NOT final performance; no operating point committed; `ood_test` untouched.

### Milestone 7C — Stain-aug calibration + candidate operating-point audit (CSV/logit-only)

A CSV/logit-only extension of 7B (no training, no inference, no MC-dropout/ensemble;
`ood_test` never read; input prediction CSVs unmodified; **no operating point committed**).
It (a) temperature-scales the **stain-aug 7A** model on `ood_val` logits (development-stage),
(b) reports raw-vs-calibrated confidence reliability vs plain-ERM 5A, (c) audits
high-confidence false negatives before/after calibration, and (d) audits **candidate**
`id_val`-selected operating points for both models on `ood_val`. Reuses
`src/calibration/temperature_scaling.py` + `src/evaluation/*`; `scripts/18`.

**Stain-aug temperature (fit on ood_val logits) = T 3.210174** (NLL 1.1060→0.6803).

Calibration reliability on ood_val (development):

| model | variant | ECE | Brier | NLL |
|-------|---------|----:|------:|----:|
| plain ERM     | raw        | 0.2330 | 0.2759 | 0.9152 |
| plain ERM     | calibrated | 0.0918 | 0.2160 | 0.6088 |
| stain-aug ERM | raw        | 0.3560 | 0.3571 | 1.1060 |
| stain-aug ERM | calibrated | 0.1908 | 0.2474 | 0.6803 |

High-confidence missed tumors on ood_val (count, `label==1 & pred==0 & conf≥t`): stain-aug
raw 399/181/4 → calibrated 0/0/0 at t=0.90/0.95/0.99; plain ERM raw 222/133/33 → calibrated
0/0/0. Stain-aug classification is **unchanged** raw→calibrated (argmax-invariant): acc 0.5290,
sens 0.0960, FN 904.

Candidate operating points (id_val-selected, raw prob_1) — ood_val at fixed specificity:

| target spec | plain ERM ood_val sens | stain-aug ood_val sens |
|------------:|-----------------------:|-----------------------:|
| 0.80 | 0.9340 | 0.6770 |
| 0.85 | 0.9010 | 0.5870 |
| 0.90 | 0.8330 | 0.4570 |
| 0.95 | 0.7570 | 0.2700 |

**Finding:** calibration IMPROVED stain-aug confidence reliability (ECE ↓46%, NLL ↓38%; removed
high-confidence missed tumors) but is **argmax-invariant** — it does not fix the 0.5 sensitivity
collapse. **Plain ERM remains stronger than stain-aug at every matched fixed-specificity
candidate point.** **Recommendation:** RETAIN the stain-aug baseline with a calibrated confidence
output and **candidate (not committed)** operating points — do not reject on the 0.5 result, do
not claim it beats plain ERM. **OVERALL: PASS** (all assertions). Development analysis on capped
balanced subsets — NOT final performance; no operating point committed; `ood_test` untouched.

### Milestone 7B — Threshold-normalized plain-ERM vs stain-aug comparison (CSV-only)

A CSV-only diagnostic (no training, no inference, no calibration, no MC-dropout/ensemble;
`ood_test` never read; input prediction CSVs unmodified) that answers *why* the 7A stain-aug
baseline looked worse on `ood_val` at the default 0.5 threshold. Operating thresholds are
chosen on `id_val` ONLY (matched specificity 0.80/0.85/0.90/0.95 and matched sensitivity
0.80/0.90/0.95) and the *same* threshold is applied to `ood_val` for development comparison.
New module `src/evaluation/metrics_operating_points.py`; `scripts/17`.

Key numbers (tumor = positive):

| view | plain ERM | stain-aug (7A) | Δ (stain−plain) |
|------|----------:|---------------:|----------------:|
| ood_val AUROC          | 0.7696 | 0.7492 | −0.0204 |
| ood_val AUPRC          | 0.6793 | 0.6881 | +0.0088 |
| ood_val sensitivity @0.5 | 0.4050 | 0.0960 | −0.3090 |
| ood_val sensitivity @ id-spec≥0.90 threshold | 0.8330 | 0.4570 | — |

**Finding:** OOD *ranking* is largely preserved (|ΔAUROC| 0.0204 < 0.03; AUPRC even improved),
but the fixed 0.5 threshold is far more conservative for the stain-aug model (it pushes
center-1 scores below 0.5, collapsing sensitivity to 0.096). Selecting the operating point on
`id_val` recovers a large share of the lost sensitivity on `ood_val` (0.096→0.457 at
spec≥0.90), i.e. **the 7A "sensitivity collapse" is mostly a threshold artifact, not a genuine
ranking failure**. **Recommendation:** TUNE / RETAIN the stain-aug model with a calibrated
operating point rather than rejecting it on the default-threshold result; the natural next
step is calibration + operating-point selection (development-stage). **OVERALL: PASS** (all
assertions). Development analysis on capped balanced subsets — NOT final performance;
`ood_test` untouched.

### Milestone 7A — Stain/color augmentation ERM baseline (development)

A stain-robustness / domain-generalization baseline for reviewer risk R3. Trained ResNet18
for 5 epochs on the SAME cached balanced `train` subset used by plain-ERM 4A, but with
**moderate stain/color augmentation applied to train only** (`RandomHorizontalFlip` 0.5,
`RandomVerticalFlip` 0.5, `RandomRotation` 15°, `ColorJitter` brightness/contrast/saturation
0.20, hue 0.05; torchvision), `id_val` on the deterministic eval transform. Best by
`id_val_loss` ONLY (epoch 5, `id_val_loss` 0.122964). `ood_val` was NEVER used for training
or selection — read only AFTER training (script 16) for development-stage comparison.
`ood_test` never touched. New modules `src/data/stain_transforms.py`,
`configs/experiments/exp07a_stainaug_resnet18.yaml`, `scripts/15`, `scripts/16`.

Per-epoch (train_loss / train_acc / id_val_loss / id_val_acc; best ✓):

| epoch | train_loss | train_acc | id_val_loss | id_val_acc | is_best |
|-------|-----------:|----------:|------------:|-----------:|:-------:|
| 1 | 0.326070 | 0.8645 | 0.182616 | 0.9280 | ✓ |
| 2 | 0.244792 | 0.9000 | 0.163943 | 0.9400 | ✓ |
| 3 | 0.222708 | 0.9090 | 0.200999 | 0.9220 |   |
| 4 | 0.193224 | 0.9195 | 0.177804 | 0.9380 |   |
| 5 | 0.198658 | 0.9190 | 0.122964 | 0.9480 | ✓ |

Evaluation (tumor = positive, threshold 0.5) and comparison vs plain ERM (4A id_val / 5A raw
ood_val):

| split | model | accuracy | auroc | sensitivity | FN | tn/fp/fn/tp |
|-------|-------|---------:|------:|------------:|---:|:-----------:|
| id_val  | plain ERM (4A) | 0.9640 | 0.9933 | 0.9440 |  14 | 246/4/14/236 |
| id_val  | stain-aug (7A) | 0.9480 | 0.9938 | 0.9080 |  23 | 247/3/23/227 |
| ood_val | plain ERM (5A raw) | 0.6195 | 0.7696 | 0.4050 | 595 | 834/166/595/405 |
| ood_val | stain-aug (7A)     | 0.5290 | 0.7492 | 0.0960 | 904 | 962/38/904/96 |

**Finding (development, honest negative result):** moderate stain/color augmentation did
**not** improve OOD reliability on this capped balanced subset. ood_val accuracy fell
0.6195→0.5290 (Δ−0.0905) and AUROC 0.7696→0.7492 (Δ−0.0204); ood_val tumor **sensitivity
collapsed 0.405→0.096** (FN 595→904) — the augmented model became more conservative
(specificity rose 0.834→0.962 but it now misses ~90% of tumors under center-1 shift). id_val
was essentially preserved (accuracy 0.964→0.948, AUROC ~unchanged 0.9938). **OVERALL: PASS**
(both scripts; all assertions passed). This is a DEVELOPMENT baseline on capped balanced
subsets — NOT final performance; it establishes that a *naive/moderate* stain-augmentation
ERM baseline is insufficient here (motivating stronger/tuned stain-robustness or a dedicated
DG method), and it does not license any performance/superiority claim. `ood_test` untouched.

### Milestone 6B — manuscript / readiness audit package (documentation-only)

Paper-facing audit built from the 6A reporting artifacts (no code, no inference, no
training, no new methods; result inputs unmodified; `ood_test` never read). Created six
deliverables: `docs/MANUSCRIPT_PLAN.md` (title, abstract skeleton, section outlines,
journal positioning, locked-test note), `docs/RESULTS_INTERPRETATION.md` (ID→OOD
degradation, calibration, high-confidence FN reduction, uncertainty, selective prediction,
argmax-invariance, OOD abstention ceiling, supports/does-not-support), `docs/
CLAIMS_AND_LIMITATIONS.md` (two-column allowed vs forbidden claim map), `docs/
FIGURE_TABLE_MANIFEST.md` (all 6A tables/figures with main-vs-supp placement), `docs/
REVIEWER_RISK_AUDIT.md` (9 anticipated reviewer risks with severity/mitigation/timing), and
`results/tables/report_claims_audit.csv` (21 claim rows: 9 allowed / 5 conditional / 7
forbidden). All current results explicitly labeled development / capped-subset. **OVERALL:
PASS** (documentation package). `ood_test` untouched.

### Milestone 6A — reliability reporting package from saved results (CSV/JSON-only)

Consolidated the accepted 4B/5A/5B/5C outputs into publication-style report tables +
figures + a text summary via `scripts/14_make_reliability_report_package.py` — no
inference, no training, no MC-dropout, no ensemble; input files unmodified; `ood_test`
never read. `ood_val` classification was sourced from the 5A JSON (`classification.raw`),
since the 4B classification CSV holds only `train`/`id_val`.

- Tables: `results/tables/report_table_classification_shift.csv`,
  `report_table_calibration.csv`, `report_table_uncertainty.csv`,
  `report_table_selective_prediction.csv`.
- Figures: `results/figures/report_fig_domain_shift_bar.png`,
  `report_fig_calibration_metrics.png`, `report_fig_uncertainty_error_detection.png`,
  `report_fig_risk_coverage_entropy.png`, `report_fig_false_negative_abstention.png`.
- Summary: `results/logs/report_reliability_package_summary.txt`.

Consolidated headline numbers (development):

| view | id_val | ood_val (raw) |
|------|-------:|--------------:|
| accuracy    | 0.9640 | 0.6195 |
| AUROC       | 0.9933 | 0.7696 |
| sensitivity | 0.9440 | 0.4050 |
| FN count    | 14     | 595    |

Calibration on ood_val (T=3.1224): ECE 0.2330→0.0918 (−61%), Brier 0.2759→0.2160 (−22%),
NLL 0.9152→0.6088 (−33%); classification unchanged (argmax-invariant). High-confidence
(≥0.90) ood_val FNs 222 (raw) → 0 (calibrated). Entropy error-detection AUROC id_val 0.9267
/ ood_val 0.6883 (raw = calibrated). Selective prediction @50% coverage: id_val risk
0.036→0.000 (all 14 FNs abstained); ood_val_calibrated risk 0.3805→0.2500 with 198/595 FNs
retained (66.7% abstained). **OVERALL: PASS.** All assertions passed; no warnings/errors.
DEVELOPMENT consolidation on capped balanced subsets — NOT final performance; `ood_test`
untouched.

### Milestone 5C — selective prediction / abstention from saved uncertainty CSVs (CSV-only)

Converted the 5B uncertainty scores into selective-prediction / abstention tables — no
inference, no training, no MC-dropout, no ensemble, `ood_test` never read, no operating
threshold committed. Retain the lowest-uncertainty (most-confident) rows; abstain on the
highest-uncertainty rows; coverage = retained_n/n; risk = retained_errors/retained_n.

Entropy selective prediction (retained risk / retained false negatives):

| coverage | id_val risk | id_val ret_FN | ood_val risk | ood_val ret_FN | ood_val FNs abstained |
|---------:|------------:|--------------:|-------------:|---------------:|----------------------:|
| 1.00 | 0.0360 | 14 | 0.3805 | 595 |   0 |
| 0.90 | 0.0111 |  3 | 0.3633 | 517 |  78 |
| 0.70 | 0.0029 |  1 | 0.3243 | 356 | 239 |
| 0.50 | 0.0000 |  0 | 0.2500 | 198 | 397 |

id_val: abstaining 50% drives risk to 0 and abstains all 14 FNs. ood_val: abstention helps
far less under shift — 50% coverage only lowers risk 0.380→0.250 and 198/595 FNs stay in
the retained set. **Raw vs calibrated ood_val share identical risk/coverage/FN rows**
(temperature scaling is monotone → same entropy ranking); only entropy thresholds differ
(coverage 0.50: raw 0.2944 vs calibrated 0.6267). **OVERALL: PASS.** DEVELOPMENT-stage
selective-prediction analysis on capped balanced subsets — NOT final clinical performance.

### Milestone 5B — deterministic uncertainty metrics from saved predictions (CSV-only)

Computed uncertainty / error-detection metrics **purely from the saved 4B/5A prediction
CSVs** — no inference, no training, no MC-dropout, no ensemble, no selective thresholds,
`ood_test` never read. Positive = tumor (1); error = `correct==False`; false negative =
`label==1 and pred_label==0`; confidence = `max(prob_0, prob_1)`; all uncertainty scores
oriented higher = more uncertain.

| variant | n | errors | error_rate | FN | err-detect AUROC (entropy) | entropy AURC |
|---------|--:|-------:|-----------:|---:|---------------------------:|-------------:|
| id_val             |  500 |  18 | 0.0360 |  14 | 0.9267 | 0.003558 |
| ood_val_raw        | 2000 | 761 | 0.3805 | 595 | 0.6883 | 0.219861 |
| ood_val_calibrated | 2000 | 761 | 0.3805 | 595 | 0.6883 | 0.219861 |

Within a variant, confidence/entropy/margin give identical error-detection AUROC (monotone
transforms in the 2-class case); energy differs slightly (id_val 0.9259; ood_val
0.6743 raw / 0.6583 cal). **Raw vs calibrated ood_val have identical AUROC/AURC** —
temperature scaling preserves ranking. It shifts confidence magnitudes: high-confidence
(≥0.90) errors fall n_high 1049 (278 err, 222 FN) raw → 256 (6 err, **0 FN**) calibrated.
High-confidence FN counts at t=0.90/0.95/0.99: id_val 6/3/2; ood_val_raw 222/133/33;
ood_val_calibrated 0/0/0. **OVERALL: PASS.** DEVELOPMENT analysis on capped balanced
subsets — NOT final performance; no thresholds committed (risk–coverage is descriptive).



Loaded 4A `best.pt` (epoch 5; **weights unchanged**), built a balanced `ood_val`
(center 1) subset `{0:1000, 1:1000}` (n=2000), exported raw predictions, fit a single
temperature on the `ood_val` logits (NLL minimization, Adam, 200 iters), applied it, and
exported calibrated predictions. Positive class = tumor (1), threshold 0.5, 15 ECE bins.

**Fitted temperature T = 3.1224** (NLL 0.915158 → 0.608814; `converged=false` at the
200-iter cap — acceptable, loss still improving slowly).

| variant | accuracy | auroc | auprc | ECE | Brier | NLL |
|---------|---------:|------:|------:|----:|------:|----:|
| raw        | 0.6195 | 0.7696 | 0.6793 | 0.2330 | 0.2759 | 0.9152 |
| calibrated | 0.6195 | 0.7696 | 0.6793 | 0.0918 | 0.2160 | 0.6088 |

Classification metrics are **identical** raw vs calibrated (temperature scaling is
argmax-invariant — it rescales confidence, not the decision). Calibration improved on all
three proper-scoring/calibration metrics (ECE ↓61%, NLL ↓33%, Brier ↓22%). Confusion at
0.5: tn/fp/fn/tp = 834/166/595/405 (both variants). **OVERALL: PASS.**

_This is the first OOD result and shows the ID-trained model is over-confident and only
moderately accurate under center-1 shift (acc 0.62, AUROC 0.77). It is a DEVELOPMENT
result on a 2000-sample balanced `ood_val` subset — NOT final performance; `ood_test`
was not touched._

### Milestone 4B — evaluation scaffold (predictions + classification metrics)

Loaded `results/checkpoints/exp04a_dev_train_resnet18/best.pt` (epoch 5) and ran
inference (eval transform, `shuffle=False`, batch_size 64) over the **cached** `train`
(n=2000) and `id_val` (n=500) subsets — cache HIT, no HF streaming, no training.
Positive class = tumor (1), threshold 0.5.

| split | n | accuracy | balanced_acc | auroc | auprc | sensitivity | specificity | precision | f1 | tn/fp/fn/tp |
|-------|--:|---------:|-------------:|------:|------:|------------:|------------:|----------:|---:|:-----------:|
| train  | 2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1000/0/0/1000 |
| id_val |  500 | 0.9640 | 0.9640 | 0.9933 | 0.9943 | 0.9440 | 0.9840 | 0.9833 | 0.9633 | 246/4/14/236 |

`id_val` accuracy 0.9640 matches the checkpoint's stored `id_val_accuracy` exactly — a
consistency check that the reload + eval path agrees with training-time evaluation.
Probability sanity passed (finite, in [0,1], `prob_0+prob_1≈1`). **OVERALL: PASS.**
_Development evaluation on cached balanced subsets — NOT final performance. `ood_val` /
`ood_test` untouched. train=1.0 reflects the model being evaluated on its own training
data, not generalization._

### Milestone 4A — ResNet18, checkpointed dev run, 5 epochs

Balanced subsets: train `{0:1000, 1:1000}` (n=2000), id_val `{0:250, 1:250}` (n=500).
Selection: best by `id_val_loss` (mode=min). Both caches were **CREATED** this run
(first build at `per_class` 1000/250); later runs will cache-HIT.

| epoch | train_loss | train_acc | id_val_loss | id_val_acc | is_best |
|-------|-----------:|----------:|------------:|-----------:|:-------:|
| 1 | 0.261794 | 0.8955 | 0.162842 | 0.9440 | ✓ |
| 2 | 0.098321 | 0.9665 | 0.344737 | 0.8020 |   |
| 3 | 0.040005 | 0.9865 | 0.173314 | 0.9380 |   |
| 4 | 0.035312 | 0.9885 | 0.117608 | 0.9620 | ✓ |
| 5 | 0.026127 | 0.9900 | 0.112202 | 0.9640 | ✓ |

Best epoch **5**, best `id_val_loss` **0.112202**. All losses finite; `train_n=2000`,
`id_val_n=500`; labels `[0,1]`/`[0,1]`. Checkpoints written: `best.pt` and `last.pt`
(~128 MB each) under `results/checkpoints/exp04a_dev_train_resnet18/`. **Reload
verification: PASS** — `best.pt` reloaded into a fresh model reproduced id_val_loss
0.112202 / id_val_acc 0.9640 exactly (match within 1e-5). **OVERALL: PASS.**

_These are development / mechanics-validation numbers on a 2000-sample balanced subset,
NOT a scientific result and NOT final performance. `ood_val` / `ood_test` were not
touched._

### Prior accepted results (Milestone 3C short train — ResNet18, 3 epochs)

Balanced subsets: train `{0:128, 1:128}` (n=256), id_val `{0:64, 1:64}` (n=128).

| epoch | train_loss | train_acc | id_val_loss | id_val_acc |
|-------|-----------:|----------:|------------:|-----------:|
| 1 | 0.471775 | 0.8086 | 1.484480 | 0.5000 |
| 2 | 0.187952 | 0.9531 | 0.619032 | 0.6953 |
| 3 | 0.087263 | 0.9805 | 0.272071 | 0.8672 |

All losses finite. `train_n=256`, `id_val_n=128`. **OVERALL: PASS.** (Deterministic on
cache-hit re-run — identical metrics.) Mechanics-validation numbers, NOT a result.

## 9. Key generated artifacts

- `results/logs/`: `env_check.txt`, `dataloader_smoke_test.txt`,
  `model_forward_smoke_test.txt`, `train_mechanics_smoke_test.txt`,
  `short_train_resnet18_report.txt`, `label_order_diagnostic.txt`,
  `hf_cache_status.txt`, `exp04a_dev_train_resnet18_report.txt`,
  `exp04a_dev_train_resnet18_log.csv`, `exp04a_run_console.txt`,
  `exp04a_eval_cached_checkpoint_report.txt`, `exp05a_calibrate_ood_val_report.txt`,
  `exp05a_run_console.txt`.
- `results/tables/`: `hf_split_verification.csv`, `dataset_split_summary.csv`,
  `dataset_label_counts.csv`, `dataset_center_counts.csv`,
  `label_order_diagnostic.csv`, `exp04a_dev_train_resnet18_classification_metrics.csv`,
  `exp05a_calibrate_ood_val_resnet18_calibration_metrics.csv`.
- `results/logs/short_train_resnet18_log.csv` (Milestone 3C per-epoch metrics).
- `results/figures/sample_patch_grid.png`.
- `results/cache/*.pt` (balanced subsets: pc128/pc64, pc1000/pc250, ood_val pc1000).
- `results/checkpoints/exp04a_dev_train_resnet18/{best.pt,last.pt}` (Milestone 4A).
- `results/predictions/exp04a_dev_train_resnet18/{train,id_val}_cached_predictions.csv`
  (Milestone 4B).
- `results/metrics/exp04a_dev_train_resnet18/classification_metrics.json` (Milestone 4B).
- `results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_{raw,calibrated}_predictions.csv`
  (Milestone 5A).
- `results/metrics/exp05a_calibrate_ood_val_resnet18/{calibration_metrics.json,temperature.json}`
  (Milestone 5A).
- `results/predictions/exp05b_uncertainty/{id_val,ood_val_raw,ood_val_calibrated}_uncertainty_scores.csv`
  (Milestone 5B).
- `results/metrics/exp05b_uncertainty/uncertainty_metrics.json`;
  `results/tables/exp05b_{uncertainty_metrics,risk_coverage,high_confidence_errors}.csv`;
  `results/logs/exp05b_uncertainty_metrics_report.txt` (Milestone 5B).
- `results/tables/exp05c_selective_prediction_all_scores.csv`,
  `exp05c_selective_prediction_entropy_summary.csv`,
  `exp05c_false_negative_abstention_summary.csv`;
  `results/metrics/exp05c_selective_prediction/selective_prediction_metrics.json`;
  `results/logs/exp05c_selective_prediction_report.txt` (Milestone 5C).
- `results/tables/report_table_{classification_shift,calibration,uncertainty,selective_prediction}.csv`;
  `results/figures/report_fig_{domain_shift_bar,calibration_metrics,uncertainty_error_detection,risk_coverage_entropy,false_negative_abstention}.png`;
  `results/logs/report_reliability_package_summary.txt` (Milestone 6A).
- `docs/MANUSCRIPT_PLAN.md`, `docs/RESULTS_INTERPRETATION.md`,
  `docs/CLAIMS_AND_LIMITATIONS.md`, `docs/FIGURE_TABLE_MANIFEST.md`,
  `docs/REVIEWER_RISK_AUDIT.md`; `results/tables/report_claims_audit.csv` (Milestone 6B).
- Milestone 7A: `results/checkpoints/exp07a_stainaug_resnet18/{best.pt,last.pt}`;
  `results/logs/exp07a_stainaug_resnet18_{log.csv,report.txt,eval_report.txt}`;
  `results/predictions/exp07a_stainaug_resnet18/{id_val,ood_val}_predictions.csv`;
  `results/metrics/exp07a_stainaug_resnet18/classification_metrics.json`;
  `results/tables/exp07a_stainaug_resnet18_classification_metrics.csv`.
- Milestone 7B (CSV-only): `results/tables/exp07b_default_threshold_comparison.csv`,
  `exp07b_threshold_free_comparison.csv`, `exp07b_fixed_specificity_operating_points.csv`,
  `exp07b_fixed_sensitivity_operating_points.csv`, `exp07b_threshold_sweeps.csv`;
  `results/metrics/exp07b_operating_point_comparison/operating_point_metrics.json`;
  `results/figures/exp07b_{roc_plain_vs_stainaug_id_val,roc_plain_vs_stainaug_ood_val,
  pr_plain_vs_stainaug_ood_val,sensitivity_specificity_tradeoff_ood_val}.png`;
  `results/logs/exp07b_operating_point_comparison_report.txt`.
- Milestone 7C (CSV/logit-only):
  `results/predictions/exp07c_stainaug_calibration/ood_val_calibrated_predictions.csv`;
  `results/metrics/exp07c_stainaug_calibration/{temperature.json,calibration_metrics.json}`;
  `results/tables/exp07c_{calibration_comparison,high_confidence_fn_comparison,
  candidate_operating_point_audit}.csv`;
  `results/figures/exp07c_{calibration_raw_vs_calibrated,high_confidence_fn_comparison,
  fixed_specificity_sensitivity_comparison}.png`;
  `results/logs/exp07c_stainaug_calibration_operating_audit_report.txt`.
- Milestone 7D: `results/checkpoints/exp07d_stainaug_v2_resnet18/{best.pt,last.pt}`;
  `results/logs/exp07d_stainaug_v2_resnet18_{log.csv,train_report.txt,eval_report.txt}`;
  `results/predictions/exp07d_stainaug_v2_resnet18/{id_val,ood_val}_predictions.csv`;
  `results/metrics/exp07d_stainaug_v2_resnet18/{classification_metrics.json,comparison_metrics.json}`;
  `results/tables/exp07d_stainaug_v2_resnet18_classification_metrics.csv`,
  `exp07d_{default_threshold_comparison,threshold_free_comparison,fixed_specificity_comparison}.csv`;
  `results/figures/exp07d_{roc_plain_vs_7a_vs_7d_ood_val,pr_plain_vs_7a_vs_7d_ood_val,
  fixed_specificity_sensitivity_comparison}.png`.
- Milestone 7E-0 (transform-only): `results/figures/exp07e0_stain_space_aug_preview.png`;
  `results/logs/exp07e0_stain_space_aug_smoke_report.txt`.
- Milestone 7E: `results/checkpoints/exp07e_stainspace_resnet18/{best.pt,last.pt}`;
  `results/logs/exp07e_stainspace_resnet18_{log.csv,train_report.txt,eval_report.txt}`;
  `results/predictions/exp07e_stainspace_resnet18/{id_val,ood_val}_predictions.csv`;
  `results/metrics/exp07e_stainspace_resnet18/{classification_metrics.json,comparison_metrics.json}`;
  `results/tables/exp07e_stainspace_resnet18_classification_metrics.csv`,
  `exp07e_{default_threshold_comparison,threshold_free_comparison,fixed_specificity_comparison}.csv`;
  `results/figures/exp07e_{roc_plain_vs_7a_vs_7d_vs_7e_ood_val,pr_plain_vs_7a_vs_7d_vs_7e_ood_val,
  fixed_specificity_sensitivity_comparison}.png`.
- Milestone 7F-0 (metadata/cache audit only): `results/tables/exp07f0_cache_metadata_summary.csv`,
  `exp07f0_group_label_counts.csv`, `exp07f0_dg_method_feasibility.csv`;
  `results/logs/exp07f0_dg_group_metadata_audit_report.txt`, `exp07f0_run_console.txt`.
- Milestone 7F-1 (center-stratified cache rebuild; no training):
  `results/cache/center_strat_train_c034_percell300_seed1337_shuf.pt`,
  `center_strat_id_val_c034_percell75_seed1337_shuf.pt`;
  `results/tables/exp07f1_center_stratified_cache_availability.csv`,
  `exp07f1_center_stratified_cache_summary.csv`,
  `exp07f1_center_stratified_label_center_counts.csv`;
  `results/logs/exp07f1_center_stratified_cache_rebuild_report.txt`, `exp07f1_run_console.txt`.
- Milestone 7F (matched center-stratified ERM + GroupDRO):
  `results/checkpoints/{exp07f_centerstrat_erm_resnet18,exp07f_groupdro_resnet18}/{best.pt,last.pt}`;
  `results/predictions/{exp07f_centerstrat_erm_resnet18,exp07f_groupdro_resnet18}/{id_val,ood_val}_predictions.csv`;
  `results/metrics/exp07f_centerstrat_erm_resnet18/classification_metrics.json`,
  `results/metrics/exp07f_groupdro_resnet18/{classification_metrics.json,comparison_metrics.json}`;
  `results/tables/exp07f_{overall_metrics_comparison,id_val_per_center_metrics,ood_val_metrics_comparison,fixed_specificity_comparison}.csv`;
  `results/figures/exp07f_{roc_centerstrat_erm_vs_groupdro_ood_val,pr_centerstrat_erm_vs_groupdro_ood_val,fixed_specificity_sensitivity_comparison,id_val_per_center_accuracy}.png`;
  `results/logs/{exp07f_centerstrat_erm_groupdro_train_report,exp07f_centerstrat_erm_groupdro_eval_report}.txt`,
  `results/logs/{exp07f_centerstrat_erm_resnet18_log,exp07f_groupdro_resnet18_log}.csv`.
- Milestone 7G (calibration + candidate operating-point audit of the 7F models; CSV/logit-only):
  `results/predictions/exp07g_7f_calibration/{centerstrat_erm,groupdro}_ood_val_calibrated_predictions.csv`;
  `results/metrics/exp07g_7f_calibration/{temperature.json,calibration_metrics.json}`;
  `results/tables/exp07g_{calibration_comparison,high_confidence_fn_comparison,candidate_operating_point_audit}.csv`;
  `results/figures/exp07g_{calibration_raw_vs_calibrated,high_confidence_fn_comparison,fixed_specificity_sensitivity_comparison}.png`;
  `results/logs/exp07g_7f_calibration_operating_audit_report.txt`.
- Milestone 8B (full development-scale evaluation; inference-only):
  `results/predictions/exp08b_full_dev_eval/{groupdro,centerstrat_erm}_{id_val,ood_val}_predictions.csv`;
  `results/metrics/exp08b_full_dev_eval/full_dev_metrics.json`;
  `results/tables/exp08b_{overall_metrics,id_val_per_center_metrics,candidate_operating_points,high_confidence_false_negatives}.csv`;
  `results/figures/exp08b_{ood_val_roc_groupdro_vs_erm,ood_val_pr_groupdro_vs_erm,candidate_fixed_specificity_sensitivity,id_val_per_center_accuracy,high_confidence_fn_comparison}.png`;
  `results/logs/exp08b_full_dev_eval_inference_report.txt`, `exp08b_full_development_eval_report.txt`.

## 10. Historical milestone-plan archive — superseded; do not follow as current instructions

> The material below preserves then-current handoff plans from Milestones 7F–8D. Those steps were
> subsequently completed. This quoted historical handoff is superseded by Milestone 9B; the current
> next action is literature/citation verification and journal-template formatting, with no second
> `ood_test` run.

Milestones through **7E** are complete and accepted; the no-training **Milestone 7F-0 (DG
feasibility / group-label audit)**, the gated no-training **Milestone 7F-1 (center-stratified
cache rebuild)**, the gated **Milestone 7F (matched center-stratified ERM + GroupDRO training)**,
the gated CSV/logit-only **Milestone 7G (calibration + candidate operating-point audit of the two
7F models)**, the reporting-only **Milestone 7H (consolidated DG reporting package)**, the
docs/tables/logs-only **Milestone 7I (pre-final-evaluation decision audit)**, the planning-only
**Milestone 8A (full development-scale evaluation planning package)**, and the inference-only
**Milestone 8B (execution of that full development-scale evaluation)**, the CSV/logit-only
**Milestone 8C (full-development calibration + reliability audit)**, and the docs/tables/logs-only
**Milestone 8D (final protocol freeze / final-evaluation readiness package)** were complete at that
historical handoff point. **Milestone 8B evaluated the two frozen 7F
checkpoints (inference only) on the FULL development splits (`id_val` {0,3,4} n=33,560; `ood_val`
{1} n=34,904); GroupDRO still beat its matched ERM on OOD (ood_val AUROC 0.8956 vs 0.8673; matched
fixed-specificity sensitivity 4/4; improved worst-center id_val), and full-split AUROC was much
higher than the capped 2000-sample `ood_val`.** **Milestone 8C** then temperature-scaled those
full-split logits on the full `ood_val` (development-stage; GroupDRO T=2.9749, ERM T=3.4963),
improving ECE/Brier/NLL for both and confirming GroupDRO remains better; **Milestone 8D** produced
the final-evaluation **readiness / protocol-freeze package** (froze checkpoint artifacts/paths;
proposed the candidate + control pair; specified the operating-point family + 8C temperatures;
drafted the single-shot `ood_test` protocol,
output schema, stop rules, claims-unlock map, and authorization template; **no** final model /
threshold / calibration policy / `ood_test` committed). **Historical 8D recommendation, now
superseded:** review the protocol or write the manuscript skeleton; the manuscript and later audits
were subsequently completed in 8E–8H. The single final `ood_test`
run stays LOCKED and may proceed **only** after the user completes
`docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md`; model-based uncertainty (DEFERRED / optional) is not the
default next step.
7F-0 found that full group metadata IS preserved in all
caches, but the capped `train` cache was center↔label confounded with center 4 absent, so
center-based DG was runnable-but-not-sound (**Recommendation A: rebuild required**). **7F-1 then
performed that rebuild** — `scripts/25_rebuild_center_stratified_caches.py` built new
center×label-balanced caches (train 300/cell → n=1800, 900/900, 600 per center; id_val 75/cell →
n=450, 225/225, 150 per center) by per-`(center,label)` quota sampling across centers {0,3,4},
preserving all seven metadata fields and the serializer schema, on new filenames with no overwrite —
so **DG-by-center is now SOUND**. 5B/5C added deterministic uncertainty /
selective-prediction tables; **6A** consolidated them into a reliability reporting package;
**6B** produced the manuscript/readiness audit package; **7A** added the first
stain-robustness / DG baseline (stain/color augmentation ERM) and found that a *moderate*
augmentation did not improve OOD at the default 0.5 threshold; **7B** (CSV-only) showed that
7A's apparent OOD sensitivity collapse is **mostly a threshold artifact**; **7C** (CSV/logit-only)
calibrated the stain-aug model (T=3.2102; ECE 0.356→0.191; high-confidence missed tumors 399→0)
and confirmed calibration is **argmax-invariant** and that **plain ERM remains stronger at every
matched fixed-specificity candidate point**; **7D** trained a **tuned torchvision stain/color
augmentation v2** baseline and found it **did not improve OOD** (ood_val AUROC 0.7397 < plain
0.7696 and 7A 0.7492; 7D sensitivity below both baselines at every matched fixed-specificity
target) — establishing that naive→tuned **torchvision** stain/color augmentation is an
insufficient stain-robustness lever here; **7E-0** implemented and **validated** an
optical-density (HED-style) **stain-space** augmentation (transform-only smoke test, PASS —
numerically safe + pathology-plausible); **7E** then **trained** an ERM with that stain-space
augmentation and ran a 4-model development comparison — the OD-space transform did **not** lift
OOD *ranking* (ood_val AUROC 0.7317, lowest of the four) but is the **strongest stain-augmentation
variant at every matched fixed-specificity operating point** (beats 7A and 7D at 4/4, still below
plain ERM at 0/4). **Net conclusion across 7A→7E: no stain-augmentation lever (torchvision colour
jitter or OD-space) beats plain-ERM OOD ranking on this capped subset.** **Milestone 7F then trained
a dedicated DG objective (GroupDRO by center) with a matched center-stratified ERM control on the new
sound caches — and GroupDRO beat its matched ERM on OOD (ood_val AUROC 0.7129 vs 0.6180; matched
fixed-specificity sensitivity 4/4), the first DG lever to beat its matched control on OOD ranking.**
Remaining candidate steps (each gated behind an explicit written go-ahead, none started):

0. **DG feasibility / group-label audit (Milestone 7F-0 — ✅ COMPLETE; no training).** Done —
   `scripts/24_dg_group_metadata_audit.py`. All three caches preserve full group metadata
   (`center`/`patient`/`slide`/`node`/`image_id`/`x_coord`/`y_coord`), but the capped `train`
   cache is **center↔label confounded** (center 0 ~9% tumor, center 3 100% tumor; center 4 absent),
   so center-based DG is **runnable but not sound**. **Verdict: Recommendation A — a
   center-stratified cache rebuild is required before sound DG-by-center training.**
1. **Center-stratified metadata-rich cache rebuild (Milestone 7F-1 — ✅ COMPLETE; gated; no
   training).** Done — `scripts/25_rebuild_center_stratified_caches.py` filled per-`(center,label)`
   quotas across source centers `{0,3,4}` (train 300/cell → n=1800, 900/900, 600 per center; id_val
   75/cell → n=450, 225/225, 150 per center), preserving the locked mapping, all seven metadata
   fields, and the serializer schema, on **new** cache filenames
   (`center_strat_train_c034_percell300_seed1337_shuf.pt`,
   `center_strat_id_val_c034_percell75_seed1337_shuf.pt`; no overwrite). **DG-by-center is now
   SOUND** (new caches support GroupDRO / CORAL / DANN by center). OVERALL PASS; no `ood_test`;
   existing caches unmodified.
2. **Dedicated domain-generalization method (Milestone 7F — ✅ COMPLETE; gated training step).**
   Done — `src/training/groupdro.py` + `scripts/26` + `scripts/27`. Trained a matched
   center-stratified ERM control and a GroupDRO-by-center model on the new sound center-stratified
   caches (`id_val_loss` selection only; `ood_val` read only after training). **GroupDRO beat its
   matched ERM control on OOD** — ood_val AUROC **0.7129 vs 0.6180 (+0.0949)**, AUPRC 0.6287 vs
   0.5214, default-0.5 sensitivity 0.0700 vs 0.0100, matched fixed-specificity ood_val sensitivity
   4/4 targets — the first DG lever to beat its matched control on OOD ranking. It did **not** improve
   worst-center id_val behavior. Old plain/7A/7D/7E are reference-only (old confounded cache, not
   matched). OVERALL PASS; no calibration; no committed threshold; `ood_test` untouched.
   **Recommendation: RETAIN GroupDRO.**
3. **Calibration + candidate operating-point audit of the 7F models (Milestone 7G — ✅ COMPLETE;
   gated; CSV/logit-only).** Done — `scripts/28_calibrate_7f_groupdro_operating_audit.py`
   temperature-scaled each matched 7F model on its `ood_val` logits (center-strat ERM T=3.9239,
   GroupDRO T=3.6570) and audited candidate operating points. **Calibration improved confidence
   reliability for both** (ERM ECE 0.4984→0.3698, GroupDRO 0.4258→0.2529; high-confidence missed
   tumors reduced sharply, both →0 at conf≥0.99) but is **argmax-invariant**. **GroupDRO remains
   better than its matched ERM after the audit** — higher ood_val AUROC (0.7129 vs 0.6180) and ood_val
   sensitivity ≥ ERM at all 4 fixed-specificity targets (4/4 strictly greater); it is also the
   better-calibrated model both raw and calibrated. OVERALL PASS; no new training/inference; no
   committed threshold; `ood_test` untouched. **Recommendation: RETAIN GroupDRO with calibrated
   confidence; candidate (not committed) operating points only.**
4. **Consolidated DG reporting package (Milestone 7H — ✅ COMPLETE; reporting-only).** Done —
   `scripts/29_make_dg_reporting_package.py` consolidated the saved deterministic-reliability
   (4B/5A/5B/5C/6A), stain-augmentation (7A–7E), center-stratified-cache (7F-1), GroupDRO (7F), and
   calibration + operating-point-audit (7G) results into paper-facing DG tables/figures/docs. **No
   training, no inference, no calibration fitting, no MC-dropout, no ensemble, no CORAL/DANN, no
   `ood_test`, no committed threshold, no final-performance claim.** Delivered 6 tables
   (`results/tables/exp07h_*`), 5 figures (`results/figures/exp07h_*`), 2 new docs
   (`docs/DG_RESULTS_INTERPRETATION.md`, `docs/DG_MANUSCRIPT_UPDATE.md`), updated 3 existing docs
   (`docs/CLAIMS_AND_LIMITATIONS.md`, `docs/REVIEWER_RISK_AUDIT.md`, `docs/FIGURE_TABLE_MANIFEST.md`),
   and a report (`results/logs/exp07h_dg_reporting_package_summary.txt`). Clearly separates the **fair
   controlled comparison (7F ERM vs 7F GroupDRO)** from **context-only** old plain ERM / 7A / 7D / 7E;
   R3 reduced HIGH→partly-mitigated, R1/R4/R7 remain open. OVERALL PASS (all assertions; inputs
   verified unmodified; no warnings/errors). `ood_test` untouched.
5. **Pre-final-evaluation decision audit (Milestone 7I — ✅ COMPLETE; docs/tables/logs only).**
   Done — `scripts/30_make_pre_final_decision_audit.py`. **No training, no inference, no calibration
   fitting, no MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final-performance claim, no
   committed clinical threshold.** Recorded what is **frozen** vs **not frozen** before any larger/full
   development evaluation or final `ood_test`; identified the **primary controlled candidate model**
   (**7F GroupDRO with temperature-scaled confidence**); identified the **matched control** (**7F
   center-stratified ERM**); clearly separated the **context-only baselines** (**old plain ERM, 7A, 7D,
   7E**); recommended a full `ood_val`/full development evaluation before `ood_test` **if compute/time
   allows** (Path A); kept model-based uncertainty **DEFERRED / optional**; and produced a **decision
   tree for next experiments**. Delivered `docs/PRE_FINAL_EVALUATION_AUDIT.md`,
   `docs/FROZEN_DECISIONS.md`, `docs/NEXT_EXPERIMENT_DECISION_TREE.md`,
   `results/tables/exp07i_{pre_final_decision_matrix,model_readiness_table,forbidden_claims_before_test}.csv`,
   `results/logs/exp07i_pre_final_evaluation_audit_report.txt`. Inputs verified unmodified; OVERALL PASS.
   **Historical 7I recommendation (completed/superseded):** Path A was full development-scale
   evaluation planning; Path E was the manuscript skeleton. Both were subsequently addressed.
   `ood_test` remained untouched.
6. **Full development-scale evaluation planning (Milestone 8A — ✅ COMPLETE; planning-only).**
   Done — `scripts/31_make_full_development_eval_plan.py`. **No training, no inference, no calibration
   fitting, no MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`, no final-performance claim, no
   committed clinical threshold** (does not import torch; frozen inputs read-only). Produced the
   `docs/FULL_DEVELOPMENT_EVALUATION_{PROTOCOL,RESOURCE_PLAN,OUTPUT_SCHEMA,STOP_RULES}.md` docs, the four
   `results/tables/exp08a_*` tables, and `results/logs/exp08a_full_development_evaluation_planning_report.txt`.
   Froze the required checkpoints (**7F GroupDRO** primary + **7F center-stratified ERM** control; old
   plain ERM/7A/7D/7E optional context-only), defined development splits (full/larger `id_val` {0,3,4} +
   full/larger `ood_val` {1}; **`ood_test` excluded**), the threshold/calibration policy, required
   metrics/output schema, compute/storage risks, and nine stop rules. OVERALL PASS (23/23 assertions).
7. **Execute the full development-scale evaluation (Milestone 8B — ✅ COMPLETE; INFERENCE ONLY, no
   training).** Done — `scripts/32_run_full_development_eval.py` +
   `scripts/33_summarize_full_development_eval.py`. Loaded the two FROZEN 7F checkpoints (exact-path
   verified; weights unchanged) and ran eval-transform inference on the FULL development splits
   (`id_val` {0,3,4} n=33,560; `ood_val` {1} n=34,904); four prediction CSVs + CSV-only metrics/
   summary. **No training, no weight update, no calibration fitting, no MC-dropout, no ensemble, no
   CORAL/DANN, no `ood_test`, no committed threshold.** **GroupDRO still beat its matched ERM at
   full/larger development scale** — ood_val AUROC **0.8956 vs 0.8673 (+0.0283)**, AUPRC 0.8958 vs
   0.8757, default-0.5 sensitivity 0.6407 vs 0.6074, matched fixed-specificity ood_val sensitivity
   4/4 targets, and (new at full scale) improved worst-center (center 4) id_val behavior (acc
   0.6901→0.8031). Full-split OOD AUROC is much higher than the capped 2000-sample `ood_val`
   (GroupDRO 0.7129→0.8956; ERM 0.6180→0.8673) — the capped subset understated both models. OVERALL
   PASS (both scripts; all assertions). **Historical 8B recommendation (completed/superseded):
   retain GroupDRO and proceed to development-stage calibration; 8C subsequently completed that
   calibration.**
8. **Development-stage calibration of the 8B full-split logits + reliability audit (Milestone 8C —
   ✅ COMPLETE; CSV/logit-only).** Done — `scripts/34_calibrate_full_development_eval.py`
   temperature-scaled the two 7F models on the FULL `ood_val` logits (GroupDRO T=2.9749, ERM
   T=3.4963; development-only, argmax-invariant), reported raw-vs-calibrated ECE/Brier/NLL, audited
   high-confidence false negatives, and confirmed **GroupDRO remains better than its matched ERM
   after full-scale calibration**. OVERALL PASS; development-stage (fit+eval on same `ood_val`, R7
   open); no committed threshold; `ood_test` never read.
9. **Final protocol freeze / final-evaluation readiness package (Milestone 8D — ✅ COMPLETE;
   docs/tables/logs only).** Done — `scripts/35_make_final_eval_readiness_package.py` (does not
   import torch; reads only saved 8B/8C artifacts read-only). **No training, no inference, no
   calibration fitting, no MC-dropout, no ensemble, no CORAL/DANN, no `ood_test`.** Produced the
   **readiness / protocol-freeze package — NOT a final evaluation**: froze the checkpoint
   artifacts/paths; proposed the 7F GroupDRO primary + 7F center-stratified ERM matched-control
   final-test pair; specified the candidate operating-point family + 8C temperatures; drafted the
   single-shot `ood_test` protocol, output schema, stop
   rules, claims-unlock map, and the `ood_test` authorization template; committed **no** final model
   / threshold / calibration policy / `ood_test` access. OVERALL PASS. **Historical 8D handoff
   recommendation (completed/superseded):** review the protocol or prepare the manuscript skeleton.
10. **Model-based uncertainty estimation (DEFERRED / optional; NOT the default next step)** —
   MC-dropout / deep ensembles (requires inference); on `id_val` + `ood_val`. Only if explicitly
   approved.
11. **Final locked evaluation (historical plan; completed by 9A)** — the single, one-time
   `ood_test` (center 2) run was explicitly authorized and completed exactly once.

Current rule: `ood_test` has been read once; do not access or infer on it again. See
`docs/NEXT_ACTIONS.md`.
