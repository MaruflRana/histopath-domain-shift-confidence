# MANUSCRIPT METHODS DRAFT — DEVELOPMENT ONLY (Milestone 8E)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology.

> **Development-stage methods draft. No experiment was run in Milestone 8E.** Prose is technical
> and deliberately conservative. The reserved test hospital (`ood_test`, center 2) is **LOCKED and
> unread**; §M8 describes its protocol only. Numbers appearing here are from completed
> development-stage milestones (7F-1/7F/7G/8B/8C) and are labeled as such.

---

## M1. Dataset and locked split mapping

We use the Camelyon17-WILDS histopathology patch dataset accessed through the Hugging Face
`datasets` streaming interface (`wltjr1007/Camelyon17-WILDS`). Each example is a 96×96 RGB patch
(images arriving as RGBA are converted to RGB) with a binary label (0 = non-tumor, 1 = tumor) and
metadata fields `center, patient, slide, node, image_id, x_coord, y_coord`. Hospitals are encoded
by `center`. We fix a **locked split mapping** for the entire study:

| Logical split | HF split | Center(s) | Role |
|---------------|----------|-----------|------|
| `train` | `train` | {0, 3, 4} | model fitting |
| `id_val` | `validation` | {0, 3, 4} | in-distribution validation / model & threshold selection |
| `ood_val` | `validation` | {1} | out-of-distribution *development* decisions only |
| `ood_test` | `test` | {2} | **reserved** unseen test hospital (single final evaluation) |

The mapping is never altered. `ood_val` (center 1) is used strictly for development-stage OOD
decisions — comparison, calibration, uncertainty and selective analysis — and never for training or
for final reporting. `ood_test` (center 2) is a **reserved unseen hospital**: it is touched exactly
once, at the very end, under written authorization, and is unread throughout the present work.

## M2. Center-stratified cache construction

The Hugging Face stream is label-ordered (all tumor patches stream before non-tumor patches), so a
balanced subset built by scanning the stream to fill per-class quotas becomes **center↔label
confounded**: in the naive cache, center 0 was ~9% tumor, center 3 was 100% tumor, and center 4 was
absent. A center-grouped domain-generalization objective trained on such a cache would learn
*center == label* rather than a domain-invariant tumor signal. We therefore rebuild **sound
center-stratified caches** by per-`(center,label)` quota sampling across source centers {0,3,4}
(train 300 patches per cell → n=1,800, 900/900, 600 per center; `id_val` 75 per cell → n=450,
225/225, 150 per center), preserving all metadata fields and the serializer schema and writing to
new filenames (no overwrite of prior caches). Each center then carries both labels, making
center-grouped DG training sound. The old confounded caches are retained only for reproducibility of
earlier baselines and are excluded from DG training.

## M3. Backbone and matched ERM baseline

The classifier is a ResNet-18 backbone with the final fully-connected layer replaced by a 2-logit
head (tumor vs non-tumor). The **matched center-stratified ERM control** is trained with standard
cross-entropy on the sound center-stratified `train` cache, with model selection by `id_val_loss`
only (minimum). It shares initialization, seed, optimizer, and schedule with the GroupDRO model so
that the only difference between the two is the training objective; this makes it the sole *fair*
baseline for the GroupDRO claim. Baselines trained on the old confounded cache (old plain ERM; stain
augmentation variants) are reported only as **context**, never as controlled comparators.

## M4. GroupDRO-by-center

We minimize a group-distributionally-robust objective over the center groups {0,3,4}. For each
mini-batch we compute per-group mean cross-entropy `L_g`, maintain group weights `q_g`, and apply an
exponentiated-gradient update `q_g ← q_g · exp(step · detach(L_g))` followed by renormalization;
groups absent from a batch carry their weight forward. The training loss is the weighted sum
`Σ_g q_g · L_g`, which emphasizes the worst-performing center. Training data, initialization, seed,
optimizer, and schedule match the ERM control exactly; model selection uses `id_val_loss` only.
`ood_val` is read **only after** training completes, for the development-stage comparison.

## M5. Calibration / temperature scaling (development-stage)

We apply single-parameter temperature scaling: a scalar temperature `T > 0` divides the logits
before softmax, fit by minimizing negative log-likelihood (NLL) on `ood_val` logits (Adam, bounded
`T`, fixed iteration cap). Temperature scaling is **argmax-invariant** — it rescales confidence but
does not change the predicted class — so accuracy, AUROC, AUPRC, sensitivity, specificity, and the
confusion matrix are unchanged; only ECE, Brier, NLL, and confidence magnitudes change. **Important
limitation (reviewer risk R7):** in the present development-stage work the temperature is both *fit*
and *evaluated* on the same `ood_val` split, so the reported calibration is an optimistic
development-stage reliability audit, **not** an independent-test calibration claim. The pre-frozen
development temperatures (GroupDRO T=2.974907; ERM T=3.496293) are the values that would later be
applied — never refit — on the reserved test.

## M6. Candidate operating points

Operating thresholds are selected on `id_val` **only** — at fixed target specificities
(0.80/0.85/0.90/0.95) and fixed target sensitivities (0.80/0.90/0.95) — and the selected threshold is
then applied unchanged to `ood_val`. These are **candidate, development-selected** operating points
reported for context; **no operating point is committed** as a clinical or deployment threshold, and
no threshold is ever chosen on OOD data.

## M7. High-confidence false-negative audit

Because a confidently missed tumor is the most consequential failure under shift, we audit
high-confidence false negatives: examples with `label == 1`, predicted negative, and confidence
≥ t for t ∈ {0.90, 0.95, 0.99}, reported for both raw and calibrated confidence. This quantifies how
many tumors are missed *with high stated confidence*, and how calibration changes that count (while
leaving the hard decisions unchanged).

## M8. Reserved `ood_test` protocol (drafted, NOT executed)

The reserved unseen test hospital (`ood_test`, center 2) is evaluated **exactly once**, under the
frozen protocol in `docs/FROZEN_FINAL_EVAL_PROTOCOL_DRAFT.md`, and only after explicit written
authorization (`docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md`). The protocol fixes: (i) frozen model
weights — no retraining, no fine-tuning, no post-hoc model/epoch selection; (ii) thresholds selected
on `id_val` only, before the test is read; (iii) calibration using the pre-frozen 8C temperatures
only, never refit on the test; (iv) a single run. Reported outputs mirror the development metric plan
(threshold-free, default-threshold, candidate operating points, calibration, high-confidence FN,
per-center/metadata). **Milestone 8E performs none of this**; the test remains LOCKED and unread.

## M9. Reproducibility

Global seeding covers Python, NumPy, and PyTorch/CUDA. All configuration lives under `configs/`;
all artifacts under `results/`. Scripts run in numbered order from the project root; each imports a
bootstrap module first that sets the path and installs a combined CA bundle to handle
TLS-intercepting environments before any `datasets` import. Streaming with a seeded balanced cache is
the default data access mode; transient network read errors are auto-retried by the loader.
