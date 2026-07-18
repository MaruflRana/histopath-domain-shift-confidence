# RESULTS INTERPRETATION

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in
Histopathology: Calibration, Uncertainty, and Selective Prediction

**Scope (read first):** every number below is **development-stage, capped balanced subset**
analysis produced through Milestone 6A. These are **not** final, tuned, full-data, or
clinical performance figures. `ood_test` (center 2) is **LOCKED** and was never read.
Sources: `results/tables/report_table_{classification_shift,calibration,uncertainty,
selective_prediction}.csv` and `results/logs/report_reliability_package_summary.txt`.

Subset sizes: `id_val` n=500 (balanced), `ood_val` n=2000 (balanced, center 1). Positive
class = tumor (1). "FN" = false negative = a tumor patch predicted non-tumor (a *missed
tumor*, the clinically worst error).

---

## 1. ID → OOD degradation

| metric | id_val | ood_val (raw) | change |
|--------|-------:|--------------:|-------:|
| accuracy    | 0.9640 | 0.6195 | −0.3445 |
| AUROC       | 0.9933 | 0.7696 | −0.2237 |
| sensitivity | 0.9440 | 0.4050 | −0.5390 |
| FN count    | 14 / 500 | 595 / 2000 | — |

**Interpretation.** Moving from seen hospitals to the unseen center-1 hospital collapses
performance. The most alarming shift is **sensitivity 0.944 → 0.405**: on the OOD hospital
the model misses the majority of tumor patches. AUROC (0.770) degrades less than accuracy,
indicating the *ranking* of tumor-likelihood is partially preserved but the operating point
(threshold 0.5) is badly mismatched under shift. This is the core motivating result: a model
that looks near-perfect ID is unsafe on a new hospital.

---

## 2. Calibration effect (temperature scaling on `ood_val`)

Fitted temperature **T = 3.1224** (T>1 ⇒ the raw model was over-confident).

| metric | raw | calibrated | change |
|--------|----:|-----------:|-------:|
| ECE   | 0.2330 | 0.0918 | −61% |
| Brier | 0.2759 | 0.2160 | −22% |
| NLL   | 0.9152 | 0.6088 | −33% |

**Interpretation.** Temperature scaling substantially improves all three calibration/proper-
scoring metrics. The raw model was badly over-confident on the OOD hospital (ECE 0.23);
dividing logits by T≈3.12 softens the probabilities toward their empirical accuracy, cutting
ECE by ~61%. Brier and NLL — which reward *correct* confident probabilities — improve less
than ECE because calibration cannot fix the underlying errors, only their confidence.

---

## 3. High-confidence false-negative reduction

From the 5B high-confidence-error table, at confidence ≥ 0.90 on `ood_val`:

| variant | high-confidence FNs (≥0.90) |
|---------|----------------------------:|
| raw        | 222 |
| calibrated |   0 |

**Interpretation.** This is the most safety-relevant calibration benefit. Before calibration,
222 missed tumors were asserted with ≥90% confidence — exactly the "confidently wrong"
pattern that a downstream reviewer or triage rule would trust. After temperature scaling, no
`ood_val` FN retains ≥0.90 confidence, because the global T>1 pushes all confidences down.
Note this is a *magnitude* effect at a *chosen threshold*, not a change in which patches are
called tumor (see §6). It is real and useful, but subset- and threshold-specific.

---

## 4. Uncertainty / error-detection behavior

Entropy error-detection AUROC (higher ⇒ uncertainty better separates errors from correct):

| variant | entropy error-detection AUROC | entropy AURC | errors |
|---------|------------------------------:|-------------:|-------:|
| id_val             | 0.9267 | 0.003558 | 18 |
| ood_val_raw        | 0.6883 | 0.219861 | 761 |
| ood_val_calibrated | 0.6883 | 0.219861 | 761 |

**Interpretation.** ID, entropy is a strong error detector (0.93) — the model "knows when it
is wrong." Under shift this drops to 0.69: uncertainty still carries signal but is far weaker
at flagging OOD errors, and there are far more errors to flag (761 vs 18). AURC rises ~60× ID
→ OOD. Raw and calibrated are **identical** because temperature scaling is a monotone
transform and error-detection AUROC / AURC depend only on ranking (see §6).

---

## 5. Selective-prediction behavior

Entropy abstention (retain the most-confident rows; abstain the least-confident):

| coverage | id_val risk | id_val ret_FN | ood_val_cal risk | ood_val_cal ret_FN | ood FNs abstained |
|---------:|------------:|--------------:|-----------------:|-------------------:|------------------:|
| 1.00 | 0.0360 | 14 | 0.3805 | 595 | 0 (0.0%) |
| 0.90 | 0.0111 |  3 | 0.3633 | 517 | 78 (13.1%) |
| 0.70 | 0.0029 |  1 | 0.3243 | 356 | 239 (40.2%) |
| 0.50 | 0.0000 |  0 | 0.2500 | 198 | 397 (66.7%) |

**Interpretation.** ID, abstaining on the least-confident 50% drives retained risk to **0**
and abstains **all 14** false negatives — selective prediction works essentially perfectly ID.
OOD, the same 50% abstention only lowers risk 0.380 → 0.250, and **198 of 595 missed tumors
remain in the retained (acted-on) set**. Abstention helps, but cannot make the OOD retained
set safe, because many OOD errors are made *with high confidence* and are therefore retained,
not abstained.

---

## 6. Why calibration changes confidence but not argmax predictions

Temperature scaling replaces logits `z` with `z / T` for a single scalar `T > 0`. Softmax of
`z/T` is a **monotone** rescaling: it never changes which logit is largest, so `argmax`
(the class decision at a symmetric threshold) is unchanged. Consequently:
- **Unchanged:** accuracy, balanced accuracy, sensitivity, specificity, precision, F1, the
  confusion matrix (tn/fp/fn/tp = 834/166/595/405), AUROC/AUPRC, and any *ranking*-based
  metric (error-detection AUROC, AURC, risk–coverage ordering).
- **Changed:** the *magnitude* of confidence and every metric that reads magnitude — ECE,
  Brier, NLL, and threshold-indexed counts like "FNs at confidence ≥ 0.90."

This is why raw and calibrated share identical classification, error-detection, and
risk–coverage rows, yet calibration still removes high-confidence false negatives.

---

## 7. Why OOD risk stays high even after abstention

Selective prediction ranks patches by uncertainty and drops the most uncertain. It can only
remove errors that are *also uncertain*. Under hospital shift the model produces **confident
errors** — low-entropy mistakes — which sit in the retained (most-confident) region and
survive abstention. At 50% coverage on `ood_val`, 198/595 missed tumors are still retained
precisely because the model was confident about them. Since temperature scaling is monotone,
it does not re-order these patches and cannot move them out of the retained set. The ceiling
is therefore a property of the *uncertainty signal*, not of the abstention rule: fixing it
needs better uncertainty (e.g. model-based / ensemble) or better OOD robustness (domain
generalization) — both currently out of scope.

---

## 8. What these results support — and what they do not

**Supported (development-scope claims):**
- Hospital-domain shift causes severe reliability degradation in this capped development
  setting (accuracy, sensitivity, calibration all worsen sharply).
- Temperature scaling improves calibration metrics (ECE/Brier/NLL) on the `ood_val` subset.
- Calibrated confidence removes high-confidence false negatives at chosen thresholds on this
  `ood_val` subset.
- Deterministic uncertainty is a weaker error detector under OOD (0.69) than ID (0.93).
- Confidence-based abstention has a hard ceiling under shift (confident errors are retained).

**NOT supported (must not be claimed):**
- Any statement about `ood_test` (center 2) — it was never evaluated.
- Final generalization or clinical/deployment performance — this is a capped development run.
- Superiority over domain-generalization or model-based-uncertainty methods — none were run.
- That calibration improves *decisions* or accuracy — it does not (argmax-invariant).
- That abstention makes OOD predictions safe — it does not (risk stays 0.25 at 50% coverage).

See `docs/CLAIMS_AND_LIMITATIONS.md` and `results/tables/report_claims_audit.csv` for the
authoritative allowed/forbidden list.
