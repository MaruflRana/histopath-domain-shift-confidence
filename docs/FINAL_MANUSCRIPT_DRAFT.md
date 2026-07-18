# When Development Gains Do Not Transfer: Confidence-Aware Tumor Detection Under Reserved-Hospital Shift

## Abstract

Hospital domain shift can change not only discrimination but also confidence reliability and the operating behavior of histopathology classifiers. We evaluated this problem using a locked Camelyon17-WILDS hospital split, a center-stratified empirical-risk-minimization (ERM) control, and a Group Distributionally Robust Optimization (GroupDRO) model grouped by source hospital. Model development used centers 0, 3, and 4 for training and in-distribution validation and center 1 for out-of-distribution development. Center 2 was reserved for one explicitly authorized final evaluation. On full development center 1, the predeclared GroupDRO primary candidate exceeded the matched ERM control in AUROC (0.8956 versus 0.8673). This ordering reversed on the reserved center: ERM achieved higher AUROC (0.6984 versus 0.6634), AUPRC (0.6556 versus 0.6364), accuracy (0.5712 versus 0.5337), sensitivity (0.2411 versus 0.1106), and fewer false negatives (32,275 versus 37,825). GroupDRO retained higher specificity (0.9569 versus 0.9012) and slightly higher precision.

Pre-frozen temperature scaling was applied without refitting on the test hospital. It improved expected calibration error, Brier score, and negative log-likelihood for both models, while leaving hard predictions and total false-negative counts unchanged. Calibration also sharply reduced the number of missed tumors assigned high confidence, showing that confidence correction is not error correction. Fourteen thresholds selected only on in-distribution validation data did not reliably preserve their nominal sensitivity or specificity targets on the reserved hospital and remained candidate/non-clinical operating points.

The negative model result is paired with a positive protocol finding: strict separation of development and reserved hospitals exposed a development-to-test reversal that would have been hidden if the development OOD center had been treated as final performance. These results support hospital-specific external validation, explicit calibration auditing, and predeclared operating policies before clinical translation.

## Keywords

Histopathology; domain shift; hospital generalization; GroupDRO; calibration; temperature scaling; selective prediction; false negatives; external validation; Camelyon17-WILDS.

## 1. Introduction

Machine-learning systems for histopathology are commonly developed on data from a limited set of institutions, scanners, staining protocols, and laboratory workflows. Differences in those conditions can alter image appearance and case mix, creating domain shift when a model is evaluated at a new hospital [CITATION NEEDED: histopathology domain shift and stain/scanner variability]. External performance under such shift is a central concern for computational pathology because strong in-distribution discrimination does not guarantee reliable behavior at another institution [CITATION NEEDED: external validation of computational pathology models].

Reliability under shift has several dimensions. Discrimination describes whether positive examples tend to receive higher scores than negative examples. Calibration describes whether predicted probabilities agree with observed frequencies. Threshold-specific metrics describe the operating consequences of converting scores into decisions. These properties can fail differently: a model may preserve ranking while becoming overconfident, or a threshold selected at one hospital may not preserve its intended sensitivity or specificity at another [CITATION NEEDED: discrimination calibration and operating-point transportability]. For tumor detection, false negatives are particularly important, and a confidently stated false negative can be more concerning than an uncertain error because it may be less likely to trigger review [CITATION NEEDED: high-confidence errors and diagnostic safety].

Domain-generalization methods attempt to improve performance on unseen domains by using variation among source domains during training [CITATION NEEDED: domain generalization overview]. Group Distributionally Robust Optimization (GroupDRO) emphasizes groups with higher training loss and can improve worst-group performance when relevant groups are known [CITATION NEEDED: original GroupDRO method]. However, gains on a development OOD domain may not transfer to a different unseen hospital. Treating an OOD validation hospital as final evidence after repeated model assessment would obscure this distinction.

This study combines a controlled GroupDRO-versus-ERM comparison with calibration, candidate operating-point analysis, and a single-shot reserved-hospital protocol. We used centers 0, 3, and 4 as source hospitals, center 1 for OOD development, and center 2 as the reserved final hospital. The model pair, checkpoint artifacts, temperatures, thresholds, metrics, and run limit were frozen before center 2 was accessed. Our primary question was not only whether GroupDRO improved development OOD performance, but whether that advantage and the associated confidence policies transferred to the reserved hospital.

The principal result was a development-to-test reversal. GroupDRO outperformed matched ERM on development center 1, but the matched control outperformed the predeclared primary on final center 2. Frozen temperature scaling improved reliability metrics for both models without changing classification errors, while development-selected operating targets transferred poorly. The study therefore contributes both a negative model result and a positive protocol lesson: hospital-specific model rankings and operating policies require genuinely reserved evaluation.

## 2. Related work

### 2.1 Domain shift in computational pathology

Histopathology models can be sensitive to site-specific staining, scanners, tissue preparation, and acquisition workflows [CITATION NEEDED: computational pathology domain shift]. Stain normalization and augmentation are common responses, but their effectiveness can vary by dataset and target domain [CITATION NEEDED: stain normalization and augmentation review]. External validation across institutions is therefore recommended before clinical interpretation [CITATION NEEDED: medical AI external validation guidance].

### 2.2 Camelyon17-WILDS

Camelyon17 contains lymph-node histopathology patches from multiple hospitals and is used in WILDS as a benchmark for distribution shift [CITATION NEEDED: Camelyon17 dataset paper] [CITATION NEEDED: WILDS benchmark paper]. Hospitals are represented by center metadata, enabling source, development OOD, and held-out domain comparisons. The present study follows a fixed center mapping throughout and does not create random splits.

### 2.3 Group-based robust optimization

GroupDRO minimizes a worst-group-weighted objective using known group labels and has been proposed for settings where average risk can hide poor subgroup performance [CITATION NEEDED: GroupDRO]. Its success depends on whether the training groups capture variation relevant to future domains. Hospital-center labels may represent meaningful source differences but can also be too coarse to capture scanner, stain-batch, slide, or patient variation within a center [CITATION NEEDED: limitations of group-defined robustness].

### 2.4 Calibration and selective prediction

Temperature scaling is a post-hoc calibration method that divides logits by a learned scalar before softmax and is widely used because it preserves score ordering and predicted classes [CITATION NEEDED: temperature scaling]. Expected calibration error (ECE), Brier score, and negative log-likelihood (NLL) summarize different aspects of probabilistic reliability [CITATION NEEDED: calibration metrics]. Selective prediction and abstention use uncertainty or confidence to defer selected predictions, but confidence policies can themselves degrade under shift [CITATION NEEDED: selective prediction under distribution shift]. In this study, selective concepts motivate the high-confidence false-negative audit; no clinical abstention policy is validated.

## 3. Materials and methods

### 3.1 Dataset and locked hospital split

We used the Camelyon17-WILDS histopathology patch dataset through the Hugging Face dataset identifier `wltjr1007/Camelyon17-WILDS`. Each observation is a 96×96 image patch with binary label 0 for non-tumor and 1 for tumor, together with center, patient, slide, node, image identifier, and spatial-coordinate metadata. RGBA inputs were converted to RGB before model preprocessing.

The split mapping was fixed before experimentation. Training used the Hugging Face `train` split restricted to centers 0, 3, and 4. In-distribution validation (`id_val`) used the Hugging Face `validation` split restricted to centers 0, 3, and 4. Development OOD validation (`ood_val`) used the same Hugging Face `validation` split restricted to center 1. The final reserved evaluation (`ood_test`) used the Hugging Face `test` split restricted to center 2.

| Logical split | Hugging Face split | Center(s) | Total rows | Study role |
|---|---|---|---:|---|
| `train` | `train` | {0,3,4} | 302,436 | Source pool for model fitting |
| `id_val` | `validation` | {0,3,4} | 33,560 | Model and threshold selection |
| `ood_val` | `validation` | {1} | 34,904 | OOD development decisions |
| `ood_test` | `test` | {2} | 85,054 | Single reserved final evaluation |

Center 1 was used for development comparison and calibration and was never treated as final performance. Center 2 remained unread until the model pair and all reporting policies were frozen. No random split, cap, sample, or silent truncation was applied to the full development or final evaluations.

### 3.2 Center-stratified cache construction

The source stream was label ordered. A naive balanced cache built only by filling class quotas became confounded between center and label: center 0 was predominantly non-tumor, center 3 was entirely tumor in the cache, and center 4 was absent. Such a cache would make center-grouped training scientifically unsound because center identity would encode label.

We therefore constructed new center-stratified caches by sampling per `(center, label)` cell across source centers 0, 3, and 4. The training cache contained 300 patches per center-label cell, totaling 1,800 patches: 900 tumor and 900 non-tumor, with 600 patches per center. The validation cache contained 75 patches per center-label cell, totaling 450 patches: 225 per class and 150 per center. All seven metadata fields were preserved, and the caches were written to new files without overwriting earlier development caches. The older confounded caches were retained only for reproduction of earlier context baselines and were not used for the controlled GroupDRO comparison.

### 3.3 Models and controlled comparison

Both controlled models used a ResNet-18 backbone with a two-logit classification head. The matched ERM control minimized standard cross-entropy. The GroupDRO model used center as the group label. For each mini-batch, per-center mean cross-entropy losses were computed, group weights were updated with an exponentiated-gradient rule, and the training objective was the group-weighted loss. Groups absent from a batch retained their previous weight. [CITATION NEEDED: GroupDRO objective and exponentiated-gradient update]

Training initialization, source cache, seed, optimizer, schedule, and model-selection rule were matched. Checkpoint selection used `id_val_loss` only. Center 1 was accessed only after training for development-stage evaluation. The final frozen artifacts were the 7F GroupDRO checkpoint as the predeclared primary and the 7F center-stratified ERM checkpoint as the matched control. Older plain ERM and stain-augmentation models used different caches and were context-only, not controlled comparators.

### 3.4 Development-stage evaluation

The two frozen checkpoints were evaluated on the full `id_val` split of 33,560 patches and the full `ood_val` center-1 split of 34,904 patches. We reported accuracy, balanced accuracy, AUROC, AUPRC, sensitivity, specificity, precision, F1, and confusion counts with tumor as the positive class. Per-center `id_val` metrics assessed behavior on source centers 0, 3, and 4.

Development evidence was used to choose the predeclared final primary and matched control, but it was not reported as held-out final performance. This distinction is essential because center 1 informed development decisions.

### 3.5 One-shot reserved-center protocol

The final protocol fixed the exact checkpoint paths and SHA256 hashes, the GroupDRO-primary and ERM-control roles, raw and calibrated reporting, the default threshold of 0.5, all 14 `id_val`-selected operating thresholds, the high-confidence audit thresholds, the output schema, and a maximum of one inference attempt. Written authorization required seven explicit approvals and a canonical authorization phrase.

Before test access, the runner verified the authorization document, checkpoint provenance, frozen configuration, threshold manifest, output freshness, run counter, and absence of prior run sentinels. It then atomically wrote a durable `RUN_STARTED` record before importing the locked dataset loader. The final dataset was instantiated once, restricted to center 2, validated to contain exactly 85,054 rows, and traversed once. Both models were evaluated in the same batch loop under inference mode. A completed-inference sentinel was written only after all four raw/calibrated prediction files passed validation. The CSV-only summary required that sentinel and did not load a model or dataset.

No training, optimizer construction, backward pass, checkpoint save, model modification, calibration fitting, threshold tuning, or post-test model selection was permitted. A crash after test access would not have authorized an automatic retry. The completed attempt count was one.

### 3.6 Temperature scaling

During development, a scalar temperature was fit to each model's center-1 logits by minimizing NLL. The resulting temperatures—GroupDRO T=2.974907 and ERM T=3.496293—were frozen before test access. On center 2, these values were applied only; they were never refit.

Temperature scaling divides both logits by the same positive scalar before softmax. It therefore preserves score ordering and argmax predictions while changing probability magnitudes [CITATION NEEDED: temperature scaling and argmax invariance]. We reported ECE with 15 bins, Brier score, and NLL for raw and calibrated probabilities.

### 3.7 Candidate operating points

Thresholds were selected on `id_val` only. For each model, four thresholds targeted specificity 0.80, 0.85, 0.90, and 0.95, and three thresholds targeted sensitivity 0.80, 0.90, and 0.95. These 14 thresholds were frozen and applied unchanged to center 1 during development and center 2 during final evaluation.

The thresholds were descriptive candidate/non-clinical operating points. No threshold was selected, optimized, or tuned on center 2, and no threshold was adopted for deployment.

### 3.8 High-confidence false-negative audit

We defined a high-confidence false negative as an observation with true label 1, predicted label 0, and predicted confidence at or above 0.90, 0.95, or 0.99. Counts were computed for raw and calibrated probabilities. Because calibration does not change predicted labels, total false negatives were expected to remain constant; the audit measured how many misses were stated with high confidence.

### 3.9 Statistical and reporting policy

The analysis reports prespecified descriptive point estimates and confusion counts for the complete locked splits. No hypothesis test, confidence interval, bootstrap resampling, threshold search, or post-test model-selection procedure was performed in the final run. The patch-level observations are not treated as independent patient-level outcomes, and no clinical utility analysis is claimed.

The primary comparative interpretation was fixed before test access: GroupDRO remained the predeclared primary and ERM the matched control regardless of the final ordering. Development and final metrics are shown separately. All unfavorable findings are retained.

## 4. Results

### 4.1 Full-development evaluation favored GroupDRO

On the full `id_val` split, GroupDRO achieved accuracy 0.841567 and AUROC 0.932256, compared with ERM accuracy 0.798659 and AUROC 0.908538. On development `ood_val` center 1, GroupDRO achieved accuracy 0.784724, AUROC 0.895609, AUPRC 0.895795, sensitivity 0.640672, specificity 0.928776, and 6,271 false negatives. The matched ERM control achieved accuracy 0.767477, AUROC 0.867271, AUPRC 0.875723, sensitivity 0.607380, specificity 0.927573, and 6,852 false negatives.

Thus, the development AUROC advantage for GroupDRO was approximately 0.0283. GroupDRO also improved the worst source-validation center: on center 4, accuracy increased from 0.6901 for ERM to 0.8031 for GroupDRO, and AUROC increased from 0.8323 to 0.8980. These findings supported the predeclared selection of GroupDRO as the final primary candidate, but they remained development evidence.

### 4.2 The matched ERM control outperformed the predeclared primary on center 2

The final center-2 dataset contained 85,054 patches, evenly divided between 42,527 tumor and 42,527 non-tumor labels. Both models produced predictions for all rows, and all required metadata and probability validations passed.

| Metric | Predeclared GroupDRO primary | Matched ERM control |
|---|---:|---:|
| Accuracy | 0.5337197545089002 | 0.5711547957768006 |
| Balanced accuracy | 0.5337197545089002 | 0.5711547957768006 |
| AUROC | 0.6633704256200204 | 0.6984352121958427 |
| AUPRC | 0.63641261236421 | 0.6556283801091156 |
| Sensitivity | 0.11056505278999224 | 0.24107037881816257 |
| Specificity | 0.9568744562278082 | 0.9012392127354386 |
| Precision | 0.7194002447980417 | 0.7093827843897038 |
| F1 | 0.19167193200578847 | 0.3598518752522859 |
| TN | 40,693 | 38,327 |
| FP | 1,834 | 4,200 |
| FN | 37,825 | 32,275 |
| TP | 4,702 | 10,252 |

The matched ERM control exceeded the predeclared GroupDRO primary by 0.0351 AUROC, 0.0192 AUPRC, 0.0374 accuracy, 0.1305 sensitivity, and 0.1682 F1, and produced 5,550 fewer false negatives. GroupDRO retained specificity higher by 0.0556 and precision higher by approximately 0.0100. The development GroupDRO advantage did not generalize to center 2.

The reversal is visible in the GroupDRO-minus-ERM AUROC contrast: approximately +0.0283 on development center 1 and −0.0351 on final center 2. ERM remains the matched control rather than being relabeled as the primary after the result.

### 4.3 Frozen temperature scaling improved held-out reliability for both models

For GroupDRO, applying T=2.974907 reduced ECE from 0.4006439581474795 to 0.2583399203510463, Brier score from 0.41233859378344173 to 0.3096680541395694, and NLL from 1.9441627836686814 to 0.8743126064429281. For ERM, applying T=3.496293 reduced ECE from 0.3084182197890404 to 0.1489841758680231, Brier score from 0.3446598909605201 to 0.2555235818479269, and NLL from 1.4183276475805955 to 0.7130964662716416.

Calibrated ERM remained better calibrated than calibrated GroupDRO on all three metrics. Raw and calibrated AUROC, AUPRC, accuracy, sensitivity, specificity, precision, F1, and confusion counts were identical within each model. The final result therefore supports held-out improvement of these frozen temperatures on center 2, but not universal calibration validity.

### 4.4 Development-selected operating targets were not reliably preserved

The fixed-sensitivity thresholds showed pronounced underachievement on the reserved hospital. GroupDRO thresholds targeting sensitivity 0.80, 0.90, and 0.95 achieved center-2 sensitivities 0.1430, 0.2884, and 0.4748. The corresponding ERM sensitivities were 0.3573, 0.5223, and 0.6692. None preserved its nominal sensitivity target.

Fixed-specificity targets were closer in some cases but also shifted. GroupDRO achieved specificities 0.8497, 0.8957, 0.9385, and 0.9748 for nominal targets 0.80, 0.85, 0.90, and 0.95, with sensitivities declining from 0.2981 to 0.0726. ERM achieved specificities 0.7809, 0.8334, 0.8879, and 0.9431, with sensitivities declining from 0.4282 to 0.1582.

These values are observations of transfer failure, not a basis for choosing a new threshold. All operating points remain candidate/non-clinical, and no threshold was tuned on the final hospital.

### 4.5 Calibration sharply reduced high-confidence missed tumors without reducing total misses

For GroupDRO, high-confidence false negatives declined from 29,775 raw to 5,479 calibrated at confidence ≥0.90, from 25,620 to 755 at ≥0.95, and from 15,485 to 1 at ≥0.99. The total false-negative count remained 37,825.

For ERM, high-confidence false negatives declined from 19,142 to 1,828 at ≥0.90, from 14,626 to 513 at ≥0.95, and from 7,354 to 0 at ≥0.99. The total false-negative count remained 32,275.

Calibration therefore corrected the confidence attached to many errors but did not recover missed tumors. This distinction is clinically important in principle but does not establish a safe triage workflow.

## 5. Discussion

### 5.1 The negative model result is scientifically informative

The matched ERM control outperformed the predeclared GroupDRO primary on the reserved hospital despite the opposite ordering on development center 1. A selective presentation could have emphasized only the development result and concluded that GroupDRO improved hospital generalization. The reserved protocol prevents that interpretation.

The result does not invalidate the development findings. GroupDRO improved center-1 discrimination and worst-center source-validation behavior under the controlled setup. Instead, the reversal shows that improvement on one OOD hospital was insufficient evidence for a different hospital. Hospital identity is not a single axis of shift, and an objective that improves robustness to source-center variation may not address unseen acquisition or tissue characteristics at every target center.

### 5.2 Possible explanations remain hypotheses

Center 2 may differ from the source hospitals and center 1 along stain, scanner, tissue, or case-mix dimensions not represented by the center grouping [CITATION NEEDED: hospital-specific histopathology shift factors]. Center labels may also be too coarse to capture relevant within-hospital heterogeneity. Another possibility is project-level overfitting to center 1: repeated development comparisons and calibration decisions can make a development OOD hospital functionally similar to a validation set even when it is never used for gradient updates.

The study did not perform post-test image, representation, or subgroup exploration on center 2, because such analyses could encourage test-driven method development. These explanations should therefore be treated as hypotheses for a new study, not conclusions from the final test.

### 5.3 Calibration transferred better than classification performance

The pre-frozen temperatures improved ECE, Brier score, and NLL on center 2 for both models. This is stronger evidence than the development calibration audit because the temperatures were not estimated from center 2. Nevertheless, the result concerns one held-out hospital and two fixed models. It does not establish that the same temperatures will generalize to other centers or that temperature scaling solves distribution-shift calibration broadly.

The high-confidence false-negative analysis illustrates the value and limitation of calibration. Many errors moved below extreme-confidence thresholds, which can improve the interpretability of stated confidence. But the models still missed the same tumors. Confidence correction is not error correction, and calibration should not be described as improving sensitivity or clinical safety.

### 5.4 Operating-point instability is a separate transport problem

The failure of development-selected sensitivity targets to transfer shows that threshold policies can be less stable than aggregate ranking metrics. A threshold is tied to the score distribution, class conditional behavior, and population on which it was selected [CITATION NEEDED: threshold transportability under dataset shift]. Even when specificity targets were approximately retained, the associated sensitivity could be unacceptable.

The study therefore supports a conservative reporting policy: threshold-free metrics describe ranking, the 0.5 threshold provides a prespecified reference, and frozen candidate operating points show transport behavior. None is a validated clinical decision rule.

### 5.5 The protocol contribution

The strict separation between `ood_val` and `ood_test` exposed a development-to-test reversal that would have been hidden if center 1 had been reported as final performance. The one-shot design also prevented post-test calibration fitting, threshold tuning, checkpoint replacement, or model relabeling. This is a reproducibility and model-governance contribution rather than evidence that the evaluated models are clinically ready.

Similar discipline is recommended for medical-AI studies in which several hospitals are available: at least one institution should remain outside all model, calibration, and operating-policy decisions until a preregistered final evaluation [CITATION NEEDED: locked external validation and medical AI reporting guidance].

## 6. Limitations

The final evaluation contains one reserved hospital. It cannot establish robustness across all hospitals, scanners, populations, or health systems. All evidence comes from Camelyon17-WILDS, so there is no external non-Camelyon cohort.

The evaluation is patch-level. Patches from the same slide or patient may be correlated, and patch-level accuracy or calibration does not establish whole-slide, patient-level, or clinical-workflow effectiveness. No slide aggregation, patient-level endpoint, reader study, prospective evaluation, or clinical utility analysis was performed.

The models use ResNet-18 and short training on capped center-stratified caches rather than all source patches. The cache construction made the center-grouped comparison sound, but may omit source variation needed for stronger transfer. The results are not a performance ceiling for ERM, GroupDRO, or histopathology models.

Only GroupDRO was evaluated as a dedicated DG objective in the final controlled pair. CORAL, DANN, and other DG methods were not tested. MC-dropout and ensembles were also not evaluated, so the study does not characterize model-based epistemic uncertainty.

Development calibration used center 1 both for fitting and development-stage evaluation, which made those development calibration estimates optimistic. Applying the frozen temperatures to center 2 provides a held-out check, but only for one target hospital. Independent multi-center calibration validity remains unproven.

The final default-threshold sensitivities were low and false-negative counts high. No clinically validated threshold was selected. The candidate operating points transferred poorly and cannot be interpreted as deployment recommendations.

Finally, the one-shot protocol prevents iterative debugging or repeated estimation on center 2. This is a strength for unbiased interpretation but means that residual uncertainty cannot be reduced by rerunning the test. Future methods must be developed and evaluated in a new, separately reserved study rather than by reopening this final dataset.

## 7. Conclusion

In a controlled development comparison, GroupDRO appeared stronger than matched ERM on an OOD development hospital. That advantage reversed on a separately reserved hospital, where the matched ERM control achieved better discrimination, accuracy, sensitivity, F1, and false-negative count than the predeclared GroupDRO primary. Frozen temperature scaling improved held-out calibration metrics and sharply reduced high-confidence false negatives for both models, but did not change hard predictions or total misses. Development-selected operating targets were not reliably preserved.

The study's main implication is that development OOD gains, calibration results, and threshold policies should not be treated as final evidence for another hospital. A locked reserved-hospital protocol can reveal model-selection and reliability risks that development evaluation alone conceals. The completed results do not establish clinical readiness or universal method rankings; they support stricter multi-center validation and transparent reporting before clinical translation.

## Data availability

The source dataset is publicly available through the CAMELYON17-WILDS benchmark and the Hugging Face dataset mirror identified in the Methods. The study did not redistribute source histopathology images. Aggregate result tables, non-image figures, protocol documentation, and reproducibility materials supporting this article are publicly available at https://github.com/MaruflRana/histopath-domain-shift-confidence. Raw patch-level predictions, source images, trained checkpoints, authorization records, and local run-state artifacts are not publicly distributed.

## Code availability

Source code, configurations, guarded evaluation scripts, manuscript-generation scripts, and reproducibility documentation are publicly available at https://github.com/MaruflRana/histopath-domain-shift-confidence. The repository intentionally excludes source histopathology images, trained checkpoints, raw patch-level predictions, credentials, authorization records, run sentinels, and environment-specific caches.

## Ethics statement

Ethics approval and informed consent were not required for this study because it involved secondary analysis of a publicly available, de-identified benchmark dataset. No participants were prospectively recruited, no intervention was performed, and no identifiable private information was accessed. Responsibility for the original data collection and its associated ethical approvals remained with the original dataset creators.

## Conflict of interest

The authors declare that they have no known competing financial interests or personal relationships that could have influenced the work reported in this article.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Author contributions

Jishan Islam Maruf: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review and editing, and Project administration.

Ishtiak Al Mamoon: Supervision, Validation, and Writing – review and editing.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During preparation of this work, the corresponding author used OpenAI ChatGPT and Codex and Anthropic Claude Code to support code drafting, workflow documentation, literature-search assistance, citation verification, content organization, and language refinement. All experimental decisions, code execution, source verification, statistical results, scientific interpretation, and manuscript revisions were reviewed and validated by the corresponding author. Both authors reviewed and approved the final manuscript and this disclosure. No generative AI or AI-assisted image-generation tool was used to create or alter scientific figures, images, data, or experimental results.

## Acknowledgments

[AUTHOR TO COMPLETE: acknowledgments, computing support, and data-provider recognition]
