# When Development Gains Do Not Transfer: Confidence-Aware Tumor Detection Under Reserved-Hospital Shift

*Original Research Article*

## Abstract

Hospital domain shift can alter discrimination, confidence reliability, and threshold behavior in computational pathology. We evaluated tumor-patch classification using a locked Camelyon17-WILDS design: centers 0, 3, and 4 for training and in-distribution validation, center 1 for out-of-distribution development, and center 2 for one reserved final evaluation. A center-stratified empirical risk minimization (ERM) control was matched to a predeclared Group Distributionally Robust Optimization (GroupDRO) candidate. On development center 1, GroupDRO achieved higher area under the receiver operating characteristic curve (AUROC) than ERM (0.8956 vs 0.8673). This ordering reversed on center 2: ERM achieved AUROC 0.6984 vs 0.6634 and sensitivity at threshold 0.5 of 0.2411 vs 0.1106, with 32,275 vs 37,825 false negatives. GroupDRO retained higher specificity (0.9569 vs 0.9012). Temperatures fixed before test access improved expected calibration error, Brier score, and negative log-likelihood for both models without changing hard predictions or total false negatives. Fourteen operating points selected only on in-distribution validation data did not reliably preserve nominal sensitivity or specificity on the reserved hospital. A strict one-shot, predeclared hospital evaluation exposed a development-to-test reversal that would have been hidden by reporting the development out-of-distribution center as final evidence. These findings show that development gains, calibrated confidence, and operating thresholds require separately reserved hospital validation; they do not establish clinical readiness.

**Keywords:** Computational pathology; Domain shift; External validation; Model calibration; GroupDRO; Histopathology; Reliability

## 1. Introduction

Machine-learning systems for histopathology are commonly developed on data from a limited set of institutions, scanners, staining protocols, and laboratory workflows. Differences in those conditions can alter image appearance and case mix, creating domain shift when a model is evaluated at a new hospital.<sup>1-3</sup> External performance under such shift is a central concern for computational pathology because strong in-distribution discrimination does not guarantee reliable behavior at another institution.<sup>4-6</sup>

Reliability under shift has several dimensions. Discrimination describes whether positive examples tend to receive higher scores than negative examples. Calibration describes whether predicted probabilities agree with observed frequencies. Threshold-specific metrics describe the operating consequences of converting scores into decisions. These properties can fail differently: a model may preserve ranking while becoming overconfident, or a threshold selected at one hospital may not preserve its intended sensitivity or specificity at another.<sup>7-9</sup> For tumor detection, false negatives are particularly important. Because confidence displays and decision aids can affect human reliance, confidently stated errors warrant separate audit rather than being treated as equivalent to uncertain errors.<sup>10-12</sup>

Domain-generalization methods attempt to improve performance on unseen domains by using variation among source domains during training.<sup>13-14</sup> Group Distributionally Robust Optimization (GroupDRO) emphasizes groups with higher training loss and can improve worst-group performance when relevant groups are known.<sup>15</sup> However, gains on a development OOD domain may not transfer to a different unseen hospital. Treating an OOD validation hospital as final evidence after repeated model assessment would obscure this distinction.

This study combines a controlled GroupDRO-versus-ERM comparison with calibration, candidate operating-point analysis, and a single-shot reserved-hospital protocol. We used centers 0, 3, and 4 as source hospitals, center 1 for OOD development, and center 2 as the reserved final hospital. The model pair, checkpoint artifacts, temperatures, thresholds, metrics, and run limit were frozen before center 2 was accessed. Our primary question was not only whether GroupDRO improved development OOD performance, but whether that advantage and the associated confidence policies transferred to the reserved hospital.

The principal result was a development-to-test reversal. GroupDRO outperformed matched ERM on development center 1, but the matched control outperformed the predeclared primary on final center 2. Frozen temperature scaling improved reliability metrics for both models without changing classification errors, while development-selected operating targets transferred poorly. The study therefore contributes both a negative model result and a positive protocol lesson: hospital-specific model rankings and operating policies require genuinely reserved evaluation.

## 2. Materials and methods



### 2.1 Dataset and locked hospital split

We used the Camelyon17-WILDS histopathology patch dataset through the Hugging Face dataset identifier wltjr1007/Camelyon17-WILDS. Each observation is a 96 x 96 image patch with binary label 0 for non-tumor and 1 for tumor, together with center, patient, slide, node, image identifier, and spatial-coordinate metadata. RGBA inputs were converted to RGB before model preprocessing. The underlying challenge dataset and standardized WILDS benchmark are described elsewhere.<sup>16-18</sup>

The split mapping was fixed before experimentation. Training used the Hugging Face train split restricted to centers 0, 3, and 4. In-distribution validation (id_val) used the Hugging Face validation split restricted to centers 0, 3, and 4. Development OOD validation (ood_val) used the same Hugging Face validation split restricted to center 1. The final reserved evaluation (ood_test) used the Hugging Face test split restricted to center 2.

Center 1 was used for development comparison and calibration and was never treated as final performance. Center 2 remained unread until the model pair and all reporting policies were frozen. No random split, cap, sample, or silent truncation was applied to the full development or final evaluations.

The locked split mapping is summarized in Table 1.

**Table 1. Locked hospital split mapping**

| Logical split | HF split | Center(s) | Rows | Study role |
| --- | --- | --- | --- | --- |
| train | train | {0,3,4} | 302,436 | Model fitting |
| id_val | validation | {0,3,4} | 33,560 | Model/threshold selection |
| ood_val | validation | {1} | 34,904 | OOD development |
| ood_test | test | {2} | 85,054 | One reserved final run |

*Note.* HF indicates Hugging Face. Center 1 was development-only; center 2 was accessed once after all policies were frozen.

### 2.2 Center-stratified cache construction

The source stream was label ordered. A naive balanced cache built only by filling class quotas became confounded between center and label: center 0 was predominantly non-tumor, center 3 was entirely tumor in the cache, and center 4 was absent. Such a cache would make center-grouped training scientifically unsound because center identity would encode label.

We therefore constructed new center-stratified caches by sampling per (center, label) cell across source centers 0, 3, and 4. The training cache contained 300 patches per center-label cell, totaling 1,800 patches: 900 tumor and 900 non-tumor, with 600 patches per center. The validation cache contained 75 patches per center-label cell, totaling 450 patches: 225 per class and 150 per center. All seven metadata fields were preserved, and the caches were written to new files without overwriting earlier development caches. The older confounded caches were retained only for reproduction of earlier context baselines and were not used for the controlled GroupDRO comparison.

### 2.3 Model architecture

Both controlled models used a ResNet-18 convolutional neural-network backbone with a two-logit classification head. Inputs were 96 × 96 histopathology patches; RGBA images were converted to RGB before preprocessing. The controlled pair shared the same source cache, seed, optimizer, schedule, and model-selection rule.

### 2.4 Matched ERM control

The matched empirical risk minimization (ERM) control minimized standard cross-entropy. Checkpoint selection used in-distribution validation loss only. The final matched control was the frozen 7F center-stratified ERM checkpoint.

### 2.5 GroupDRO-by-center

The GroupDRO model used center as the group label. For each mini-batch, per-center mean cross-entropy losses were computed, group weights were updated with an exponentiated-gradient rule, and the training objective was the group-weighted loss. Groups absent from a batch retained their previous weight.<sup>15</sup> The final frozen artifact was the 7F GroupDRO-by-center checkpoint, retained as the predeclared primary candidate regardless of the final ordering.

### 2.6 Development evaluation

The two frozen checkpoints were evaluated on the full id_val split of 33,560 patches and the full ood_val center-1 split of 34,904 patches. We reported accuracy, balanced accuracy, AUROC, AUPRC, sensitivity, specificity, precision, F1, and confusion counts with tumor as the positive class. Per-center id_val metrics assessed behavior on source centers 0, 3, and 4.

Development evidence was used to choose the predeclared final primary and matched control, but it was not reported as held-out final performance. This distinction is essential because center 1 informed development decisions.

### 2.7 Reserved-hospital final protocol

The final protocol fixed the exact checkpoint paths and SHA256 hashes, the GroupDRO-primary and ERM-control roles, raw and calibrated reporting, the default threshold of 0.5, all 14 id_val-selected operating thresholds, the high-confidence audit thresholds, the output schema, and a maximum of one inference attempt. Written authorization required seven explicit approvals and a canonical authorization phrase.

Before test access, the runner verified the authorization document, checkpoint provenance, frozen configuration, threshold manifest, output freshness, run counter, and absence of prior run sentinels. It then atomically wrote a durable RUN_STARTED record before importing the locked dataset loader. The final dataset was instantiated once, restricted to center 2, validated to contain exactly 85,054 rows, and traversed once. Both models were evaluated in the same batch loop under inference mode. A completed-inference sentinel was written only after all four raw/calibrated prediction files passed validation. The CSV-only summary required that sentinel and did not load a model or dataset.

No training, optimizer construction, backward pass, checkpoint save, model modification, calibration fitting, threshold tuning, or post-test model selection was permitted. A crash after test access would not have authorized an automatic retry. The completed attempt count was one.

### 2.8 Temperature scaling

During development, a scalar temperature was fit to each model's center-1 logits by minimizing NLL. The resulting temperatures—GroupDRO T=2.974907 and ERM T=3.496293—were frozen before test access. On center 2, these values were applied only; they were never refit.

Temperature scaling divides both logits by the same positive scalar before softmax. It therefore preserves score ordering and argmax predictions while changing probability magnitudes.<sup>19</sup> We reported ECE with 15 bins, Brier score, and NLL for raw and calibrated probabilities.

### 2.9 Candidate operating points

Thresholds were selected on id_val only. For each model, four thresholds targeted specificity 0.80, 0.85, 0.90, and 0.95, and three thresholds targeted sensitivity 0.80, 0.90, and 0.95. These 14 thresholds were frozen and applied unchanged to center 1 during development and center 2 during final evaluation.

The thresholds were descriptive candidate/non-clinical operating points. No threshold was selected, optimized, or tuned on center 2, and no threshold was adopted for deployment.

### 2.10 High-confidence false-negative audit

We defined a high-confidence false negative as an observation with true label 1, predicted label 0, and predicted confidence at or above 0.90, 0.95, or 0.99. Counts were computed for raw and calibrated probabilities. Because calibration does not change predicted labels, total false negatives were expected to remain constant; the audit measured how many misses were stated with high confidence.

### 2.11 Metrics and reporting policy

The analysis reports prespecified descriptive point estimates and confusion counts for the complete locked splits. No hypothesis test, confidence interval, bootstrap resampling, threshold search, or post-test model-selection procedure was performed in the final run. The patch-level observations are not treated as independent patient-level outcomes, and no clinical utility analysis is claimed.

The primary comparative interpretation was fixed before test access: GroupDRO remained the predeclared primary and ERM the matched control regardless of the final ordering. Development and final metrics are shown separately. All unfavorable findings are retained.

## 3. Results



### 3.1 Full development evaluation

On the full id_val split, GroupDRO achieved accuracy 0.841567 and AUROC 0.932256, compared with ERM accuracy 0.798659 and AUROC 0.908538. On development ood_val center 1, GroupDRO achieved accuracy 0.784724, AUROC 0.895609, AUPRC 0.895795, sensitivity 0.640672, specificity 0.928776, and 6,271 false negatives. The matched ERM control achieved accuracy 0.767477, AUROC 0.867271, AUPRC 0.875723, sensitivity 0.607380, specificity 0.927573, and 6,852 false negatives.

Thus, the development AUROC advantage for GroupDRO was approximately 0.0283. GroupDRO also improved the worst source-validation center: on center 4, accuracy increased from 0.6901 for ERM to 0.8031 for GroupDRO, and AUROC increased from 0.8323 to 0.8980. These findings supported the predeclared selection of GroupDRO as the final primary candidate, but they remained development evidence.

Development and final discrimination are summarized in Table 2 and Figure 1.

**Table 2. Development and final model comparison**

| Stage | Split | Center | Model | n | AUROC | AUPRC | Sens. | Spec. | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Development | id_val | {0,3,4} | GroupDRO | 33,560 | 0.9323 | 0.9322 | 0.7572 | 0.9243 | 4,033 |
| Development | ood_val | {1} | GroupDRO | 34,904 | 0.8956 | 0.8958 | 0.6407 | 0.9288 | 6,271 |
| Development | id_val | {0,3,4} | Matched ERM | 33,560 | 0.9085 | 0.9131 | 0.6821 | 0.9129 | 5,280 |
| Development | ood_val | {1} | Matched ERM | 34,904 | 0.8673 | 0.8757 | 0.6074 | 0.9276 | 6,852 |
| Final held-out | ood_test | {2} | GroupDRO | 85,054 | 0.6634 | 0.6364 | 0.1106 | 0.9569 | 37,825 |
| Final held-out | ood_test | {2} | Matched ERM | 85,054 | 0.6984 | 0.6556 | 0.2411 | 0.9012 | 32,275 |

*Note.* AUROC indicates area under the receiver operating characteristic curve; AUPRC, area under the precision-recall curve; FN, false negatives. Development center 1 informed model assessment and is not final evidence.

### 3.2 Reserved-hospital final evaluation

The final center-2 dataset contained 85,054 patches, evenly divided between 42,527 tumor and 42,527 non-tumor labels. Both models produced predictions for all rows, and all required metadata and probability validations passed.

The matched ERM control exceeded the predeclared GroupDRO primary by 0.0351 AUROC, 0.0192 AUPRC, 0.0374 accuracy, 0.1305 sensitivity, and 0.1682 F1, and produced 5,550 fewer false negatives. GroupDRO retained specificity higher by 0.0556 and precision higher by approximately 0.0100. The development GroupDRO advantage did not generalize to center 2.

Final default-threshold metrics are summarized in Table 3 and Figure 2.

**Table 3. Final held-out center-2 performance at threshold 0.5**

| Model | Acc. | AUROC | AUPRC | Sens. | Spec. | Prec. | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GroupDRO | 0.5337 | 0.6634 | 0.6364 | 0.1106 | 0.9569 | 0.7194 | 0.1917 | 40,693 | 1,834 | 37,825 | 4,702 |
| Matched ERM | 0.5712 | 0.6984 | 0.6556 | 0.2411 | 0.9012 | 0.7094 | 0.3599 | 38,327 | 4,200 | 32,275 | 10,252 |

*Note.* Acc. indicates accuracy; Sens., sensitivity; Spec., specificity; Prec., precision; TN, true negatives; FP, false positives; FN, false negatives; TP, true positives. GroupDRO was the predeclared primary candidate; ERM was the matched control.

### 3.3 Development-to-test reversal

The reversal is visible in the GroupDRO-minus-ERM AUROC contrast: approximately +0.0283 on development center 1 and -0.0351 on final center 2. ERM remains the matched control rather than being relabeled as the primary after the result.

The predeclared AUROC reversal is displayed in Figure 3.

### 3.4 Held-out calibration

For GroupDRO, applying T=2.974907 reduced ECE from 0.4006439581474795 to 0.2583399203510463, Brier score from 0.41233859378344173 to 0.3096680541395694, and NLL from 1.9441627836686814 to 0.8743126064429281. For ERM, applying T=3.496293 reduced ECE from 0.3084182197890404 to 0.1489841758680231, Brier score from 0.3446598909605201 to 0.2555235818479269, and NLL from 1.4183276475805955 to 0.7130964662716416.

Calibrated ERM remained better calibrated than calibrated GroupDRO on all three metrics. Raw and calibrated AUROC, AUPRC, accuracy, sensitivity, specificity, precision, F1, and confusion counts were identical within each model. The final result therefore supports held-out improvement of these frozen temperatures on center 2, but not universal calibration validity.

Raw and calibrated reliability metrics are summarized in Table 4 and Figure 4.

**Table 4. Final held-out raw and calibrated reliability**

| Model | Variant | Temperature | ECE | Brier | NLL | Hard predictions changed |
| --- | --- | --- | --- | --- | --- | --- |
| GroupDRO | Raw | 1.000000 | 0.4006 | 0.4123 | 1.9442 | No |
| GroupDRO | Calibrated | 2.974907 | 0.2583 | 0.3097 | 0.8743 | No |
| Matched ERM | Raw | 1.000000 | 0.3084 | 0.3447 | 1.4183 | No |
| Matched ERM | Calibrated | 3.496293 | 0.1490 | 0.2555 | 0.7131 | No |

*Note.* ECE indicates expected calibration error; NLL, negative log-likelihood. Temperatures were frozen before test access and were applied without refitting.

### 3.5 Operating-point transportability

The fixed-sensitivity thresholds showed pronounced underachievement on the reserved hospital. GroupDRO thresholds targeting sensitivity 0.80, 0.90, and 0.95 achieved center-2 sensitivities 0.1430, 0.2884, and 0.4748. The corresponding ERM sensitivities were 0.3573, 0.5223, and 0.6692. None preserved its nominal sensitivity target.

Fixed-specificity targets were closer in some cases but also shifted. GroupDRO achieved specificities 0.8497, 0.8957, 0.9385, and 0.9748 for nominal targets 0.80, 0.85, 0.90, and 0.95, with sensitivities declining from 0.2981 to 0.0726. ERM achieved specificities 0.7809, 0.8334, 0.8879, and 0.9431, with sensitivities declining from 0.4282 to 0.1582.

These values are observations of transfer failure, not a basis for choosing a new threshold. All operating points remain candidate/non-clinical, and no threshold was tuned on the final hospital.

Transfer of the 14 frozen candidate/non-clinical operating points is shown in Figure 5 and detailed in Supplementary Table S3.

### 3.6 High-confidence false negatives

For GroupDRO, high-confidence false negatives declined from 29,775 raw to 5,479 calibrated at confidence >=0.90, from 25,620 to 755 at >=0.95, and from 15,485 to 1 at >=0.99. The total false-negative count remained 37,825.

For ERM, high-confidence false negatives declined from 19,142 to 1,828 at >=0.90, from 14,626 to 513 at >=0.95, and from 7,354 to 0 at >=0.99. The total false-negative count remained 32,275.

Calibration therefore corrected the confidence attached to many errors but did not recover missed tumors. This distinction is clinically important in principle but does not establish a safe triage workflow.

Counts are summarized in Table 5 and Figure 6.

**Table 5. High-confidence false negatives on final held-out center 2**

| Model | Variant | Confidence threshold | High-confidence FN | Total FN | Fraction of total FN |
| --- | --- | --- | --- | --- | --- |
| GroupDRO | Raw | 0.90 | 29,775 | 37,825 | 0.7872 |
| GroupDRO | Raw | 0.95 | 25,620 | 37,825 | 0.6773 |
| GroupDRO | Raw | 0.99 | 15,485 | 37,825 | 0.4094 |
| GroupDRO | Calibrated | 0.90 | 5,479 | 37,825 | 0.1449 |
| GroupDRO | Calibrated | 0.95 | 755 | 37,825 | 0.0200 |
| GroupDRO | Calibrated | 0.99 | 1 | 37,825 | 0.0000 |
| Matched ERM | Raw | 0.90 | 19,142 | 32,275 | 0.5931 |
| Matched ERM | Raw | 0.95 | 14,626 | 32,275 | 0.4532 |
| Matched ERM | Raw | 0.99 | 7,354 | 32,275 | 0.2279 |
| Matched ERM | Calibrated | 0.90 | 1,828 | 32,275 | 0.0566 |
| Matched ERM | Calibrated | 0.95 | 513 | 32,275 | 0.0159 |
| Matched ERM | Calibrated | 0.99 | 0 | 32,275 | 0.0000 |

*Note.* FN indicates false negatives. Calibration changes confidence magnitudes but not the total number of missed tumors.

## 4. Discussion



### 4.1 Principal findings

The matched ERM control outperformed the predeclared GroupDRO primary on the reserved hospital despite the opposite ordering on development center 1. A selective presentation could have emphasized only the development result and concluded that GroupDRO improved hospital generalization. The reserved protocol prevents that interpretation.

The result does not invalidate the development findings. GroupDRO improved center-1 discrimination and worst-center source-validation behavior under the controlled setup. Instead, the reversal shows that improvement on one OOD hospital was insufficient evidence for a different hospital. Hospital identity is not a single axis of shift, and an objective that improves robustness to source-center variation may not address unseen acquisition or tissue characteristics at every target center.

### 4.2 Implications for domain-generalization evaluation

Center 2 may differ from the source hospitals and center 1 along stain, scanner, tissue, or case-mix dimensions not represented by the center grouping.<sup>1-3</sup> Center labels may also be too coarse to capture relevant within-hospital heterogeneity. Another possibility is project-level overfitting to center 1: repeated development comparisons and calibration decisions can make a development OOD hospital functionally similar to a validation set even when it is never used for gradient updates.<sup>5,14</sup>

The study did not perform post-test image, representation, or subgroup exploration on center 2, because such analyses could encourage test-driven method development. These explanations should therefore be treated as hypotheses for a new study, not conclusions from the final test.

The strict separation between ood_val and ood_test exposed a development-to-test reversal that would have been hidden if center 1 had been reported as final performance. The one-shot design also prevented post-test calibration fitting, threshold tuning, checkpoint replacement, or model relabeling. This is a reproducibility and model-governance contribution rather than evidence that the evaluated models are clinically ready.

Medical-AI reporting and validation guidance consistently emphasizes transparent data partitioning, external validation, complete performance reporting, and clear specification of model-evaluation procedures.<sup>6,20-22</sup> Where several hospitals are available, this study illustrates the value of retaining at least one institution outside model, calibration, and operating-policy decisions until a prespecified final evaluation. That design recommendation is an interpretation of the present protocol, not a universal requirement imposed by the cited guidelines.

### 4.3 Calibration interpretation

The pre-frozen temperatures improved ECE, Brier score, and NLL on center 2 for both models. This is stronger evidence than the development calibration audit because the temperatures were not estimated from center 2. Nevertheless, the result concerns one held-out hospital and two fixed models. It does not establish that the same temperatures will generalize to other centers or that temperature scaling solves distribution-shift calibration broadly.<sup>23-24</sup>

The high-confidence false-negative analysis illustrates the value and limitation of calibration. Many errors moved below extreme-confidence thresholds, which can improve the interpretability of stated confidence. But the models still missed the same tumors. Confidence correction is not error correction, and calibration should not be described as improving sensitivity or clinical safety.

### 4.4 Operating-point instability

The failure of development-selected sensitivity targets to transfer shows that threshold policies can be less stable than aggregate ranking metrics. A threshold is tied to the score distribution, class-conditional behavior, and population on which it was selected.<sup>8-9</sup> Even when specificity targets were approximately retained, the associated sensitivity could be unacceptable. Decision-analytic usefulness would require a separately justified clinical context and consequence model, neither of which was evaluated here.<sup>25</sup>

The study therefore supports a conservative reporting policy: threshold-free metrics describe ranking, the 0.5 threshold provides a prespecified reference, and frozen candidate operating points show transport behavior. None is a validated clinical decision rule.

### 4.5 Limitations

The final evaluation contains one reserved hospital. It cannot establish robustness across all hospitals, scanners, populations, or health systems. All evidence comes from Camelyon17-WILDS, so there is no external non-Camelyon cohort.

The evaluation is patch-level. Patches from the same slide or patient may be correlated, and patch-level accuracy or calibration does not establish whole-slide, patient-level, or clinical-workflow effectiveness. Whole-slide systems require aggregation, workflow validation, and fit-for-purpose technical evaluation beyond patch classification.<sup>26-28</sup> No slide aggregation, patient-level endpoint, reader study, prospective evaluation, or clinical utility analysis was performed.

The models use ResNet-18 and short training on capped center-stratified caches rather than all source patches. The cache construction made the center-grouped comparison sound, but may omit source variation needed for stronger transfer. The results are not a performance ceiling for ERM, GroupDRO, or histopathology models.

Only GroupDRO was evaluated as a dedicated DG objective in the final controlled pair. CORAL, DANN, and other DG methods were not tested. MC-dropout and ensembles were also not evaluated, so the study does not characterize model-based epistemic uncertainty.

Development calibration used center 1 both for fitting and development-stage evaluation, which made those development calibration estimates optimistic. Applying the frozen temperatures to center 2 provides a held-out check, but only for one target hospital. Independent multi-center calibration validity remains unproven.

The final default-threshold sensitivities were low and false-negative counts high. No clinically validated threshold was selected. The candidate operating points transferred poorly and cannot be interpreted as deployment recommendations.

Finally, the one-shot protocol prevents iterative debugging or repeated estimation on center 2. This is a strength for unbiased interpretation but means that residual uncertainty cannot be reduced by rerunning the test. Future methods must be developed and evaluated in a new, separately reserved study rather than by reopening this final dataset.

## 5. Conclusion

In a controlled development comparison, GroupDRO appeared stronger than matched ERM on an OOD development hospital. That advantage reversed on a separately reserved hospital, where the matched ERM control achieved better discrimination, accuracy, sensitivity, F1, and false-negative count than the predeclared GroupDRO primary. Frozen temperature scaling improved held-out calibration metrics and sharply reduced high-confidence false negatives for both models, but did not change hard predictions or total misses. Development-selected operating targets were not reliably preserved.

The study's main implication is that development OOD gains, calibration results, and threshold policies should not be treated as final evidence for another hospital. A locked reserved-hospital protocol can reveal model-selection and reliability risks that development evaluation alone conceals. The completed results do not establish clinical readiness or universal method rankings; they support stricter multi-center validation and transparent reporting before clinical translation.<sup>22,29-30</sup>

## Data availability

The source dataset is publicly available through the CAMELYON17-WILDS benchmark and the dataset mirror identified in the Methods. Source histopathology images were not redistributed. Aggregate result tables, non-image figures, protocol documentation, and reproducibility materials are available in a public repository. The identifying repository address is withheld from the blinded manuscript and is provided separately to the editorial office.

## Code availability

Source code, configurations, guarded evaluation scripts, manuscript-generation scripts, and reproducibility documentation are available in a public repository. The identifying repository address is withheld from the blinded manuscript and is provided separately to the editorial office. Source images, trained checkpoints, raw patch-level predictions, authorization records, run sentinels, credentials, and environment-specific caches are excluded.

## Ethics approval and informed consent

Ethics approval and informed consent were not required for this study because it involved secondary analysis of a publicly available, de-identified benchmark dataset. No participants were prospectively recruited, no intervention was performed, and no identifiable private information was accessed. Responsibility for the original data collection and its associated ethical approvals remained with the original dataset creators.

## Consent for publication

Not applicable. The manuscript contains no identifiable individual-level information.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Declaration of competing interests

The authors declare that they have no known competing financial interests or personal relationships that could have influenced the work reported in this article.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During preparation of this work, the corresponding author used OpenAI ChatGPT and Codex and Anthropic Claude Code to support code drafting, workflow documentation, literature-search assistance, citation verification, content organization, and language refinement. All experimental decisions, code execution, source verification, statistical results, scientific interpretation, and manuscript revisions were reviewed and validated by the corresponding author. Both authors reviewed and approved the final manuscript and this disclosure. No generative AI or AI-assisted image-generation tool was used to create or alter scientific figures, images, data, or experimental results.

## References

1. Stacke K, Eilertsen G, Unger J, Lundström C. Measuring Domain Shift for Deep Learning in Histopathology. IEEE J Biomed Health Inform. 2021;25(2):325-336. https://doi.org/10.1109/jbhi.2020.3032060

2. Howard FM, Dolezal J, Kochanny S, et al. The Impact of Site-Specific Digital Histology Signatures on Deep Learning Model Accuracy and Bias. Nat Commun. 2021;12(1):4423. https://doi.org/10.1038/s41467-021-24698-1

3. Tellez D, Litjens G, Bándi P, et al. Quantifying the Effects of Data Augmentation and Stain Color Normalization in Convolutional Neural Networks for Computational Pathology. Med Image Anal. 2019;58:101544. https://doi.org/10.1016/j.media.2019.101544

4. Kleppe A, Skrede OJ, De Raedt S, Liestøl K, Kerr DJ, Danielsen HE. Designing Deep Learning Studies in Cancer Diagnostics. Nat Rev Cancer. 2021;21(3):199-211. https://doi.org/10.1038/s41568-020-00327-9

5. Varoquaux G, Cheplygina V. Machine Learning for Medical Imaging: Methodological Failures and Recommendations for the Future. NPJ Digit Med. 2022;5(1):48. https://doi.org/10.1038/s41746-022-00592-y

6. Steyerberg EW, Harrell FE. Prediction Models Need Appropriate Internal, Internal--External, and External Validation. J Clin Epidemiol. 2016;69:245-247. https://doi.org/10.1016/j.jclinepi.2015.04.005

7. Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: The Achilles Heel of Predictive Analytics. BMC Med. 2019;17(1):230. https://doi.org/10.1186/s12916-019-1466-7

8. Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, Steyerberg EW. A Calibration Hierarchy for Risk Models Was Defined: From Utopia to Empirical Data. J Clin Epidemiol. 2016;74:167-176. https://doi.org/10.1016/j.jclinepi.2015.12.005

9. Leeflang MMG, Rutjes AWS, Reitsma JB, Hooft L, Bossuyt PMM. Variation of a Test's Sensitivity and Specificity with Disease Prevalence. CMAJ. 2013;185(11):E537-E544. https://doi.org/10.1503/cmaj.121286

10. Goddard K, Roudsari A, Wyatt JC. Automation Bias: A Systematic Review of Frequency, Effect Mediators, and Mitigators. J Am Med Inform Assoc. 2012;19(1):121-127. https://doi.org/10.1136/amiajnl-2011-000089

11. Gaube S, Suresh H, Raue M, et al. Do as AI Say: Susceptibility in Deployment of Clinical Decision-Aids. NPJ Digit Med. 2021;4(1):31. https://doi.org/10.1038/s41746-021-00385-9

12. Kiani A, Uyumazturk B, Rajpurkar P, et al. Impact of a Deep Learning Assistant on the Histopathologic Classification of Liver Cancer. NPJ Digit Med. 2020;3(1):23. https://doi.org/10.1038/s41746-020-0232-8

13. Zhou K, Liu Z, Qiao Y, Xiang T, Loy CC. Domain Generalization: A Survey. IEEE Trans Pattern Anal Mach Intell. 2023;45(4):4396-4415. https://doi.org/10.1109/tpami.2022.3195549

14. Gulrajani I, Lopez-Paz D. In Search of Lost Domain Generalization. International Conference on Learning Representations. 2021. https://openreview.net/forum?id=lQdXeXDoWtI

15. Sagawa S, Koh PW, Hashimoto TB, Liang P. Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. International Conference on Learning Representations. 2020. https://openreview.net/forum?id=ryxGuJrFvS

16. Bandi P, et al. From Detection of Individual Metastases to Classification of Lymph Node Status at the Patient Level: The CAMELYON17 Challenge. IEEE Trans Med Imaging. 2019;38(2):550-560. https://doi.org/10.1109/tmi.2018.2867350

17. Koh PW, Sagawa S, Marklund H, et al. WILDS: A Benchmark of in-the-Wild Distribution Shifts. Proceedings of the 38th International Conference on Machine Learning. 2021. https://proceedings.mlr.press/v139/koh21a.html

18. WILDS. Camelyon17 Dataset. Official WILDS dataset page. n.d. https://wilds.stanford.edu/datasets/

19. Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. Proceedings of the 34th International Conference on Machine Learning. 2017. https://proceedings.mlr.press/v70/guo17a.html

20. Park SH, Han K. Methodologic Guide for Evaluating Clinical Performance and Effect of Artificial Intelligence Technology for Medical Diagnosis and Prediction. Radiology. 2018;286(3):800-809. https://doi.org/10.1148/radiol.2017171920

21. Tejani AS, et al. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update. Radiol Artif Intell. 2024;6(4):e240300. https://doi.org/10.1148/ryai.240300

22. Collins GS, et al. TRIPOD+AI Statement: Updated Guidance for Reporting Clinical Prediction Models That Use Regression or Machine Learning Methods. BMJ. 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378

23. Ovadia Y, Fertig E, Ren J, et al. Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. Advances in Neural Information Processing Systems. 2019. https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

24. Minderer M, Djolonga J, Romijnders R, et al. Revisiting the Calibration of Modern Neural Networks. Advances in Neural Information Processing Systems. 2021. https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html

25. Vickers AJ, Elkin EB. Decision Curve Analysis: A Novel Method for Evaluating Prediction Models. Med Decis Making. 2006;26(6):565-574. https://doi.org/10.1177/0272989x06295361

26. Campanella G, Hanna MG, Geneslaw L, et al. Clinical-Grade Computational Pathology Using Weakly Supervised Deep Learning on Whole Slide Images. Nat Med. 2019;25(8):1301-1309. https://doi.org/10.1038/s41591-019-0508-1

27. Lu MY, Williamson DFK, Chen TY, Chen RJ, Barbieri M, Mahmood F. Data-Efficient and Weakly Supervised Computational Pathology on Whole-Slide Images. Nat Biomed Eng. 2021;5(6):555-570. https://doi.org/10.1038/s41551-020-00682-w

28. Pantanowitz L, Sinard JH, Henricks WH, et al. Validating Whole Slide Imaging for Diagnostic Purposes in Pathology: Guideline from the College of American Pathologists Pathology and Laboratory Quality Center. Arch Pathol Lab Med. 2013;137(12):1710-1722. https://doi.org/10.5858/arpa.2013-0093-cp

29. Wiens J, Saria S, Sendak M, et al. Do No Harm: A Roadmap for Responsible Machine Learning for Health Care. Nat Med. 2019;25(9):1337-1340. https://doi.org/10.1038/s41591-019-0548-6

30. Kelly CJ, Karthikesalingam A, Suleyman M, Corrado G, King D. Key Challenges for Delivering Clinical Impact with Artificial Intelligence. BMC Med. 2019;17(1):195. https://doi.org/10.1186/s12916-019-1426-2
