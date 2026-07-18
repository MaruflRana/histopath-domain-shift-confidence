# FULL DEVELOPMENT-SCALE EVALUATION STOP RULES (Milestone 8A)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology:
Calibration, Uncertainty, and Selective Prediction

_Milestone 8A — **planning only**. Hard **STOP** conditions the later execution milestone (8B)
must enforce. If any trigger fires during 8B, the executor **halts immediately**, writes the
failure to the run log, and does **not** proceed or write further artifacts until the issue is
resolved and (where the rule requires) re-authorized. 8A performs no inference; these rules
govern 8B. `ood_test` is **LOCKED**._

---

## Stop rules (all mandatory; any single trigger halts 8B)

| # | Stop condition | Trigger / check | Action on trigger |
|---|---|---|---|
| **S-1** | **Any `ood_test` path appears** | Any file path, split value, center filter, or dataloader references the HF `test` split or center 2 / `ood_test`. | HALT immediately. `ood_test` is LOCKED; never read in 8B. Requires separate written authorization to ever touch — not in scope here. |
| **S-2** | **Split mapping differs from locked mapping** | Effective mapping is not exactly `id_val`=validation∩{0,3,4}, `ood_val`=validation∩{1}, `train`=train∩{0,3,4}, `ood_test`=test∩{2}; or a random/invented split is used. | HALT. Do not evaluate on an unlocked mapping. Fix the split derivation before re-running. |
| **S-3** | **Checkpoint differs from frozen paths** | Required-model checkpoint path is not exactly `results/checkpoints/exp07f_groupdro_resnet18/best.pt` (primary) or `results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt` (matched control); or a checkpoint file's size/hash changed vs its Milestone 7F record. | HALT. Do not evaluate an unfrozen/altered checkpoint. Restore/confirm the frozen checkpoint first. |
| **S-4** | **Prediction row count mismatch** | Rows written for a split ≠ the intended full/subset size for that split (e.g. `ood_val` rows ≠ evaluated center-1 count), or a partial/appended CSV is detected. | HALT. Re-run that split cleanly to a fresh output; never compute metrics on a truncated set; disclose any intended capping explicitly (no silent truncation). |
| **S-5** | **Invalid probabilities** | Any `prob_0`/`prob_1` is NaN/Inf, outside [0,1], or `prob_0+prob_1` deviates from 1 beyond 1e-5. | HALT. Investigate the inference/softmax path; do not report metrics from invalid probabilities. |
| **S-6** | **Non-binary labels** | Any `label` value ∉ {0,1}, or a split unexpectedly single-class. | HALT. The task is binary tumor/non-tumor; a non-binary or degenerate label set indicates a data/loader fault. |
| **S-7** | **Missing metadata fields** | Any of `center, patient, slide, node, image_id, x_coord, y_coord` (or any required prediction column incl. `confidence`) is absent or null in the output. | HALT. Group/per-center analysis and provenance require full metadata; do not drop columns. |
| **S-8** | **Calibration accidentally fit on `ood_test`** | Any temperature-scaling fit reads `ood_test`/center-2 logits, or a calibration input path resolves to the test split. | HALT. Calibration is development-only, fit on `ood_val` (or reused 7G temperatures) — **never** on `ood_test`. |
| **S-9** | **Would overwrite existing development artifacts without a new run name** | An output path would overwrite any existing checkpoint, prediction CSV, metrics JSON, cache, or table from Milestones 4A–7I (i.e. not under a fresh `exp08b_*` / new run directory). | HALT. Use new run names only (`exp08b_full_dev_eval/...`); never overwrite frozen inputs or prior-milestone artifacts. |

## Additional guardrails (halt-worthy, consistent with the milestone scope)

- **No training / weight updates.** If any optimizer step, `loss.backward()`, or checkpoint
  re-save occurs, HALT — 8B is inference-only.
- **No MC-dropout / ensemble / CORAL / DANN.** These remain deferred/optional and out of 8B
  scope; if invoked, HALT.
- **No committed clinical threshold.** Operating points are candidate-only; committing/deploying
  a threshold is a separate gated step. If an output labels a threshold "final"/"committed", HALT.
- **No final-performance claim.** All outputs are development-stage; if any artifact asserts final
  or clinical performance, HALT and correct the wording.

## Precedence

These stop rules **override** throughput, convenience, or completeness. When a stop rule
conflicts with "finish the evaluation", the stop rule wins: halt, log, and escalate. Resolution
of **S-1** or **S-8** (any `ood_test` involvement) additionally requires explicit written
authorization before any further test-adjacent action — which is out of scope for both 8A and 8B.
