# FINAL-TEST STOP RULES (Milestone 8D)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology.

> Hard **STOP** conditions the later authorized final-test milestone must enforce. If any trigger
> fires, the executor **halts immediately**, writes the failure to the run log, and does **not**
> proceed. These rules are specified in Milestone 8D (planning); 8D itself reads no `ood_test`.

---

| # | Stop condition | Action on trigger |
|---|----------------|-------------------|
| **FT-1** | **Checkpoint path differs** from the frozen `results/checkpoints/exp07f_groupdro_resnet18/best.pt` (primary) or `results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt` (control), or a checkpoint hash changed vs its 7F/8B record. | HALT. Do not evaluate an unfrozen/altered checkpoint. |
| **FT-2** | **`ood_test` row count ≠ locked expected 85054**, or a partial/appended/truncated set is detected. | HALT. Never compute metrics on a wrong-sized test set; no silent truncation. |
| **FT-3** | **Center is not exactly {2}** for `ood_test` (any other/extra center present). | HALT. The locked test is center 2 only. |
| **FT-4** | **Any training / weight update is attempted** (optimizer step, `loss.backward()`, checkpoint re-save). | HALT. The final test is inference-only; weights are frozen. |
| **FT-5** | **Calibration fitting is attempted on `ood_test`** (temperature fit reads test/center-2 logits). | HALT. Use pre-frozen 8C temperatures only (GroupDRO T=2.974907, ERM T=3.496293). |
| **FT-6** | **Any threshold is selected using `ood_test`** (operating point tuned on the test set). | HALT. Thresholds are selected on `id_val` only, before the test is read. |
| **FT-7** | **Output path would overwrite development outputs** (any 4A–8D checkpoint/prediction/metrics/table/figure), i.e. not under a fresh final-test run name. | HALT. Use a new run name (e.g. `exp09_*`) only. |
| **FT-8** | **Labels / probabilities / metadata invalid** — `label ∉ {0,1}`; `prob` NaN/Inf/out of [0,1]; `prob_0+prob_1` off by >1e-5; any required metadata column missing/null. | HALT. Investigate the loader/softmax path. |
| **FT-9** | **More than one final-test run is attempted** without an explicit written override. | HALT. `ood_test` is a single one-time evaluation. |

## Additional guardrails

- **No MC-dropout / ensemble / CORAL / DANN** in the final test unless separately authorized.
- **No post-test model selection** — no switching model / epoch / checkpoint / threshold after
  reading `ood_test`.
- **No committed clinical threshold** — final-test operating points are candidate/context unless a
  separate threshold-commit step is authorized.
- **No final claim** until the authorized run completes; development-scope tags remain on all
  pre-test artifacts.

## Precedence

These stop rules **override** throughput or completeness. Any `ood_test`-adjacent action
additionally requires the completed `docs/OOD_TEST_AUTHORIZATION_TEMPLATE.md`. Milestone 8D does
not authorize the test.
