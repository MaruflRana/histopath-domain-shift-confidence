# FULL DEVELOPMENT-SCALE EVALUATION RESOURCE PLAN (Milestone 8A)

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology:
Calibration, Uncertainty, and Selective Prediction

_Milestone 8A — **planning only**. Compute/storage estimates for the later execution
milestone (8B). **No training, no inference, no calibration fitting, no `ood_test`** happens in
8A. Estimates are order-of-magnitude planning figures on the validated environment, not
measurements. `ood_test` (center 2) is **LOCKED and unused**._

---

## 1. Expected dataset sizes (from previously verified split counts)

| Logical split | HF split | Centers | Verified total | Used by 8B? |
|---|---|---|---|---|
| `train` | `train` | {0,3,4} | **302,436** | No (training split; not evaluated) |
| `id_val` | `validation` | {0,3,4} | **33,560** | Yes — full or larger in-distribution eval |
| `ood_val` | `validation` | {1} | **34,904** | Yes — full or larger OOD (center 1) eval |
| `ood_test` | `test` | {2} | **85,054** | **No — LOCKED / not used** |

8B inference volume (required models only) ≈ (`id_val` 33,560 + `ood_val` 34,904) × 2 models ≈
**~137k forward passes total** if full splits are used. Optional context-only models add
~68k per model. `train` (302,436) and `ood_test` (85,054) are **not** part of this budget.

## 2. Runtime considerations (RTX 3060, validated environment)

- Environment: Windows, Python 3.12.10, PyTorch 2.9.0+cu128, CUDA True, RTX 3060 (12 GB),
  `.venv/Scripts/python.exe`. Inputs are 96×96 RGB patches; ResNet-18 backbone.
- ResNet-18 inference at 96×96 is cheap; the RTX 3060 handles it comfortably. Throughput is
  expected to be dominated by **data access (HF streaming / decode)**, not GPU compute.
- Order-of-magnitude planning estimate: at a few thousand patches/minute (streaming-bound,
  `num_workers` modest on Windows), full `id_val` (~33.6k) is roughly single-digit minutes and
  full `ood_val` (~34.9k) similar, **per model** — call it **~10–30 min per split per model**
  as a conservative planning band. Two required models × two splits → **plausibly under ~1–2
  GPU-hours total**. This is a **planning estimate only**; 8B must record actual timings.
- If the executor reuses **already-cached** patches (e.g. re-runs at current subset sizes) the
  runtime is trivial; the cost is entirely in any **new** larger streaming.

## 3. Disk usage estimate for prediction CSVs

- Current 7F `ood_val_predictions.csv` (n=2000, 16 columns incl. metadata + logits/probs) is on
  the order of a few hundred KB. Scaling linearly:
  - Full `id_val` (~33,560 rows) ≈ **~4–7 MB** per model.
  - Full `ood_val` (~34,904 rows) ≈ **~4–7 MB** per model.
- Two required models × two splits ≈ **~20–30 MB** of prediction CSVs total. Optional
  context-only models add proportionally. Metrics/calibration/operating-point tables and the
  report are negligible (KB).
- Budget **well under ~100 MB** for the required-model outputs. Project drive D: had ~71.87 GB
  free at last check — no storage concern.

## 4. Batch size recommendations

- Use **batch_size 64** (matches 4B/5A/7F eval batch sizes) as the default; the RTX 3060 has
  ample memory for ResNet-18 at 96×96. batch_size 128–256 is safe if throughput needs a lift.
- `shuffle=False` for all evaluation dataloaders (deterministic row order, reproducible CSVs).
- `num_workers` modest on Windows (0–4); prefer correctness/stability over maximal throughput.

## 5. Cache predictions only, not images

- **Persist prediction CSVs + metrics/tables only.** Do **not** create new large image caches
  for the full splits, and do **not** rebuild or overwrite existing caches in 8B (that is a
  separate gated cache milestone, out of scope here).
- Rationale: images are large and already streamable; the reusable scientific artifact is the
  per-example prediction table (logits/probs/metadata), which is small and sufficient for all
  downstream metric/calibration/operating-point analysis (as 5B/5C/6A/7B/7C/7G already
  demonstrated on saved CSVs).
- If the executor chooses to keep a larger balanced image cache for reproducibility, it must use
  **new filenames** (no overwrite) and treat it as a distinct, separately-authorized step.

## 6. Failure / retry plan

- **HF streaming/network instability** (`Server disconnected`, `peer closed connection`) is
  expected and auto-retried by `datasets` — not a failure (per `CLAUDE.md` §6). Log transient
  retries; do not abort on them.
- **Checkpoint-in-hand, stream-out** design: load the frozen checkpoint once per model, then
  stream/evaluate. If a split evaluation fails mid-way, **re-run that split from scratch to a
  new/temp output**, then atomically finalize — never append to a partial CSV.
- **Idempotence:** each 8B run writes under a clearly named run directory (e.g.
  `results/predictions/exp08b_full_dev_eval/...`); re-running overwrites only that run's own
  outputs, never the frozen 7F artifacts.
- **Row-count guard:** after each split, assert the prediction row count equals the intended
  split/subset size before computing metrics (see `FULL_DEVELOPMENT_EVALUATION_STOP_RULES.md`).

## 7. Checkpoint loading plan

- Load `best.pt` for each required model with `map_location` to the active device; rebuild
  `CamelyonClassifier` from the checkpoint's stored `config` (backbone resnet18, fc→logits[B,2]),
  then `model.eval()` under `torch.no_grad()` (reuse `src/evaluation/predict.py`).
- **Do not** modify, re-save, or re-train any checkpoint. Verify each loaded checkpoint's stored
  `id_val` metric against its Milestone 7F record as a sanity check before evaluating (optional
  but recommended), without altering the file.

## 8. Order: `id_val` first, then `ood_val`

- **Run `id_val` first**, for both required models. `id_val` is needed to (a) sanity-check the
  reload against the stored training-time metric and (b) select the candidate operating
  thresholds. Only after `id_val` thresholds exist should `ood_val` be evaluated and the
  thresholds applied.
- This ordering also fails fast: any reload/schema/metadata problem surfaces on the
  in-distribution split before spending time on the larger OOD stream.

## 9. Warning — `ood_test` is not included

`ood_test` (center 2, n=85,054) is **excluded** from this resource plan and from Milestone 8B.
No runtime, storage, batch, or retry budget above covers `ood_test`. It remains **LOCKED** and
is reserved for the single final locked evaluation only, after a frozen final
model/calibration/threshold **and** explicit written authorization.
