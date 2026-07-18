# RUNBOOK

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in
Histopathology: Calibration, Uncertainty, and Selective Prediction

How to run every script in the pipeline, what it needs, what it produces, and how long
it takes. All commands run from the **project root** with the venv interpreter.

---

## 0. Prerequisites

- Interpreter: `.venv/Scripts/python.exe` (Python 3.12.10).
- No manual activation needed — call the interpreter path directly.
- Network: HF streaming requires internet. Transient `Server disconnected` /
  `peer closed connection` warnings are auto-retried and are **not** failures.
- SSL: handled automatically. Every script does `import _bootstrap` first, which runs
  `src/utils/ssl_setup.py` (Norton CA fix) **before** any `datasets` import. Do not
  reorder imports around this.
- The benign `Warning: You are sending unauthenticated requests to the HF Hub` can be
  ignored (or filtered with `grep -vE "unauthenticated|HF_TOKEN"`).

General form:
```
.venv/Scripts/python.exe scripts/<NN_name>.py
```

---

## 1. Script reference

| # | Script | Purpose | Reads | Writes | Notes |
|---|--------|---------|-------|--------|-------|
| 00 | `00_check_env.py` | Environment report | — | `results/logs/env_check.txt` | Fast. |
| 01 | `01_verify_hf_dataset.py` | Verify HF splits/fields/centers vs locked mapping | HF (stream) | `results/tables/hf_split_verification.csv` | Streaming scan. |
| 02 | `02_dataset_sanity.py` | Counts + label/center breakdown + sample grid | HF (stream) | `results/tables/dataset_*_*.csv`, `results/figures/sample_patch_grid.png` | Streaming scan. |
| 03 | `03_dataloader_smoke_test.py` | DataLoader tensor/shape/dtype check, all splits | HF (stream, 128/split) | `results/logs/dataloader_smoke_test.txt` | max_samples=128, bs=16. |
| 04 | `04_model_forward_smoke_test.py` | One forward pass per split → logits `[B,2]` | HF (stream, 32/split) | `results/logs/model_forward_smoke_test.txt` | GPU. |
| 05 | `05_train_mechanics_smoke_test.py` | 20 optimizer steps on one batch | HF (stream, 64) | `results/logs/train_mechanics_smoke_test.txt` | Mechanics only. |
| 06 | `06_train_short_run.py` | Short balanced train + id_val eval (3 epochs) | **cache** or HF | `results/logs/short_train_resnet18_{report.txt,log.csv}`, `results/cache/*.pt` | See §2. |
| 07 | `07_label_order_diagnostic.py` | Quantify label ordering (labels only) | HF (stream) | `results/tables/label_order_diagnostic.csv`, `results/logs/label_order_diagnostic.txt` | train max_scan 5k, id_val 20k. |
| 08 | `08_hf_cache_status.py` | HF cache + disk status (read-only) | filesystem/env | `results/logs/hf_cache_status.txt` | No download. Fast. |

---

## 2. Script 06 — first run vs cached run (important)

Script 06 builds **balanced** subsets, which is expensive **the first time** because the
HF stream is label-ordered (train's label-0 block starts ~128k in).

- **First run (cache miss):** streams ~128k train records + ~14k id_val records to
  reach class balance, then writes:
  - `results/cache/balanced_train_pc128_seed1337_shuf.pt`
  - `results/cache/balanced_id_val_pc64_seed1337_shuf.pt`
  This is download-bound and takes several minutes. **Run it in the background** and
  monitor `results/logs/` progress (progress prints every 5,000 scanned).
- **Subsequent runs (cache HIT):** load the `.pt` files instantly — no HF streaming,
  completes in seconds, deterministic metrics.

Config: `configs/experiments/exp03b_short_train_resnet18.yaml`
(`max_scan_train: 200000`, `max_scan_id_val: 30000`, `shuffle_collected: true`,
`use_cache: true`).

**To force a rebuild:** delete the relevant `results/cache/*.pt` file (or call
`load_or_build_balanced_examples(..., force_rebuild=True)`). Changing `per_class` or
`seed` produces a new cache filename automatically.

---

## 3. Milestone-1 reproduction (only if needed)

Milestone 1 is complete and accepted; **do not re-run unless necessary**.
```
.venv/Scripts/python.exe scripts/00_check_env.py
.venv/Scripts/python.exe scripts/01_verify_hf_dataset.py
.venv/Scripts/python.exe scripts/02_dataset_sanity.py
```

## 4. Fastest way to confirm the current pipeline is healthy

With caches already present, this is the quickest end-to-end signal (seconds):
```
.venv/Scripts/python.exe scripts/06_train_short_run.py
```
Expect: `cache HIT` for both subsets, 3 epochs, `OVERALL: PASS`.

## 5. Windows / shell notes

- Paths are Windows-native; scripts resolve paths from the project root via
  `src/utils/io.py`, so they are CWD-robust, but running from the root is recommended.
- Long-running collection should use a background run + a log tail rather than a
  blocking foreground call.
- Do not run long scripts when the task says not to (e.g., handoff-only sessions).
