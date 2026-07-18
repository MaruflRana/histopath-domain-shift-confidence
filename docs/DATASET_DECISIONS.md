# DATASET DECISIONS

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in
Histopathology: Calibration, Uncertainty, and Selective Prediction

The rationale behind how data is sourced, split, and loaded. Read this before changing
anything in `src/data/` or `configs/dataset/`. `PROJECT_PROTOCOL.md` is the binding
source of truth; this document explains the *why*.

---

## 1. Source: Hugging Face only

- **Decision:** load exclusively from Hugging Face `wltjr1007/Camelyon17-WILDS` via the
  `datasets` package (streaming by default).
- **Why:** the official WILDS / CodaLab automatic download
  (`wilds.get_dataset(..., download=True)`) is **unreliable in this environment**
  (broken/timing-out downloads). The HF mirror is stable and streamable.
- **Consequence:** `wilds` / CodaLab download paths are **forbidden**. A full local
  dataset copy is not required; streaming suffices, and shards are cached by HF on
  demand under the user's configured Hugging Face cache directory.

## 2. Locked split mapping (no random splits)

We derive four **logical** splits from three HF splits using the `center` field.

| Logical split | HF split     | Center filter | Expected total |
|---------------|--------------|---------------|----------------|
| `train`       | `train`      | {0, 3, 4}     | 302,436        |
| `id_val`      | `validation` | {0, 3, 4}     | 33,560         |
| `ood_val`     | `validation` | {1}           | 34,904         |
| `ood_test`    | `test`       | {2}           | 85,054         |

- **Why center-based:** the scientific question is confidence-awareness **under hospital
  (center) domain shift**. ID = centers seen in training; OOD = held-out centers.
  `ood_val` (center 1) drives all OOD decisions; `ood_test` (center 2) is a truly
  held-out final estimate.
- **Why no random splits:** random splits would leak domain (center) information across
  train/val/test and destroy the domain-shift design.
- Encoded in `configs/dataset/camelyon17_hf.yaml`; enforced in
  `src/data/hf_camelyon17.py`.

## 3. Usage rules for the splits

- `train` — training only.
- `id_val` — in-distribution model selection / sanity.
- `ood_val` — OOD decisions only (temperature scaling, selective-prediction thresholds,
  model selection under shift). **Not** authorized for use yet.
- `ood_test` — final locked evaluation only; touched once at the very end (or an
  explicit, requested sanity check). Never for training/calibration/selection.

## 4. RGBA → RGB

- HF may decode patches as **RGBA**. Every image is converted to **RGB** before model
  use (`to_rgb` in `hf_camelyon17.py`; also defensively in `transforms.py`).
- Patches are natively **96×96**; `transforms.py` asserts this size so a wrong-sized
  patch fails loudly. Pipeline: RGB PIL → float32 [0,1] CHW → ImageNet mean/std norm.

## 5. TLS interception (Norton) — solved

- **Problem:** Norton MITMs HTTPS with its own root CA; Python's `certifi` bundle does
  not trust it → `CERTIFICATE_VERIFY_FAILED` against huggingface.co.
- **Fix (no verification disabled):** `src/utils/ssl_setup.py` builds a **combined CA
  bundle** (certifi + Norton/corporate CA from `NODE_EXTRA_CA_CERTS` /
  `HISTOPATH_EXTRA_CA_CERTS`) into
  `configs/certs/combined_ca_bundle.pem`, and exports `SSL_CERT_FILE`,
  `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`.
- **Ordering:** must run **before** `datasets` / `huggingface_hub` import. It is invoked
  from `scripts/_bootstrap.py`, which every script imports first.

## 6. The HF stream is label-ordered (key finding)

Quantified by `scripts/07_label_order_diagnostic.py`:

- **train:** label-1 (tumor) records stream first; **label-0 (non-tumor) not seen in
  the first 5,000**. The training build revealed the label-0 block begins at scan
  **~128,182** (balance reached at `scanned=128,310`).
- **id_val:** first label-0 at index **14,033**.
- **Both splits:** the first 1,000 samples are **single-class** (a naive early batch is
  label-skewed — exactly the failure seen in Milestone 3A, where the first `train`
  batch was all label 1).

**Implication:** you cannot rely on stream order for balanced batches. Do not build
training batches by taking the first N streamed samples.

## 7. Balanced subset cache (required)

- **Decision (user-approved in Milestone 3C):** build a **class-balanced** subset once
  by scanning far enough to reach both classes, then **cache it to disk**; reuse it
  instantly on later runs.
- **Why this over alternatives:**
  - *Raise `max_scan` and re-scan every run* — rejected: re-streams ~128k train records
    (multi-GB, minutes) on every run; violates "controlled runtime" / "no duplicate
    downloads".
  - *Non-streaming full materialization* — rejected for now: downloads all shards.
  - *Cache to disk* — chosen: one heavy scan ever, then seconds-long cache HITs.
- **Efficiency:** collection uses **lazy decode** — labels are read with
  `decode_images=False`, and only the kept (`2 × per_class`) images are decoded, not
  the ~128k scanned-and-skipped ones.
- **Implementation:** `src/data/balanced_subset.py::load_or_build_balanced_examples`.
  Serializes examples as uint8 `[H,W,3]` arrays + metadata via `torch.save`; reloads as
  RGB PIL. Cache filename encodes `(split, per_class, seed, shuf|noshuf)`, so changing
  those regenerates a distinct cache. A payload check re-validates class balance on
  load.
- **Current cache files:**
  - `results/cache/balanced_train_pc128_seed1337_shuf.pt` (~12.3 MB, `{0:128,1:128}`)
  - `results/cache/balanced_id_val_pc64_seed1337_shuf.pt` (~6.1 MB, `{0:64,1:64}`)
- **Note on scale-up:** larger `per_class` for `train` requires scanning at least to the
  label-0 block (~128k) once; budget `max_scan_train` accordingly (currently 200,000).

## 8. Transient network errors

`datasets` auto-retries `Server disconnected` / `peer closed connection` on parquet
shards. These are expected under streaming and are not pipeline failures.
