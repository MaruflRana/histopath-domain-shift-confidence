# PROJECT PROTOCOL

**Project:** Confidence-Aware Tumor Detection Under Hospital Domain Shift in Histopathology: Calibration, Uncertainty, and Selective Prediction

This document is the **single source of truth** for how data is loaded, split, and used.
It is intentionally strict. Any deviation must be recorded here before code changes.

---

## 1. Scientific goal

Study confidence-awareness (calibration, uncertainty, selective prediction) of a
tumor / non-tumor patch classifier **under hospital domain shift**. The domain shift
is defined by the acquisition *center* (hospital) in Camelyon17-WILDS.

We separate:
- **in-distribution (ID)** performance: centers seen during training,
- **out-of-distribution (OOD)** performance: a held-out center for validation and a
  disjoint held-out center for the final test.

---

## 2. Data source (LOCKED)

- **Loader:** Hugging Face `datasets` package **only**.
- **Dataset ID:** `wltjr1007/Camelyon17-WILDS`.
- **Forbidden:** `wilds.get_dataset(..., download=True)` and any CodaLab / official
  WILDS automatic download. These are unreliable in this environment.
- Any local WILDS dataset root is **not** used by this pipeline; the original machine-local path is intentionally omitted from the public repository.

### Dataset fields (validated present)
`image`, `label`, `center`, `image_id`, `patient`, `node`, `x_coord`, `y_coord`, `slide`

- `label`: 0 = non-tumor, 1 = tumor.
- `center`: integer hospital id.
- `image`: may be **RGBA** when decoded → must be converted to **RGB** before model use.

---

## 3. Split mapping (LOCKED)

We do **not** create random splits. We derive four logical splits from three HF splits
using the `center` field.

| Logical split | HF split     | Center filter        | Expected total | Label counts (validated)          |
|---------------|--------------|----------------------|----------------|-----------------------------------|
| `train`       | `train`      | center in {0, 3, 4}  | 302,436        | tumor 151,390; non-tumor 151,046  |
| `id_val`      | `validation` | center in {0, 3, 4}  | 33,560         | —                                 |
| `ood_val`     | `validation` | center == 1          | 34,904         | —                                 |
| `ood_test`    | `test`       | center == 2          | 85,054         | non-tumor 42,527; tumor 42,527    |

Notes:
- HF `train` split contains only centers {0, 3, 4}; no center filter is strictly
  required for `train`, but the pipeline still asserts the invariant.
- HF `validation` split contains centers {0, 1, 3, 4}. It is partitioned by center into
  `id_val` (0,3,4) and `ood_val` (1). These are disjoint and cover the whole HF
  validation split.
- HF `test` split contains only center {2}.

---

## 4. Usage rules (LOCKED)

1. `train` — model training only.
2. `id_val` — in-distribution model selection / sanity monitoring.
3. `ood_val` — **all** OOD decisions: calibration fitting (temperature scaling),
   selective-prediction threshold selection, model selection under shift.
4. `ood_test` — **final reporting only**. It is NEVER used for training, calibration,
   threshold selection, or model selection. Touched once, at the very end.

Rationale: `ood_test` (center 2) must remain a truly held-out estimate of performance
under an unseen hospital domain.

---

## 5. Reproducibility

- Global seed set via `src/utils/seed.py` (Python, NumPy, PyTorch, CUDA).
- All configuration lives in `configs/`. The dataset config is
  `configs/dataset/camelyon17_hf.yaml`.
- All generated artifacts are written under `results/` (`tables/`, `figures/`, `logs/`).
- Scripts are numbered and run in order. No notebooks.

---

## 6. Milestones

- **Milestone 1 (this stage):** project foundation + dataset verification.
  Scripts `00`, `01`, `02`. No model training.
- Later milestones (NOT started yet): model training, calibration, uncertainty,
  selective prediction, reporting.

**Current status: Milestone 1. Model training is intentionally NOT implemented.**
