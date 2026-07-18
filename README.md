# Hospital Domain Shift Can Reverse Development Gains in Histopathology Tumor Detection: A Reserved-Center Study of Calibration and Operating-Point Transfer

This repository contains the reproducible, submission-aligned code and public aggregate evidence for a patch-level Camelyon17-WILDS study of hospital domain shift, confidence calibration, uncertainty, and selective prediction. The protocol used source hospitals for training, a distinct development hospital for domain-shift decisions, and one reserved hospital for a single final evaluation. It is a research artifact, not a clinical product.

## Central finding

GroupDRO outperformed its matched center-stratified ERM control on development center 1, but that advantage did not transfer to the reserved center 2: matched ERM achieved higher final AUROC, AUPRC, sensitivity, and fewer false negatives. Frozen temperature scaling improved confidence reliability for both models without changing hard predictions or total errors. Fourteen operating points selected on `id_val` transferred poorly and remain candidate, descriptive, and non-clinical.

## Locked hospital split

| Logical split | Hugging Face split | Centers | Rows | Role |
|---|---|---:|---:|---|
| `train` | `train` | 0, 3, 4 | 302,436 | Model fitting |
| `id_val` | `validation` | 0, 3, 4 | 33,560 | In-distribution selection and operating-point provenance |
| `ood_val` | `validation` | 1 | 34,904 | Domain-shift development and calibration fitting |
| `ood_test` | `test` | 2 | 85,054 | One completed reserved final evaluation |

No random split was used. Center 1 was never training data. Center 2 was not used for model development, calibration fitting, threshold selection, or post-test model selection.

## One-shot evaluation policy

The reserved center-2 evaluation was completed exactly once, with 85,054 rows per model in one dataloader traversal. It must not be rerun or replaced. `scripts/38_run_final_ood_test.py` and `scripts/39_summarize_final_ood_test.py` are retained as guarded implementation records only; do not execute them for this accepted result. Use the committed aggregate exp09 evidence for reporting.

## Repository structure

```text
configs/                 Safe dataset and experiment configuration
src/                     Data, model, training, calibration, and evaluation modules
scripts/                 Numbered research and reporting pipeline
docs/                    Protocol, interpretation, manuscript, and audit documentation
references/              Verified bibliography
results/metrics/         Portable aggregate metric JSON files
results/tables/          Aggregate, non-row-level result tables
results/figures/         Public chart figures plus the workflow SVG/PDF/PNG
submission/jpi/          Sanitized JPI manuscript and submission materials
PROJECT_PROTOCOL.md      Locked data-use protocol
REPRODUCIBILITY.md       Frozen provenance and permitted reproduction workflow
```

Datasets, caches, checkpoints, raw predictions, authorization/run-state records, certificates, local environments, private contact documents, TIFF copies, and duplicate archives are intentionally absent.

## Installation

The validated environment was Windows, Python 3.12.10, PyTorch 2.9.0+cu128, and CUDA 12.8.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

CPU or other CUDA builds may be used for independent development, but they are not the validated environment for the accepted runs.

## Data access

Data are not redistributed. The governed loader uses only the Hugging Face dataset `wltjr1007/Camelyon17-WILDS` through `datasets`; the WILDS/CodaLab automatic download is not part of this pipeline. Users must obtain access under the dataset's terms and keep local data and Hugging Face caches outside version control.

## Data availability

The source dataset is publicly available through the CAMELYON17-WILDS benchmark and the Hugging Face dataset mirror identified in the Methods. The study did not redistribute source histopathology images. Aggregate result tables, non-image figures, protocol documentation, and reproducibility materials supporting this article are publicly available at https://github.com/MaruflRana/histopath-domain-shift-confidence. Raw patch-level predictions, source images, trained checkpoints, authorization records, and local run-state artifacts are not publicly distributed.

## Code availability

Source code, configurations, guarded evaluation scripts, manuscript-generation scripts, and reproducibility documentation are publicly available at https://github.com/MaruflRana/histopath-domain-shift-confidence. The repository intentionally excludes source histopathology images, trained checkpoints, raw patch-level predictions, credentials, authorization records, run sentinels, and environment-specific caches.

## Reproduction workflow

1. Read `PROJECT_PROTOCOL.md`, `REPRODUCIBILITY.md`, and `docs/SESSION_STATE.md`.
2. Create the validated environment and configure any required extra CA certificate through `NODE_EXTRA_CA_CERTS` or `HISTOPATH_EXTRA_CA_CERTS`.
3. For new development work, follow the numbered scripts and their configs without changing the locked center mapping. Never use center 2 for development.
4. Use the committed aggregate exp09 outputs for the accepted final result. Do not rerun scripts 38 or 39.
5. Manuscript-facing tables and figures can be rebuilt from permitted aggregate inputs; JPI TIFF copies and the local code-package ZIP are generated locally by the package builder and are not versioned.

Exact commands, frozen provenance, expected public outputs, and limitations are documented in [REPRODUCIBILITY.md](./REPRODUCIBILITY.md).

## Checkpoint policy

Checkpoints are not distributed. The accepted SHA256 fingerprints are:

| Model | Role | SHA256 |
|---|---|---|
| 7F GroupDRO-by-center | Predeclared primary | `CE0DC65DC6106648F2ABB77C603746A1DB2B856F9672B01C2E8F540BEEAD8502` |
| 7F center-stratified ERM | Matched control | `8BF990BE517A41AA74D111E0A4F4111A05A9FE416919261D979F2FD65AE974FF` |

The hashes document the exact local artifacts used once; they are not download links.

## Final aggregate results

| Model | AUROC | AUPRC | Sensitivity at 0.5 | Specificity at 0.5 | Precision at 0.5 | False negatives |
|---|---:|---:|---:|---:|---:|---:|
| GroupDRO, predeclared primary | 0.663370 | 0.636413 | 0.110565 | 0.956874 | 0.719400 | 37,825 |
| Center-stratified ERM, matched control | 0.698435 | 0.655628 | 0.241070 | 0.901239 | 0.709383 | 32,275 |

The unfavorable result is explicit: the development GroupDRO advantage on center 1 did not generalize to center 2. GroupDRO retained higher specificity and slightly higher precision; this does not establish universal superiority or failure for either method.

## Calibration and operating-point caveats

Frozen temperatures were 2.974907 for GroupDRO and 3.496293 for ERM. Temperature scaling improved final ECE, Brier score, and NLL, but it was argmax-invariant and did not change total false negatives. All 14 thresholds were selected on `id_val`, applied without test-set tuning, transferred poorly, and are not clinically validated thresholds.

## Clinical scope

This work evaluates 96x96 histopathology patches on one benchmark with one reserved hospital. It does not demonstrate clinical readiness, WSI- or patient-level effectiveness, prospective utility, robustness across all hospitals, or a deployable clinical threshold.

## Manuscript and submission status

The manuscript package is prepared for the Journal of Pathology Informatics and has not yet been submitted. Journal submission is intentionally on hold; no portal action is currently authorized. Portal requirements, APC or waiver terms, file designations, and the portal-generated proof require future human verification. This repository does not perform submission.

## Citation

Software citation metadata are in [CITATION.cff](./CITATION.cff). The manuscript title is:

> Hospital Domain Shift Can Reverse Development Gains in Histopathology Tumor Detection: A Reserved-Center Study of Calibration and Operating-Point Transfer

No DOI, journal publication metadata, release date, or software version has been assigned.

## Authors and contact

- Jishan Islam Maruf — first author and principal contributor
- Ishtiak Al Mamoon — supervision, validation, and manuscript review
- Shared affiliation: Department of Computer Science and Engineering, IUBAT—International University of Business Agriculture and Technology

Corresponding author:

- Jishan Islam Maruf
- jishanislammaruf62@gmail.com

Authorship metadata was updated after the initial public release. Scientific results, protected exp09 evidence, model roles, and final interpretations were unchanged.

The manuscript title and figure numbering were updated in Milestone 9G. A deterministic, programmatically rendered methodology workflow is now Figure 1, and the six accepted scientific figures were renumbered as Figures 2-7 without changing their pixels or scientific content.

## License status

No open-source license has yet been assigned. All rights are reserved until a license is added. License selection remains a manual decision.

## AI-use disclosure summary

During preparation of this work, the corresponding author used OpenAI ChatGPT and Codex and Anthropic Claude Code to support code drafting, workflow documentation, literature-search assistance, citation verification, content organization, and language refinement. All experimental decisions, code execution, source verification, statistical results, scientific interpretation, and manuscript revisions were reviewed and validated by the corresponding author. Both authors reviewed and approved the final manuscript and this disclosure. No generative AI or AI-assisted image-generation tool was used to create or alter scientific figures, images, data, or experimental results.

The methodology workflow diagram was rendered deterministically from the locked study protocol with standard vector-drawing code; it contains no generated imagery, source images, or inferred scientific results.

## Reproducibility limitations

The repository omits restricted data, trained weights, raw row-level predictions, local caches, and private one-shot governance state. Exact accepted inference therefore cannot be independently replayed from this repository alone, and project policy forbids a second center-2 attempt. Reproducibility is supported through source code, locked split definitions, checkpoint hashes, frozen calibration and operating-point provenance, aggregate results, and manuscript-facing evidence.
