# JPI Title and Methodology Figure Update Audit

**Milestone:** 9G
**Journal:** Journal of Pathology Informatics
**Verdict:** **PASS**
**Final submission status:** **ON HOLD — READY FOR FUTURE HUMAN REVIEW**

## Title revision

- Historical prior title: “When Development Gains Do Not Transfer: Confidence-Aware Tumor Detection Under Reserved-Hospital Shift.”
- Current binding title: “Hospital Domain Shift Can Reverse Development Gains in Histopathology Tumor Detection: A Reserved-Center Study of Calibration and Operating-Point Transfer.”
- Reason: the current title leads with hospital domain shift, uses the cautious formulation “can
  reverse,” identifies tumor detection and the reserved-center design, and retains calibration and
  operating-point transfer without making a universal model or clinical-readiness claim.
- Active submission files contain the current title. The prior title remains only in this
  explicitly historical audit field and the package builder's `OLD_TITLE` audit constant.

## Deterministic methodology diagram

Figure 1 is titled “Locked multi-hospital development and reserved-center evaluation workflow.”
It shows the locked center roles; center-stratified 1,800/450-patch caches; matched ERM and
GroupDRO branches; full `id_val`/center-1 development evaluation; the pre-center-2 freeze gate;
one center-2 dataset instance and dataloader traversal for 85,054 patches; attempt count 1; and
the four discrimination, calibration, operating-point, and high-confidence-false-negative output
categories. Its footer forbids center-2 refitting, retuning, model switching, and repeat
evaluation.

- Renderer: `scripts/44_make_methodology_workflow_figure.py`, matplotlib patches/arrows only.
- Source/output SVG SHA256: `26C194376CDAF2157B9DDD690BEB2B4E924AA0A3D18866543C31092D28C3B674`.
- Vector PDF SHA256: `1E2A8906AE0E7D1D13AF302FABA376CDDCBED45438636332BF77C0DC42255ED0`.
- 600-dpi review PNG SHA256: `2BD0A9000F76DFF262034273B987385EA37BAC069A0BA72C64790155C6316587`.
- 300-dpi submission PNG SHA256: `34D5EAB8AF6ABF7039726BBA85E295C16EEF522140E1A11B24DA62E22FA03CE5`.
- No generative image model, stock image, external icon, synthetic data, histopathology image, or
  screenshot was used. Content came entirely from locked protocol records; no result was inferred
  or altered.

## Figure renumbering and unchanged-content proof

| Old | New | Accepted source | Old submission PNG SHA256 | New submission PNG SHA256 | Dimensions | Status |
|---:|---:|---|---|---|---|---|
| — | 1 | `figure_methodology_workflow.png` | — | `34D5EAB8AF6ABF7039726BBA85E295C16EEF522140E1A11B24DA62E22FA03CE5` | 3754×2542, 300 dpi, RGB | New deterministic schematic; visual QA PASS |
| 1 | 2 | `exp09b_development_to_final_auroc_auprc.png` | `46DC5FC4386BA87E5EDC5FD37BC96AA574D3D726C9BABADE88809CDE1CBD6DF3` | same | 2289×1075, 300 dpi, RGB | Pixel-identical |
| 2 | 3 | `exp09b_final_default_threshold_metrics.png` | `E02BCC98B7F733419E02DB0AF31202C27CAC8646A13CD776589A9D3F117567FD` | same | 2179×1133, 300 dpi, RGB | Pixel-identical |
| 3 | 4 | `exp09b_development_test_reversal.png` | `D5B53BDF762618315987A4CB9709354902C9DC8431EB4A87F916E1959D769D89` | same | 2043×1544, 300 dpi, RGB | Pixel-identical |
| 4 | 5 | `exp09b_final_calibration_raw_vs_calibrated.png` | `F4FA7FE8F21C51BBA1B5B54CD1B5B9C437EB8B2FE8B84BA48DFB57A2A939B2D6` | same | 2839×1074, 300 dpi, RGB | Pixel-identical |
| 5 | 6 | `exp09b_operating_point_transfer.png` | `20561962EE5B4B3F4AEEE0CF51180995E400A26FCB25902F89575F5A2011A054` | same | 2505×1117, 300 dpi, RGB | Pixel-identical |
| 6 | 7 | `exp09b_high_confidence_fn_raw_vs_calibrated.png` | `0C3644191ED920BA1202F80CF54AF506EE382BE59CD61DFC21EE1B6AEC5A7479` | same | 2508×1117, 300 dpi, RGB | Pixel-identical |

The full old/new paths, source hashes, pixel hashes, TIFF hashes, PNG hashes, dimensions, dpi,
color modes, unchanged-content flags, and visual-QA results are in
`submission/jpi/JPI_Figure_Manifest.csv`. Accepted `results/figures/exp09b_*` sources were not
overwritten.

## Manuscript and package updates

- Section 2.1 now cites Figure 1 after the locked split description; the diagram and caption
  follow Table 1 in the editable manuscript and PDF proof.
- Prior result citations shifted sequentially to Figures 2-7. All seven figures are cited.
- The title page reports seven figures. Captions, CLAIM item 39, supplement inventory, checklist,
  upload map, manifest, public sources, README/CITATION metadata, builders, audits, and project
  context were synchronized.
- The anonymized proof contains 12 pages. Every page and all affected DOCX renders were inspected;
  the title, diagram, arrows, tables, and references are unclipped and readable.
- Citation ordering and all scientific numeric tokens remained unchanged.

## Anonymization and reviewer-package result

- Blinded DOCX/PDF/Markdown, captions, tables, supplement, and CLAIM checklist: PASS.
- Names, emails, IUBAT, `MaruflRana`, exact repository URL, telephone/room/extension details,
  local paths, creator identity, comments, tracked changes, hidden text, and custom properties:
  absent from blinded materials.
- Workflow SVG/PDF/PNG metadata: PASS; no identity or local path, and PNG Author is blank.
- Reviewer ZIP: PASS, 91 entries, SHA256
  `27FF9ED54AA4C4A1898216B8E098D6716DB0709FC81D6845E835FA6ABDE1ED2C`;
  no identity, repository address, old title, credential, certificate, governance file, sentinel,
  raw prediction, or local path was found. Rebuild was not required.

## Scientific integrity and protected evidence

- Protected exp09 files: 9/9 SHA256 values unchanged.
- Result-table tree: 99 files, 256,850 bytes, digest
  `17AF8514119F666F701CD7245ED9C7594A381D0687647BFB89A7A77F266EAF77` before and after.
- Pre-existing scientific-figure tree, excluding the three new workflow outputs: 52 files,
  8,210,666 bytes, digest
  `A141388875BE48C6B64465A98EAEE767F07491E3674BB803E1407781EF608EAB` before and after.
- GroupDRO remains the predeclared primary candidate; ERM remains the matched control. The
  development-to-test reversal remains explicit. Calibration remains confidence correction, not
  classification correction; total false negatives are unchanged; operating points remain
  candidate/non-clinical.
- No dataset or Hugging Face split was loaded; no histopathology image or checkpoint was accessed;
  no inference, training, calibration fitting, threshold tuning, test-driven analysis, or second
  `ood_test` attempt occurred. Scripts 38 and 39 were not run.

## Repository result and files changed

Safe public changes comprise the title/metadata sources, deterministic workflow source and
SVG/PDF/PNG, public/anonymized manuscript package, renumbered PNG figure copies, manifests,
builders, audits, and project context. TIFFs, title page, cover letter, reviewer ZIP, datasets,
checkpoints, raw predictions, authorization/run-state/sentinel records, caches, and private
correspondence remain excluded from GitHub. Staged security, privacy, path, and size checks passed;
the safe update was pushed without force, and local/remote SHA equality is recorded in the final
Milestone 9G handoff.

## Disposition

**ON HOLD — READY FOR FUTURE HUMAN REVIEW**

Nothing was submitted automatically. Journal submission is intentionally on hold. No portal
action is currently authorized, model development remains closed, and `ood_test` remains
permanently closed.
