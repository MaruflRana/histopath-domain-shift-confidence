# GitHub Publication Audit

## Publication record

- Milestone: 9E
- Repository: <https://github.com/MaruflRana/histopath-domain-shift-confidence>
- Visibility: public
- Branch: `main`
- Initial public commit: `9907d6efb546c529fa5226fd0182edac17878b86`
- Initial push verification: 2026-07-18T00:46:10Z
- Remote state before the initial push: empty; no branches or tags
- GitHub authentication: authenticated as `MaruflRana` through GitHub CLI
- Force push: not used

The documentation follow-up commit containing this audit advances `main`; its final SHA is
verified after push and recorded in the Milestone 9E closeout report. This avoids embedding a
self-referential commit identifier in its own contents.

## Initial committed content

- File count: 324
- Total working-tree size of staged files: 6,527,274 bytes
- Total GitHub tree blob size: 6,524,610 bytes
- Largest committed blob: 171,631 bytes
- Files over 10 MB: 0
- Files over 25 MB: 0
- Files over 50 MB: 0
- Files over 100 MB: 0

The public tree includes reproducible code, scripts, configurations, tests, references,
documentation, aggregate metrics and tables, accepted PNG figures, manuscript source, and
sanitized submission-aligned documents.

## Intentionally excluded artifacts

- datasets, source histopathology images, Hugging Face caches, and local data caches;
- Python environments and local IDE/OS/cache/temp files;
- checkpoints, trained weights, saved models, and model-export formats;
- raw patch-level predictions, logits, and per-image/row-level outputs;
- private authorization JSON, one-shot run state, completion summaries, and sentinels;
- credentials, environment files, certificates, private keys, and local CA bundles;
- the JPI reproducibility ZIP and duplicate local code-release tree;
- redundant TIFF submission copies and other medical/source-image formats;
- identity-bearing title-page and cover-letter documents containing private telephone details.

PNG publication figures and aggregate, non-row-level result artifacts were retained. TIFF
submission copies remain locally regenerable through the package builder.

## Security, privacy, and portability audits

- Staged text files scanned: 258
- Secret/token/API-key/private-key/password/bearer/database/certificate findings: 0
- Unexpected email findings: 0; the intended author contact was allowed
- Absolute Windows path findings: 0
- Local username/home-path findings: 0
- Private authorization-phrase findings: 0
- Candidate DOCX files scanned through package XML: 8; findings: 0
- Anonymized manuscript PDF checked through text and metadata: findings: 0
- `git diff --cached --check`: passed
- Python syntax parse: 75 files passed
- JSON parse: 13 files passed
- YAML/CFF parse: 9 files passed

Two identity-bearing JPI documents containing a private telephone detail were excluded rather
than modified. No credential value or private telephone number was printed or committed.

## Large-file and remote-content audits

The local project inventory contained 32 files over 25 MB, 31 over 50 MB, and 23 over 100 MB.
They were confined to the local Python environment, result caches, and checkpoints and were all
excluded. The initial remote tree contained 324 blobs, was not truncated by the GitHub API, had
no forbidden artifact paths, and contained no blob over 10 MB.

The rendered GitHub README was verified to contain the project heading, the unfavorable
development-to-test GroupDRO reversal, the clinical-scope disclaimer, and the unlicensed/all
rights reserved notice. GitHub reported public visibility and default branch `main`.

## Scientific immutability and execution boundary

All nine protected exp09 evidence/governance files retained their pre-publication SHA256 hashes.
No accepted exp09 result, authorization record, run-state record, sentinel, checkpoint, or raw
prediction was modified. The public scientific narrative still states that GroupDRO was stronger
on development center 1 but matched ERM was stronger on reserved center 2; calibration improved
confidence reliability without reducing total errors; validation-selected operating points
transferred poorly; and no clinical-readiness conclusion is supported.

During Milestone 9E:

- no dataset was loaded and no Hugging Face split was accessed;
- no histopathology image was read;
- no model or checkpoint was loaded;
- no inference, training, fine-tuning, calibration fitting, or threshold tuning ran;
- scripts 38 and 39 were not run;
- `ood_test` was not accessed and no second attempt was created.

## GitHub metadata and remaining decisions

Description set to: “Reserved-hospital evaluation of domain shift, calibration, and
operating-point transportability in histopathology.”

Topics set: `computational-pathology`, `histopathology`, `domain-shift`,
`domain-generalization`, `model-calibration`, `groupdro`, `external-validation`, `pytorch`,
`medical-ai`, and `reproducibility`.

Visibility and homepage were not changed; the homepage remains blank. No release, DOI, GitHub
Pages site, package, checkpoint, preprint, or manuscript submission was created. The repository
remains unlicensed pending a manual license decision.

The next action remains human visual review in Microsoft Word, re-verification of current JPI
APC/waiver and portal requirements, and manual JPI submission. Model development must not be
reopened and `ood_test` must not be rerun.
