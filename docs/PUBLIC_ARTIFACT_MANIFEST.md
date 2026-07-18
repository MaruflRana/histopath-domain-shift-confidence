# Public artifact manifest

This classification governs the GitHub publication. It does not delete or alter local scientific evidence.

## Public and committed

- Root protocol, publication, contribution, security, citation, and reproducibility documentation.
- Public two-author metadata, confirmation checklist, CRediT roles, and availability statements.
- `src/`, the governed numbered `scripts/` pipeline, safe configs, verified references, and manuscript/source documentation.
- Portable aggregate metric JSON files and aggregate, non-row-level CSV tables.
- Accepted chart PNGs, including final exp09 and manuscript-facing figures.
- The anonymized JPI manuscript, PDF proof, highlights, declarations, captions, tables, supplement, CLAIM checklist, manifests, and audits that contain no private telephone detail.

## Public but regenerated

- JPI TIFF submission copies are generated locally from accepted PNG/source figures by the package builder.
- The local JPI reproducibility ZIP is generated from the public code and manifests when needed; the ZIP itself is not versioned.
- Machine-local logs can be regenerated during a new, separately governed development workflow and are not public evidence for the accepted run.

## Excluded because oversized

- All trained checkpoints and weights, including the approximately 128 MiB ResNet-18 checkpoint files.
- The local Python environment and its CUDA/PyTorch binaries.

## Excluded because dataset-restricted

- Dataset files, patches, medical/source images, Hugging Face downloads and caches, and serialized balanced subset caches.
- `results/figures/sample_patch_grid.png` and `results/figures/exp07e0_stain_space_aug_preview.png`, because they contain dataset-derived patch imagery rather than aggregate charts.

## Excluded because sensitive or governance-bound

- The final-test authorization JSON, frozen private-governance config, run-state JSON, completion sentinels, and pre-authorization template/checklist files.
- Local CA certificates, environment files, credentials, IDE/agent state, and temporary files.
- `JPI_Title_Page.docx` and `JPI_Cover_Letter.docx`, because identity-bearing editorial files remain local under repository policy. Their Milestone 9F copies contain the finalized two-author metadata and no telephone, room, or extension detail.
- Historical aggregate JSON files that embed machine-local absolute provenance paths; equivalent scientific values remain available in portable aggregate tables and documentation.

## Excluded because redundant or non-portable

- `submission/jpi/code_release/` and `JPI_Reproducibility_Code_Package.zip`, which duplicate the repository's sanitized code.
- TIFF figure copies, local logs, temporary render/build directories, and obsolete root-level data-inspection helpers outside the governed numbered pipeline.

## Scientific evidence policy

The committed exp09 artifacts are aggregate only. Raw per-patch predictions, logits, image or patient metadata, authorization state, and sentinels are not public. Checkpoint identity is represented by SHA256 hashes in `README.md` and `REPRODUCIBILITY.md`.

## Milestone 9F authorship and anonymity policy

- Public metadata identifies Jishan Islam Maruf first and Ishtiak Al Mamoon second.
- Jishan Islam Maruf remains corresponding author and principal contributor.
- The anonymized manuscript, supplement, CLAIM checklist, and reviewer ZIP contain no author,
  affiliation, or identifying repository address.
- The existing reviewer ZIP passed its identity/repository/security scan and was not rebuilt.
- Scientific result tables, figures, exp09 evidence, model roles, and final interpretation were
  unchanged.
