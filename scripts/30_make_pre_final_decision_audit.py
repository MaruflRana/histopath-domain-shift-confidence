"""Milestone 7I / Script 30: pre-final-evaluation decision audit package.

**Audit / documentation only. NO training. NO model inference. NO calibration fitting.
NO MC-dropout. NO ensemble. NO CORAL/DANN. NO ood_test. NO final-performance claim.
NO committed clinical threshold.**

This script produces the machine-readable half of the Milestone 7I pre-final-evaluation
decision audit: three decision/readiness/forbidden-claim tables and a text report. It reads
the *already-saved* Milestone 7H consolidated DG artifacts read-only (fingerprint-verified
unmodified) purely to pull a few provenance numbers into the tables; every other value is a
copied/derived decision record. It performs no computation on images, logits, or model
forward passes.

The three narrative docs (docs/PRE_FINAL_EVALUATION_AUDIT.md, docs/FROZEN_DECISIONS.md,
docs/NEXT_EXPERIMENT_DECISION_TREE.md) and the four context docs (CLAUDE.md,
docs/SESSION_STATE.md, docs/MILESTONE_LOG.md, docs/NEXT_ACTIONS.md) are hand-authored /
hand-updated this milestone; this script only VERIFIES they exist and carry a 7I marker.

Interpretation guardrails (baked into every artifact):
  * Primary controlled candidate = 7F GroupDRO + temperature-scaled confidence.
  * Matched control = 7F center-stratified ERM.
  * Context-only baselines = old plain ERM, 7A, 7D, 7E (old confounded cache / different
    id_val subset) — NEVER controlled comparators for GroupDRO.
  * GroupDRO is retained because it beat its MATCHED ERM control; it has NO controlled claim
    against old plain ERM.
  * All numbers are DEVELOPMENT-stage on capped balanced subsets — NOT final performance.
  * Calibration is development-stage, fit AND evaluated on the same ood_val subset.
  * `ood_val` is development-only; `ood_test` is LOCKED and is NEVER read.

Outputs:
  Tables (results/tables/exp07i_*.csv):
    1. exp07i_pre_final_decision_matrix.csv
    2. exp07i_model_readiness_table.csv
    3. exp07i_forbidden_claims_before_test.csv
  Report (results/logs/):
    exp07i_pre_final_evaluation_audit_report.txt
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup; kept for repo convention)

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
DOCS = ROOT / "docs"

# --------------------------------------------------------------------------------------
# Inputs (read-only). Parsed for a few provenance numbers; fingerprinted to guarantee we
# never mutate any saved result.
# --------------------------------------------------------------------------------------
PARSED_INPUTS = [
    TABLES / "exp07h_consolidated_model_comparison.csv",
    TABLES / "exp07h_controlled_dg_comparison.csv",
]
CONTEXT_INPUTS = [
    TABLES / "exp07h_context_vs_controlled_warning_table.csv",
    TABLES / "exp07h_groupdro_calibration_summary.csv",
    TABLES / "exp07h_groupdro_operating_point_summary.csv",
    TABLES / "exp07h_reviewer_risk_update.csv",
    LOGS / "exp07h_dg_reporting_package_summary.txt",
]

# Narrative docs hand-authored this milestone (script VERIFIES existence + 7I marker).
NEW_DOCS = [
    DOCS / "PRE_FINAL_EVALUATION_AUDIT.md",
    DOCS / "FROZEN_DECISIONS.md",
    DOCS / "NEXT_EXPERIMENT_DECISION_TREE.md",
]
# Context docs hand-updated this milestone (verified after the edits are applied).
CONTEXT_DOCS = [
    ROOT / "CLAUDE.md",
    DOCS / "SESSION_STATE.md",
    DOCS / "MILESTONE_LOG.md",
    DOCS / "NEXT_ACTIONS.md",
]
MARKER = "7I"

WARNINGS: list[str] = []
ERRORS: list[str] = []


def _fingerprint(paths):
    fp = {}
    for p in paths:
        if p.exists():
            st = p.stat()
            fp[str(p)] = (st.st_size, st.st_mtime_ns)
    return fp


def _write_csv(path: Path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main():
    for d in (TABLES, LOGS, DOCS):
        d.mkdir(parents=True, exist_ok=True)

    # ---- guardrail: ood_test must never appear in any input/output path ----
    all_paths = PARSED_INPUTS + CONTEXT_INPUTS + NEW_DOCS + CONTEXT_DOCS
    for p in all_paths:
        assert "ood_test" not in str(p).lower(), f"ood_test referenced in path: {p}"

    # ---- required inputs must exist ----
    missing = [str(p) for p in PARSED_INPUTS if not p.exists()]
    assert not missing, f"required 7H inputs missing: {missing}"

    # ---- read-only fingerprint (before) ----
    fp_before = _fingerprint(PARSED_INPUTS + CONTEXT_INPUTS)

    # ==================================================================================
    # LOAD (provenance numbers only)
    # ==================================================================================
    consolidated = pd.read_csv(TABLES / "exp07h_consolidated_model_comparison.csv")
    controlled = pd.read_csv(TABLES / "exp07h_controlled_dg_comparison.csv")

    def auroc_of(model_key):
        row = consolidated.loc[consolidated["model"] == model_key]
        return float(row["ood_val_auroc"].iloc[0]) if len(row) else None

    erm_auroc = auroc_of("centerstrat_erm_7f_controlled")
    gdro_auroc = auroc_of("groupdro_7f_controlled")
    plain_auroc = auroc_of("old_plain_erm_context")
    a7_auroc = auroc_of("stain_aug_7a_context")
    d7_auroc = auroc_of("stain_aug_7d_context")
    e7_auroc = auroc_of("stain_space_7e_context")

    # controlled AUROC delta (sanity, from the 7H controlled table)
    dauroc_row = controlled.loc[controlled["metric"] == "ood_val AUROC"]
    dauroc = float(dauroc_row["delta_groupdro_minus_erm"].iloc[0]) if len(dauroc_row) else None

    # sanity: controlled numbers must be internally consistent
    assert erm_auroc is not None and gdro_auroc is not None, "missing controlled AUROC rows"
    assert gdro_auroc > erm_auroc, "controlled: GroupDRO ood_val AUROC must exceed matched ERM"
    assert plain_auroc is not None and plain_auroc > gdro_auroc, (
        "context: old plain ERM AUROC expected higher than GroupDRO (reference gap)"
    )

    # ==================================================================================
    # TABLE 1 — pre-final decision matrix
    # ==================================================================================
    dm_header = [
        "decision_area",
        "current_status",
        "recommendation",
        "frozen_status",
        "reason",
        "risk_if_ignored",
        "next_action",
    ]
    dm_rows = [
        [
            "primary model",
            "7F GroupDRO (+ temperature-scaled confidence) is the primary CONTROLLED candidate; not yet frozen as final.",
            "Retain GroupDRO as primary candidate; freeze only at final-eval protocol time.",
            "conditional",
            f"Beat its matched ERM control on ood_val (AUROC {gdro_auroc:.4f} vs {erm_auroc:.4f}); first DG lever to do so.",
            "Testing an unvetted/undocumented model on ood_test; wasting the one-time test run.",
            "Confirm candidate in final-eval protocol (Path B) before any ood_test.",
        ],
        [
            "matched control",
            "7F center-stratified ERM is the fair matched control (same sound 7F-1 cache).",
            "Keep as the controlled baseline; report alongside GroupDRO.",
            "frozen",
            "Only matched baseline that isolates the GroupDRO objective from cache/subset differences.",
            "Losing the controlled comparison; reverting to an unfair (context) baseline.",
            "Report ERM vs GroupDRO as the controlled comparison in all outputs.",
        ],
        [
            "context baselines",
            "old plain ERM / 7A / 7D / 7E are context-only (old confounded cache / different id_val subset).",
            "Report as context/reference only; never as controlled comparators for GroupDRO.",
            "frozen",
            f"Different training cache/subset; e.g. old plain ERM AUROC {plain_auroc:.4f} is a reference gap, not a controlled loss.",
            "Falsely claiming GroupDRO beats/loses to plain ERM in a controlled comparison.",
            "Keep context-only labelling in tables/manuscript.",
        ],
        [
            "calibration policy",
            "7G temperature scaling (GroupDRO T=3.6570) improves ECE/Brier/NLL but is fit+eval on the same ood_val subset (R7); argmax-invariant.",
            "Report as development-stage calibration; do NOT claim it generalizes.",
            "not_frozen",
            "No separate calibration/eval split; optimistic estimate.",
            "Over-claiming calibration; leaky calibration estimate at test time.",
            "Decide a validated calibration policy (separate split) before any final calibration claim.",
        ],
        [
            "operating threshold",
            "Only candidate id_val-selected fixed-specificity points exist; NONE committed.",
            "Do not commit a clinical threshold yet.",
            "not_frozen",
            "Candidate points are development-stage; no deployable threshold chosen.",
            "Committing a threshold that does not transfer to the unseen test hospital.",
            "Optionally commit an operating point on ood_val later (gated), then freeze for test.",
        ],
        [
            "full ood_val evaluation",
            "All OOD numbers are on a capped balanced ood_val subset (n=2000).",
            "Recommended before final test if compute/time allows (Path A).",
            "not_frozen",
            "Capped subset fragility (R1/R4); larger ood_val de-risks the locked test.",
            "Final test conclusions resting on a small balanced subset.",
            "Plan a larger/full id_val + ood_val development evaluation (inference only; no ood_test).",
        ],
        [
            "final ood_test timing",
            "ood_test is LOCKED and unread; touched once at the very end.",
            "Do NOT run until final model+calibration+threshold are frozen AND explicitly authorized.",
            "frozen",
            "Reserved-test discipline; single held-out generalization estimate.",
            "Burning the one-time test on an unfrozen configuration.",
            "Run only after Path B protocol is frozen and written authorization is given.",
        ],
        [
            "additional DG method",
            "Only GroupDRO tested; no CORAL/DANN (R3 residual).",
            "Optional (Path C); only if DG breadth is worth the training scope.",
            "not_frozen",
            "Broader DG comparison further reduces R3 but adds training scope.",
            "Reviewer asks for DG breadth; single-method DG story.",
            "Decide whether to add CORAL/DANN on the sound 7F-1 caches (gated).",
        ],
        [
            "model-based uncertainty",
            "Deterministic uncertainty + temperature scaling only (R8).",
            "DEFERRED / optional (Path D); NOT the default next step.",
            "not_frozen",
            "MC-dropout/ensemble not required to freeze candidate or run final test.",
            "Reviewer expects model-based uncertainty for an uncertainty paper.",
            "Only on explicit approval, after this 7I decision.",
        ],
        [
            "manuscript drafting",
            "6A/6B/7H reporting + claims governance exist; no draft written.",
            "Recommended (Path E), especially as a fallback if compute/time is limited.",
            "not_frozen",
            "Zero-compute consolidation of the controlled DG + reliability story.",
            "Delaying write-up; losing the current evidence framing.",
            "Start methods/results skeleton around current evidence (no new experiments).",
        ],
    ]
    _write_csv(TABLES / "exp07i_pre_final_decision_matrix.csv", dm_header, dm_rows)

    # ==================================================================================
    # TABLE 2 — model readiness table
    # ==================================================================================
    mr_header = [
        "model",
        "role",
        "training_cache",
        "controlled_for_final_story",
        "ood_val_evidence",
        "calibration_evidence",
        "operating_point_evidence",
        "readiness_status",
        "recommended_use_in_manuscript",
    ]
    mr_rows = [
        [
            "old_plain_erm_context",
            "context-only baseline",
            "old confounded balanced cache (pc1000; center<->label confound; center 4 absent)",
            "no",
            f"ood_val AUROC {plain_auroc:.4f} (highest ranking, but context-only)",
            "5A temperature scaling (T=3.1224) on ood_val (development-stage)",
            "candidate id_val-selected points (7B/7C); none committed",
            "context-only; NOT a controlled comparator",
            "Report as prior/context reference and motivation; never as a controlled DG comparator.",
        ],
        [
            "stain_aug_7a_context",
            "context-only baseline (moderate torchvision stain aug)",
            "old confounded balanced cache (pc1000) + moderate stain aug",
            "no",
            f"ood_val AUROC {a7_auroc:.4f}; did not beat plain-ERM ranking",
            "7C temperature scaling on ood_val (development-stage)",
            "candidate id_val-selected points (7C); none committed",
            "context-only; negative/mixed stain lever",
            "Report as a stain-augmentation lever that did not beat plain-ERM ranking.",
        ],
        [
            "stain_aug_7d_context",
            "context-only baseline (tuned torchvision stain aug v2)",
            "old confounded balanced cache (pc1000) + tuned stain aug v2",
            "no",
            f"ood_val AUROC {d7_auroc:.4f}; below plain and 7A",
            "none (no calibration in 7D)",
            "candidate id_val-selected points (7D); none committed",
            "context-only; negative stain lever",
            "Report as a negative/mixed comparator; do not promote.",
        ],
        [
            "stain_space_7e_context",
            "context-only baseline (OD/HED stain-space aug)",
            "old confounded balanced cache (pc1000) + OD/HED stain-space aug",
            "no",
            f"ood_val AUROC {e7_auroc:.4f} (lowest); strongest stain variant at fixed-spec operating points",
            "none (no calibration in 7E)",
            "candidate id_val-selected points (7E); none committed",
            "context-only; strongest stain operating-point behavior, lowest AUROC",
            "Report as the strongest stain variant at operating points but below plain-ERM ranking.",
        ],
        [
            "centerstrat_erm_7f_control",
            "matched control (fair DG baseline)",
            "NEW center-stratified sound cache (7F-1; centers {0,3,4} x labels balanced)",
            "yes",
            f"ood_val AUROC {erm_auroc:.4f} (matched control for GroupDRO)",
            "7G temperature scaling (T=3.9239) on ood_val (development-stage; R7)",
            "candidate id_val-selected fixed-specificity points (7G); none committed",
            "controlled baseline; development-stage",
            "Report as the matched ERM control in the controlled DG comparison.",
        ],
        [
            "groupdro_7f_candidate",
            "PRIMARY controlled candidate (+ temperature-scaled confidence)",
            "NEW center-stratified sound cache (7F-1; centers {0,3,4} x labels balanced)",
            "yes",
            f"ood_val AUROC {gdro_auroc:.4f} vs matched ERM {erm_auroc:.4f} (+{dauroc:.4f}); 4/4 fixed-spec sensitivity wins",
            "7G temperature scaling (T=3.6570) on ood_val; better-calibrated than ERM raw+cal (development-stage; R7)",
            "candidate id_val-selected fixed-specificity points (7G); none committed",
            "primary candidate; development-stage; NOT frozen as final",
            "Report as the first positive controlled DG lever (vs matched ERM control only).",
        ],
    ]
    _write_csv(TABLES / "exp07i_model_readiness_table.csv", mr_header, mr_rows)

    # ==================================================================================
    # TABLE 3 — forbidden claims before test
    # ==================================================================================
    fc_header = [
        "forbidden_claim",
        "why_forbidden",
        "allowed_replacement_wording",
        "what_would_be_needed_to_unlock",
    ]
    fc_rows = [
        [
            "Final generalization performance is known.",
            "All numbers are development-stage on capped balanced subsets; ood_test unread.",
            "In the capped development setting, GroupDRO improves ood_val ranking over its matched ERM control.",
            "The single locked ood_test (center 2) evaluation under written authorization.",
        ],
        [
            "The model is ready for clinical deployment.",
            "Patch-level, capped-subset, development-stage; no clinical endpoint or WSI aggregation.",
            "This is a development-stage reliability/DG audit, not a clinical performance study.",
            "Full/final evaluation, WSI/patient-level aggregation, external cohorts, clinical validation.",
        ],
        [
            "GroupDRO beats old plain ERM in a controlled comparison.",
            "Old plain ERM used a different (confounded) cache and id_val subset; not matched.",
            "GroupDRO beats its MATCHED center-stratified ERM control; old plain ERM is context-only.",
            "An old-plain-ERM re-train on the sound center-stratified cache (matched) — not planned.",
        ],
        [
            "The selected candidate operating thresholds are final.",
            "Only id_val-selected candidate points exist; none committed; development-stage.",
            "Candidate id_val-selected operating points are reported for illustration only (not committed).",
            "A committed operating-point selection on ood_val (gated), then frozen for the test.",
        ],
        [
            "Calibrated probabilities are validated on an independent test.",
            "Temperature was fit AND evaluated on the same ood_val subset (R7).",
            "Temperature scaling improves calibration on this ood_val subset (argmax-invariant).",
            "A separate calibration vs evaluation split (or fit-on-one-hospital / eval-on-another).",
        ],
        [
            "The ood_test result can be inferred from ood_val.",
            "ood_test (center 2) is a distinct unseen hospital; ood_val (center 1) cannot substitute.",
            "ood_val is a development-stage OOD proxy; the held-out test hospital is unread.",
            "The single locked ood_test evaluation (center 2) under written authorization.",
        ],
        [
            "The model is robust to all hospital shifts.",
            "Only one OOD-val hospital (center 1) tested; single dataset; capped subset.",
            "GroupDRO improves development-stage robustness to the center-1 shift over its matched control.",
            "Multiple held-out hospitals / external cohorts + the final locked test.",
        ],
        [
            "Stain augmentation is ineffective universally.",
            "7A/7D/7E were on the old confounded cache and capped subsets; scope-limited negative result.",
            "On this capped development subset, the tested stain-augmentation variants did not beat plain-ERM ranking.",
            "Broader stain-normalization studies (e.g. Macenko/Vahadane) on sound full-scale data.",
        ],
    ]
    _write_csv(TABLES / "exp07i_forbidden_claims_before_test.csv", fc_header, fc_rows)

    # ==================================================================================
    # VERIFY DOCS (existence + 7I marker). Context docs are edited by hand this milestone.
    # ==================================================================================
    doc_marker_ok = {}
    for p in NEW_DOCS + CONTEXT_DOCS:
        if not p.exists():
            ERRORS.append(f"missing doc: {p}")
            doc_marker_ok[str(p)] = False
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        ok = MARKER in txt
        doc_marker_ok[str(p)] = ok
        if not ok:
            WARNINGS.append(f"doc missing 7I marker: {p}")

    # ---- read-only fingerprint (after) ----
    fp_after = _fingerprint(PARSED_INPUTS + CONTEXT_INPUTS)
    inputs_unmodified = fp_before == fp_after
    if not inputs_unmodified:
        ERRORS.append("input artifact fingerprint changed (inputs were modified!)")

    # ==================================================================================
    # ASSERTIONS
    # ==================================================================================
    out_tables = [
        TABLES / "exp07i_pre_final_decision_matrix.csv",
        TABLES / "exp07i_model_readiness_table.csv",
        TABLES / "exp07i_forbidden_claims_before_test.csv",
    ]
    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    check("no ood_test in any path", all("ood_test" not in str(p).lower() for p in all_paths))
    check("input artifacts unmodified (size+mtime fingerprint)", inputs_unmodified)
    for t in out_tables:
        check(f"table saved non-empty: {t.name}", t.exists() and t.stat().st_size > 0)
    # row-count sanity
    check("decision matrix has 10 rows", len(dm_rows) == 10)
    check("model readiness has 6 rows", len(mr_rows) == 6)
    check("forbidden claims has 8 rows", len(fc_rows) == 8)
    # controlled-vs-context integrity
    check("controlled: GroupDRO ood_val AUROC > matched ERM", gdro_auroc > erm_auroc)
    check("context: old plain ERM AUROC > GroupDRO (reference gap)", plain_auroc > gdro_auroc)
    # docs present with marker
    for p in NEW_DOCS:
        check(f"new doc present + 7I marker: {p.name}", doc_marker_ok.get(str(p), False))
    for p in CONTEXT_DOCS:
        check(f"context doc present + 7I marker: {p.name}", doc_marker_ok.get(str(p), False))

    all_pass = all(ok for _, ok in checks) and not ERRORS

    # ==================================================================================
    # REPORT
    # ==================================================================================
    lines = []
    A = lines.append
    A("MILESTONE 7I - PRE-FINAL-EVALUATION DECISION AUDIT (report)")
    A("=" * 78)
    A("Audit / documentation only: no training, no inference, no calibration fitting,")
    A("no MC-dropout, no ensemble, no CORAL/DANN, no ood_test, no final claim, no")
    A("committed clinical threshold. All numbers are DEVELOPMENT-stage (capped subsets).")
    A("")
    A("PRIMARY CONTROLLED CANDIDATE:")
    A("  7F GroupDRO-by-center + temperature-scaled confidence (T=3.6570).")
    A(f"    ood_val AUROC {gdro_auroc:.4f}  (matched ERM control {erm_auroc:.4f}; +{dauroc:.4f})")
    A("MATCHED CONTROL:")
    A(f"  7F center-stratified ERM (ood_val AUROC {erm_auroc:.4f}); same sound 7F-1 cache.")
    A("CONTEXT-ONLY BASELINES (NOT controlled comparators):")
    A(f"  old plain ERM {plain_auroc:.4f} | 7A {a7_auroc:.4f} | 7D {d7_auroc:.4f} | 7E {e7_auroc:.4f}")
    A("")
    A("KEY GROUPDRO EVIDENCE (controlled, development-stage):")
    A(f"  - ood_val AUROC {erm_auroc:.4f} -> {gdro_auroc:.4f} (+{dauroc:.4f})")
    A("  - ood_val AUPRC 0.5214 -> 0.6287 (+0.1073)")
    A("  - default sensitivity 0.0100 -> 0.0700; default FN 990 -> 930")
    A("  - fixed-spec ood_val sensitivity: GroupDRO wins 4/4 targets")
    A("  - better-calibrated than matched ERM (raw+cal); high-conf FN 688->25 (cal)")
    A("")
    A("KEY LIMITATIONS (why NOT ready for final ood_test):")
    A("  - Capped balanced subsets only (R1/R4 open); no full-scale evidence.")
    A("  - No frozen final model / calibration policy / operating threshold.")
    A("  - Calibration fit AND evaluated on the same ood_val subset (R7 open).")
    A("  - No controlled claim vs old plain ERM (context-only reference gap).")
    A("  - Only GroupDRO tested (no CORAL/DANN; R3 residual).")
    A("  - Deterministic + temperature-scaling uncertainty only (R8; MBU deferred).")
    A("")
    A("RECOMMENDED NEXT PATH:")
    A("  Path A (full development-scale evaluation planning) if compute/time allows;")
    A("  otherwise Path E (manuscript skeleton). Path B (freeze candidate + write")
    A("  final-eval protocol) alongside. Path C (CORAL/DANN) optional. Path D")
    A("  (model-based uncertainty) DEFERRED / optional - NOT the default next step.")
    A("  The default next action is a DECISION, not automatic model-based uncertainty.")
    A("")
    A("FROZEN (see docs/FROZEN_DECISIONS.md): locked split mapping; ood_test locked;")
    A("  ood_val development-only; old confounded caches barred from DG training;")
    A("  center-stratified caches required for DG; fair DG = 7F ERM vs 7F GroupDRO;")
    A("  old plain ERM / 7A / 7D / 7E context-only.")
    A("NOT FROZEN: final model; final threshold; final calibration policy; full-eval")
    A("  scope; CORAL/DANN; model-based uncertainty.")
    A("")
    A("GENERATED ARTIFACTS:")
    A("  docs/PRE_FINAL_EVALUATION_AUDIT.md")
    A("  docs/FROZEN_DECISIONS.md")
    A("  docs/NEXT_EXPERIMENT_DECISION_TREE.md")
    A("  results/tables/exp07i_pre_final_decision_matrix.csv")
    A("  results/tables/exp07i_model_readiness_table.csv")
    A("  results/tables/exp07i_forbidden_claims_before_test.csv")
    A("  results/logs/exp07i_pre_final_evaluation_audit_report.txt (this file)")
    A("")
    A("ASSERTIONS:")
    for name, ok in checks:
        A(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    A("")
    A(f"WARNINGS: {len(WARNINGS)}")
    for w in WARNINGS:
        A(f"  - {w}")
    A(f"ERRORS: {len(ERRORS)}")
    for e in ERRORS:
        A(f"  - {e}")
    A("")
    A("EXPLICIT ood_test STATEMENT:")
    A("  ood_test (center 2) remains FULLY LOCKED and unread. It may be used once, for")
    A("  the single final locked evaluation, only after a final model/calibration/")
    A("  threshold are frozen AND explicit written authorization is given. No 7I")
    A("  artifact reads or references ood_test.")
    A("")
    A(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")

    report = "\n".join(lines) + "\n"
    (LOGS / "exp07i_pre_final_evaluation_audit_report.txt").write_text(report, encoding="utf-8")

    # console (ASCII-safe)
    print(report)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
