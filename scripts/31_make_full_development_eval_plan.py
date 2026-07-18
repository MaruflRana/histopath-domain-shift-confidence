"""Milestone 8A - Full development-scale evaluation PLANNING package.

PLANNING ONLY. This script performs NO training, NO model inference, NO
calibration fitting, NO MC-dropout, NO ensemble, NO CORAL/DANN, and NEVER reads
`ood_test`. It does not import torch and does not load model weights or images.
It only:
  - writes four planning tables under results/tables/exp08a_*.csv,
  - writes a planning report under results/logs/,
  - verifies (read-only, existence + size) that the frozen checkpoints and 7F
    prediction CSVs referenced by the plan are present,
  - runs assertions and prints an OVERALL PASS/FAIL.

It does NOT modify any checkpoint, prediction CSV, cache, or prior-milestone
artifact. The companion planning docs live under docs/FULL_DEVELOPMENT_*.
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup; no network used here)

import csv
import os

# ---------------------------------------------------------------------------
# Paths (repo root = parent of scripts/)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")
LOGS = os.path.join(ROOT, "results", "logs")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# Frozen checkpoints for the two REQUIRED (controlled) models.
CKPT_GROUPDRO = os.path.join(
    ROOT, "results", "checkpoints", "exp07f_groupdro_resnet18", "best.pt"
)
CKPT_ERM = os.path.join(
    ROOT, "results", "checkpoints", "exp07f_centerstrat_erm_resnet18", "best.pt"
)
# Optional context-only checkpoints.
CKPT_PLAIN = os.path.join(ROOT, "results", "checkpoints", "exp04a_dev_train_resnet18", "best.pt")
CKPT_7A = os.path.join(ROOT, "results", "checkpoints", "exp07a_stainaug_resnet18", "best.pt")
CKPT_7D = os.path.join(ROOT, "results", "checkpoints", "exp07d_stainaug_v2_resnet18", "best.pt")
CKPT_7E = os.path.join(ROOT, "results", "checkpoints", "exp07e_stainspace_resnet18", "best.pt")

# Existing 7F predictions (current capped artifacts; read-only reference only).
PRED_GROUPDRO = os.path.join(ROOT, "results", "predictions", "exp07f_groupdro_resnet18")
PRED_ERM = os.path.join(ROOT, "results", "predictions", "exp07f_centerstrat_erm_resnet18")


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# 5. exp08a_evaluation_scope_matrix.csv
# ---------------------------------------------------------------------------
scope_header = [
    "evaluation_component", "included", "models", "split_or_data_source", "purpose",
    "uses_ood_val", "uses_ood_test", "execution_milestone", "notes",
]
scope_rows = [
    ["GroupDRO full id_val prediction", "yes", "7F GroupDRO (primary candidate)",
     "full/larger id_val (validation, centers {0,3,4})",
     "reload sanity + candidate threshold selection + per-center analysis",
     "no", "no", "8B",
     "inference only; thresholds selected here"],
    ["GroupDRO full ood_val prediction", "yes", "7F GroupDRO (primary candidate)",
     "full/larger ood_val (validation, center {1})",
     "development OOD estimate; apply id_val-selected thresholds",
     "yes", "no", "8B",
     "development-stage OOD analysis (authorized)"],
    ["matched ERM full id_val prediction", "yes", "7F center-stratified ERM (matched control)",
     "full/larger id_val (validation, centers {0,3,4})",
     "matched control reload sanity + threshold selection + per-center analysis",
     "no", "no", "8B",
     "keeps controlled comparison scaling with candidate"],
    ["matched ERM full ood_val prediction", "yes", "7F center-stratified ERM (matched control)",
     "full/larger ood_val (validation, center {1})",
     "matched control OOD estimate (fair comparison vs GroupDRO)",
     "yes", "no", "8B",
     "the FAIR controlled comparison is 7F ERM vs 7F GroupDRO"],
    ["candidate threshold selection", "yes", "7F GroupDRO + 7F center-stratified ERM",
     "id_val only (thresholds), applied to ood_val",
     "fixed-specificity (+optional fixed-sensitivity) candidate operating points",
     "yes", "no", "8B",
     "CANDIDATE only; NO committed clinical threshold"],
    ["calibration fitting", "optional", "7F GroupDRO + 7F center-stratified ERM",
     "ood_val logits (development-only) OR reuse saved 7G temperatures",
     "development-only temperature scaling (argmax-invariant)",
     "yes", "no", "8B",
     "R7 open; NO independent-test calibration claim; 8A fits nothing"],
    ["high-confidence FN audit", "yes", "7F GroupDRO + 7F center-stratified ERM",
     "ood_val predictions",
     "count label==1 & pred==0 & confidence>=t (t=0.90/0.95/0.99) raw vs cal",
     "yes", "no", "8B",
     "missed-tumor safety audit"],
    ["uncertainty/selective prediction", "optional", "7F GroupDRO + 7F center-stratified ERM",
     "id_val + ood_val predictions",
     "deterministic error-detection + risk-coverage/selective tables",
     "yes", "no", "8B",
     "deterministic only; NO MC-dropout/ensemble (model-based uncertainty deferred)"],
    ["context-only model evaluation", "optional", "old plain ERM / 7A / 7D / 7E",
     "id_val + ood_val (old confounded-cache-trained models)",
     "situate GroupDRO among prior levers (reference/context)",
     "yes", "no", "8B",
     "NOT required for controlled story; never a controlled comparator"],
    ["final ood_test evaluation", "no", "(none in 8A/8B)",
     "test split, center {2} -- LOCKED",
     "single final locked evaluation (separate, later, authorized)",
     "no", "no", "excluded",
     "LOCKED and unread; excluded from 8A and 8B"],
]

# ---------------------------------------------------------------------------
# 6. exp08a_model_checkpoint_freeze_table.csv
# ---------------------------------------------------------------------------
freeze_header = [
    "model", "role", "checkpoint_path", "prediction_path_current_if_any",
    "controlled_comparison_status", "include_in_full_dev_eval", "reason",
    "frozen_for_next_execution",
]
freeze_rows = [
    ["7F GroupDRO", "primary controlled candidate (+ temperature-scaled confidence)",
     rel(CKPT_GROUPDRO), rel(PRED_GROUPDRO),
     "controlled (vs 7F center-stratified ERM)", "yes",
     "First DG lever to beat its matched control on OOD (ood_val AUROC 0.7129 vs 0.6180); better-calibrated",
     "yes"],
    ["7F center-stratified ERM", "matched control",
     rel(CKPT_ERM), rel(PRED_ERM),
     "controlled (matched baseline for GroupDRO)", "yes",
     "Only fair baseline (same sound 7F-1 cache, matched init/seed/schedule); required so comparison scales",
     "yes"],
    ["old plain ERM", "context-only baseline",
     rel(CKPT_PLAIN),
     "results/predictions/exp04a_dev_train_resnet18 + exp05a_calibrate_ood_val_resnet18",
     "context-only (old confounded cache; different id_val subset)", "optional",
     "Reference/motivation only; ood_val AUROC 0.7696 is a reference gap, NOT a controlled loss",
     "no"],
    ["7A moderate stain/color augmentation", "context-only baseline",
     rel(CKPT_7A), "results/predictions/exp07a_stainaug_resnet18",
     "context-only (old confounded cache + stain aug)", "optional",
     "Negative/mixed stain lever; did not beat plain-ERM ranking (ood_val AUROC 0.7492)",
     "no"],
    ["7D tuned stain/color augmentation v2", "context-only baseline",
     rel(CKPT_7D), "results/predictions/exp07d_stainaug_v2_resnet18",
     "context-only (old confounded cache + tuned stain aug v2)", "optional",
     "Negative stain lever; below plain and 7A (ood_val AUROC 0.7397)",
     "no"],
    ["7E OD/HED-style stain-space augmentation", "context-only baseline",
     rel(CKPT_7E), "results/predictions/exp07e_stainspace_resnet18",
     "context-only (old confounded cache + OD/HED stain-space aug)", "optional",
     "Strongest stain variant at fixed-spec operating points but lowest AUROC (0.7317)",
     "no"],
]

# ---------------------------------------------------------------------------
# 7. exp08a_metric_output_plan.csv
# ---------------------------------------------------------------------------
metric_header = [
    "metric_family", "required", "input_needed", "output_artifact", "split", "notes",
]
metric_rows = [
    ["classification", "yes", "predictions (label, prob_1) per model/split",
     "results/metrics/exp08b_full_dev_eval/<model>/classification_metrics.json",
     "id_val + ood_val", "accuracy/balanced_acc/precision/f1; tumor=1 positive"],
    ["AUROC/AUPRC", "yes", "predictions (label, prob_1)",
     "classification_metrics.json + report", "id_val + ood_val",
     "threshold-free ranking; primary controlled comparison metric"],
    ["fixed-specificity candidate operating points", "yes",
     "id_val + ood_val predictions", "results/tables/exp08b_operating_points.csv",
     "id_val (select) -> ood_val (apply)", "spec 0.80/0.85/0.90/0.95; CANDIDATE only"],
    ["fixed-sensitivity candidate operating points", "optional",
     "id_val + ood_val predictions", "results/tables/exp08b_operating_points.csv",
     "id_val (select) -> ood_val (apply)", "sens 0.80/0.90/0.95; CANDIDATE only"],
    ["calibration", "optional", "ood_val logits (dev) OR reuse 7G temperatures",
     "results/metrics/exp08b_full_dev_eval/<model>/calibration_metrics.json + exp08b_calibration_comparison.csv",
     "ood_val", "ECE/Brier/NLL raw vs cal; argmax-invariant; R7 open; NO independent-test claim"],
    ["high-confidence false negatives", "yes", "ood_val predictions (label, pred, confidence)",
     "results/tables/exp08b_high_confidence_fn.csv", "ood_val",
     "count label==1 & pred==0 & confidence>=t (0.90/0.95/0.99) raw vs cal"],
    ["uncertainty/error detection", "optional", "predictions (probs/logits)",
     "results/tables/exp08b_uncertainty_metrics.csv", "id_val + ood_val",
     "deterministic entropy/confidence/margin/energy; NO MC-dropout/ensemble"],
    ["risk-coverage", "optional", "predictions + uncertainty scores",
     "results/tables/exp08b_risk_coverage.csv + exp08b_selective_prediction.csv",
     "id_val + ood_val", "descriptive selective-prediction; no committed threshold"],
    ["per-center id_val analysis", "yes", "id_val predictions + center metadata",
     "results/tables/exp08b_id_val_per_center_metrics.csv", "id_val (centers 0/3/4 + overall)",
     "worst-center behavior (e.g. GroupDRO up-weighted center 4)"],
]

# ---------------------------------------------------------------------------
# 8. exp08a_compute_storage_risk_table.csv
# ---------------------------------------------------------------------------
risk_header = ["risk", "severity", "mitigation", "stop_rule", "owner_next_step"]
risk_rows = [
    ["full id_val prediction runtime", "low",
     "ResNet18 @96x96 is cheap on RTX 3060; ~10-30 min planning band per split/model; streaming-bound",
     "S-4 (row-count guard)", "8B executor records actual timing"],
    ["full ood_val prediction runtime", "low",
     "Similar to id_val; batch_size 64 (up to 128-256 safe); shuffle=False",
     "S-4 (row-count guard)", "8B executor records actual timing"],
    ["large prediction CSV size", "low",
     "~4-7 MB per model/split; <~30 MB total for required models; cache predictions only, not images",
     "S-9 (no overwrite)", "8B writes under exp08b_full_dev_eval/ (new names)"],
    ["HF streaming/network instability", "medium",
     "Transient disconnects auto-retried by datasets (expected per CLAUDE.md 6); re-run split cleanly on hard fail",
     "S-4 (no partial/appended CSV)", "8B executor logs retries; atomic finalize"],
    ["GPU memory", "low",
     "12 GB ample for ResNet18 @96x96; reduce batch size if needed",
     "(none; inference-only)", "8B executor sets batch size"],
    ["accidental ood_test access", "high",
     "ood_test LOCKED; no path/split/center-2 reference permitted; id_val first then ood_val only",
     "S-1 (halt on any ood_test path)", "8B halts + escalates; separate auth required"],
    ["overwriting old artifacts", "high",
     "New run names only (exp08b_*); never overwrite 4A-7I checkpoints/predictions/caches/tables",
     "S-9 (no overwrite without new run name)", "8B uses fresh run directory"],
    ["calibration leakage", "high",
     "Calibration development-only, fit on ood_val or reuse 7G temps; NEVER on ood_test; R7 stays open",
     "S-8 (halt if calibration touches ood_test)", "8B keeps calibration dev-only; no generalization claim"],
    ["metadata loss", "medium",
     "Preserve all 7 metadata fields + confidence in every prediction CSV",
     "S-7 (halt on missing metadata/columns)", "8B reuses predict.py schema + confidence column"],
]

TABLE_SPECS = [
    ("exp08a_evaluation_scope_matrix.csv", scope_header, scope_rows),
    ("exp08a_model_checkpoint_freeze_table.csv", freeze_header, freeze_rows),
    ("exp08a_metric_output_plan.csv", metric_header, metric_rows),
    ("exp08a_compute_storage_risk_table.csv", risk_header, risk_rows),
]

for name, header, rows in TABLE_SPECS:
    write_csv(os.path.join(TABLES, name), header, rows)

# ---------------------------------------------------------------------------
# Guard: no generated CSV may contain an ood_test DATA reference.
# (Docs/tables may say "ood_test excluded/LOCKED"; the guard bans it as a
#  split value / data-source cell, which would imply usage.)
# ---------------------------------------------------------------------------
FORBIDDEN_DATA_TOKENS = ["ood_test prediction", "test split, center {2} -- included",
                         "uses_ood_test,yes"]

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
assertions = []


def check(name, ok):
    assertions.append((name, bool(ok)))
    return bool(ok)


# frozen checkpoints exist (read-only existence)
gd_ok = os.path.isfile(CKPT_GROUPDRO)
erm_ok = os.path.isfile(CKPT_ERM)
check("frozen GroupDRO checkpoint present (read-only)", gd_ok)
check("frozen center-strat ERM checkpoint present (read-only)", erm_ok)

# 7F prediction CSVs present (reference only)
check("7F GroupDRO predictions present (reference)",
      os.path.isfile(os.path.join(PRED_GROUPDRO, "ood_val_predictions.csv")))
check("7F center-strat ERM predictions present (reference)",
      os.path.isfile(os.path.join(PRED_ERM, "ood_val_predictions.csv")))

# tables saved + row-count checks
for name, header, rows in TABLE_SPECS:
    p = os.path.join(TABLES, name)
    check(f"table saved non-empty: {name}", os.path.isfile(p) and os.path.getsize(p) > 0)

check("scope matrix has 10 rows", len(scope_rows) == 10)
check("freeze table has 6 rows", len(freeze_rows) == 6)
check("metric plan has 9 rows", len(metric_rows) == 9)
check("risk table has 9 rows", len(risk_rows) == 9)

# uses_ood_test column is 'no' for every scope/plan row (no test usage anywhere)
scope_ot_idx = scope_header.index("uses_ood_test")
check("every scope row uses_ood_test == no", all(r[scope_ot_idx] == "no" for r in scope_rows))

# excluded final ood_test row present and marked excluded
check("final ood_test row present and excluded",
      any(r[0] == "final ood_test evaluation" and r[1] == "no"
          and r[scope_header.index("execution_milestone")] == "excluded" for r in scope_rows))

# no generated table cell asserts ood_test data usage
def no_forbidden(path):
    with open(path, encoding="utf-8") as f:
        txt = f.read().replace(" ", "").lower()
    return not any(tok.replace(" ", "").lower() in txt for tok in FORBIDDEN_DATA_TOKENS)

check("no generated table asserts ood_test data usage",
      all(no_forbidden(os.path.join(TABLES, n)) for n, _, _ in TABLE_SPECS))

# required (controlled) models are frozen_for_next_execution == yes
fidx = freeze_header.index("frozen_for_next_execution")
iidx = freeze_header.index("include_in_full_dev_eval")
req = [r for r in freeze_rows if r[0].startswith("7F")]
check("both 7F models frozen_for_next_execution == yes",
      all(r[fidx] == "yes" for r in req) and len(req) == 2)
check("both 7F models include_in_full_dev_eval == yes",
      all(r[iidx] == "yes" for r in req))
ctx = [r for r in freeze_rows if not r[0].startswith("7F")]
check("all context-only models include == optional",
      all(r[iidx] == "optional" for r in ctx) and len(ctx) == 4)

# planning docs present
DOCS = os.path.join(ROOT, "docs")
for d in [
    "FULL_DEVELOPMENT_EVALUATION_PROTOCOL.md",
    "FULL_DEVELOPMENT_EVALUATION_RESOURCE_PLAN.md",
    "FULL_DEVELOPMENT_EVALUATION_OUTPUT_SCHEMA.md",
    "FULL_DEVELOPMENT_EVALUATION_STOP_RULES.md",
]:
    check(f"planning doc present: {d}", os.path.isfile(os.path.join(DOCS, d)))

# no torch imported (planning-only, no inference)
import sys
check("torch NOT imported (no inference in 8A)", "torch" not in sys.modules)

n_pass = sum(1 for _, ok in assertions if ok)
n_fail = sum(1 for _, ok in assertions if not ok)
overall = "PASS" if n_fail == 0 else "FAIL"

# ---------------------------------------------------------------------------
# 9. Report
# ---------------------------------------------------------------------------
report_path = os.path.join(LOGS, "exp08a_full_development_evaluation_planning_report.txt")
lines = []
lines.append("MILESTONE 8A - FULL DEVELOPMENT-SCALE EVALUATION PLANNING (report)")
lines.append("=" * 78)
lines.append("PLANNING ONLY: no training, no inference, no calibration fitting, no")
lines.append("MC-dropout, no ensemble, no CORAL/DANN, no ood_test, no final claim, no")
lines.append("committed clinical threshold. Frozen inputs read-only (existence only).")
lines.append("")
lines.append("RECOMMENDED EXECUTION ORDER (for the gated Milestone 8B):")
lines.append("  1. Load frozen 7F GroupDRO + 7F center-stratified ERM checkpoints (read-only).")
lines.append("  2. Evaluate full/larger id_val (centers {0,3,4}) FIRST for both models")
lines.append("     (reload sanity + candidate threshold selection + per-center analysis).")
lines.append("  3. Evaluate full/larger ood_val (center {1}) for both models; apply the")
lines.append("     id_val-selected candidate thresholds (development comparison).")
lines.append("  4. (optional) development-only calibration (reuse 7G temps or refit on")
lines.append("     ood_val) + high-confidence FN audit + optional uncertainty/selective.")
lines.append("  5. (optional) context-only models (old plain ERM/7A/7D/7E) for reference.")
lines.append("  6. NEVER touch ood_test.")
lines.append("")
lines.append("FROZEN CHECKPOINTS (required, controlled):")
lines.append(f"  primary candidate : {rel(CKPT_GROUPDRO)}  present={gd_ok}")
lines.append(f"  matched control   : {rel(CKPT_ERM)}  present={erm_ok}")
lines.append("")
lines.append("INCLUDED MODELS (required): 7F GroupDRO (primary), 7F center-stratified ERM (control).")
lines.append("OPTIONAL (context-only, NOT required): old plain ERM, 7A, 7D, 7E.")
lines.append("EXCLUDED: final ood_test evaluation (LOCKED; separate, later, authorized).")
lines.append("")
lines.append("REQUIRED OUTPUTS (produced later by 8B; schemas in docs/FULL_DEVELOPMENT_"
             "EVALUATION_OUTPUT_SCHEMA.md):")
lines.append("  - prediction CSVs (adds `confidence` col to the 7F schema) per model/split")
lines.append("  - classification metrics JSON (acc/AUROC/AUPRC/sens/spec/confusion)")
lines.append("  - operating-point table (id_val-selected fixed-spec; CANDIDATE only)")
lines.append("  - per-center id_val metrics table")
lines.append("  - high-confidence false-negative table")
lines.append("  - (optional) calibration JSON/table, uncertainty/selective tables")
lines.append("  - run report/log")
lines.append("")
lines.append("STOP RULES (enforced by 8B; docs/FULL_DEVELOPMENT_EVALUATION_STOP_RULES.md):")
lines.append("  S-1 any ood_test path -> HALT      S-6 non-binary labels -> HALT")
lines.append("  S-2 split mapping differs -> HALT  S-7 missing metadata fields -> HALT")
lines.append("  S-3 checkpoint differs -> HALT     S-8 calibration on ood_test -> HALT")
lines.append("  S-4 row-count mismatch -> HALT     S-9 would overwrite artifacts -> HALT")
lines.append("  S-5 invalid probabilities -> HALT")
lines.append("")
lines.append("PLANNING TABLES:")
for name, _, _ in TABLE_SPECS:
    lines.append(f"  results/tables/{name}")
lines.append("PLANNING DOCS:")
for d in ["FULL_DEVELOPMENT_EVALUATION_PROTOCOL.md",
          "FULL_DEVELOPMENT_EVALUATION_RESOURCE_PLAN.md",
          "FULL_DEVELOPMENT_EVALUATION_OUTPUT_SCHEMA.md",
          "FULL_DEVELOPMENT_EVALUATION_STOP_RULES.md"]:
    lines.append(f"  docs/{d}")
lines.append("")
lines.append("PROJECT READY TO RUN FULL DEVELOPMENT EVALUATION (8B)?")
ready = gd_ok and erm_ok and n_fail == 0
lines.append(f"  {'YES' if ready else 'NO'} - frozen checkpoints present, plan/tables/stop-rules "
             "written, assertions pass.")
lines.append("  8B requires INFERENCE ONLY (no training) and an explicit written go-ahead.")
lines.append("")
lines.append("ASSERTIONS:")
for name, ok in assertions:
    lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
lines.append("")
lines.append(f"WARNINGS: 0")
lines.append(f"ERRORS: {n_fail}")
lines.append("")
lines.append("EXPLICIT ood_test STATEMENT:")
lines.append("  ood_test (center 2) is FULLY LOCKED and unread. It is NOT used in Milestone")
lines.append("  8A (planning) or 8B (execution). No 8A artifact reads or references ood_test")
lines.append("  as data. It may be used once, for the single final locked evaluation, only")
lines.append("  after a frozen final model/calibration/threshold AND explicit written")
lines.append("  authorization.")
lines.append("")
lines.append(f"OVERALL: {overall}")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\nReport: {rel(report_path)}")
print(f"ASSERTIONS: {n_pass} pass / {n_fail} fail -> OVERALL {overall}")
