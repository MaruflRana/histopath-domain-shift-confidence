"""Milestone 9B: build the final manuscript package from saved artifacts only.

This script deliberately imports no dataset, model, training, calibration-fitting, or
inference module. It reads accepted 8B/8C/9A CSV/JSON/log artifacts and creates only new
exp09b tables, figures, and the integration report.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
LOG_DIR = ROOT / "results" / "logs"

TABLE_PATHS = {
    "development_vs_final": TABLE_DIR / "exp09b_development_vs_final_comparison.csv",
    "final_model_comparison": TABLE_DIR / "exp09b_final_model_comparison.csv",
    "calibration": TABLE_DIR / "exp09b_final_calibration_summary.csv",
    "operating_points": TABLE_DIR / "exp09b_operating_point_transfer.csv",
    "high_confidence_fn": TABLE_DIR / "exp09b_high_confidence_fn_summary.csv",
    "claims": TABLE_DIR / "exp09b_final_claims_audit.csv",
    "manifest": TABLE_DIR / "exp09b_manuscript_table_figure_manifest.csv",
}

FIGURE_PATHS = {
    "development_vs_final": FIGURE_DIR / "exp09b_development_to_final_auroc_auprc.png",
    "default_metrics": FIGURE_DIR / "exp09b_final_default_threshold_metrics.png",
    "calibration": FIGURE_DIR / "exp09b_final_calibration_raw_vs_calibrated.png",
    "operating_points": FIGURE_DIR / "exp09b_operating_point_transfer.png",
    "high_confidence_fn": FIGURE_DIR / "exp09b_high_confidence_fn_raw_vs_calibrated.png",
    "reversal": FIGURE_DIR / "exp09b_development_test_reversal.png",
}

REPORT_PATH = LOG_DIR / "exp09b_final_manuscript_integration_report.txt"

SOURCE_PATHS = {
    "dev_metrics": ROOT / "results" / "tables" / "exp08b_overall_metrics.csv",
    "dev_centers": ROOT / "results" / "tables" / "exp08b_id_val_per_center_metrics.csv",
    "dev_ops": ROOT / "results" / "tables" / "exp08b_candidate_operating_points.csv",
    "dev_calibration": ROOT / "results" / "tables" / "exp08c_calibration_comparison.csv",
    "dev_hcfn": ROOT / "results" / "tables" / "exp08c_high_confidence_fn_comparison.csv",
    "dev_reliability": ROOT / "results" / "tables" / "exp08c_reliability_summary.csv",
    "final_metrics": ROOT / "results" / "metrics" / "exp09_final_ood_test" / "final_metrics.json",
    "final_classification": ROOT / "results" / "tables" / "exp09_final_ood_test_classification.csv",
    "final_calibration": ROOT / "results" / "tables" / "exp09_final_ood_test_calibration.csv",
    "final_ops": ROOT / "results" / "tables" / "exp09_final_ood_test_frozen_operating_points.csv",
    "final_hcfn": ROOT / "results" / "tables" / "exp09_final_ood_test_high_confidence_false_negatives.csv",
    "authorization": ROOT / "configs" / "experiments" / "exp09_final_ood_test_authorization.json",
    "run_state": ROOT / "results" / "logs" / "exp09_final_ood_test_run_state.json",
    "inference_sentinel": ROOT / "results" / "logs" / "exp09_final_ood_test_completed_run.json",
    "summary_sentinel": ROOT / "results" / "logs" / "exp09_final_ood_test_summary_completed.json",
}

IMMUTABLE_HASHES = {
    "configs/experiments/exp09_final_ood_test_authorization.json":
        "0B5977E3D92990B6FC59E1138AC2D398C737F4BDCBACCDD1025D0CE0E3B96C45",
    "results/metrics/exp09_final_ood_test/final_metrics.json":
        "831F77ED24FD0DB0A261EB056D3945E4682F7D85A99435E16B808C6CAF5C49F8",
    "results/tables/exp09_final_ood_test_classification.csv":
        "9873E90501F556CCDF4DB75420C8659939B55F4C7FAD7DD406EF04DA8A50F65E",
    "results/tables/exp09_final_ood_test_calibration.csv":
        "38475335B8C2BA1B070F6A4FC5B101CF8930E0BDD09BEFEB0E2B38EE8C30EBCA",
    "results/tables/exp09_final_ood_test_frozen_operating_points.csv":
        "428326EA54C905A621C6A365207283B704F81AE7E41F599930E5AA960A5CD19C",
    "results/tables/exp09_final_ood_test_high_confidence_false_negatives.csv":
        "A7116E60FEE503D433F73F25D8B294CC8741ED4F48859426ED7D925237322519",
    "results/logs/exp09_final_ood_test_run_state.json":
        "946282AF934E91CD614C75E333EB965B856DEA9DCB5E81A527379407498B57EF",
    "results/logs/exp09_final_ood_test_completed_run.json":
        "C5FB65D3C2F20381B97766826357B392BEA04AFC689CCCAE796E5BB5C6AA92F5",
    "results/logs/exp09_final_ood_test_summary_completed.json":
        "6F1153EF0687D54CA3BAE34CC76F81C70CB05B49D9757C59DE7C66612A2E0B14",
    "results/predictions/exp09_final_ood_test/groupdro_ood_test_raw_predictions.csv":
        "D8EB2DFB87FA1B162ABC1033C130E7FBC5758EF1E990A237349FEF492BB691C8",
    "results/predictions/exp09_final_ood_test/groupdro_ood_test_calibrated_predictions.csv":
        "5B1DD8100D4CFE451F0F0FDA7B29B9D33988BE22000289F1BDCC0989B6EAEDBC",
    "results/predictions/exp09_final_ood_test/centerstrat_erm_ood_test_raw_predictions.csv":
        "CBFDBCB0C4530909877F179DE2B595DBB2C2DA3EBE70DCABEF698A912123ED3A",
    "results/predictions/exp09_final_ood_test/centerstrat_erm_ood_test_calibrated_predictions.csv":
        "738A69F197CAEF8C38D08650BEC46A090A8DBBBCBDF75E1421B4656903D41693",
    "results/figures/exp09_final_ood_test_roc.png":
        "69ADB31DCDBF5557498EC0A194623301143332E3A9BB11A6AC47602E073FAE84",
    "results/figures/exp09_final_ood_test_precision_recall.png":
        "0F9DEB188E0FE0278E5EEAAE1864785386F26FDE2BDF50101C460991A5C50284",
    "results/figures/exp09_final_ood_test_calibration_reliability.png":
        "B2ABECB0A1B7C521B3366A331B2C2D4E5C34794DF2D49BEE6A4EC09E189C2803",
    "results/figures/exp09_final_ood_test_operating_points.png":
        "5A12A08B535833F5A89E6BFB81A79FB4A7B9056D1AA6DC2F0E4395C874209ADA",
    "results/figures/exp09_final_ood_test_high_confidence_fn.png":
        "9247D581BF230EA8622E2E4A4301A27405527D13F7CDF834727CEF54DD0F77F8",
}

MODEL_LABELS = {
    "groupdro": "GroupDRO",
    "centerstrat_erm": "Matched ERM",
}
MODEL_ROLES = {
    "groupdro": "predeclared_primary",
    "centerstrat_erm": "matched_control",
}
COLORS = {
    "groupdro": "#C44E52",
    "centerstrat_erm": "#4C78A8",
    "raw": "#6B7280",
    "calibrated": "#E69F00",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def assert_sources_immutable() -> None:
    for relative, expected in IMMUTABLE_HASHES.items():
        path = ROOT / relative
        assert path.is_file(), f"Missing immutable source: {relative}"
        actual = sha256(path)
        assert actual == expected, f"Immutable source changed: {relative}: {actual} != {expected}"


def assert_output_scope() -> None:
    for path in [*TABLE_PATHS.values(), *FIGURE_PATHS.values(), REPORT_PATH]:
        relative = path.relative_to(ROOT).as_posix()
        assert "exp09b_" in path.name, f"Refusing non-9B output path: {relative}"


def load_sources() -> dict[str, object]:
    sources: dict[str, object] = {}
    for key, path in SOURCE_PATHS.items():
        assert path.is_file(), f"Missing source: {path}"
        if path.suffix == ".csv":
            sources[key] = pd.read_csv(path)
        elif path.suffix == ".json":
            sources[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            sources[key] = path.read_text(encoding="utf-8")
    return sources


def validate_run_provenance(sources: dict[str, object]) -> None:
    state = sources["run_state"]
    inference = sources["inference_sentinel"]
    summary = sources["summary_sentinel"]
    assert isinstance(state, dict) and isinstance(inference, dict) and isinstance(summary, dict)
    assert state["status"] == "COMPLETE"
    assert state["attempt_counter"] == state["run_counter"] == 1
    assert state["test_dataset_instances"] == 1
    assert state["test_dataloader_traversals"] == 1
    assert inference["run_status"] == "INFERENCE_COMPLETE"
    assert inference["attempt_counter"] == inference["run_counter"] == 1
    assert inference["row_count"] == 85054 and inference["required_center"] == 2
    assert inference["test_dataset_instances"] == 1
    assert inference["test_dataloader_traversals"] == 1
    assert inference["models_per_batch"] == 2
    assert inference["calibration_refit"] is False
    assert inference["threshold_selection_on_test"] is False
    assert summary["summary_status"] == "SUMMARY_COMPLETE"
    assert summary["inference_attempt_counter"] == 1
    assert summary["calibration_refit"] is False
    assert summary["threshold_selection_on_test"] is False


def create_development_vs_final(sources: dict[str, object]) -> pd.DataFrame:
    dev = sources["dev_metrics"].copy()
    final = sources["final_classification"].copy()
    assert isinstance(dev, pd.DataFrame) and isinstance(final, pd.DataFrame)
    final = final[final["variant"] == "raw"].copy()

    rows: list[dict[str, object]] = []
    metrics = [
        "n", "accuracy", "balanced_accuracy", "auroc", "auprc", "sensitivity",
        "specificity", "precision", "f1", "tn", "fp", "fn", "tp",
    ]
    for r in dev.itertuples(index=False):
        is_id = r.split == "id_val"
        row = {
            "stage": "full_development",
            "split": r.split,
            "center": "{0,3,4}" if is_id else "{1}",
            "model": r.model,
            "role": MODEL_ROLES[r.model],
            "scope_note": (
                "Full-development id_val; centers 0,3,4; development evidence, not final."
                if is_id else
                "Full-development ood_val; center 1; development evidence used for model assessment, not final."
            ),
        }
        row.update({m: getattr(r, m) for m in metrics})
        rows.append(row)

    for r in final.itertuples(index=False):
        row = {
            "stage": "final_held_out",
            "split": "ood_test",
            "center": "{2}",
            "model": r.model,
            "role": MODEL_ROLES[r.model],
            "scope_note": (
                "Single authorized reserved-center final evaluation; attempt 1; no post-test selection."
            ),
        }
        row.update({m: getattr(r, m) for m in metrics})
        rows.append(row)

    out = pd.DataFrame(rows)
    expected_cols = [
        "stage", "split", "center", "model", "role", "n", "accuracy",
        "balanced_accuracy", "auroc", "auprc", "sensitivity", "specificity",
        "precision", "f1", "tn", "fp", "fn", "tp", "scope_note",
    ]
    out = out[expected_cols]
    assert len(out) == 6
    return out


def create_final_model_comparison(sources: dict[str, object]) -> pd.DataFrame:
    final = sources["final_classification"]
    assert isinstance(final, pd.DataFrame)
    final = final[final["variant"] == "raw"].set_index("model")
    metrics = [
        "accuracy", "balanced_accuracy", "auroc", "auprc", "sensitivity",
        "specificity", "precision", "f1", "tn", "fp", "fn", "tp",
    ]
    lower_better = {"fp", "fn"}
    rows: list[dict[str, object]] = []
    for metric in metrics:
        gd = float(final.loc["groupdro", metric])
        erm = float(final.loc["centerstrat_erm", metric])
        delta = gd - erm
        if metric in lower_better:
            winner = "groupdro" if gd < erm else "centerstrat_erm" if erm < gd else "tie"
        else:
            winner = "groupdro" if gd > erm else "centerstrat_erm" if erm > gd else "tie"
        for model, value in [("groupdro", gd), ("centerstrat_erm", erm)]:
            rows.append({
                "model": model,
                "role": MODEL_ROLES[model],
                "metric": metric,
                "value": value,
                "delta_groupdro_minus_erm": delta,
                "winner_for_metric": winner,
                "interpretation": (
                    "Matched control outperformed the predeclared primary on the reserved center."
                    if winner == "centerstrat_erm" else
                    "Predeclared primary retained the favorable value for this metric."
                    if winner == "groupdro" else
                    "Models tied on this metric."
                ),
            })
    out = pd.DataFrame(rows)
    assert len(out) == 24
    return out


def create_calibration_summary(sources: dict[str, object]) -> pd.DataFrame:
    final = sources["final_calibration"]
    assert isinstance(final, pd.DataFrame)
    rows = []
    for model in ["groupdro", "centerstrat_erm"]:
        model_rows = final[final["model"] == model].set_index("variant")
        raw = model_rows.loc["raw"]
        for variant in ["raw", "calibrated"]:
            r = model_rows.loc[variant]
            deltas = {
                "ece": float(r.ece - raw.ece),
                "brier": float(r.brier - raw.brier),
                "nll": float(r.nll - raw.nll),
            }
            rows.append({
                "model": model,
                "variant": variant,
                "temperature": float(r.temperature),
                "ece": float(r.ece),
                "brier": float(r.brier),
                "nll": float(r.nll),
                "delta_from_raw": json.dumps(deltas, sort_keys=True),
                "hard_predictions_changed": False,
                "scope_note": (
                    "Held-out center-2 reliability with pre-frozen temperature; no refit. "
                    "Improved confidence metrics do not imply corrected classification errors."
                ),
            })
    out = pd.DataFrame(rows)
    assert len(out) == 4
    for model in ["groupdro", "centerstrat_erm"]:
        cal = out[(out.model == model) & (out.variant == "calibrated")].iloc[0]
        raw = out[(out.model == model) & (out.variant == "raw")].iloc[0]
        assert cal.ece < raw.ece and cal.brier < raw.brier and cal.nll < raw.nll
    return out


def transfer_status(target_type: str, gap: float) -> str:
    if target_type == "sensitivity" and gap < -0.10:
        return "substantial_underachievement"
    if abs(gap) <= 0.01:
        return "near_nominal_descriptive_only"
    if abs(gap) <= 0.05:
        return "shifted_from_nominal"
    return "poor_transfer"


def create_operating_point_transfer(sources: dict[str, object]) -> pd.DataFrame:
    ops = sources["final_ops"]
    assert isinstance(ops, pd.DataFrame) and len(ops) == 14
    rows = []
    for r in ops.itertuples(index=False):
        selected_metric = (
            float(r.id_val_specificity) if r.target_type == "specificity"
            else float(r.id_val_sensitivity)
        )
        achieved_metric = (
            float(r.ood_test_specificity) if r.target_type == "specificity"
            else float(r.ood_test_sensitivity)
        )
        gap = achieved_metric - float(r.target_value)
        rows.append({
            "model": r.model,
            "target_type": r.target_type,
            "nominal_development_target": float(r.target_value),
            "frozen_threshold": float(r.threshold_selected_on_id_val),
            "id_val_metric_at_selection": selected_metric,
            "final_test_sensitivity": float(r.ood_test_sensitivity),
            "final_test_specificity": float(r.ood_test_specificity),
            "target_gap": gap,
            "transfer_status": transfer_status(r.target_type, gap),
            "candidate_nonclinical_note": (
                "Frozen id_val-selected candidate; explicitly non-clinical; no test-set selection or tuning."
            ),
        })
    out = pd.DataFrame(rows)
    assert len(out) == 14
    return out


def create_high_confidence_fn_summary(sources: dict[str, object]) -> pd.DataFrame:
    hc = sources["final_hcfn"]
    assert isinstance(hc, pd.DataFrame) and len(hc) == 12
    counts = {
        (r.model, r.variant, float(r.confidence_threshold)):
            int(r.high_confidence_false_negatives)
        for r in hc.itertuples(index=False)
    }
    rows = []
    for r in hc.itertuples(index=False):
        key = (r.model, float(r.confidence_threshold))
        delta = (
            counts[(r.model, "calibrated", key[1])] - counts[(r.model, "raw", key[1])]
            if r.variant == "calibrated" else np.nan
        )
        rows.append({
            "model": r.model,
            "variant": r.variant,
            "confidence_threshold": float(r.confidence_threshold),
            "high_confidence_fn": int(r.high_confidence_false_negatives),
            "total_fn": int(r.total_false_negatives),
            "fraction_of_total_fn_high_confidence": (
                float(r.high_confidence_false_negatives) / float(r.total_false_negatives)
            ),
            "delta_after_calibration": delta,
            "interpretation": (
                "Calibration reduced confidently stated misses but total FN was unchanged."
                if r.variant == "calibrated" else
                "Raw-confidence baseline; total FN is fixed by hard predictions."
            ),
        })
    out = pd.DataFrame(rows)
    assert len(out) == 12
    return out


def create_claims_audit() -> pd.DataFrame:
    rows = [
        {
            "claim_id": "A1",
            "claim": "Final held-out center-2 metrics for both frozen models",
            "status": "allowed",
            "supporting_artifact": "results/metrics/exp09_final_ood_test/final_metrics.json",
            "exact_allowed_wording": "On the single reserved center-2 evaluation, the frozen GroupDRO primary and matched ERM control achieved the reported exp09 metrics.",
            "required_caveat": "Single hospital, patch-level, one authorized attempt; not all-hospital or clinical performance.",
            "manuscript_location": "Results: Final held-out evaluation",
        },
        {
            "claim_id": "A2",
            "claim": "Matched ERM outperformed the predeclared GroupDRO primary on center 2",
            "status": "allowed",
            "supporting_artifact": "results/tables/exp09b_final_model_comparison.csv",
            "exact_allowed_wording": "The matched control outperformed the predeclared primary on AUROC, AUPRC, accuracy, sensitivity, F1, and false-negative count on the reserved center.",
            "required_caveat": "Do not relabel ERM as the predeclared primary or imply post-test model selection.",
            "manuscript_location": "Abstract; Results; Discussion",
        },
        {
            "claim_id": "A3",
            "claim": "Development GroupDRO advantage reversed on the reserved hospital",
            "status": "allowed",
            "supporting_artifact": "results/tables/exp09b_development_vs_final_comparison.csv",
            "exact_allowed_wording": "The GroupDRO advantage observed on development center 1 did not generalize to reserved center 2.",
            "required_caveat": "This is a two-hospital development-to-test contrast, not a universal DG conclusion.",
            "manuscript_location": "Abstract; Results; Discussion",
        },
        {
            "claim_id": "A4",
            "claim": "Frozen temperature scaling improved held-out ECE, Brier, and NLL",
            "status": "allowed",
            "supporting_artifact": "results/tables/exp09b_final_calibration_summary.csv",
            "exact_allowed_wording": "Applying the pre-frozen temperatures improved held-out ECE, Brier score, and NLL for both models on center 2.",
            "required_caveat": "Temperatures were not refit; hard predictions and total FN did not change; no universal calibration-validity claim.",
            "manuscript_location": "Results: Calibration; Discussion",
        },
        {
            "claim_id": "A5",
            "claim": "Development-selected operating targets transferred poorly",
            "status": "allowed",
            "supporting_artifact": "results/tables/exp09b_operating_point_transfer.csv",
            "exact_allowed_wording": "The frozen id_val-selected operating targets were not reliably preserved under the center-2 hospital shift.",
            "required_caveat": "Candidate/non-clinical thresholds; no threshold selected on test.",
            "manuscript_location": "Results: Operating-point transfer; Discussion",
        },
        {
            "claim_id": "A6",
            "claim": "Calibration reduced high-confidence false negatives but not total false negatives",
            "status": "allowed",
            "supporting_artifact": "results/tables/exp09b_high_confidence_fn_summary.csv",
            "exact_allowed_wording": "Calibration sharply reduced confidently stated missed tumors while leaving the underlying total number of missed tumors unchanged.",
            "required_caveat": "Confidence correction is not error correction.",
            "manuscript_location": "Results: High-confidence FN; Discussion",
        },
        {
            "claim_id": "A7",
            "claim": "Reserved-test discipline exposed model-selection risk",
            "status": "allowed",
            "supporting_artifact": "results/logs/exp09_final_ood_test_report.txt",
            "exact_allowed_wording": "The reserved-hospital protocol exposed a development-to-test reversal that would have been hidden by reporting ood_val as final performance.",
            "required_caveat": "Methodological contribution within this study; not proof that every reserved-hospital protocol will show reversal.",
            "manuscript_location": "Discussion; Conclusion",
        },
        {
            "claim_id": "C1",
            "claim": "ERM may be preferred for future clinical use",
            "status": "conditional",
            "supporting_artifact": "results/tables/exp09b_final_model_comparison.csv",
            "exact_allowed_wording": "The center-2 result motivates further independent validation of the matched ERM control.",
            "required_caveat": "No post-test model switch, no deployment recommendation, and no threshold validation.",
            "manuscript_location": "Discussion: Future work",
        },
        {
            "claim_id": "F1",
            "claim": "Clinical deployment readiness",
            "status": "forbidden",
            "supporting_artifact": "none",
            "exact_allowed_wording": "Not allowed.",
            "required_caveat": "Patch-level single-dataset evidence does not establish clinical safety or utility.",
            "manuscript_location": "Prohibited throughout",
        },
        {
            "claim_id": "F2",
            "claim": "Robustness to all hospitals",
            "status": "forbidden",
            "supporting_artifact": "none",
            "exact_allowed_wording": "Not allowed.",
            "required_caveat": "Only one reserved test hospital was evaluated.",
            "manuscript_location": "Prohibited throughout",
        },
        {
            "claim_id": "F3",
            "claim": "Universal ERM superiority",
            "status": "forbidden",
            "supporting_artifact": "none",
            "exact_allowed_wording": "Not allowed.",
            "required_caveat": "ERM outperformed GroupDRO on center 2 only; development center 1 showed the opposite ordering.",
            "manuscript_location": "Prohibited throughout",
        },
        {
            "claim_id": "F4",
            "claim": "Universal GroupDRO failure",
            "status": "forbidden",
            "supporting_artifact": "none",
            "exact_allowed_wording": "Not allowed.",
            "required_caveat": "GroupDRO improved development center 1 and worst-center id_val behavior.",
            "manuscript_location": "Prohibited throughout",
        },
        {
            "claim_id": "F5",
            "claim": "Independent multi-center calibration validity",
            "status": "forbidden",
            "supporting_artifact": "none",
            "exact_allowed_wording": "Not allowed.",
            "required_caveat": "Frozen temperatures improved metrics on one reserved center only.",
            "manuscript_location": "Prohibited throughout",
        },
        {
            "claim_id": "F6",
            "claim": "Clinically validated operating threshold",
            "status": "forbidden",
            "supporting_artifact": "none",
            "exact_allowed_wording": "Not allowed.",
            "required_caveat": "All operating points remain candidate/non-clinical.",
            "manuscript_location": "Prohibited throughout",
        },
        {
            "claim_id": "F7",
            "claim": "WSI-level or patient-level effectiveness",
            "status": "forbidden",
            "supporting_artifact": "none",
            "exact_allowed_wording": "Not allowed.",
            "required_caveat": "Evaluation is patch-level.",
            "manuscript_location": "Prohibited throughout",
        },
        {
            "claim_id": "F8",
            "claim": "Superiority over CORAL, DANN, or untested DG methods",
            "status": "forbidden",
            "supporting_artifact": "none",
            "exact_allowed_wording": "Not allowed.",
            "required_caveat": "These methods were not implemented or evaluated.",
            "manuscript_location": "Prohibited throughout",
        },
    ]
    return pd.DataFrame(rows)


def create_manifest() -> pd.DataFrame:
    rows = [
        ("T1", "table", "results/tables/exp09b_development_vs_final_comparison.csv", "Results: development and final comparison", "main", "9B", "Development centers 0/3/4 and 1 versus final held-out center 2", "Do not merge development and final scope; one-shot final result.", "created"),
        ("T2", "table", "results/tables/exp09b_final_model_comparison.csv", "Results: final controlled comparison", "main", "9B", "Final held-out center 2", "GroupDRO remains the predeclared primary; ERM remains the matched control.", "created"),
        ("T3", "table", "results/tables/exp09b_final_calibration_summary.csv", "Results: held-out calibration", "main", "9B", "Final held-out center 2; frozen temperatures", "No refit; hard predictions and total FN unchanged.", "created"),
        ("T4", "table", "results/tables/exp09b_operating_point_transfer.csv", "Results: operating-point transfer", "supplement", "9B", "Frozen id_val candidates applied to final center 2", "Candidate/non-clinical; no test tuning.", "created"),
        ("T5", "table", "results/tables/exp09b_high_confidence_fn_summary.csv", "Results: high-confidence false negatives", "main", "9B", "Final held-out center 2", "Confidence correction is not error correction.", "created"),
        ("T6", "table", "results/tables/exp09b_final_claims_audit.csv", "Governance and supplement", "supplement", "9B", "Post-test claims map", "Allowed wording remains hospital- and patch-scoped.", "created"),
        ("F1", "figure", "results/figures/exp09b_development_to_final_auroc_auprc.png", "Results: development-to-final comparison", "main", "9B", "Development center 1 versus final held-out center 2", "Development and final stages must remain visibly distinct.", "created"),
        ("F2", "figure", "results/figures/exp09b_final_default_threshold_metrics.png", "Results: final default-threshold comparison", "main", "9B", "Final center 2; threshold 0.5", "Not a clinically validated operating threshold.", "created"),
        ("F3", "figure", "results/figures/exp09b_final_calibration_raw_vs_calibrated.png", "Results: calibration", "main", "9B", "Final center 2; raw versus frozen-temperature calibrated", "No calibration refit and no hard-prediction change.", "created"),
        ("F4", "figure", "results/figures/exp09b_operating_point_transfer.png", "Results: operating-point transfer", "supplement", "9B", "Nominal id_val target versus achieved final metric", "Candidate/non-clinical operating points.", "created"),
        ("F5", "figure", "results/figures/exp09b_high_confidence_fn_raw_vs_calibrated.png", "Results: high-confidence false negatives", "main", "9B", "Final center 2", "Total FN is unchanged by calibration.", "created"),
        ("F6", "figure", "results/figures/exp09b_development_test_reversal.png", "Discussion: development-to-test reversal", "main", "9B", "Development center 1 versus final center 2", "Do not imply universal ERM superiority or universal GroupDRO failure.", "created"),
        ("S1", "figure", "results/figures/exp09_final_ood_test_roc.png", "Supplement: final ROC", "supplement", "9A", "Final held-out center 2", "Raw/calibrated curves overlap because ranking is invariant.", "existing"),
        ("S2", "figure", "results/figures/exp09_final_ood_test_precision_recall.png", "Supplement: final precision-recall", "supplement", "9A", "Final held-out center 2", "Raw/calibrated curves overlap because ranking is invariant.", "existing"),
        ("S3", "figure", "results/figures/exp09_final_ood_test_calibration_reliability.png", "Supplement: reliability diagrams", "supplement", "9A", "Final held-out center 2", "Frozen temperatures; no refit.", "existing"),
        ("S4", "figure", "results/figures/exp09_final_ood_test_operating_points.png", "Supplement: candidate operating points", "supplement", "9A", "Final held-out center 2", "Candidate/non-clinical.", "existing"),
        ("S5", "figure", "results/figures/exp09_final_ood_test_high_confidence_fn.png", "Supplement: high-confidence FN", "supplement", "9A", "Final held-out center 2", "Total FN unchanged.", "existing"),
        ("D1", "document", "docs/FINAL_MANUSCRIPT_DRAFT.md", "Primary manuscript", "main", "9B", "Complete venue-neutral manuscript", "Citation markers require later literature verification.", "created"),
        ("D2", "document", "docs/FINAL_TITLE_ABSTRACT_KEYWORDS.md", "Front matter", "main", "9B", "Title, abstract, keywords", "No title claims GroupDRO superiority.", "created"),
        ("D3", "document", "docs/FINAL_RESULTS_INTERPRETATION.md", "Interpretation", "supplement", "9B", "Exact scientific narrative", "Negative model result and positive protocol contribution separated.", "created"),
        ("D4", "document", "docs/FINAL_DISCUSSION_AND_LIMITATIONS.md", "Discussion and limitations", "main", "9B", "Evidence-supported and hypothetical explanations", "No clinical-readiness or all-hospital claim.", "created"),
        ("D5", "document", "docs/FINAL_CLAIMS_AND_LIMITATIONS.md", "Claims governance", "supplement", "9B", "Post-test claims map", "Machine-readable counterpart T6.", "created"),
        ("D6", "document", "docs/LITERATURE_CITATION_GAPS.md", "Citation checklist", "supplement", "9B", "Unverified external literature needs", "No invented bibliography.", "created"),
        ("D7", "document", "docs/SUBMISSION_PACKAGE_README.md", "Submission handoff", "main", "9B", "Package inventory and remaining work", "Not yet journal-formatted.", "created"),
        ("D8", "document", "docs/FINAL_REPRODUCIBILITY_CHECKLIST.md", "Reproducibility", "supplement", "9B", "One-shot provenance", "Second run prohibited.", "created"),
        ("D9", "document", "docs/FINAL_MANUSCRIPT_CONSISTENCY_AUDIT.md", "Final QA", "supplement", "9B", "Source-to-claim audit", "Must pass before handoff.", "created"),
    ]
    return pd.DataFrame(rows, columns=[
        "item_id", "type", "artifact_path", "manuscript_section", "main_or_supplement",
        "source_milestone", "scope", "caption_caveat", "status",
    ])


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.edgecolor": "#333333",
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#D1D5DB",
        "grid.alpha": 0.55,
        "legend.frameon": False,
    })


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    assert path.is_file() and path.stat().st_size > 0


def plot_development_vs_final(table: pd.DataFrame) -> None:
    data = table[table["split"].isin(["ood_val", "ood_test"])].copy()
    stages = [("ood_val", "Development center 1"), ("ood_test", "Final held-out center 2")]
    metrics = [("auroc", "AUROC"), ("auprc", "AUPRC")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    x = np.arange(len(stages))
    width = 0.34
    for ax, (metric, label) in zip(axes, metrics):
        for j, model in enumerate(["groupdro", "centerstrat_erm"]):
            vals = [
                float(data[(data.split == split_name) & (data.model == model)][metric].iloc[0])
                for split_name, _ in stages
            ]
            bars = ax.bar(
                x + (j - 0.5) * width, vals, width,
                color=COLORS[model], edgecolor="#333333", linewidth=0.7,
                label=MODEL_LABELS[model],
            )
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=3, fontsize=9)
        ax.set_title(label)
        ax.set_xticks(x, [s[1] for s in stages])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(label)
        ax.grid(axis="y")
    axes[0].legend(loc="lower left")
    fig.suptitle("Development center 1 versus final held-out center 2")
    fig.text(
        0.5, 0.01,
        "Development ood_val was used for model assessment; center 2 was reserved for one authorized final run.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, FIGURE_PATHS["development_vs_final"])


def plot_default_metrics(sources: dict[str, object]) -> None:
    final = sources["final_classification"]
    assert isinstance(final, pd.DataFrame)
    final = final[final.variant == "raw"].set_index("model")
    metrics = ["accuracy", "sensitivity", "specificity", "precision", "f1"]
    labels = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1"]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for j, model in enumerate(["groupdro", "centerstrat_erm"]):
        vals = [float(final.loc[model, m]) for m in metrics]
        bars = ax.bar(
            x + (j - 0.5) * width, vals, width,
            color=COLORS[model], edgecolor="#333333", linewidth=0.7,
            label=MODEL_LABELS[model],
        )
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=3, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Final held-out center-2 metrics at the predeclared 0.5 threshold")
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    fig.text(
        0.5, 0.01,
        "The 0.5 threshold is a reporting default, not a clinically validated operating threshold.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, FIGURE_PATHS["default_metrics"])


def plot_calibration(calibration: pd.DataFrame) -> None:
    metrics = [("ece", "ECE"), ("brier", "Brier score"), ("nll", "NLL")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    x = np.arange(2)
    width = 0.34
    models = ["groupdro", "centerstrat_erm"]
    for ax, (metric, label) in zip(axes, metrics):
        for j, variant in enumerate(["raw", "calibrated"]):
            vals = [
                float(calibration[(calibration.model == model) & (calibration.variant == variant)][metric].iloc[0])
                for model in models
            ]
            bars = ax.bar(
                x + (j - 0.5) * width, vals, width,
                color=COLORS[variant], edgecolor="#333333", linewidth=0.7,
                label=variant.capitalize(),
            )
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=3, fontsize=8)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in models])
        ax.set_ylim(bottom=0)
        ax.set_title(label)
        ax.grid(axis="y")
    axes[0].legend(loc="upper right")
    fig.suptitle("Final held-out center-2 reliability: raw versus frozen-temperature calibrated")
    fig.text(
        0.5, 0.01,
        "Temperatures were frozen before test access and were not refit; hard predictions and total false negatives were unchanged.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, FIGURE_PATHS["calibration"])


def plot_operating_points(ops: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharex=False, sharey=False)
    panels = [
        ("specificity", "Fixed-specificity candidates", "Achieved final specificity"),
        ("sensitivity", "Fixed-sensitivity candidates", "Achieved final sensitivity"),
    ]
    for ax, (target_type, title, ylabel) in zip(axes, panels):
        subset = ops[ops.target_type == target_type]
        for model in ["groupdro", "centerstrat_erm"]:
            m = subset[subset.model == model].sort_values("nominal_development_target")
            y_col = "final_test_specificity" if target_type == "specificity" else "final_test_sensitivity"
            ax.plot(
                m["nominal_development_target"], m[y_col],
                marker="o", linewidth=2, color=COLORS[model], label=MODEL_LABELS[model],
            )
        low = subset["nominal_development_target"].min() - 0.03
        ax.plot([low, 1.0], [low, 1.0], color="#333333", linestyle="--", linewidth=1.2, label="Nominal preserved")
        ax.set_xlim(low, 0.98)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(f"Nominal development {target_type} target")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
    axes[0].legend(loc="lower right")
    fig.suptitle("Candidate/non-clinical operating-point transfer to final held-out center 2")
    fig.text(
        0.5, 0.01,
        "Thresholds were selected on id_val only. Deviation from the diagonal indicates target instability under hospital shift.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, FIGURE_PATHS["operating_points"])


def plot_high_confidence_fn(hc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
    thresholds = [0.90, 0.95, 0.99]
    x = np.arange(len(thresholds))
    width = 0.36
    for ax, model in zip(axes, ["groupdro", "centerstrat_erm"]):
        m = hc[hc.model == model]
        for j, variant in enumerate(["raw", "calibrated"]):
            vals = [
                int(m[(m.variant == variant) & np.isclose(m.confidence_threshold, t)]["high_confidence_fn"].iloc[0])
                for t in thresholds
            ]
            bars = ax.bar(
                x + (j - 0.5) * width, vals, width,
                color=COLORS[variant], edgecolor="#333333", linewidth=0.7,
                label=variant.capitalize(),
            )
            ax.bar_label(bars, labels=[f"{v:,}" for v in vals], padding=3, fontsize=8)
        total_fn = int(m["total_fn"].iloc[0])
        ax.set_xticks(x, [f"≥{t:.2f}" for t in thresholds])
        ax.set_title(f"{MODEL_LABELS[model]} (total FN = {total_fn:,})")
        ax.set_xlabel("Confidence threshold")
        ax.grid(axis="y")
    axes[0].set_ylabel("High-confidence false negatives")
    axes[0].legend(loc="upper right")
    fig.suptitle("Final held-out center-2 high-confidence false negatives")
    fig.text(
        0.5, 0.01,
        "Frozen calibration reduced confidently stated misses but did not change the underlying total false-negative count.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, FIGURE_PATHS["high_confidence_fn"])


def plot_reversal(table: pd.DataFrame) -> None:
    data = table[table["split"].isin(["ood_val", "ood_test"])].copy()
    stages = [("ood_val", "Development center 1"), ("ood_test", "Final held-out center 2")]
    values = {
        model: [
            float(data[(data.split == split_name) & (data.model == model)]["auroc"].iloc[0])
            for split_name, _ in stages
        ]
        for model in ["groupdro", "centerstrat_erm"]
    }
    deltas = np.array(values["groupdro"]) - np.array(values["centerstrat_erm"])

    fig = plt.figure(figsize=(9.5, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.35], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(2)
    width = 0.34
    for j, model in enumerate(["groupdro", "centerstrat_erm"]):
        bars = ax1.bar(
            x + (j - 0.5) * width, values[model], width,
            color=COLORS[model], edgecolor="#333333", linewidth=0.7,
            label=MODEL_LABELS[model],
        )
        ax1.bar_label(bars, labels=[f"{v:.4f}" for v in values[model]], padding=3)
    ax1.set_xticks(x, [s[1] for s in stages])
    ax1.set_ylim(0, 1.03)
    ax1.set_ylabel("AUROC")
    ax1.set_title("AUROC by evaluation stage")
    ax1.grid(axis="y")
    ax1.legend(loc="lower left")

    ax2 = fig.add_subplot(gs[1])
    colors = ["#C44E52" if d < 0 else "#4C78A8" for d in deltas]
    bars = ax2.bar(x, deltas, width=0.55, color=colors, edgecolor="#333333", linewidth=0.7)
    ax2.axhline(0, color="#333333", linewidth=1.2)
    ax2.bar_label(bars, labels=[f"{d:+.4f}" for d in deltas], padding=4)
    limit = max(abs(deltas)) * 1.55
    ax2.set_ylim(-limit, limit)
    ax2.set_xticks(x, [s[1] for s in stages])
    ax2.set_ylabel("GroupDRO − ERM AUROC")
    ax2.set_title("Controlled AUROC difference: advantage reversed on the reserved center")
    ax2.grid(axis="y")

    fig.suptitle("Development-to-final reversal: development center 1 versus final held-out center 2")
    fig.text(
        0.5, 0.01,
        "GroupDRO remains the predeclared primary and ERM the matched control; the result does not imply universal ERM superiority.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.13, hspace=0.42)
    save_figure(fig, FIGURE_PATHS["reversal"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    assert path.is_file() and path.stat().st_size > 0


def validate_generated(
    sources: dict[str, object],
    comparison: pd.DataFrame,
    final_model: pd.DataFrame,
    calibration: pd.DataFrame,
    ops: pd.DataFrame,
    hc: pd.DataFrame,
    claims: pd.DataFrame,
    manifest: pd.DataFrame,
) -> list[str]:
    checks: list[str] = []
    final = sources["final_classification"]
    dev = sources["dev_metrics"]
    assert isinstance(final, pd.DataFrame) and isinstance(dev, pd.DataFrame)
    final_raw = final[final.variant == "raw"].set_index("model")
    dev_ood = dev[dev.split == "ood_val"].set_index("model")

    assert np.isclose(dev_ood.loc["groupdro", "auroc"], 0.895609, atol=1e-6)
    assert np.isclose(dev_ood.loc["centerstrat_erm", "auroc"], 0.867271, atol=1e-6)
    assert np.isclose(final_raw.loc["groupdro", "auroc"], 0.6633704256200204)
    assert np.isclose(final_raw.loc["centerstrat_erm", "auroc"], 0.6984352121958427)
    assert final_raw.loc["groupdro", "fn"] == 37825
    assert final_raw.loc["centerstrat_erm", "fn"] == 32275
    checks.append("exact development and final headline metrics verified")

    delta = (
        final_raw.loc["groupdro", "auroc"] -
        final_raw.loc["centerstrat_erm", "auroc"]
    )
    assert delta < 0
    assert (
        dev_ood.loc["groupdro", "auroc"] -
        dev_ood.loc["centerstrat_erm", "auroc"]
    ) > 0
    checks.append("development-to-test AUROC reversal verified")

    assert len(comparison) == 6
    assert len(final_model) == 24
    assert len(calibration) == 4
    assert len(ops) == 14
    assert len(hc) == 12
    assert {"allowed", "conditional", "forbidden"} <= set(claims.status)
    assert len(manifest) >= 20
    checks.append("all required table row counts and statuses verified")

    assert all(calibration[calibration.variant == "calibrated"]["hard_predictions_changed"] == False)  # noqa: E712
    assert all(ops["candidate_nonclinical_note"].str.contains("non-clinical", case=False))
    assert all(ops["candidate_nonclinical_note"].str.contains("no test-set selection", case=False))
    checks.append("calibration and operating-point caveats verified")

    docs = [
        "docs/FINAL_MANUSCRIPT_DRAFT.md",
        "docs/FINAL_TITLE_ABSTRACT_KEYWORDS.md",
        "docs/FINAL_RESULTS_INTERPRETATION.md",
        "docs/FINAL_DISCUSSION_AND_LIMITATIONS.md",
        "docs/FINAL_CLAIMS_AND_LIMITATIONS.md",
        "docs/LITERATURE_CITATION_GAPS.md",
        "docs/SUBMISSION_PACKAGE_README.md",
        "docs/FINAL_REPRODUCIBILITY_CHECKLIST.md",
        "docs/FINAL_MANUSCRIPT_CONSISTENCY_AUDIT.md",
    ]
    for relative in docs:
        path = ROOT / relative
        assert path.is_file() and path.stat().st_size > 0, relative
    checks.append("required manuscript and submission-package documents present")

    manuscript = (ROOT / "docs" / "FINAL_MANUSCRIPT_DRAFT.md").read_text(encoding="utf-8")
    required_phrases = [
        "predeclared GroupDRO primary",
        "matched ERM control",
        "did not generalize to center 2",
        "candidate/non-clinical",
        "confidence correction is not error correction",
        "[CITATION NEEDED:",
    ]
    for phrase in required_phrases:
        assert phrase in manuscript, phrase
    forbidden_phrases = [
        "ready for clinical deployment",
        "robust to all hospitals",
        "GroupDRO is universally inferior",
        "ERM is universally superior",
    ]
    for phrase in forbidden_phrases:
        assert phrase.lower() not in manuscript.lower(), phrase
    checks.append("manuscript role labels, negative result, caveats, and citation markers verified")

    return checks


def write_report(checks: list[str]) -> None:
    lines = [
        "MILESTONE 9B FINAL MANUSCRIPT INTEGRATION REPORT",
        "=" * 80,
        "execution_mode=saved_artifacts_only",
        "dataset_loaded=false",
        "hf_test_accessed=false",
        "center_2_images_read=false",
        "model_inference=false",
        "training=false",
        "calibration_fitting=false",
        "threshold_tuning=false",
        "checkpoint_files_accessed=false",
        "script_38_rerun=false",
        "second_inference_attempt=false",
        "",
        "SOURCE VALIDATION",
    ]
    lines.extend(f"- PASS: {check}" for check in checks)
    lines.extend([
        "",
        "IMMUTABILITY",
        f"- immutable_exp09_files_verified={len(IMMUTABLE_HASHES)}",
        "- authorization_file_modified=false",
        "- exp09_predictions_modified=false",
        "- exp09_metrics_modified=false",
        "- exp09_sentinels_modified=false",
        "",
        "OUTPUTS",
    ])
    lines.extend(f"- {path.relative_to(ROOT)}" for path in TABLE_PATHS.values())
    lines.extend(f"- {path.relative_to(ROOT)}" for path in FIGURE_PATHS.values())
    lines.extend([
        "",
        "SCIENTIFIC VERDICT",
        "- Development center 1: GroupDRO AUROC exceeded matched ERM.",
        "- Final held-out center 2: matched ERM outperformed the predeclared GroupDRO primary on AUROC, AUPRC, accuracy, sensitivity, F1, and false-negative count.",
        "- GroupDRO retained higher specificity and slightly higher precision.",
        "- Frozen temperature scaling improved held-out ECE/Brier/NLL for both models without changing hard predictions or total false negatives.",
        "- Frozen candidate operating-point targets were not reliably preserved and remain explicitly non-clinical.",
        "",
        "OVERALL=PASS",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    assert_output_scope()
    assert_sources_immutable()
    sources = load_sources()
    validate_run_provenance(sources)

    comparison = create_development_vs_final(sources)
    final_model = create_final_model_comparison(sources)
    calibration = create_calibration_summary(sources)
    ops = create_operating_point_transfer(sources)
    hc = create_high_confidence_fn_summary(sources)
    claims = create_claims_audit()
    manifest = create_manifest()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for key, df in [
        ("development_vs_final", comparison),
        ("final_model_comparison", final_model),
        ("calibration", calibration),
        ("operating_points", ops),
        ("high_confidence_fn", hc),
        ("claims", claims),
        ("manifest", manifest),
    ]:
        write_csv(df, TABLE_PATHS[key])

    set_style()
    plot_development_vs_final(comparison)
    plot_default_metrics(sources)
    plot_calibration(calibration)
    plot_operating_points(ops)
    plot_high_confidence_fn(hc)
    plot_reversal(comparison)

    checks = validate_generated(
        sources, comparison, final_model, calibration, ops, hc, claims, manifest
    )
    assert_sources_immutable()
    checks.append("all exp09 authorization/prediction/metric/sentinel/figure hashes unchanged after generation")
    write_report(checks)

    print("Milestone 9B saved-artifact manuscript package generation: PASS")
    print("dataset_loaded=false")
    print("hf_test_accessed=false")
    print("model_inference=false")
    print("second_inference_attempt=false")
    print(f"tables_created={len(TABLE_PATHS)}")
    print(f"figures_created={len(FIGURE_PATHS)}")
    print(f"report={REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
