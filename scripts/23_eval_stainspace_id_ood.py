"""Milestone 7E / Script 23: evaluate stain-space aug on id_val + ood_val; compare 4 models.

Loads the best **stain-space** (OD/HED-style) checkpoint (script 22), runs inference over the
cached balanced ``id_val`` and ``ood_val`` subsets (loaded from disk — no HF streaming), writes
per-example prediction CSVs + classification metrics, and then compares FOUR models on the
development splits:

    1. plain ERM ResNet18            (4A id_val + 5A raw ood_val prediction CSVs)
    2. 7A moderate stain-aug ERM     (7A id_val + ood_val prediction CSVs)
    3. 7D tuned stain-aug v2 ERM     (7D id_val + ood_val prediction CSVs)
    4. 7E stain-space aug ERM        (this run's freshly written prediction CSVs)

Comparison views (tumor = positive = 1, y_score = prob_1):
    A. default-threshold (0.50) metrics per model/split,
    B. threshold-free ranking (AUROC / AUPRC) per model/split,
    C. id_val-selected FIXED-SPECIFICITY operating points (0.80/0.85/0.90/0.95), applied to
       ood_val — a *candidate* (NOT committed) operating-point audit.

Strict scope (Milestone 7E eval):
  * ``ood_val`` is read ONLY here, AFTER training, for development-stage comparison (never for
    training or model selection). ``id_val`` is used to select CANDIDATE thresholds only.
  * ``ood_test`` is NEVER read. Eval transform only (no augmentation at eval).
  * No calibration (deferred), no committed operating threshold, no MC-dropout, no ensemble.
  * The three baseline input-CSV pairs are read-only and verified unmodified (mtime + size).

DEVELOPMENT comparison on capped balanced subsets — NOT final performance, NOT a superiority
claim. Inputs (must already exist):
  * results/checkpoints/exp07e_stainspace_resnet18/best.pt   (script 22)
  * results/cache/balanced_id_val_pc250_seed1337_shuf.pt
  * results/cache/balanced_ood_val_pc1000_seed1337_noshuf.pt
  * plain ERM + 7A + 7D prediction CSVs (see PATHS below)
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / non-interactive; no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from data.balanced_subset import BalancedSubsetDataset, load_or_build_balanced_examples
from data.transforms import eval_transform
from evaluation.metrics_classification import compute_binary_classification_metrics
from evaluation.metrics_operating_points import (
    METRIC_FIELDS,
    apply_operating_points,
    compute_metrics_at_threshold,
)
from evaluation.predict import predict_to_dataframe
from models.classifier import CamelyonClassifier
from training.checkpointing import load_checkpoint
from utils.io import ensure_dir, ensure_results_tree, load_yaml, project_root, write_csv, write_text
from utils.seed import set_seed

CONFIG_REL = "configs/experiments/exp07e_stainspace_resnet18.yaml"

# --- baseline prediction CSVs for the 4-model comparison (read-only) ---------- #
PLAIN_ID_VAL_CSV = "results/predictions/exp04a_dev_train_resnet18/id_val_cached_predictions.csv"
PLAIN_OOD_VAL_CSV = "results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_raw_predictions.csv"
STAIN7A_ID_VAL_CSV = "results/predictions/exp07a_stainaug_resnet18/id_val_predictions.csv"
STAIN7A_OOD_VAL_CSV = "results/predictions/exp07a_stainaug_resnet18/ood_val_predictions.csv"
STAIN7D_ID_VAL_CSV = "results/predictions/exp07d_stainaug_v2_resnet18/id_val_predictions.csv"
STAIN7D_OOD_VAL_CSV = "results/predictions/exp07d_stainaug_v2_resnet18/ood_val_predictions.csv"

COMPARISON_INPUTS = [
    PLAIN_ID_VAL_CSV, PLAIN_OOD_VAL_CSV,
    STAIN7A_ID_VAL_CSV, STAIN7A_OOD_VAL_CSV,
    STAIN7D_ID_VAL_CSV, STAIN7D_OOD_VAL_CSV,
]

# --- output tables / metrics / figures ---------------------------------------- #
OUT_DEFAULT_CSV = "results/tables/exp07e_default_threshold_comparison.csv"
OUT_THRESHOLDFREE_CSV = "results/tables/exp07e_threshold_free_comparison.csv"
OUT_FIXED_SPEC_CSV = "results/tables/exp07e_fixed_specificity_comparison.csv"
OUT_METRICS_JSON = "results/metrics/exp07e_stainspace_resnet18/comparison_metrics.json"
OUT_REPORT_TXT = "results/logs/exp07e_stainspace_resnet18_eval_report.txt"

OUT_FIG_ROC_OOD = "results/figures/exp07e_roc_plain_vs_7a_vs_7d_vs_7e_ood_val.png"
OUT_FIG_PR_OOD = "results/figures/exp07e_pr_plain_vs_7a_vs_7d_vs_7e_ood_val.png"
OUT_FIG_FIXED_SPEC = "results/figures/exp07e_fixed_specificity_sensitivity_comparison.png"

REQUIRED_COLUMNS = ["label", "prob_0", "prob_1", "pred_label", "correct"]
TARGET_SPECIFICITIES = [0.80, 0.85, 0.90, 0.95]
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 64
NUM_WORKERS = 0

_METRIC_FIELDS = [
    "n", "accuracy", "balanced_accuracy", "auroc", "auprc",
    "sensitivity", "specificity", "precision", "f1",
    "tn", "fp", "fn", "tp", "threshold", "positive_class",
]

_CACHE_RE = re.compile(r"^balanced_(?P<split>.+)_pc(?P<pc>\d+)_seed(?P<seed>\d+)_(?P<order>shuf|noshuf)\.pt$")

MODEL_KEYS = ("plain", "stain7a", "stain7d", "stain7e")
MODEL_LABEL = {
    "plain": "plain ERM (4A/5A)",
    "stain7a": "stain-aug 7A",
    "stain7d": "stain-aug v2 7D",
    "stain7e": "stain-space 7E",
}


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def _cache_params(cache_rel: str) -> tuple[str, int, int, bool]:
    name = Path(cache_rel).name
    m = _CACHE_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized balanced-cache filename: {name!r}")
    return m["split"], int(m["pc"]), int(m["seed"]), (m["order"] == "shuf")


def _load_cached_subset(cache_rel: str) -> tuple[list[dict], Path, int]:
    """Load a balanced subset strictly from its existing disk cache (no HF streaming)."""
    path = _resolve(cache_rel)
    if not path.exists():
        raise FileNotFoundError(f"required cache not found: {path}")
    split_name, per_class, seed, shuffle_collected = _cache_params(cache_rel)
    examples = load_or_build_balanced_examples(
        split_name, per_class=per_class, seed=seed,
        shuffle_collected=shuffle_collected, use_cache=True, verbose=True,
    )
    return examples, path, 2 * per_class


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "None"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _prob_checks(df, name) -> None:
    p0 = df["prob_0"].to_numpy(dtype=float)
    p1 = df["prob_1"].to_numpy(dtype=float)
    assert np.isfinite(p0).all(), f"{name}: non-finite prob_0"
    assert np.isfinite(p1).all(), f"{name}: non-finite prob_1"
    assert ((p0 >= 0) & (p0 <= 1)).all(), f"{name}: prob_0 out of [0,1]"
    assert ((p1 >= 0) & (p1 <= 1)).all(), f"{name}: prob_1 out of [0,1]"
    assert (np.abs(p0 + p1 - 1.0) < 1e-5).all(), f"{name}: prob_0+prob_1 != 1"


def main() -> int:
    root = project_root()
    cfg = load_yaml(root / CONFIG_REL)
    ensure_results_tree()

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    o_cfg = cfg["output"]
    run_name = o_cfg["run_name"]

    set_seed(int(cfg.get("training", {}).get("seed", 1337)))

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = _resolve(o_cfg["checkpoint_dir"]) / "best.pt"
    pred_dir = _resolve(o_cfg["prediction_dir"])
    id_val_pred_csv = pred_dir / "id_val_predictions.csv"
    ood_val_pred_csv = pred_dir / "ood_val_predictions.csv"
    metrics_json = _resolve(o_cfg["metrics_json"])
    metrics_csv = _resolve(o_cfg["metrics_csv"])
    report_path = _resolve(OUT_REPORT_TXT)

    emit("=" * 78)
    emit("Milestone 7E - stain-space aug eval + 4-model comparison (script 23)")
    emit(f"run_name={run_name}")
    emit(f"device={device}")
    emit(f"checkpoint={ckpt_path}")
    emit("scope: cached id_val + ood_val ONLY (eval transform). ood_val = post-training")
    emit("       development comparison (authorized). NO training, NO calibration, NO ood_test.")
    emit("=" * 78)

    ok = True

    try:
        # --- guard: no ood_test path anywhere ----------------------------------- #
        all_out_paths = COMPARISON_INPUTS + [
            OUT_DEFAULT_CSV, OUT_THRESHOLDFREE_CSV, OUT_FIXED_SPEC_CSV, OUT_METRICS_JSON,
            OUT_REPORT_TXT, OUT_FIG_ROC_OOD, OUT_FIG_PR_OOD, OUT_FIG_FIXED_SPEC,
            str(id_val_pred_csv), str(ood_val_pred_csv), str(metrics_json), str(metrics_csv),
        ]
        assert not any("ood_test" in str(p) for p in all_out_paths), "ood_test must never be referenced"

        assert ckpt_path.exists(), f"checkpoint not found (run script 22 first): {ckpt_path}"

        # --- load model from best checkpoint ------------------------------------ #
        model = CamelyonClassifier(
            backbone_name=m_cfg["backbone_name"],
            pretrained=bool(m_cfg["pretrained"]),
            num_classes=int(m_cfg["num_classes"]),
        )
        payload = load_checkpoint(ckpt_path, model, optimizer=None, map_location="cpu")
        model.to(device)
        emit("")
        emit(
            f"loaded checkpoint: epoch={payload.get('epoch')}  "
            f"metrics={ {k: payload['metrics'][k] for k in ('id_val_loss', 'id_val_accuracy') if k in payload.get('metrics', {})} }"
        )

        # --- cached subsets: id_val + ood_val (eval transform only) ------------- #
        id_val_examples, id_val_cache_path, expected_id_val_n = _load_cached_subset(d_cfg["id_val_cache"])
        ood_val_examples, ood_val_cache_path, expected_ood_val_n = _load_cached_subset(d_cfg["ood_val_cache"])

        id_val_ds = BalancedSubsetDataset(id_val_examples, transform=eval_transform)
        ood_val_ds = BalancedSubsetDataset(ood_val_examples, transform=eval_transform)

        id_val_loader = DataLoader(
            id_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            drop_last=False, pin_memory=False,
        )
        ood_val_loader = DataLoader(
            ood_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            drop_last=False, pin_memory=False,
        )

        emit("")
        emit(f"id_val  cache : {id_val_cache_path}  (n={len(id_val_ds)})")
        emit(f"ood_val cache : {ood_val_cache_path}  (n={len(ood_val_ds)})")

        # --- inference (7E) ----------------------------------------------------- #
        emit("")
        emit("--- running inference (stain-space aug / 7E) ---")
        id_val_df = predict_to_dataframe(model, id_val_loader, device, "id_val")
        ood_val_df = predict_to_dataframe(model, ood_val_loader, device, "ood_val")
        emit(f"id_val predictions : n={len(id_val_df)}")
        emit(f"ood_val predictions: n={len(ood_val_df)}")

        _prob_checks(id_val_df, "id_val")
        _prob_checks(ood_val_df, "ood_val")

        # --- save 7E prediction CSVs -------------------------------------------- #
        ensure_dir(pred_dir)
        id_val_df.to_csv(id_val_pred_csv, index=False)
        ood_val_df.to_csv(ood_val_pred_csv, index=False)

        # --- 7E classification metrics (tumor = positive, threshold 0.5) -------- #
        emit("")
        emit("--- 7E classification metrics (positive class = tumor = 1, threshold 0.5) ---")
        id_val_metrics = compute_binary_classification_metrics(
            id_val_df["label"].to_numpy(), id_val_df["prob_1"].to_numpy(), threshold=0.5
        )
        ood_val_metrics = compute_binary_classification_metrics(
            ood_val_df["label"].to_numpy(), ood_val_df["prob_1"].to_numpy(), threshold=0.5
        )

        def _emit_metrics(name, m) -> None:
            emit(f"  [{name}]")
            for k in _METRIC_FIELDS:
                emit(f"      {k:<18}: {_fmt(m[k]) if not isinstance(m[k], int) else m[k]}")
            for w in m["warnings"]:
                emit(f"      warning           : {w}")

        _emit_metrics("id_val (7E)", id_val_metrics)
        _emit_metrics("ood_val (7E)", ood_val_metrics)

        # --- save 7E metrics (JSON + flat CSV) ---------------------------------- #
        ensure_dir(metrics_json.parent)
        metrics_payload_7e = {
            "run_name": run_name,
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "threshold": 0.5,
            "positive_class": 1,
            "splits": {"id_val": id_val_metrics, "ood_val": ood_val_metrics},
            "note": "Stain-space (OD/HED-style) ERM baseline (Milestone 7E). DEVELOPMENT "
                    "comparison on capped balanced subsets. ood_val used post-training for "
                    "development comparison only. NOT a final result. ood_test never touched.",
        }
        write_text(metrics_json, json.dumps(metrics_payload_7e, indent=2) + "\n")

        csv_rows = []
        for split_name, m in (("id_val", id_val_metrics), ("ood_val", ood_val_metrics)):
            row = {"split": split_name}
            for k in _METRIC_FIELDS:
                row[k] = m[k]
            csv_rows.append(row)
        write_csv(metrics_csv, csv_rows, header=["split", *_METRIC_FIELDS])

        # ------------------------------------------------------------------ #
        # Load the three baseline models' prediction CSVs (read-only)        #
        # ------------------------------------------------------------------ #
        resolved_inputs = {name: _resolve(name) for name in COMPARISON_INPUTS}
        for name, path in resolved_inputs.items():
            assert path.exists(), f"required comparison input not found: {path}"
        pre_mtime = {name: resolved_inputs[name].stat().st_mtime_ns for name in COMPARISON_INPUTS}
        pre_size = {name: resolved_inputs[name].stat().st_size for name in COMPARISON_INPUTS}

        frames = {
            ("plain", "id_val"): pd.read_csv(resolved_inputs[PLAIN_ID_VAL_CSV]),
            ("plain", "ood_val"): pd.read_csv(resolved_inputs[PLAIN_OOD_VAL_CSV]),
            ("stain7a", "id_val"): pd.read_csv(resolved_inputs[STAIN7A_ID_VAL_CSV]),
            ("stain7a", "ood_val"): pd.read_csv(resolved_inputs[STAIN7A_OOD_VAL_CSV]),
            ("stain7d", "id_val"): pd.read_csv(resolved_inputs[STAIN7D_ID_VAL_CSV]),
            ("stain7d", "ood_val"): pd.read_csv(resolved_inputs[STAIN7D_OOD_VAL_CSV]),
            ("stain7e", "id_val"): id_val_df,
            ("stain7e", "ood_val"): ood_val_df,
        }

        expected_n = {"id_val": 500, "ood_val": 2000}
        for (mdl, split), df in frames.items():
            tag = f"{mdl}/{split}"
            for col in REQUIRED_COLUMNS:
                assert col in df.columns, f"{tag}: missing required column {col!r}"
            assert len(df) == expected_n[split], f"{tag}: expected n=={expected_n[split]}, got {len(df)}"
            labs = set(np.unique(df["label"].to_numpy()).tolist())
            assert labs.issubset({0, 1}), f"{tag}: labels outside {{0,1}}: {sorted(labs)}"
            _prob_checks(df, tag)

        # ------------------------------------------------------------------ #
        # A. Default-threshold (0.5) metrics for all 4 models                #
        # ------------------------------------------------------------------ #
        default_by_key: dict[tuple[str, str], dict] = {}
        default_rows: list[dict] = []
        for (mdl, split), df in frames.items():
            m = compute_metrics_at_threshold(df["label"].to_numpy(), df["prob_1"].to_numpy(), DEFAULT_THRESHOLD)
            default_by_key[(mdl, split)] = m
            row = {"model": mdl, "model_label": MODEL_LABEL[mdl], "split": split}
            for k in METRIC_FIELDS:
                row[k] = m[k]
            default_rows.append(row)

        emit("")
        emit("A. Default threshold (0.50) comparison  [tumor = positive]")
        emit(f"   {'model':<20}{'split':<8}{'acc':>8}{'sens':>8}{'spec':>8}{'FN':>6}{'FP':>6}")
        for mdl in MODEL_KEYS:
            for split in ("id_val", "ood_val"):
                m = default_by_key[(mdl, split)]
                emit(f"   {MODEL_LABEL[mdl]:<20}{split:<8}"
                     f"{_fmt(m['accuracy']):>8}{_fmt(m['sensitivity']):>8}"
                     f"{_fmt(m['specificity']):>8}{m['fn']:>6}{m['fp']:>6}")

        # ------------------------------------------------------------------ #
        # B. Threshold-free ranking metrics (AUROC / AUPRC) for all 4 models #
        # ------------------------------------------------------------------ #
        auc_by_key: dict[tuple[str, str], dict] = {}
        thresholdfree_rows: list[dict] = []
        for (mdl, split), df in frames.items():
            y = df["label"].to_numpy().astype(int)
            s = df["prob_1"].to_numpy(dtype=float)
            if len(set(y.tolist())) < 2:
                auroc = auprc = None
            else:
                auroc = float(roc_auc_score(y, s))
                auprc = float(average_precision_score(y, s))
            auc_by_key[(mdl, split)] = {"auroc": auroc, "auprc": auprc}
            thresholdfree_rows.append({
                "model": mdl, "model_label": MODEL_LABEL[mdl], "split": split,
                "n": int(len(df)), "auroc": auroc, "auprc": auprc,
            })

        emit("")
        emit("B. Threshold-free ranking metrics (AUROC / AUPRC)")
        emit(f"   {'model':<20}{'split':<8}{'AUROC':>9}{'AUPRC':>9}")
        for mdl in MODEL_KEYS:
            for split in ("id_val", "ood_val"):
                a = auc_by_key[(mdl, split)]
                emit(f"   {MODEL_LABEL[mdl]:<20}{split:<8}{_fmt(a['auroc']):>9}{_fmt(a['auprc']):>9}")

        # ------------------------------------------------------------------ #
        # C. id_val-selected FIXED-SPECIFICITY candidate operating points    #
        #    applied to ood_val, for all 4 models                            #
        # ------------------------------------------------------------------ #
        op_rows: dict[str, list[dict]] = {}
        for mdl in MODEL_KEYS:
            rows = apply_operating_points(
                frames[(mdl, "id_val")], frames[(mdl, "ood_val")],
                TARGET_SPECIFICITIES, [],  # fixed-specificity only in 7E
            )
            for r in rows:
                r["model"] = mdl
                r["model_label"] = MODEL_LABEL[mdl]
            op_rows[mdl] = rows

        def _op_lookup(mdl: str, target: float, split: str) -> dict | None:
            for r in op_rows[mdl]:
                if r["criterion"] == "specificity" and abs(r["target"] - target) < 1e-9 and r["split"] == split:
                    return r
            return None

        fixed_spec_rows: list[dict] = []
        for mdl in MODEL_KEYS:
            for r in op_rows[mdl]:
                if r["criterion"] == "specificity":
                    fixed_spec_rows.append(r)

        emit("")
        emit("C. id_val-selected FIXED-SPECIFICITY candidate operating points, applied to ood_val")
        emit("   (threshold chosen on id_val so id_val specificity >= target; same t on ood_val)")
        emit("   *** CANDIDATE points only - NO operating threshold is committed in 7E ***")
        emit(f"   {'target_spec':>11}  {'model':<20}{'thr':>8}{'ood_sens':>10}{'ood_spec':>10}{'ood_FN':>8}")
        for spec in TARGET_SPECIFICITIES:
            for mdl in MODEL_KEYS:
                oor = _op_lookup(mdl, spec, "ood_val")
                idr = _op_lookup(mdl, spec, "id_val")
                thr = idr["threshold"] if idr else None
                emit(f"   {spec:>11.2f}  {MODEL_LABEL[mdl]:<20}{_fmt(thr):>8}"
                     f"{_fmt(oor['sensitivity'] if oor else None):>10}"
                     f"{_fmt(oor['specificity'] if oor else None):>10}"
                     f"{(oor['fn'] if oor and oor['fn'] is not None else 'None'):>8}")

        # ------------------------------------------------------------------ #
        # Save comparison tables                                             #
        # ------------------------------------------------------------------ #
        op_field_order = [
            "model", "model_label", "criterion", "target", "split", "achieved_on_id_val",
            *METRIC_FIELDS,
        ]
        write_csv(_resolve(OUT_DEFAULT_CSV), default_rows,
                  header=["model", "model_label", "split", *METRIC_FIELDS])
        write_csv(_resolve(OUT_THRESHOLDFREE_CSV), thresholdfree_rows,
                  header=["model", "model_label", "split", "n", "auroc", "auprc"])
        write_csv(_resolve(OUT_FIXED_SPEC_CSV), fixed_spec_rows, header=op_field_order)

        # ------------------------------------------------------------------ #
        # Figures (matplotlib only)                                          #
        # ------------------------------------------------------------------ #
        color_by_model = {
            "plain": "tab:blue", "stain7a": "tab:orange",
            "stain7d": "tab:green", "stain7e": "tab:red",
        }

        # ROC - ood_val (all 4 models)
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        for mdl in MODEL_KEYS:
            df = frames[(mdl, "ood_val")]
            y = df["label"].to_numpy().astype(int)
            s = df["prob_1"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=color_by_model[mdl],
                    label=f"{MODEL_LABEL[mdl]} (AUROC {_fmt(auc_by_key[(mdl, 'ood_val')]['auroc'])})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title("ROC - ood_val (development): plain vs 7A vs 7D vs 7E")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(_resolve(OUT_FIG_ROC_OOD), dpi=150)
        plt.close(fig)

        # PR - ood_val (all 4 models)
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        for mdl in MODEL_KEYS:
            df = frames[(mdl, "ood_val")]
            y = df["label"].to_numpy().astype(int)
            s = df["prob_1"].to_numpy(dtype=float)
            prec, rec, _ = precision_recall_curve(y, s)
            ax.plot(rec, prec, color=color_by_model[mdl],
                    label=f"{MODEL_LABEL[mdl]} (AUPRC {_fmt(auc_by_key[(mdl, 'ood_val')]['auprc'])})")
        ax.set_xlabel("Recall (sensitivity)")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall - ood_val (development): plain vs 7A vs 7D vs 7E")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(_resolve(OUT_FIG_PR_OOD), dpi=150)
        plt.close(fig)

        # Fixed-specificity ood_val sensitivity (grouped bars, 4 models)
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        x = np.arange(len(TARGET_SPECIFICITIES))
        width = 0.2
        for i, mdl in enumerate(MODEL_KEYS):
            vals = []
            for spec in TARGET_SPECIFICITIES:
                oor = _op_lookup(mdl, spec, "ood_val")
                v = oor["sensitivity"] if (oor and oor["sensitivity"] is not None) else np.nan
                vals.append(v)
            ax.bar(x + (i - 1.5) * width, vals, width, color=color_by_model[mdl], label=MODEL_LABEL[mdl])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.2f}" for s in TARGET_SPECIFICITIES])
        ax.set_xlabel("id_val-selected target specificity")
        ax.set_ylabel("ood_val sensitivity (candidate)")
        ax.set_title("Candidate fixed-specificity ood_val sensitivity\n(plain vs 7A vs 7D vs 7E; NOT committed)")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(_resolve(OUT_FIG_FIXED_SPEC), dpi=150)
        plt.close(fig)

        figure_paths = [OUT_FIG_ROC_OOD, OUT_FIG_PR_OOD, OUT_FIG_FIXED_SPEC]

        # ------------------------------------------------------------------ #
        # Comparison print-out (7E vs plain + 7A + 7D) + interpretation      #
        # ------------------------------------------------------------------ #
        emit("")
        emit("--- 7E vs plain ERM, 7A, and 7D (ood_val default-0.5 + ranking) ---")
        emit(f"   {'metric':<14}{'plain':>12}{'7A':>12}{'7D':>12}{'7E':>12}")
        _rows = [
            ("accuracy", "accuracy"), ("auroc", None), ("auprc", None),
            ("sensitivity", "sensitivity"), ("specificity", "specificity"), ("FN", "fn"),
        ]
        for label, mkey in _rows:
            def _val(mdl):
                if label == "auroc":
                    return _fmt(auc_by_key[(mdl, "ood_val")]["auroc"])
                if label == "auprc":
                    return _fmt(auc_by_key[(mdl, "ood_val")]["auprc"])
                v = default_by_key[(mdl, "ood_val")][mkey]
                return str(v) if label == "FN" else _fmt(v)
            emit(f"   {label:<14}{_val('plain'):>12}{_val('stain7a'):>12}"
                 f"{_val('stain7d'):>12}{_val('stain7e'):>12}")

        # interpretation signals (development-stage; capped subsets)
        plain_ood_auroc = auc_by_key[("plain", "ood_val")]["auroc"]
        s7a_ood_auroc = auc_by_key[("stain7a", "ood_val")]["auroc"]
        s7d_ood_auroc = auc_by_key[("stain7d", "ood_val")]["auroc"]
        s7e_ood_auroc = auc_by_key[("stain7e", "ood_val")]["auroc"]

        def _delta(a, b):
            return (a - b) if (a is not None and b is not None) else None

        d_auroc_7e_vs_plain = _delta(s7e_ood_auroc, plain_ood_auroc)
        d_auroc_7e_vs_7a = _delta(s7e_ood_auroc, s7a_ood_auroc)
        d_auroc_7e_vs_7d = _delta(s7e_ood_auroc, s7d_ood_auroc)

        plain_def_sens = default_by_key[("plain", "ood_val")]["sensitivity"]
        s7a_def_sens = default_by_key[("stain7a", "ood_val")]["sensitivity"]
        s7d_def_sens = default_by_key[("stain7d", "ood_val")]["sensitivity"]
        s7e_def_sens = default_by_key[("stain7e", "ood_val")]["sensitivity"]
        d_def_sens_7e_vs_7d = _delta(s7e_def_sens, s7d_def_sens)

        # fixed-specificity sensitivity: 7E vs each baseline, count wins across targets
        def _fixedspec_wins(other_key: str) -> tuple[int, int]:
            flags = []
            for spec in TARGET_SPECIFICITIES:
                o = _op_lookup(other_key, spec, "ood_val")
                e = _op_lookup("stain7e", spec, "ood_val")
                ov = o["sensitivity"] if (o and o["sensitivity"] is not None) else None
                ev = e["sensitivity"] if (e and e["sensitivity"] is not None) else None
                if ov is not None and ev is not None:
                    flags.append(ev >= ov)
            return sum(1 for x in flags if x), len(flags)

        n_7e_ge_plain, n_fixedspec = _fixedspec_wins("plain")
        n_7e_ge_7a, _ = _fixedspec_wins("stain7a")
        n_7e_ge_7d, _ = _fixedspec_wins("stain7d")

        ranking_improved_vs_plain = (d_auroc_7e_vs_plain is not None and d_auroc_7e_vs_plain > 0.0)
        ranking_improved_vs_7a = (d_auroc_7e_vs_7a is not None and d_auroc_7e_vs_7a > 0.0)
        ranking_improved_vs_7d = (d_auroc_7e_vs_7d is not None and d_auroc_7e_vs_7d > 0.0)
        beats_plain_ranking = (d_auroc_7e_vs_plain is not None and d_auroc_7e_vs_plain >= 0.01)
        fixedspec_improved_vs_7d = (n_fixedspec > 0 and n_7e_ge_7d > n_fixedspec / 2)
        def_sens_improved_vs_7d = (d_def_sens_7e_vs_7d is not None and d_def_sens_7e_vs_7d > 0.0)

        emit("")
        emit("INTERPRETATION (development-stage; capped balanced subsets; NOT final performance)")
        emit("-" * 66)
        emit(f"* ood_val AUROC: plain {_fmt(plain_ood_auroc)}  7A {_fmt(s7a_ood_auroc)}  "
             f"7D {_fmt(s7d_ood_auroc)}  7E {_fmt(s7e_ood_auroc)}")
        emit(f"    7E - plain = {_fmt(d_auroc_7e_vs_plain)} ; 7E - 7A = {_fmt(d_auroc_7e_vs_7a)} ; "
             f"7E - 7D = {_fmt(d_auroc_7e_vs_7d)}")
        emit(f"* 7E {'IMPROVED' if ranking_improved_vs_plain else 'did NOT improve'} ood_val ranking (AUROC) over plain ERM.")
        emit(f"* 7E {'IMPROVED' if ranking_improved_vs_7a else 'did NOT improve'} ood_val ranking (AUROC) over 7A.")
        emit(f"* 7E {'IMPROVED' if ranking_improved_vs_7d else 'did NOT improve'} ood_val ranking (AUROC) over 7D.")
        emit(f"* Default-0.5 ood_val sensitivity: plain {_fmt(plain_def_sens)}  7A {_fmt(s7a_def_sens)}  "
             f"7D {_fmt(s7d_def_sens)}  7E {_fmt(s7e_def_sens)}")
        emit(f"* Matched fixed-specificity ood_val sensitivity wins (7E >= baseline): "
             f"vs plain {n_7e_ge_plain}/{n_fixedspec}; vs 7A {n_7e_ge_7a}/{n_fixedspec}; "
             f"vs 7D {n_7e_ge_7d}/{n_fixedspec}.")

        # overall recommendation
        if beats_plain_ranking and (n_fixedspec > 0 and n_7e_ge_plain > n_fixedspec / 2):
            recommendation = (
                "RETAIN / PROMOTE (development): stain-space (OD/HED-style) augmentation improved "
                "OOD ranking over plain ERM and matched-or-beat it at most fixed-specificity "
                "operating points on this capped subset. Natural next step: calibration + "
                "candidate operating-point audit (still development-stage)."
            )
            verdict = "retain_promote"
        elif (ranking_improved_vs_plain or ranking_improved_vs_7a or ranking_improved_vs_7d
              or fixedspec_improved_vs_7d or def_sens_improved_vs_7d):
            recommendation = (
                "TUNE FURTHER (development): stain-space augmentation improved on at least one "
                "torchvision baseline (7A/7D) or on plain ERM in at least one view, but does NOT "
                "clearly beat plain ERM ranking by >= 0.01 AUROC and/or at most fixed-specificity "
                "targets. Keep as the pathology-specific stain-robust comparator; consider tuning "
                "the OD ranges or a dedicated DG objective (GroupDRO/CORAL/DANN) next -- gated. Do "
                "NOT claim superiority over plain ERM."
            )
            verdict = "tune_further"
        else:
            recommendation = (
                "MIXED / DO NOT PROMOTE (development): stain-space (OD/HED-style) augmentation did "
                "not improve OOD ranking over plain ERM and did not clearly beat the torchvision "
                "baselines (7A/7D) on this capped subset. Stain augmentation (colour-jitter 7A/7D "
                "and OD-space 7E) remains an insufficient stain-robustness lever here; a dedicated "
                "DG method (GroupDRO/CORAL/DANN) is the next candidate -- gated. Do NOT claim "
                "superiority over plain ERM."
            )
            verdict = "mixed_do_not_promote"

        emit("")
        emit(f"RECOMMENDATION: {recommendation}")
        emit("(Development-stage comparison only - NOT a final-performance or superiority claim; "
             "no operating point committed; no calibration in 7E.)")

        # ------------------------------------------------------------------ #
        # Comparison metrics JSON                                            #
        # ------------------------------------------------------------------ #
        def _op_json(mdl: str) -> list[dict]:
            return [
                {k: r[k] for k in ["criterion", "target", "split", "achieved_on_id_val", *METRIC_FIELDS]}
                for r in op_rows[mdl]
            ]

        comparison_payload = {
            "milestone": "7E",
            "description": "Stain-space (OD/HED-style) augmentation ERM baseline: eval + 4-model "
                           "comparison (plain ERM vs 7A vs 7D vs 7E). id_val selects candidate "
                           "thresholds; ood_val is development comparison only.",
            "positive_class": 1,
            "default_threshold": DEFAULT_THRESHOLD,
            "target_specificities": TARGET_SPECIFICITIES,
            "models": {k: MODEL_LABEL[k] for k in MODEL_KEYS},
            "comparison_inputs": {name: str(resolved_inputs[name]) for name in COMPARISON_INPUTS},
            "default_threshold_metrics": {
                f"{mdl}_{split}": {k: default_by_key[(mdl, split)][k] for k in ["n", *METRIC_FIELDS]}
                for mdl in MODEL_KEYS for split in ("id_val", "ood_val")
            },
            "threshold_free": {
                f"{mdl}_{split}": auc_by_key[(mdl, split)]
                for mdl in MODEL_KEYS for split in ("id_val", "ood_val")
            },
            "fixed_specificity_operating_points": {mdl: _op_json(mdl) for mdl in MODEL_KEYS},
            "interpretation": {
                "ood_val_auroc": {
                    "plain": plain_ood_auroc, "stain7a": s7a_ood_auroc,
                    "stain7d": s7d_ood_auroc, "stain7e": s7e_ood_auroc,
                },
                "delta_auroc_7e_vs_plain": d_auroc_7e_vs_plain,
                "delta_auroc_7e_vs_7a": d_auroc_7e_vs_7a,
                "delta_auroc_7e_vs_7d": d_auroc_7e_vs_7d,
                "default_sensitivity": {
                    "plain": plain_def_sens, "stain7a": s7a_def_sens,
                    "stain7d": s7d_def_sens, "stain7e": s7e_def_sens,
                },
                "delta_default_sensitivity_7e_vs_7d": d_def_sens_7e_vs_7d,
                "fixedspec_7e_ge_plain": f"{n_7e_ge_plain}/{n_fixedspec}",
                "fixedspec_7e_ge_7a": f"{n_7e_ge_7a}/{n_fixedspec}",
                "fixedspec_7e_ge_7d": f"{n_7e_ge_7d}/{n_fixedspec}",
                "ranking_improved_vs_plain": bool(ranking_improved_vs_plain),
                "ranking_improved_vs_7a": bool(ranking_improved_vs_7a),
                "ranking_improved_vs_7d": bool(ranking_improved_vs_7d),
                "verdict": verdict,
                "recommendation": recommendation,
            },
            "note": "DEVELOPMENT comparison on capped balanced subsets - NOT final performance. "
                    "ood_val used for development-stage comparison only after id_val threshold "
                    "selection. No calibration, no committed operating point. ood_test never read.",
        }
        comp_json_path = _resolve(OUT_METRICS_JSON)
        ensure_dir(comp_json_path.parent)
        write_text(comp_json_path, json.dumps(comparison_payload, indent=2) + "\n")

        # ------------------------------------------------------------------ #
        # Verify baseline input CSVs were NOT modified                       #
        # ------------------------------------------------------------------ #
        for name in COMPARISON_INPUTS:
            post = resolved_inputs[name].stat()
            assert post.st_mtime_ns == pre_mtime[name], f"comparison input modified (mtime): {name}"
            assert post.st_size == pre_size[name], f"comparison input modified (size): {name}"

        # ------------------------------------------------------------------ #
        # Assertions                                                         #
        # ------------------------------------------------------------------ #
        assert len(id_val_df) == expected_id_val_n == 500, f"id_val n {len(id_val_df)} != 500"
        assert len(ood_val_df) == expected_ood_val_n == 2000, f"ood_val n {len(ood_val_df)} != 2000"
        assert id_val_pred_csv.exists(), "7E id_val prediction CSV missing"
        assert ood_val_pred_csv.exists(), "7E ood_val prediction CSV missing"
        assert metrics_json.exists(), "7E metrics JSON missing"
        assert metrics_csv.exists(), "7E metrics CSV missing"
        for p in (OUT_DEFAULT_CSV, OUT_THRESHOLDFREE_CSV, OUT_FIXED_SPEC_CSV):
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"table missing/empty: {p}"
        for p in figure_paths:
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"figure missing/empty: {p}"
        assert comp_json_path.exists() and comp_json_path.stat().st_size > 0, "comparison JSON missing/empty"

        # ------------------------------------------------------------------ #
        # Output paths summary                                               #
        # ------------------------------------------------------------------ #
        emit("")
        emit("OUTPUT PATHS")
        emit("-" * 60)
        emit("predictions:")
        emit(f"  {id_val_pred_csv}")
        emit(f"  {ood_val_pred_csv}")
        emit("metrics:")
        emit(f"  {metrics_json}")
        emit(f"  {metrics_csv}")
        emit(f"  {OUT_METRICS_JSON}")
        emit("tables:")
        for p in (OUT_DEFAULT_CSV, OUT_THRESHOLDFREE_CSV, OUT_FIXED_SPEC_CSV):
            emit(f"  {p}")
        emit("figures:")
        for p in figure_paths:
            emit(f"  {p}")
        emit(f"report      : {OUT_REPORT_TXT}")
        emit("ood_test usage: NONE")

    except Exception as exc:
        ok = False
        emit("")
        emit(f"ERROR: {type(exc).__name__}: {exc}")

    emit("")
    emit("=" * 78)
    emit(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    emit("=" * 78)

    write_text(report_path, "\n".join(lines) + "\n")
    print(f"\nSaved report -> {report_path}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
