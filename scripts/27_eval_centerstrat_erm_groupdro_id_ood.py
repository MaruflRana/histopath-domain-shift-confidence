"""Milestone 7F / Script 27: evaluate matched center-stratified ERM + GroupDRO on id_val + ood_val.

Loads the two 7F checkpoints trained by script 26 on the NEW center-stratified caches:

    * exp07f_centerstrat_erm_resnet18  (matched ERM control)
    * exp07f_groupdro_resnet18         (GroupDRO by center)

and evaluates BOTH on the center-stratified ``id_val`` cache and the balanced ``ood_val``
cache (eval transform only). Writes per-example prediction CSVs, overall classification
metrics, per-center id_val metrics, id_val-selected fixed-specificity candidate operating
points on ood_val, and a comparison against the older models.

FAIR comparison  = 7F center-strat ERM vs 7F GroupDRO   (both trained on the SAME new
                   center-stratified train cache).
REFERENCE compare = old plain ERM (4A/5A) / 7A / 7D / 7E — these were trained on the OLD
                   confounded balanced cache and are NOT matched to the new center-stratified
                   data; shown for context only, clearly labeled as such.

Strict scope (Milestone 7F eval):
  * ``ood_val`` read ONLY here, AFTER training, for development-stage comparison (never for
    training / model selection). ``id_val`` selects CANDIDATE thresholds only.
  * ``ood_test`` is NEVER read. Eval transform only. No calibration, no committed operating
    threshold, no MC-dropout, no ensemble, no CORAL/DANN.
  * The reference prediction CSVs are read-only and verified unmodified (mtime + size).

DEVELOPMENT comparison on capped balanced subsets — NOT final performance, NOT a superiority
claim.
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
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

from data.balanced_subset import BalancedSubsetDataset, _deserialize_examples
from data.transforms import eval_transform
from evaluation.metrics_classification import compute_binary_classification_metrics
from evaluation.metrics_operating_points import METRIC_FIELDS, apply_operating_points, compute_metrics_at_threshold
from evaluation.predict import predict_to_dataframe
from models.classifier import CamelyonClassifier
from training.checkpointing import load_checkpoint
from utils.io import ensure_dir, ensure_results_tree, load_yaml, project_root, write_csv, write_text
from utils.seed import set_seed

CONFIG_REL = "configs/experiments/exp07f_groupdro_resnet18.yaml"

# --- reference prediction CSVs (context only; trained on OLD confounded caches) ------ #
PLAIN_ID_VAL_CSV = "results/predictions/exp04a_dev_train_resnet18/id_val_cached_predictions.csv"
PLAIN_OOD_VAL_CSV = "results/predictions/exp05a_calibrate_ood_val_resnet18/ood_val_raw_predictions.csv"
STAIN7A_OOD_VAL_CSV = "results/predictions/exp07a_stainaug_resnet18/ood_val_predictions.csv"
STAIN7D_OOD_VAL_CSV = "results/predictions/exp07d_stainaug_v2_resnet18/ood_val_predictions.csv"
STAIN7E_OOD_VAL_CSV = "results/predictions/exp07e_stainspace_resnet18/ood_val_predictions.csv"

# reference ood_val CSVs used for the ood_val ranking/operating-point context table.
REF_OOD_CSVS = {
    "plain_old": PLAIN_OOD_VAL_CSV,
    "stain7a_old": STAIN7A_OOD_VAL_CSV,
    "stain7d_old": STAIN7D_OOD_VAL_CSV,
    "stain7e_old": STAIN7E_OOD_VAL_CSV,
}
# reference id_val CSV that has prob_1 for threshold selection (plain ERM only, 4A).
REF_ID_CSVS = {"plain_old": PLAIN_ID_VAL_CSV}

READONLY_INPUTS = [PLAIN_ID_VAL_CSV, PLAIN_OOD_VAL_CSV,
                   STAIN7A_OOD_VAL_CSV, STAIN7D_OOD_VAL_CSV, STAIN7E_OOD_VAL_CSV]

REF_LABEL = {
    "plain_old": "old plain ERM (4A/5A) [old cache]",
    "stain7a_old": "old 7A stain-aug [old cache]",
    "stain7d_old": "old 7D stain-aug v2 [old cache]",
    "stain7e_old": "old 7E stain-space [old cache]",
    "erm7f": "7F center-strat ERM [NEW]",
    "groupdro7f": "7F GroupDRO [NEW]",
}

# --- output tables / metrics / figures ---------------------------------------------- #
OUT_OVERALL_CSV = "results/tables/exp07f_overall_metrics_comparison.csv"
OUT_PERCENTER_CSV = "results/tables/exp07f_id_val_per_center_metrics.csv"
OUT_OOD_CSV = "results/tables/exp07f_ood_val_metrics_comparison.csv"
OUT_FIXED_SPEC_CSV = "results/tables/exp07f_fixed_specificity_comparison.csv"
OUT_REPORT_TXT = "results/logs/exp07f_centerstrat_erm_groupdro_eval_report.txt"

OUT_FIG_ROC = "results/figures/exp07f_roc_centerstrat_erm_vs_groupdro_ood_val.png"
OUT_FIG_PR = "results/figures/exp07f_pr_centerstrat_erm_vs_groupdro_ood_val.png"
OUT_FIG_FIXED_SPEC = "results/figures/exp07f_fixed_specificity_sensitivity_comparison.png"
OUT_FIG_PERCENTER = "results/figures/exp07f_id_val_per_center_accuracy.png"

TARGET_SPECIFICITIES = [0.80, 0.85, 0.90, 0.95]
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 64
NUM_WORKERS = 0
EXPECTED_ID_VAL_N = 450
EXPECTED_OOD_VAL_N = 2000

_METRIC_FIELDS_CLS = [
    "n", "accuracy", "balanced_accuracy", "auroc", "auprc",
    "sensitivity", "specificity", "precision", "f1",
    "tn", "fp", "fn", "tp",
]

# the two NEW (fair) models, in display order
NEW_MODELS = ("erm7f", "groupdro7f")


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def _load_center_strat_cache(cache_rel: str):
    path = _resolve(cache_rel)
    if not path.exists():
        raise FileNotFoundError(f"required cache not found: {path}")
    payload = torch.load(path, weights_only=False)
    return _deserialize_examples(payload), path


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


def _per_center_metrics(df) -> list[dict]:
    """Overall id_val per-center metrics from a prediction DataFrame (has 'center')."""
    rows = []
    for c in sorted(df["center"].unique().tolist()):
        sub = df[df["center"] == c]
        y = sub["label"].to_numpy()
        p1 = sub["prob_1"].to_numpy()
        m = compute_metrics_at_threshold(y, p1, DEFAULT_THRESHOLD)
        # per-center NLL-free CE (mean cross-entropy) using prob for the true class
        p_true = np.where(y == 1, p1, 1.0 - p1)
        p_true = np.clip(p_true, 1e-7, 1.0)
        loss = float(-np.log(p_true).mean())
        rows.append({
            "center": int(c), "n": int(len(sub)),
            "label0": int((y == 0).sum()), "label1": int((y == 1).sum()),
            "accuracy": m["accuracy"], "sensitivity": m["sensitivity"],
            "specificity": m["specificity"], "loss": loss,
        })
    return rows


def main() -> int:
    root = project_root()
    cfg = load_yaml(root / CONFIG_REL)
    ensure_results_tree()

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    o_cfg = cfg["output"]
    set_seed(int(cfg.get("training", {}).get("seed", 1337)))

    erm_run = str(o_cfg["erm_run_name"])
    gdro_run = str(o_cfg["groupdro_run_name"])
    ckpt_root = _resolve(o_cfg["checkpoint_root"])
    pred_root = _resolve(o_cfg["prediction_root"])
    metrics_root = _resolve(o_cfg["metrics_root"])

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emit("=" * 80)
    emit("Milestone 7F - eval matched center-stratified ERM vs GroupDRO (script 27)")
    emit(f"erm_run={erm_run}   groupdro_run={gdro_run}   device={device}")
    emit("scope: cached center-strat id_val + balanced ood_val ONLY (eval transform).")
    emit("       ood_val = post-training development comparison (authorized). NO calibration,")
    emit("       NO committed threshold, NO ood_test. FAIR = 7F ERM vs 7F GroupDRO.")
    emit("=" * 80)

    ok = True
    try:
        # --- guard: no ood_test anywhere in outputs/inputs ---------------------- #
        all_paths = READONLY_INPUTS + [
            OUT_OVERALL_CSV, OUT_PERCENTER_CSV, OUT_OOD_CSV, OUT_FIXED_SPEC_CSV, OUT_REPORT_TXT,
            OUT_FIG_ROC, OUT_FIG_PR, OUT_FIG_FIXED_SPEC, OUT_FIG_PERCENTER,
            str(d_cfg["id_val_cache"]), str(d_cfg["ood_val_cache"]),
        ]
        assert not any("ood_test" in str(p) for p in all_paths), "ood_test must never be referenced"

        erm_ckpt = ckpt_root / erm_run / "best.pt"
        gdro_ckpt = ckpt_root / gdro_run / "best.pt"
        assert erm_ckpt.exists(), f"ERM checkpoint missing (run script 26): {erm_ckpt}"
        assert gdro_ckpt.exists(), f"GroupDRO checkpoint missing (run script 26): {gdro_ckpt}"

        # --- cached subsets: center-strat id_val + balanced ood_val ------------- #
        id_val_examples, id_val_cache_path = _load_center_strat_cache(d_cfg["id_val_cache"])
        ood_val_examples, ood_val_cache_path = _load_center_strat_cache(d_cfg["ood_val_cache"])
        id_val_ds = BalancedSubsetDataset(id_val_examples, transform=eval_transform)
        ood_val_ds = BalancedSubsetDataset(ood_val_examples, transform=eval_transform)
        id_val_loader = DataLoader(id_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, drop_last=False, pin_memory=False)
        ood_val_loader = DataLoader(ood_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, drop_last=False, pin_memory=False)
        emit("")
        emit(f"id_val  cache : {id_val_cache_path}  (n={len(id_val_ds)})")
        emit(f"ood_val cache : {ood_val_cache_path}  (n={len(ood_val_ds)})")

        # --- inference for both NEW models -------------------------------------- #
        preds = {}       # model_key -> {"id_val": df, "ood_val": df}
        ckpt_epoch = {}
        for key, run_name, ckpt_path in (("erm7f", erm_run, erm_ckpt),
                                         ("groupdro7f", gdro_run, gdro_ckpt)):
            model = CamelyonClassifier(
                backbone_name=m_cfg["backbone_name"], pretrained=bool(m_cfg["pretrained"]),
                num_classes=int(m_cfg["num_classes"]),
            )
            payload = load_checkpoint(ckpt_path, model, optimizer=None, map_location="cpu")
            model.to(device)
            ckpt_epoch[key] = int(payload.get("epoch", -1))
            emit("")
            emit(f"--- inference: {REF_LABEL[key]} ({run_name}, best epoch {ckpt_epoch[key]}) ---")
            id_df = predict_to_dataframe(model, id_val_loader, device, "id_val")
            ood_df = predict_to_dataframe(model, ood_val_loader, device, "ood_val")
            _prob_checks(id_df, f"{key}/id_val")
            _prob_checks(ood_df, f"{key}/ood_val")
            preds[key] = {"id_val": id_df, "ood_val": ood_df}
            emit(f"  id_val n={len(id_df)}  ood_val n={len(ood_df)}")

            # save prediction CSVs
            pdir = pred_root / run_name
            ensure_dir(pdir)
            id_df.to_csv(pdir / "id_val_predictions.csv", index=False)
            ood_df.to_csv(pdir / "ood_val_predictions.csv", index=False)

        # --- A. overall classification metrics (both NEW models, both splits) --- #
        emit("")
        emit("A. Overall classification metrics (tumor = positive, threshold 0.5)")
        overall_rows = []
        cls_by_key = {}
        for key in NEW_MODELS:
            for split in ("id_val", "ood_val"):
                m = compute_binary_classification_metrics(
                    preds[key][split]["label"].to_numpy(),
                    preds[key][split]["prob_1"].to_numpy(), threshold=0.5)
                cls_by_key[(key, split)] = m
                row = {"model": key, "model_label": REF_LABEL[key], "split": split}
                for k in _METRIC_FIELDS_CLS:
                    row[k] = m[k]
                overall_rows.append(row)
                emit(f"  [{REF_LABEL[key]} / {split}]  acc={_fmt(m['accuracy'])} auroc={_fmt(m['auroc'])} "
                     f"auprc={_fmt(m['auprc'])} sens={_fmt(m['sensitivity'])} spec={_fmt(m['specificity'])} "
                     f"f1={_fmt(m['f1'])}  tn/fp/fn/tp={m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}")

        # save per-model classification_metrics.json
        for key, run_name in (("erm7f", erm_run), ("groupdro7f", gdro_run)):
            mp = metrics_root / run_name / "classification_metrics.json"
            ensure_dir(mp.parent)
            write_text(mp, json.dumps({
                "run_name": run_name, "checkpoint_epoch": ckpt_epoch[key],
                "threshold": 0.5, "positive_class": 1,
                "splits": {"id_val": cls_by_key[(key, "id_val")], "ood_val": cls_by_key[(key, "ood_val")]},
                "note": "Milestone 7F matched center-stratified " +
                        ("ERM" if key == "erm7f" else "GroupDRO") + " baseline. DEVELOPMENT "
                        "comparison on capped balanced subsets. ood_val used post-training for "
                        "development comparison only. NOT final. ood_test never touched.",
            }, indent=2) + "\n")

        # --- B. per-center id_val metrics (both NEW models) --------------------- #
        emit("")
        emit("B. Per-center id_val metrics (tumor = positive, threshold 0.5)")
        percenter_rows = []
        percenter_by_key = {}
        for key in NEW_MODELS:
            rows = _per_center_metrics(preds[key]["id_val"])
            percenter_by_key[key] = rows
            for r in rows:
                out = {"model": key, "model_label": REF_LABEL[key], **r}
                percenter_rows.append(out)
                emit(f"  [{REF_LABEL[key]}] center {r['center']}: n={r['n']} "
                     f"(l0={r['label0']},l1={r['label1']})  acc={_fmt(r['accuracy'])} "
                     f"sens={_fmt(r['sensitivity'])} spec={_fmt(r['specificity'])} loss={_fmt(r['loss'])}")

        # worst-center summary per model
        def _worst_center(key):
            rows = percenter_by_key[key]
            wr = max(rows, key=lambda r: (r["loss"] if r["loss"] is not None else -1))
            return wr
        erm_worst = _worst_center("erm7f")
        gdro_worst = _worst_center("groupdro7f")

        # --- C. OOD metrics on ood_val (both NEW + reference models) ------------ #
        emit("")
        emit("C. ood_val metrics comparison (NEW 7F models + old-cache reference models)")
        # reference ood_val frames (read-only)
        resolved_inputs = {name: _resolve(name) for name in READONLY_INPUTS}
        for name, p in resolved_inputs.items():
            assert p.exists(), f"required reference input not found: {p}"
        pre_mtime = {name: resolved_inputs[name].stat().st_mtime_ns for name in READONLY_INPUTS}
        pre_size = {name: resolved_inputs[name].stat().st_size for name in READONLY_INPUTS}

        ref_ood_frames = {k: pd.read_csv(resolved_inputs[v]) for k, v in REF_OOD_CSVS.items()}
        ref_id_frames = {k: pd.read_csv(resolved_inputs[v]) for k, v in REF_ID_CSVS.items()}
        for k, df in ref_ood_frames.items():
            _prob_checks(df, f"{k}/ood_val")

        ood_rows = []
        ood_metrics_all = {}  # key -> classification metrics dict on ood_val
        # NEW models first, then reference
        for key in NEW_MODELS:
            ood_metrics_all[key] = cls_by_key[(key, "ood_val")]
        for key, df in ref_ood_frames.items():
            ood_metrics_all[key] = compute_binary_classification_metrics(
                df["label"].to_numpy(), df["prob_1"].to_numpy(), threshold=0.5)

        order_all = list(NEW_MODELS) + list(REF_OOD_CSVS.keys())
        emit(f"   {'model':<34}{'acc':>8}{'auroc':>9}{'auprc':>9}{'sens':>8}{'spec':>8}{'FN':>6}")
        for key in order_all:
            m = ood_metrics_all[key]
            row = {"model": key, "model_label": REF_LABEL[key],
                   "matched_new_cache": key in NEW_MODELS}
            for k in _METRIC_FIELDS_CLS:
                row[k] = m[k]
            ood_rows.append(row)
            emit(f"   {REF_LABEL[key]:<34}{_fmt(m['accuracy']):>8}{_fmt(m['auroc']):>9}"
                 f"{_fmt(m['auprc']):>9}{_fmt(m['sensitivity']):>8}{_fmt(m['specificity']):>8}{m['fn']:>6}")

        # --- D. fixed-specificity candidate operating points -------------------- #
        # For NEW models: select on center-strat id_val, apply to ood_val.
        # For reference: plain_old uses its own 4A id_val CSV; 7A/7D/7E have no id_val
        #   prob_1 CSV wired here, so we select their thresholds on plain_old id_val is
        #   NOT valid -> we only include reference models that have a matching id_val CSV
        #   (plain_old). 7A/7D/7E ood_val are reported at default 0.5 in table C above.
        emit("")
        emit("D. id_val-selected FIXED-SPECIFICITY candidate operating points -> ood_val")
        emit("   (NEW models: threshold from center-strat id_val; plain_old: from its 4A id_val)")
        emit("   *** CANDIDATE points only - NO operating threshold committed in 7F ***")

        op_rows_by_key = {}
        for key in NEW_MODELS:
            op_rows_by_key[key] = apply_operating_points(
                preds[key]["id_val"], preds[key]["ood_val"], TARGET_SPECIFICITIES, [])
        # plain_old reference (has id_val prob_1 CSV)
        op_rows_by_key["plain_old"] = apply_operating_points(
            ref_id_frames["plain_old"], ref_ood_frames["plain_old"], TARGET_SPECIFICITIES, [])

        fixed_spec_rows = []
        for key in ("erm7f", "groupdro7f", "plain_old"):
            for r in op_rows_by_key[key]:
                if r["criterion"] != "specificity":
                    continue
                out = {"model": key, "model_label": REF_LABEL[key], **r}
                fixed_spec_rows.append(out)

        def _op(key, target, split):
            for r in op_rows_by_key.get(key, []):
                if r["criterion"] == "specificity" and abs(r["target"] - target) < 1e-9 and r["split"] == split:
                    return r
            return None

        emit(f"   {'target_spec':>11}  {'model':<34}{'thr':>8}"
             f"{'id_sens':>9}{'id_spec':>9}{'id_FN':>7}{'id_FP':>7}"
             f"{'ood_sens':>10}{'ood_spec':>10}{'ood_FN':>8}{'ood_FP':>8}{'ood_PPR':>9}")
        for spec in TARGET_SPECIFICITIES:
            for key in ("erm7f", "groupdro7f", "plain_old"):
                idr = _op(key, spec, "id_val")
                oor = _op(key, spec, "ood_val")
                thr = idr["threshold"] if idr else None
                emit(f"   {spec:>11.2f}  {REF_LABEL[key]:<34}{_fmt(thr):>8}"
                     f"{_fmt(idr['sensitivity'] if idr else None):>9}"
                     f"{_fmt(idr['specificity'] if idr else None):>9}"
                     f"{(idr['fn'] if idr and idr['fn'] is not None else 'None'):>7}"
                     f"{(idr['fp'] if idr and idr['fp'] is not None else 'None'):>7}"
                     f"{_fmt(oor['sensitivity'] if oor else None):>10}"
                     f"{_fmt(oor['specificity'] if oor else None):>10}"
                     f"{(oor['fn'] if oor and oor['fn'] is not None else 'None'):>8}"
                     f"{(oor['fp'] if oor and oor['fp'] is not None else 'None'):>8}"
                     f"{_fmt(oor['predicted_positive_rate'] if oor else None):>9}")

        # --- save tables -------------------------------------------------------- #
        write_csv(_resolve(OUT_OVERALL_CSV), overall_rows,
                  header=["model", "model_label", "split", *_METRIC_FIELDS_CLS])
        write_csv(_resolve(OUT_PERCENTER_CSV), percenter_rows,
                  header=["model", "model_label", "center", "n", "label0", "label1",
                          "accuracy", "sensitivity", "specificity", "loss"])
        write_csv(_resolve(OUT_OOD_CSV), ood_rows,
                  header=["model", "model_label", "matched_new_cache", *_METRIC_FIELDS_CLS])
        op_header = ["model", "model_label", "criterion", "target", "split",
                     "achieved_on_id_val", *METRIC_FIELDS]
        write_csv(_resolve(OUT_FIXED_SPEC_CSV), fixed_spec_rows, header=op_header)

        # --- figures ------------------------------------------------------------ #
        color = {"erm7f": "tab:blue", "groupdro7f": "tab:red",
                 "plain_old": "tab:gray", "stain7a_old": "tab:orange",
                 "stain7d_old": "tab:green", "stain7e_old": "tab:purple"}

        # ROC ood_val: two NEW models (fair) + faint reference lines
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        for key in NEW_MODELS:
            df = preds[key]["ood_val"]
            y = df["label"].to_numpy().astype(int); s = df["prob_1"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=color[key], linewidth=2.0,
                    label=f"{REF_LABEL[key]} (AUROC {_fmt(ood_metrics_all[key]['auroc'])})")
        for key in ("plain_old",):
            df = ref_ood_frames[key]
            y = df["label"].to_numpy().astype(int); s = df["prob_1"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=color[key], linewidth=1.0, alpha=0.7, linestyle=":",
                    label=f"{REF_LABEL[key]} (AUROC {_fmt(ood_metrics_all[key]['auroc'])})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title("ROC - ood_val (development): 7F ERM vs GroupDRO")
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_ROC), dpi=150); plt.close(fig)

        # PR ood_val
        fig, ax = plt.subplots(figsize=(5.8, 5.4))
        for key in NEW_MODELS:
            df = preds[key]["ood_val"]
            y = df["label"].to_numpy().astype(int); s = df["prob_1"].to_numpy(dtype=float)
            prec, rec, _ = precision_recall_curve(y, s)
            ax.plot(rec, prec, color=color[key], linewidth=2.0,
                    label=f"{REF_LABEL[key]} (AUPRC {_fmt(ood_metrics_all[key]['auprc'])})")
        for key in ("plain_old",):
            df = ref_ood_frames[key]
            y = df["label"].to_numpy().astype(int); s = df["prob_1"].to_numpy(dtype=float)
            prec, rec, _ = precision_recall_curve(y, s)
            ax.plot(rec, prec, color=color[key], linewidth=1.0, alpha=0.7, linestyle=":",
                    label=f"{REF_LABEL[key]} (AUPRC {_fmt(ood_metrics_all[key]['auprc'])})")
        ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall - ood_val (development): 7F ERM vs GroupDRO")
        ax.legend(loc="lower left", fontsize=7)
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_PR), dpi=150); plt.close(fig)

        # Fixed-specificity ood_val sensitivity: erm7f vs groupdro7f vs plain_old
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        x = np.arange(len(TARGET_SPECIFICITIES)); width = 0.25
        bar_keys = ("erm7f", "groupdro7f", "plain_old")
        for i, key in enumerate(bar_keys):
            vals = []
            for spec in TARGET_SPECIFICITIES:
                oor = _op(key, spec, "ood_val")
                vals.append(oor["sensitivity"] if (oor and oor["sensitivity"] is not None) else np.nan)
            ax.bar(x + (i - 1) * width, vals, width, color=color[key], label=REF_LABEL[key])
        ax.set_xticks(x); ax.set_xticklabels([f"{s:.2f}" for s in TARGET_SPECIFICITIES])
        ax.set_xlabel("id_val-selected target specificity")
        ax.set_ylabel("ood_val sensitivity (candidate)")
        ax.set_title("Candidate fixed-specificity ood_val sensitivity\n(7F ERM vs GroupDRO; plain_old ref; NOT committed)")
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_FIXED_SPEC), dpi=150); plt.close(fig)

        # Per-center id_val accuracy: erm7f vs groupdro7f
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        centers = sorted({r["center"] for r in percenter_by_key["erm7f"]})
        x = np.arange(len(centers)); width = 0.35
        for i, key in enumerate(NEW_MODELS):
            accs = []
            lut = {r["center"]: r for r in percenter_by_key[key]}
            for c in centers:
                accs.append(lut[c]["accuracy"] if lut[c]["accuracy"] is not None else np.nan)
            ax.bar(x + (i - 0.5) * width, accs, width, color=color[key], label=REF_LABEL[key])
        ax.set_xticks(x); ax.set_xticklabels([f"center {c}" for c in centers])
        ax.set_ylabel("id_val accuracy"); ax.set_ylim(0.0, 1.02)
        ax.set_title("Per-center id_val accuracy: 7F ERM vs GroupDRO")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(_resolve(OUT_FIG_PERCENTER), dpi=150); plt.close(fig)

        figure_paths = [OUT_FIG_ROC, OUT_FIG_PR, OUT_FIG_FIXED_SPEC, OUT_FIG_PERCENTER]

        # --- print q weights ---------------------------------------------------- #
        gdro_payload = torch.load(gdro_ckpt, map_location="cpu", weights_only=False)
        gdro_q = gdro_payload.get("metrics", {}).get("q_weights", None)
        emit("")
        emit(f"GroupDRO q weights (best checkpoint, epoch {ckpt_epoch['groupdro7f']}): {gdro_q}")

        # --- FAIR comparison + interpretation ----------------------------------- #
        erm_ood = cls_by_key[("erm7f", "ood_val")]
        gdro_ood = cls_by_key[("groupdro7f", "ood_val")]
        erm_id = cls_by_key[("erm7f", "id_val")]
        gdro_id = cls_by_key[("groupdro7f", "id_val")]

        def _delta(a, b):
            return (a - b) if (a is not None and b is not None) else None

        d_ood_auroc = _delta(gdro_ood["auroc"], erm_ood["auroc"])
        d_ood_auprc = _delta(gdro_ood["auprc"], erm_ood["auprc"])
        d_ood_def_sens = _delta(gdro_ood["sensitivity"], erm_ood["sensitivity"])
        d_worst_loss = _delta(gdro_worst["loss"], erm_worst["loss"])  # lower is better
        d_worst_acc = _delta(gdro_worst["accuracy"], erm_worst["accuracy"])

        # fixed-spec matched sensitivity: GroupDRO >= ERM count
        n_gdro_ge_erm, n_targets = 0, 0
        for spec in TARGET_SPECIFICITIES:
            e = _op("erm7f", spec, "ood_val"); g = _op("groupdro7f", spec, "ood_val")
            ev = e["sensitivity"] if (e and e["sensitivity"] is not None) else None
            gv = g["sensitivity"] if (g and g["sensitivity"] is not None) else None
            if ev is not None and gv is not None:
                n_targets += 1
                if gv >= ev:
                    n_gdro_ge_erm += 1

        worst_center_improved = (d_worst_loss is not None and d_worst_loss < 0) or \
                                 (d_worst_acc is not None and d_worst_acc > 0)
        ood_ranking_improved = (d_ood_auroc is not None and d_ood_auroc > 0)
        ood_auprc_improved = (d_ood_auprc is not None and d_ood_auprc > 0)
        def_sens_improved = (d_ood_def_sens is not None and d_ood_def_sens > 0)
        fixedspec_improved = (n_targets > 0 and n_gdro_ge_erm > n_targets / 2)

        emit("")
        emit("FAIR COMPARISON: 7F center-strat ERM vs 7F GroupDRO (both on NEW center-strat cache)")
        emit("-" * 78)
        emit(f"  id_val  accuracy : ERM {_fmt(erm_id['accuracy'])}  GroupDRO {_fmt(gdro_id['accuracy'])}")
        emit(f"  id_val  AUROC    : ERM {_fmt(erm_id['auroc'])}  GroupDRO {_fmt(gdro_id['auroc'])}")
        emit(f"  worst-center id_val loss : ERM c{erm_worst['center']}={_fmt(erm_worst['loss'])}  "
             f"GroupDRO c{gdro_worst['center']}={_fmt(gdro_worst['loss'])}  (delta {_fmt(d_worst_loss)})")
        emit(f"  worst-center id_val acc  : ERM c{erm_worst['center']}={_fmt(erm_worst['accuracy'])}  "
             f"GroupDRO c{gdro_worst['center']}={_fmt(gdro_worst['accuracy'])}  (delta {_fmt(d_worst_acc)})")
        emit(f"  ood_val AUROC    : ERM {_fmt(erm_ood['auroc'])}  GroupDRO {_fmt(gdro_ood['auroc'])}  (delta {_fmt(d_ood_auroc)})")
        emit(f"  ood_val AUPRC    : ERM {_fmt(erm_ood['auprc'])}  GroupDRO {_fmt(gdro_ood['auprc'])}  (delta {_fmt(d_ood_auprc)})")
        emit(f"  ood_val sens@0.5 : ERM {_fmt(erm_ood['sensitivity'])}  GroupDRO {_fmt(gdro_ood['sensitivity'])}  (delta {_fmt(d_ood_def_sens)})")
        emit(f"  matched fixed-spec ood_val sensitivity wins (GroupDRO >= ERM): {n_gdro_ge_erm}/{n_targets}")

        emit("")
        emit("REFERENCE COMPARISON (context only; old models trained on OLD confounded cache):")
        emit(f"  ood_val AUROC: old plain {_fmt(ood_metrics_all['plain_old']['auroc'])}  "
             f"old 7A {_fmt(ood_metrics_all['stain7a_old']['auroc'])}  "
             f"old 7D {_fmt(ood_metrics_all['stain7d_old']['auroc'])}  "
             f"old 7E {_fmt(ood_metrics_all['stain7e_old']['auroc'])}  "
             f"|  NEW 7F ERM {_fmt(erm_ood['auroc'])}  NEW 7F GroupDRO {_fmt(gdro_ood['auroc'])}")
        emit("  NOTE: old models used a DIFFERENT (confounded) train cache and a different id_val")
        emit("        subset; they are NOT matched to the 7F center-stratified data. Treat the")
        emit("        old-vs-new numbers as context, NOT a controlled comparison.")

        # interpretation bullets
        emit("")
        emit("INTERPRETATION (development-stage; capped balanced subsets; NOT final performance)")
        emit("-" * 78)
        emit(f"* GroupDRO {'IMPROVED' if worst_center_improved else 'did NOT improve'} worst-center id_val behavior vs matched ERM.")
        emit(f"* GroupDRO {'IMPROVED' if ood_ranking_improved else 'did NOT improve'} ood_val AUROC vs matched ERM (delta {_fmt(d_ood_auroc)}).")
        emit(f"* GroupDRO {'IMPROVED' if ood_auprc_improved else 'did NOT improve'} ood_val AUPRC vs matched ERM (delta {_fmt(d_ood_auprc)}).")
        emit(f"* GroupDRO {'IMPROVED' if def_sens_improved else 'did NOT improve'} default-0.5 ood_val sensitivity vs matched ERM (delta {_fmt(d_ood_def_sens)}).")
        emit(f"* GroupDRO {'IMPROVED' if fixedspec_improved else 'did NOT improve'} matched fixed-specificity ood_val sensitivity ({n_gdro_ge_erm}/{n_targets} targets).")

        # verdict
        improvements = sum([bool(worst_center_improved), bool(ood_ranking_improved),
                            bool(ood_auprc_improved), bool(fixedspec_improved)])
        if ood_ranking_improved and (fixedspec_improved or worst_center_improved):
            verdict = "retain"
            recommendation = (
                "RETAIN GroupDRO (development): on the matched center-stratified data GroupDRO "
                "improved ood_val ranking over the matched ERM and also improved worst-center "
                "and/or matched fixed-specificity behavior. Next: calibration + candidate "
                "operating-point audit (still development-stage), gated.")
        elif improvements >= 1:
            verdict = "tune"
            recommendation = (
                "TUNE FURTHER GroupDRO (development): GroupDRO helped on at least one axis "
                "(worst-center / ranking / AUPRC / fixed-specificity) but did not clearly beat "
                "the matched ERM overall on this capped subset. Consider tuning the GroupDRO "
                "step_size, more epochs, or group granularity - gated. Do NOT claim superiority.")
        else:
            verdict = "reject"
            recommendation = (
                "DO NOT PROMOTE GroupDRO (development): on the matched center-stratified data "
                "GroupDRO did not improve worst-center id_val behavior, ood_val ranking, "
                "default-threshold sensitivity, or matched fixed-specificity sensitivity vs the "
                "matched ERM. Center-group DRO is not an effective OOD lever here on this capped "
                "subset; retain the matched ERM as the control. Do NOT claim superiority.")
        emit("")
        emit(f"RECOMMENDATION: {recommendation}")
        emit("(Development-stage comparison only - NOT a final-performance or superiority claim; "
             "no operating point committed; no calibration in 7F.)")

        # --- combined comparison metrics JSON (in each metrics dir root) -------- #
        comparison_payload = {
            "milestone": "7F",
            "description": "Matched center-stratified ERM vs GroupDRO (both trained on the new "
                           "center-stratified train cache). id_val selects candidate thresholds; "
                           "ood_val is development comparison only. Old plain/7A/7D/7E are "
                           "reference-only (old confounded cache, not matched).",
            "positive_class": 1, "default_threshold": DEFAULT_THRESHOLD,
            "target_specificities": TARGET_SPECIFICITIES,
            "checkpoint_epoch": ckpt_epoch,
            "groupdro_q_weights": gdro_q,
            "overall_metrics": {f"{key}_{split}": {k: cls_by_key[(key, split)][k] for k in ["n", *_METRIC_FIELDS_CLS]}
                                for key in NEW_MODELS for split in ("id_val", "ood_val")},
            "ood_val_reference_metrics": {key: {k: ood_metrics_all[key][k] for k in ["n", *_METRIC_FIELDS_CLS]}
                                          for key in order_all},
            "per_center_id_val": {key: percenter_by_key[key] for key in NEW_MODELS},
            "fair_comparison": {
                "delta_ood_auroc_gdro_minus_erm": d_ood_auroc,
                "delta_ood_auprc_gdro_minus_erm": d_ood_auprc,
                "delta_ood_default_sensitivity_gdro_minus_erm": d_ood_def_sens,
                "erm_worst_center": erm_worst, "groupdro_worst_center": gdro_worst,
                "delta_worst_center_loss_gdro_minus_erm": d_worst_loss,
                "delta_worst_center_acc_gdro_minus_erm": d_worst_acc,
                "fixedspec_gdro_ge_erm": f"{n_gdro_ge_erm}/{n_targets}",
                "worst_center_improved": bool(worst_center_improved),
                "ood_ranking_improved": bool(ood_ranking_improved),
                "ood_auprc_improved": bool(ood_auprc_improved),
                "default_sensitivity_improved": bool(def_sens_improved),
                "fixedspec_improved": bool(fixedspec_improved),
                "verdict": verdict, "recommendation": recommendation,
            },
            "note": "DEVELOPMENT comparison on capped balanced subsets - NOT final performance. "
                    "ood_val used for development-stage comparison only after id_val threshold "
                    "selection. No calibration, no committed operating point. ood_test never read. "
                    "Old plain/7A/7D/7E are reference-only (old confounded cache, not matched).",
        }
        comp_json = metrics_root / gdro_run / "comparison_metrics.json"
        ensure_dir(comp_json.parent)
        write_text(comp_json, json.dumps(comparison_payload, indent=2) + "\n")

        # --- verify reference CSVs unmodified ----------------------------------- #
        for name in READONLY_INPUTS:
            post = resolved_inputs[name].stat()
            assert post.st_mtime_ns == pre_mtime[name], f"reference input modified (mtime): {name}"
            assert post.st_size == pre_size[name], f"reference input modified (size): {name}"

        # --- assertions --------------------------------------------------------- #
        for key in NEW_MODELS:
            assert len(preds[key]["id_val"]) == EXPECTED_ID_VAL_N, f"{key} id_val n != {EXPECTED_ID_VAL_N}"
            assert len(preds[key]["ood_val"]) == EXPECTED_OOD_VAL_N, f"{key} ood_val n != {EXPECTED_OOD_VAL_N}"
            assert "center" in preds[key]["id_val"].columns, f"{key} id_val missing center metadata"
        for key, run_name in (("erm7f", erm_run), ("groupdro7f", gdro_run)):
            assert (metrics_root / run_name / "classification_metrics.json").exists(), f"{key} metrics json missing"
        for p in (OUT_OVERALL_CSV, OUT_PERCENTER_CSV, OUT_OOD_CSV, OUT_FIXED_SPEC_CSV):
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"table missing/empty: {p}"
        for p in figure_paths:
            assert _resolve(p).exists() and _resolve(p).stat().st_size > 0, f"figure missing/empty: {p}"
        assert comp_json.exists() and comp_json.stat().st_size > 0, "comparison JSON missing/empty"

        # --- output paths ------------------------------------------------------- #
        emit("")
        emit("OUTPUT PATHS")
        emit("-" * 60)
        emit("predictions:")
        for key, run_name in (("erm7f", erm_run), ("groupdro7f", gdro_run)):
            emit(f"  {pred_root / run_name / 'id_val_predictions.csv'}")
            emit(f"  {pred_root / run_name / 'ood_val_predictions.csv'}")
        emit("metrics:")
        for run_name in (erm_run, gdro_run):
            emit(f"  {metrics_root / run_name / 'classification_metrics.json'}")
        emit(f"  {comp_json}")
        emit("tables:")
        for p in (OUT_OVERALL_CSV, OUT_PERCENTER_CSV, OUT_OOD_CSV, OUT_FIXED_SPEC_CSV):
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
        import traceback
        emit(traceback.format_exc())

    emit("")
    emit("=" * 80)
    emit(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    emit("=" * 80)

    write_text(_resolve(OUT_REPORT_TXT), "\n".join(lines) + "\n")
    print(f"\nSaved report -> {OUT_REPORT_TXT}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
