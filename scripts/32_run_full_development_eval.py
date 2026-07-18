"""Milestone 8B / Script 32: full development-scale inference for the frozen 7F models.

Executes the Milestone 8A plan (see docs/FULL_DEVELOPMENT_EVALUATION_*). This is an
INFERENCE-ONLY, development-stage evaluation — NOT final performance.

What it does
------------
Loads the two FROZEN 7F checkpoints (read-only, weights unchanged):

    * primary candidate  : results/checkpoints/exp07f_groupdro_resnet18/best.pt
    * matched control    : results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt

and runs eval-transform inference for BOTH models on the full/larger development splits
of the locked mapping (streamed from the sanctioned HF mirror wltjr1007/Camelyon17-WILDS):

    * id_val  = HF validation, centers {0,3,4}   (expected n = 33,560)
    * ood_val = HF validation, center  {1}        (expected n = 34,904)

It writes four per-example prediction CSVs under a NEW run directory
``results/predictions/exp08b_full_dev_eval/`` (no overwrite of any prior-milestone
artifact) with the extended schema (adds a ``confidence`` column).

Strict scope (Milestone 8B):
  * NO training, NO weight update, NO checkpoint modification.
  * NO calibration fitting, NO MC-dropout, NO ensemble, NO CORAL/DANN.
  * ``ood_test`` (HF test / center 2) is NEVER referenced or read.
  * ``ood_val`` is read for development-stage OOD analysis only.
  * Candidate operating thresholds (computed in script 33) are selected on id_val only
    and are NOT committed clinical thresholds.

DEVELOPMENT-stage evaluation on full/larger development splits — NOT final performance.
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.torch_dataset import Camelyon17HFDataset
from data.transforms import eval_transform
from models.classifier import CamelyonClassifier
from training.checkpointing import load_checkpoint
from utils.io import ensure_dir, ensure_results_tree, project_root, write_text
from utils.seed import set_seed

# --- FROZEN checkpoints (exact paths; Milestone 8A freeze) --------------------------- #
FROZEN = {
    "groupdro": "results/checkpoints/exp07f_groupdro_resnet18/best.pt",
    "centerstrat_erm": "results/checkpoints/exp07f_centerstrat_erm_resnet18/best.pt",
}
MODEL_LABEL = {
    "groupdro": "7F GroupDRO (primary candidate)",
    "centerstrat_erm": "7F center-stratified ERM (matched control)",
}
MODEL_ORDER = ("groupdro", "centerstrat_erm")

# --- new run directory (never overwrites prior milestones) --------------------------- #
RUN_NAME = "exp08b_full_dev_eval"
PRED_DIR = f"results/predictions/{RUN_NAME}"
OUT_REPORT_TXT = f"results/logs/{RUN_NAME}_inference_report.txt"

# prediction CSV output paths: <pred_dir>/<model>_<split>_predictions.csv
def _pred_path(model_key: str, split: str) -> str:
    return f"{PRED_DIR}/{model_key}_{split}_predictions.csv"


# --- locked development splits -------------------------------------------------------- #
SPLITS = ("id_val", "ood_val")
EXPECTED_N = {"id_val": 33560, "ood_val": 34904}
ALLOWED_CENTERS = {"id_val": {0, 3, 4}, "ood_val": {1}}
FORBIDDEN_CENTERS = {2}  # ood_test / center 2 must never appear

BATCH_SIZE = 64
NUM_WORKERS = 0
SEED = 1337

# Extended prediction schema (adds `confidence` to the Milestone-7F schema).
REQUIRED_COLUMNS = [
    "split", "label", "pred_label", "correct",
    "prob_0", "prob_1", "logit_0", "logit_1", "confidence",
    "center", "patient", "slide", "node", "image_id", "x_coord", "y_coord",
]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def main() -> int:
    root = project_root()
    ensure_results_tree()
    set_seed(SEED)

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emit("=" * 80)
    emit("Milestone 8B - full development-scale inference (script 32)")
    emit("INFERENCE ONLY. No training, no weight update, no calibration, no MC-dropout,")
    emit("no ensemble, no CORAL/DANN. ood_test (center 2) is NEVER read.")
    emit(f"device={device}   run={RUN_NAME}")
    emit("=" * 80)

    ok = True
    observed = {}   # (model, split) -> {"n":, "centers":, "labels":}
    try:
        # --- guard: no ood_test / center-2 anywhere in configured paths ---------- #
        all_paths = [OUT_REPORT_TXT, PRED_DIR] + [
            _pred_path(m, s) for m in MODEL_ORDER for s in SPLITS
        ] + list(FROZEN.values())
        assert not any("ood_test" in str(p) for p in all_paths), "ood_test must never be referenced"
        assert not any("test" == Path(str(p)).name for p in all_paths), "no HF test artifact"
        # every output must live under the explicit exp08b run name (no-overwrite of prior work)
        for m in MODEL_ORDER:
            for s in SPLITS:
                assert RUN_NAME in _pred_path(m, s), "output must be under exp08b run name"

        # --- verify FROZEN checkpoint paths exactly ------------------------------ #
        ckpt_paths = {}
        for key in MODEL_ORDER:
            expected_rel = FROZEN[key]
            p = _resolve(expected_rel)
            assert p.exists(), f"frozen checkpoint missing: {expected_rel}"
            # exact-path check: resolved path must end with the frozen relative path
            assert str(p).replace("\\", "/").endswith(expected_rel), (
                f"checkpoint path mismatch for {key}: {p} != {expected_rel}")
            ckpt_paths[key] = p
        emit("")
        emit("Frozen checkpoints verified (exact paths):")
        for key in MODEL_ORDER:
            emit(f"  {key:<16} -> {FROZEN[key]}")

        # --- no-overwrite explicit run-name check -------------------------------- #
        pred_dir = _resolve(PRED_DIR)
        existing = [p for m in MODEL_ORDER for s in SPLITS
                    if (p := _resolve(_pred_path(m, s))).exists()]
        if existing:
            emit("")
            emit("WARNING: existing exp08b prediction files found; re-running the SAME")
            emit("         exp08b_full_dev_eval run (its own outputs only, never prior milestones):")
            for p in existing:
                emit(f"         {p}")
        ensure_dir(pred_dir)

        # --- inference for both models on both splits ---------------------------- #
        for key in MODEL_ORDER:
            ckpt = ckpt_paths[key]
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            cfg = payload.get("config", {}) or {}
            m_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
            backbone = str(m_cfg.get("backbone_name", "resnet18"))
            num_classes = int(m_cfg.get("num_classes", 2))
            pretrained = bool(m_cfg.get("pretrained", False))

            model = CamelyonClassifier(
                backbone_name=backbone, pretrained=pretrained, num_classes=num_classes)
            load_checkpoint(ckpt, model, optimizer=None, map_location="cpu")
            model.to(device)
            ckpt_epoch = int(payload.get("epoch", -1))

            emit("")
            emit(f"--- inference: {MODEL_LABEL[key]} (best epoch {ckpt_epoch}) ---")

            for split in SPLITS:
                # Full/larger split via the sanctioned locked-mapping loader.
                # max_samples=None -> full logical split (validation intersect allowed centers).
                emit(f"  [{split}] materializing full split (HF validation, "
                     f"centers {sorted(ALLOWED_CENTERS[split])}) ...")
                dataset = Camelyon17HFDataset(
                    split_name=split, transform=eval_transform,
                    max_samples=None, verbose=False)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, drop_last=False, pin_memory=False)
                from evaluation.predict import predict_to_dataframe
                df = predict_to_dataframe(model, loader, device, split)

                # add confidence = max(prob_0, prob_1)
                df["confidence"] = np.maximum(
                    df["prob_0"].to_numpy(dtype=float), df["prob_1"].to_numpy(dtype=float))

                # --- per-(model,split) validation checks --------------------------- #
                p0 = df["prob_0"].to_numpy(dtype=float)
                p1 = df["prob_1"].to_numpy(dtype=float)
                centers = set(int(c) for c in df["center"].unique().tolist())
                labels = set(int(v) for v in df["label"].unique().tolist())

                assert np.isfinite(p0).all() and np.isfinite(p1).all(), \
                    f"{key}/{split}: non-finite probabilities"
                assert ((p0 >= 0) & (p0 <= 1)).all() and ((p1 >= 0) & (p1 <= 1)).all(), \
                    f"{key}/{split}: probability out of [0,1]"
                assert (np.abs(p0 + p1 - 1.0) < 1e-5).all(), \
                    f"{key}/{split}: prob_0+prob_1 != 1"
                assert labels.issubset({0, 1}), f"{key}/{split}: non-binary labels {labels}"
                assert centers.issubset(ALLOWED_CENTERS[split]), \
                    f"{key}/{split}: unexpected centers {centers} (allowed {ALLOWED_CENTERS[split]})"
                assert not (centers & FORBIDDEN_CENTERS), \
                    f"{key}/{split}: forbidden center present {centers & FORBIDDEN_CENTERS}"
                assert 1 not in centers or split == "ood_val", \
                    f"{key}/{split}: center 1 must only appear in ood_val"
                assert len(df) == EXPECTED_N[split], \
                    f"{key}/{split}: row count {len(df)} != expected {EXPECTED_N[split]}"
                for col in REQUIRED_COLUMNS:
                    assert col in df.columns, f"{key}/{split}: missing required column {col}"

                # --- write CSV (required columns first, keep helper 'index') ------- #
                out_cols = REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]
                out_path = _resolve(_pred_path(key, split))
                df[out_cols].to_csv(out_path, index=False)

                observed[(key, split)] = {
                    "n": int(len(df)),
                    "centers": sorted(centers),
                    "labels": sorted(labels),
                    "label0": int((df["label"] == 0).sum()),
                    "label1": int((df["label"] == 1).sum()),
                }
                emit(f"  [{split}] n={len(df)}  centers={sorted(centers)}  "
                     f"labels={sorted(labels)} (l0={observed[(key, split)]['label0']}, "
                     f"l1={observed[(key, split)]['label1']})  -> {out_path.name}")

        # --- summary of row counts / centers ------------------------------------- #
        emit("")
        emit("ROW COUNTS & CENTERS OBSERVED")
        emit("-" * 60)
        for key in MODEL_ORDER:
            for split in SPLITS:
                o = observed[(key, split)]
                emit(f"  {MODEL_LABEL[key]:<45} {split:<8} n={o['n']:>6} "
                     f"centers={o['centers']}")

        # --- final assertions across all four CSVs ------------------------------- #
        for key in MODEL_ORDER:
            for split in SPLITS:
                assert (key, split) in observed, f"missing predictions for {key}/{split}"
                assert observed[(key, split)]["n"] == EXPECTED_N[split]
                assert _resolve(_pred_path(key, split)).exists()
        # id_val counts identical across models; ood_val counts identical across models
        assert observed[("groupdro", "id_val")]["n"] == observed[("centerstrat_erm", "id_val")]["n"]
        assert observed[("groupdro", "ood_val")]["n"] == observed[("centerstrat_erm", "ood_val")]["n"]

        emit("")
        emit("OUTPUT PATHS")
        emit("-" * 60)
        for key in MODEL_ORDER:
            for split in SPLITS:
                emit(f"  {_pred_path(key, split)}")
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
    print(f"\nSaved inference report -> {OUT_REPORT_TXT}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
