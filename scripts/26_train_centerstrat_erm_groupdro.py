"""Milestone 7F / Script 26: matched center-stratified ERM + GroupDRO — checkpointed training.

Trains TWO ResNet18 models on the SAME new metadata-rich center-stratified ``train`` cache
(Milestone 7F-1, per-(center,label) balanced across centers {0,3,4}):

    A. center-stratified ERM baseline  (plain cross-entropy)      -> exp07f_centerstrat_erm_resnet18
    B. center-stratified GroupDRO by center group                 -> exp07f_groupdro_resnet18

The matched ERM control is REQUIRED because the previous plain-ERM 4A checkpoint was trained
on the OLD confounded balanced cache; a fair GroupDRO comparison needs an ERM trained on the
SAME center-stratified data. Groups = acquisition centers {0, 3, 4} (the domain-shift axis).

Strict scope (Milestone 7F training):
  * ``train`` + ``id_val`` (center-stratified caches) ONLY, for training and model selection.
    Best checkpoint per model is selected by ``id_val_loss`` ONLY.
  * ``ood_val`` is NEVER read in this script (used only afterwards, by script 27, for
    development-stage comparison). ``ood_test`` is NEVER read anywhere.
  * NO stain augmentation (normal train/eval transforms). NO CORAL/DANN, NO MC-dropout/
    ensemble, NO calibration, NO committed operating threshold.

DEVELOPMENT run, NOT a paper result. All knobs come from
configs/experiments/exp07f_groupdro_resnet18.yaml.
Outputs per model: per-epoch CSV, text report (results/logs/), checkpoints
(results/checkpoints/<run_name>/best.pt + last.pt). GroupDRO additionally logs q weights,
per-group losses, and worst-center id_val loss/accuracy each epoch.
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import math
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.balanced_subset import META_INT_FIELDS, _deserialize_examples, BalancedSubsetDataset
from data.transforms import eval_transform, train_transform
from models.classifier import CamelyonClassifier
from training.checkpointing import load_checkpoint, save_checkpoint
from training.groupdro import GroupDROLoss, groups_from_batch, summarize_group_counts
from training.loop import train_one_epoch
from training.losses import create_loss
from training.optimizers import create_optimizer
from utils.io import ensure_dir, ensure_results_tree, load_yaml, project_root, write_csv, write_text
from utils.seed import set_seed

CONFIG_REL = "configs/experiments/exp07f_groupdro_resnet18.yaml"

EXPECTED_TRAIN_N = 1800
EXPECTED_ID_VAL_N = 450
EXPECTED_CENTERS = {0, 3, 4}


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def _load_center_strat_cache(cache_rel: str) -> tuple[list[dict], Path]:
    """Load a center-stratified cache directly (new filename scheme; no HF streaming).

    These caches (``center_strat_*``) share the byte-for-byte serializer schema of the
    balanced caches, so ``_deserialize_examples`` reloads them exactly. We load from disk
    only — no streaming, no ood_test.
    """
    path = _resolve(cache_rel)
    if not path.exists():
        raise FileNotFoundError(f"required center-stratified cache not found (run 7F-1 first): {path}")
    payload = torch.load(path, weights_only=False)
    examples = _deserialize_examples(payload)
    return examples, path


def _group_aware_evaluate(model, dataset, group_field, group_ids, device, batch_size):
    """No-grad id_val evaluation returning overall + per-center loss/accuracy.

    Uses plain cross-entropy (reduction sum over each center) so the reported per-center
    loss is comparable across models. Returns:
        {"overall": {"loss": .., "acc": .., "n": ..},
         "per_center": {gid: {"loss": .., "acc": .., "n": .., "l0": .., "l1": ..}}}
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False, pin_memory=False,
    )
    ce_none = torch.nn.CrossEntropyLoss(reduction="none")
    sums = {gid: {"loss": 0.0, "correct": 0, "n": 0, "l0": 0, "l1": 0} for gid in group_ids}
    tot_loss, tot_correct, tot_n = 0.0, 0, 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            groups = groups_from_batch(batch, group_field, device)
            logits = model(images)
            losses = ce_none(logits, labels)          # [B]
            preds = logits.argmax(dim=1)              # [B]
            correct = (preds == labels)

            tot_loss += float(losses.sum().item())
            tot_correct += int(correct.sum().item())
            tot_n += int(labels.shape[0])

            for gid in group_ids:
                m = groups == int(gid)
                cnt = int(m.sum().item())
                if cnt == 0:
                    continue
                sums[gid]["loss"] += float(losses[m].sum().item())
                sums[gid]["correct"] += int(correct[m].sum().item())
                sums[gid]["n"] += cnt
                lab_g = labels[m]
                sums[gid]["l0"] += int((lab_g == 0).sum().item())
                sums[gid]["l1"] += int((lab_g == 1).sum().item())

    per_center = {}
    for gid in group_ids:
        n = sums[gid]["n"]
        per_center[gid] = {
            "loss": (sums[gid]["loss"] / n) if n > 0 else None,
            "acc": (sums[gid]["correct"] / n) if n > 0 else None,
            "n": n,
            "l0": sums[gid]["l0"],
            "l1": sums[gid]["l1"],
        }
    overall = {
        "loss": (tot_loss / tot_n) if tot_n > 0 else None,
        "acc": (tot_correct / tot_n) if tot_n > 0 else None,
        "n": tot_n,
    }
    return {"overall": overall, "per_center": per_center}


def _train_one_epoch_groupdro(model, loader, gdro_loss, optimizer, group_field, group_ids, device):
    """One GroupDRO training pass. Returns sample-weighted mean loss/acc + q + group losses."""
    model.train()
    n_samples, sum_loss, sum_correct = 0, 0.0, 0
    # accumulate per-group mean-loss (weighted by batch group count) for reporting
    grp_loss_sum = {gid: 0.0 for gid in group_ids}
    grp_n = {gid: 0 for gid in group_ids}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        groups = groups_from_batch(batch, group_field, device)

        logits = model(images)
        robust_loss = gdro_loss(logits, labels, groups)   # updates q in place

        optimizer.zero_grad(set_to_none=True)
        robust_loss.backward()
        optimizer.step()

        bsz = int(labels.shape[0])
        preds = logits.detach().argmax(dim=1)
        sum_correct += int((preds == labels).sum().item())
        sum_loss += float(robust_loss.detach().item()) * bsz
        n_samples += bsz

        gl = gdro_loss.last_group_loss_dict()
        for gi, gid in enumerate(group_ids):
            cnt = int(gdro_loss.last_group_count[gi].item())
            if cnt > 0 and gl[gid] is not None:
                grp_loss_sum[gid] += gl[gid] * cnt
                grp_n[gid] += cnt

    avg_group_loss = {
        gid: (grp_loss_sum[gid] / grp_n[gid]) if grp_n[gid] > 0 else None
        for gid in group_ids
    }
    return {
        "loss_mean": sum_loss / n_samples if n_samples else 0.0,
        "accuracy_mean": sum_correct / n_samples if n_samples else 0.0,
        "n_samples": n_samples,
        "q": gdro_loss.q_dict(),
        "avg_group_loss": avg_group_loss,
    }


def main() -> int:
    root = project_root()
    cfg = load_yaml(root / CONFIG_REL)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    t_cfg = cfg["training"]
    g_cfg = cfg["groupdro"]
    s_cfg = cfg["selection"]
    o_cfg = cfg["output"]

    seed = int(t_cfg["seed"])
    set_seed(seed)
    ensure_results_tree()

    group_field = str(d_cfg["group_field"])
    group_ids = [int(g) for g in d_cfg["group_ids"]]
    batch_size = int(d_cfg["batch_size"])
    num_workers = int(d_cfg["num_workers"])
    epochs = int(t_cfg["epochs"])
    sel_metric = str(s_cfg["metric"])
    sel_mode = str(s_cfg["mode"])

    erm_run = str(o_cfg["erm_run_name"])
    gdro_run = str(o_cfg["groupdro_run_name"])
    ckpt_root = _resolve(o_cfg["checkpoint_root"])
    log_root = _resolve(o_cfg["log_root"])

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emit("=" * 78)
    emit("Milestone 7F - matched center-stratified ERM + GroupDRO (script 26)")
    emit(f"erm_run={erm_run}   groupdro_run={gdro_run}")
    emit(f"backbone={m_cfg['backbone_name']}  pretrained={m_cfg['pretrained']}  num_classes={m_cfg['num_classes']}")
    emit(f"optimizer={t_cfg['optimizer']}  lr={t_cfg['learning_rate']}  wd={t_cfg['weight_decay']}  epochs={epochs}  seed={seed}")
    emit(f"group_field={group_field}  group_ids={group_ids}")
    emit(f"groupdro step_size={g_cfg['step_size']}  normalize_group_weights={g_cfg['normalize_group_weights']}")
    emit(f"batch_size={batch_size}  num_workers={num_workers}  device={device}")
    emit(f"selection: metric={sel_metric}  mode={sel_mode}")
    emit("scope: train + id_val (center-stratified) ONLY - ood_val/ood_test untouched.")
    emit("       normal transforms (NO stain aug). NO CORAL/DANN/MC-dropout/ensemble/calibration.")
    emit("=" * 78)

    ok = True

    try:
        # --- no-ood-test guard on all target paths ------------------------------- #
        assert "ood_test" not in str(CONFIG_REL)
        for p in (d_cfg["train_cache"], d_cfg["id_val_cache"]):
            assert "ood_test" not in str(p) and "test" not in Path(str(p)).name, \
                f"training must not reference ood_test/test cache: {p}"

        # --- load center-stratified caches (train + id_val ONLY) ---------------- #
        emit("")
        emit("--- loading center-stratified caches (disk load; no streaming) ---")
        train_examples, train_cache_path = _load_center_strat_cache(d_cfg["train_cache"])
        id_val_examples, id_val_cache_path = _load_center_strat_cache(d_cfg["id_val_cache"])

        train_ds = BalancedSubsetDataset(train_examples, transform=train_transform)
        id_val_ds = BalancedSubsetDataset(id_val_examples, transform=eval_transform)

        train_summary = summarize_group_counts(train_examples, group_ids)
        id_val_summary = summarize_group_counts(id_val_examples, group_ids)

        train_centers = set(train_summary["per_group"].keys())
        id_val_centers = set(id_val_summary["per_group"].keys())
        train_labels = set(train_summary["labels"].keys())
        id_val_labels = set(id_val_summary["labels"].keys())

        emit("")
        emit(f"train  cache : {train_cache_path}  (n={len(train_ds)})")
        emit(f"id_val cache : {id_val_cache_path}  (n={len(id_val_ds)})")
        emit(f"train  label counts  : {dict(sorted(train_summary['labels'].items()))}")
        emit(f"id_val label counts  : {dict(sorted(id_val_summary['labels'].items()))}")
        emit(f"train  center counts : {dict(sorted(train_summary['per_group'].items()))}")
        emit(f"id_val center counts : {dict(sorted(id_val_summary['per_group'].items()))}")
        emit("train  (center,label) cells:")
        for gid in group_ids:
            emit(f"    center {gid}: l0={train_summary['per_cell'].get((gid,0),0)}  l1={train_summary['per_cell'].get((gid,1),0)}")
        emit("id_val (center,label) cells:")
        for gid in group_ids:
            emit(f"    center {gid}: l0={id_val_summary['per_cell'].get((gid,0),0)}  l1={id_val_summary['per_cell'].get((gid,1),0)}")

        # --- data assertions (both splits) -------------------------------------- #
        assert len(train_ds) == EXPECTED_TRAIN_N, f"train_n {len(train_ds)} != {EXPECTED_TRAIN_N}"
        assert len(id_val_ds) == EXPECTED_ID_VAL_N, f"id_val_n {len(id_val_ds)} != {EXPECTED_ID_VAL_N}"
        assert train_centers == EXPECTED_CENTERS, f"train centers {sorted(train_centers)} != {sorted(EXPECTED_CENTERS)}"
        assert id_val_centers == EXPECTED_CENTERS, f"id_val centers {sorted(id_val_centers)} != {sorted(EXPECTED_CENTERS)}"
        assert train_labels == {0, 1}, f"train labels {sorted(train_labels)} != [0,1]"
        assert id_val_labels == {0, 1}, f"id_val labels {sorted(id_val_labels)} != [0,1]"
        for gid in group_ids:
            for y in (0, 1):
                assert train_summary["per_cell"].get((gid, y), 0) > 0, f"train missing cell center={gid} label={y}"
                assert id_val_summary["per_cell"].get((gid, y), 0) > 0, f"id_val missing cell center={gid} label={y}"

        # sanity: all group ids finite / known
        assert all(g in group_ids for g in train_centers), "unexpected train center outside group_ids"

        results = {}  # run_name -> {best_epoch, best_val, best_path, reload_ok, ...}

        # ------------------------------------------------------------------ #
        # A. Center-stratified ERM baseline                                  #
        # ------------------------------------------------------------------ #
        emit("")
        emit("=" * 78)
        emit(f"[A] Training center-stratified ERM baseline -> {erm_run}")
        emit("=" * 78)
        set_seed(seed)  # identical init/order start point for both models
        gen_erm = torch.Generator(); gen_erm.manual_seed(seed)
        erm_train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            drop_last=False, pin_memory=False, generator=gen_erm,
        )
        erm_id_val_loader = DataLoader(
            id_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            drop_last=False, pin_memory=False,
        )
        erm_model = CamelyonClassifier(
            backbone_name=m_cfg["backbone_name"], pretrained=bool(m_cfg["pretrained"]),
            num_classes=int(m_cfg["num_classes"]),
        ).to(device)
        erm_criterion = create_loss("cross_entropy")
        erm_optimizer = create_optimizer(
            erm_model, name=t_cfg["optimizer"], lr=float(t_cfg["learning_rate"]),
            weight_decay=float(t_cfg["weight_decay"]),
        )

        erm_ckpt_dir = ckpt_root / erm_run
        ensure_dir(erm_ckpt_dir)
        erm_best_path = erm_ckpt_dir / "best.pt"
        erm_last_path = erm_ckpt_dir / "last.pt"
        erm_csv = log_root / f"{erm_run}_log.csv"

        erm_best_val, erm_best_epoch = None, None
        all_finite_erm = True
        erm_log_rows: list[dict] = []

        def _improved(cand, best):
            if best is None:
                return True
            return cand < best if sel_mode == "min" else cand > best

        for epoch in range(1, epochs + 1):
            tr = train_one_epoch(erm_model, erm_train_loader, erm_criterion, erm_optimizer, device)
            va = _group_aware_evaluate(erm_model, id_val_ds, group_field, group_ids, device, batch_size)
            va_loss = va["overall"]["loss"]
            va_acc = va["overall"]["acc"]
            if not (math.isfinite(tr["loss_mean"]) and math.isfinite(va_loss)):
                all_finite_erm = False

            worst_center = max(group_ids, key=lambda g: (va["per_center"][g]["loss"] if va["per_center"][g]["loss"] is not None else -1))
            worst_loss = va["per_center"][worst_center]["loss"]
            worst_acc = va["per_center"][worst_center]["acc"]

            metrics = {
                "epoch": epoch, "train_loss": tr["loss_mean"], "train_accuracy": tr["accuracy_mean"],
                "id_val_loss": va_loss, "id_val_accuracy": va_acc,
                "train_n": tr["n_samples"], "id_val_n": va["overall"]["n"],
                "worst_center": worst_center, "worst_center_id_val_loss": worst_loss,
                "worst_center_id_val_accuracy": worst_acc,
            }
            sel_val = float(metrics[sel_metric])
            is_best = _improved(sel_val, erm_best_val)
            save_checkpoint(erm_last_path, erm_model, erm_optimizer, epoch, metrics, cfg)
            if is_best:
                erm_best_val, erm_best_epoch = sel_val, epoch
                save_checkpoint(erm_best_path, erm_model, erm_optimizer, epoch, metrics, cfg)

            row = {
                "epoch": epoch, "train_loss": round(tr["loss_mean"], 6),
                "train_accuracy": round(tr["accuracy_mean"], 6),
                "id_val_loss": round(va_loss, 6), "id_val_accuracy": round(va_acc, 6),
                "worst_center": worst_center,
                "worst_center_id_val_loss": round(worst_loss, 6),
                "worst_center_id_val_accuracy": round(worst_acc, 6),
            }
            for g in group_ids:
                row[f"id_val_loss_c{g}"] = round(va["per_center"][g]["loss"], 6)
                row[f"id_val_acc_c{g}"] = round(va["per_center"][g]["acc"], 6)
            row["train_n"] = tr["n_samples"]; row["id_val_n"] = va["overall"]["n"]; row["is_best"] = int(is_best)
            erm_log_rows.append(row)
            emit(f"  epoch {epoch}/{epochs}  train_loss={tr['loss_mean']:.6f} train_acc={tr['accuracy_mean']:.4f}  "
                 f"id_val_loss={va_loss:.6f} id_val_acc={va_acc:.4f}  "
                 f"worst_center={worst_center} (loss={worst_loss:.6f} acc={worst_acc:.4f})"
                 f"{'  <- best' if is_best else ''}")
            emit(f"        per-center id_val: " + "  ".join(
                f"c{g}(loss={va['per_center'][g]['loss']:.4f},acc={va['per_center'][g]['acc']:.4f},n={va['per_center'][g]['n']})"
                for g in group_ids))

        write_csv(erm_csv, erm_log_rows, header=list(erm_log_rows[0].keys()))
        assert all_finite_erm, "ERM: non-finite loss during training/eval"
        assert erm_best_path.exists(), "ERM best.pt missing"
        assert erm_last_path.exists(), "ERM last.pt missing"

        # reload verification (ERM)
        fresh_erm = CamelyonClassifier(
            backbone_name=m_cfg["backbone_name"], pretrained=bool(m_cfg["pretrained"]),
            num_classes=int(m_cfg["num_classes"]),
        )
        payload = load_checkpoint(erm_best_path, fresh_erm, optimizer=None, map_location="cpu")
        fresh_erm.to(device)
        reload_va = _group_aware_evaluate(fresh_erm, id_val_ds, group_field, group_ids, device, batch_size)
        ck_loss = float(payload["metrics"]["id_val_loss"]); ck_acc = float(payload["metrics"]["id_val_accuracy"])
        rl_loss = float(reload_va["overall"]["loss"]); rl_acc = float(reload_va["overall"]["acc"])
        erm_reload_ok = (math.isclose(rl_loss, ck_loss, rel_tol=1e-5, abs_tol=1e-5)
                         and math.isclose(rl_acc, ck_acc, rel_tol=1e-5, abs_tol=1e-5)
                         and math.isfinite(rl_loss))
        assert math.isfinite(rl_loss), "ERM reload produced non-finite id_val loss"
        emit("")
        emit(f"[A] ERM best epoch={erm_best_epoch}  id_val_loss(ckpt)={ck_loss:.6f} (reload)={rl_loss:.6f}  "
             f"reload_verify={'PASS' if erm_reload_ok else 'FAIL'}")
        if not erm_reload_ok:
            ok = False
        results[erm_run] = {"best_epoch": erm_best_epoch, "best_val": erm_best_val,
                            "best_path": str(erm_best_path), "reload_ok": erm_reload_ok}

        # ------------------------------------------------------------------ #
        # B. Center-stratified GroupDRO by center                            #
        # ------------------------------------------------------------------ #
        emit("")
        emit("=" * 78)
        emit(f"[B] Training center-stratified GroupDRO by center -> {gdro_run}")
        emit("=" * 78)
        set_seed(seed)  # matched init/order start point vs ERM
        gen_gdro = torch.Generator(); gen_gdro.manual_seed(seed)
        gdro_train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            drop_last=False, pin_memory=False, generator=gen_gdro,
        )
        gdro_model = CamelyonClassifier(
            backbone_name=m_cfg["backbone_name"], pretrained=bool(m_cfg["pretrained"]),
            num_classes=int(m_cfg["num_classes"]),
        ).to(device)
        gdro_loss = GroupDROLoss(
            group_ids=group_ids, step_size=float(g_cfg["step_size"]),
            normalize_group_weights=bool(g_cfg["normalize_group_weights"]), device=device,
        )
        gdro_loss.reset()
        gdro_optimizer = create_optimizer(
            gdro_model, name=t_cfg["optimizer"], lr=float(t_cfg["learning_rate"]),
            weight_decay=float(t_cfg["weight_decay"]),
        )

        gdro_ckpt_dir = ckpt_root / gdro_run
        ensure_dir(gdro_ckpt_dir)
        gdro_best_path = gdro_ckpt_dir / "best.pt"
        gdro_last_path = gdro_ckpt_dir / "last.pt"
        gdro_csv = log_root / f"{gdro_run}_log.csv"

        gdro_best_val, gdro_best_epoch = None, None
        all_finite_gdro = True
        q_history = []
        gdro_log_rows: list[dict] = []

        for epoch in range(1, epochs + 1):
            tr = _train_one_epoch_groupdro(gdro_model, gdro_train_loader, gdro_loss, gdro_optimizer,
                                           group_field, group_ids, device)
            va = _group_aware_evaluate(gdro_model, id_val_ds, group_field, group_ids, device, batch_size)
            va_loss = va["overall"]["loss"]; va_acc = va["overall"]["acc"]
            if not (math.isfinite(tr["loss_mean"]) and math.isfinite(va_loss)):
                all_finite_gdro = False

            worst_center = max(group_ids, key=lambda g: (va["per_center"][g]["loss"] if va["per_center"][g]["loss"] is not None else -1))
            worst_loss = va["per_center"][worst_center]["loss"]
            worst_acc = va["per_center"][worst_center]["acc"]
            q = tr["q"]
            q_history.append({"epoch": epoch, **{f"q_c{g}": q[g] for g in group_ids}})

            metrics = {
                "epoch": epoch, "train_loss": tr["loss_mean"], "train_accuracy": tr["accuracy_mean"],
                "id_val_loss": va_loss, "id_val_accuracy": va_acc,
                "train_n": tr["n_samples"], "id_val_n": va["overall"]["n"],
                "worst_center": worst_center, "worst_center_id_val_loss": worst_loss,
                "worst_center_id_val_accuracy": worst_acc,
                "q_weights": {str(g): q[g] for g in group_ids},
            }
            sel_val = float(metrics[sel_metric])
            is_best = _improved(sel_val, gdro_best_val)
            save_checkpoint(gdro_last_path, gdro_model, gdro_optimizer, epoch, metrics, cfg)
            if is_best:
                gdro_best_val, gdro_best_epoch = sel_val, epoch
                save_checkpoint(gdro_best_path, gdro_model, gdro_optimizer, epoch, metrics, cfg)

            gdro_row = {
                "epoch": epoch, "train_robust_loss": round(tr["loss_mean"], 6),
                "train_accuracy": round(tr["accuracy_mean"], 6),
                "id_val_loss": round(va_loss, 6), "id_val_accuracy": round(va_acc, 6),
                "worst_center": worst_center,
                "worst_center_id_val_loss": round(worst_loss, 6),
                "worst_center_id_val_accuracy": round(worst_acc, 6),
                **{f"q_c{g}": round(q[g], 6) for g in group_ids},
                **{f"group_loss_c{g}": (round(tr["avg_group_loss"][g], 6) if tr["avg_group_loss"][g] is not None else "")
                   for g in group_ids},
                **{f"id_val_loss_c{g}": round(va["per_center"][g]["loss"], 6) for g in group_ids},
                **{f"id_val_acc_c{g}": round(va["per_center"][g]["acc"], 6) for g in group_ids},
                "train_n": tr["n_samples"], "id_val_n": va["overall"]["n"], "is_best": int(is_best),
            }
            gdro_log_rows.append(gdro_row)
            emit(f"  epoch {epoch}/{epochs}  train_robust_loss={tr['loss_mean']:.6f} train_acc={tr['accuracy_mean']:.4f}  "
                 f"id_val_loss={va_loss:.6f} id_val_acc={va_acc:.4f}  "
                 f"worst_center={worst_center} (loss={worst_loss:.6f} acc={worst_acc:.4f})"
                 f"{'  <- best' if is_best else ''}")
            emit(f"        q weights: " + "  ".join(f"c{g}={q[g]:.4f}" for g in group_ids))
            emit(f"        avg group loss: " + "  ".join(
                f"c{g}=" + (f"{tr['avg_group_loss'][g]:.4f}" if tr['avg_group_loss'][g] is not None else "None")
                for g in group_ids))
            emit(f"        per-center id_val: " + "  ".join(
                f"c{g}(loss={va['per_center'][g]['loss']:.4f},acc={va['per_center'][g]['acc']:.4f},n={va['per_center'][g]['n']})"
                for g in group_ids))

        write_csv(gdro_csv, gdro_log_rows, header=list(gdro_log_rows[0].keys()))
        assert all_finite_gdro, "GroupDRO: non-finite loss during training/eval"
        assert gdro_best_path.exists(), "GroupDRO best.pt missing"
        assert gdro_last_path.exists(), "GroupDRO last.pt missing"

        # reload verification (GroupDRO)
        fresh_gdro = CamelyonClassifier(
            backbone_name=m_cfg["backbone_name"], pretrained=bool(m_cfg["pretrained"]),
            num_classes=int(m_cfg["num_classes"]),
        )
        payload = load_checkpoint(gdro_best_path, fresh_gdro, optimizer=None, map_location="cpu")
        fresh_gdro.to(device)
        reload_va = _group_aware_evaluate(fresh_gdro, id_val_ds, group_field, group_ids, device, batch_size)
        ck_loss = float(payload["metrics"]["id_val_loss"]); ck_acc = float(payload["metrics"]["id_val_accuracy"])
        rl_loss = float(reload_va["overall"]["loss"]); rl_acc = float(reload_va["overall"]["acc"])
        gdro_reload_ok = (math.isclose(rl_loss, ck_loss, rel_tol=1e-5, abs_tol=1e-5)
                          and math.isclose(rl_acc, ck_acc, rel_tol=1e-5, abs_tol=1e-5)
                          and math.isfinite(rl_loss))
        assert math.isfinite(rl_loss), "GroupDRO reload produced non-finite id_val loss"
        emit("")
        emit(f"[B] GroupDRO best epoch={gdro_best_epoch}  id_val_loss(ckpt)={ck_loss:.6f} (reload)={rl_loss:.6f}  "
             f"reload_verify={'PASS' if gdro_reload_ok else 'FAIL'}")
        emit(f"    final q weights: {gdro_loss.q_dict()}")
        if not gdro_reload_ok:
            ok = False
        results[gdro_run] = {"best_epoch": gdro_best_epoch, "best_val": gdro_best_val,
                             "best_path": str(gdro_best_path), "reload_ok": gdro_reload_ok,
                             "final_q": gdro_loss.q_dict()}

        # --- summary ------------------------------------------------------------ #
        emit("")
        emit("--- summary ---")
        emit(f"train_n={len(train_ds)} (expected {EXPECTED_TRAIN_N})  id_val_n={len(id_val_ds)} (expected {EXPECTED_ID_VAL_N})")
        emit(f"train centers={sorted(train_centers)}  id_val centers={sorted(id_val_centers)}")
        emit(f"ERM      best epoch={erm_best_epoch}  best {sel_metric}={erm_best_val:.6f}  reload={'PASS' if erm_reload_ok else 'FAIL'}")
        emit(f"           best.pt -> {erm_best_path}")
        emit(f"GroupDRO best epoch={gdro_best_epoch}  best {sel_metric}={gdro_best_val:.6f}  reload={'PASS' if gdro_reload_ok else 'FAIL'}")
        emit(f"           best.pt -> {gdro_best_path}")
        emit(f"GroupDRO q history: {q_history}")
        emit("ood_val usage : NONE   ood_test usage : NONE")

    except Exception as exc:
        ok = False
        emit("")
        emit(f"ERROR: {type(exc).__name__}: {exc}")
        import traceback
        emit(traceback.format_exc())

    emit("")
    emit("=" * 78)
    emit(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    emit("=" * 78)

    report_path = log_root / "exp07f_centerstrat_erm_groupdro_train_report.txt"
    write_text(report_path, "\n".join(lines) + "\n")
    print(f"\nSaved report -> {report_path}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
