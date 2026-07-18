"""Milestone 7F-0 / Script 24: DG feasibility / group-label metadata audit.

A **metadata / cache audit only** — NO training, NO model inference, NO ``ood_test``,
no cache/prediction/checkpoint/metrics modification. It answers a single question before
any dedicated domain-generalization (DG) training is proposed:

    Do the existing balanced caches preserve enough group metadata (center / patient /
    slide) for GroupDRO / CORAL / DANN, and do they contain more than one source domain
    in ``train``?

For each cached balanced subset (train / id_val / ood_val) it inspects the serialized
payload written by ``src/data/balanced_subset.py`` (a dict with keys ``version, n,
images, labels, meta``; ``meta`` holds the integer fields ``center, image_id, patient,
node, x_coord, y_coord, slide``), then:
  * records object type, fields, example count, label counts, image presence, and which
    metadata fields are present,
  * quantifies label x center and label x slide counts,
  * derives whether GroupDRO / CORAL / DANN are feasible *immediately* on the cache
    (need >= 2 distinct source domains in train), and whether id_val supports
    source-domain validation summaries and ood_val's center is identifiable,
  * if any grouping metadata is missing or single-domain, reports what is missing, notes
    the original HF dataset DOES carry these fields (Milestone 1 verification), and
    whether a metadata-preserving cache REBUILD is required (proposal only — no rebuild
    here).

Outputs (under results/): three CSV tables + one text report. Reads ONLY the three
balanced caches. ``ood_test`` is never referenced.

NOT a final result — a development-stage feasibility audit on capped balanced subsets.
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup; kept for repo convention)

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from utils.io import (
    ensure_results_tree,
    logs_dir,
    project_root,
    tables_dir,
    tee,
    write_csv,
    write_text,
)

# ---------------------------------------------------------------------------
# Configuration (audit inputs). ood_test is intentionally absent and never used.
# ---------------------------------------------------------------------------

# (logical split, relative cache path, expected example count)
CACHES = [
    ("train", "results/cache/balanced_train_pc1000_seed1337_shuf.pt", 2000),
    ("id_val", "results/cache/balanced_id_val_pc250_seed1337_shuf.pt", 500),
    ("ood_val", "results/cache/balanced_ood_val_pc1000_seed1337_noshuf.pt", 2000),
]

# All integer metadata fields the cache serializer stores (balanced_subset.META_INT_FIELDS).
META_FIELDS = ("center", "patient", "slide", "node", "image_id", "x_coord", "y_coord")

# Candidate DG grouping fields (a "domain"/"group" label for GroupDRO/CORAL/DANN).
GROUP_FIELDS = ("center", "patient", "slide")

# Source domains the locked mapping assigns to train / id_val (centers {0,3,4}).
EXPECTED_SOURCE_CENTERS = {0, 3, 4}

# DG methods audited and the grouping field they would use here.
DG_METHODS = [
    ("GroupDRO", "center"),
    ("CORAL", "center"),
    ("DANN", "center"),
]


def _load_payload(abs_path: Path):
    """Load a cache payload read-only. Returns (payload, error_str_or_None)."""
    try:
        payload = torch.load(abs_path, weights_only=False)
        return payload, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{type(exc).__name__}: {exc}"


def _field_values(payload, field):
    """Return the list of integer values for a meta field, or None if absent/malformed."""
    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta")
    if not isinstance(meta, dict) or field not in meta:
        return None
    vals = meta[field]
    try:
        return [int(v) for v in vals]
    except Exception:  # pragma: no cover - defensive
        return None


def main() -> int:
    ensure_results_tree()
    root = project_root()
    log: list[str] = []
    warnings: list[str] = []

    tee("=" * 78, log)
    tee("Milestone 7F-0 / Script 24: DG feasibility / group-label metadata audit", log)
    tee("(metadata/cache audit only - NO training, NO inference, NO ood_test)", log)
    tee("=" * 78, log)

    # Guard: none of the audited paths may reference the locked test split.
    for _, rel, _ in CACHES:
        assert "ood_test" not in rel and "test" not in rel.split("/")[-1], (
            f"ood_test must never be audited (offending path: {rel})"
        )

    per_split = {}  # split -> dict of computed facts

    # -----------------------------------------------------------------------
    # 1) Load + inspect each cache.
    # -----------------------------------------------------------------------
    for split, rel, expected_n in CACHES:
        abs_path = root / rel
        tee("", log)
        tee(f"[{split}] cache: {rel}", log)

        facts = {
            "split": split,
            "cache_file": rel,
            "expected_n": expected_n,
            "exists": abs_path.exists(),
            "object_type": "",
            "keys": [],
            "n_examples": None,
            "label_0_count": None,
            "label_1_count": None,
            "labels_finite": None,
            "images_present": False,
            "load_error": "",
        }
        for f in META_FIELDS:
            facts[f"has_{f}"] = False
        for f in GROUP_FIELDS:
            facts[f"distinct_{f}"] = None
        facts["values"] = {}  # field -> list[int] (kept in-memory for counting)
        facts["labels"] = None

        if not abs_path.exists():
            warnings.append(f"[{split}] cache file missing: {rel}")
            tee(f"  MISSING: {abs_path}", log)
            per_split[split] = facts
            continue

        payload, err = _load_payload(abs_path)
        facts["object_type"] = type(payload).__name__
        if err is not None:
            facts["load_error"] = err
            warnings.append(f"[{split}] load error: {err}")
            tee(f"  LOAD ERROR: {err}", log)
            per_split[split] = facts
            continue

        if isinstance(payload, dict):
            facts["keys"] = list(payload.keys())
        tee(f"  object_type = {facts['object_type']}; keys = {facts['keys']}", log)

        # n
        n = None
        if isinstance(payload, dict) and "n" in payload:
            try:
                n = int(payload["n"])
            except Exception:
                n = None
        facts["n_examples"] = n

        # labels
        labels = None
        if isinstance(payload, dict) and "labels" in payload:
            try:
                labels = [int(v) for v in payload["labels"]]
            except Exception:
                labels = None
        facts["labels"] = labels
        if labels is not None:
            lc = Counter(labels)
            facts["label_0_count"] = int(lc.get(0, 0))
            facts["label_1_count"] = int(lc.get(1, 0))
            finite = all(np.isfinite(labels)) and set(labels).issubset({0, 1})
            facts["labels_finite"] = bool(finite)
            if n is None:
                facts["n_examples"] = len(labels)
                n = len(labels)
            tee(
                f"  n = {facts['n_examples']}; labels: 0={facts['label_0_count']}, "
                f"1={facts['label_1_count']}; finite/binary = {facts['labels_finite']}",
                log,
            )
        else:
            warnings.append(f"[{split}] labels missing/malformed")
            tee("  labels: MISSING/MALFORMED", log)

        # images present (no decode - just structural presence)
        if isinstance(payload, dict) and "images" in payload:
            imgs = payload["images"]
            try:
                facts["images_present"] = len(imgs) == (n if n is not None else len(imgs))
            except Exception:
                facts["images_present"] = bool(imgs is not None)
        tee(f"  images_present = {facts['images_present']}", log)

        # metadata fields
        present_meta = []
        for field in META_FIELDS:
            vals = _field_values(payload, field)
            present = vals is not None and (n is None or len(vals) == n)
            facts[f"has_{field}"] = bool(present)
            if present:
                present_meta.append(field)
                facts["values"][field] = vals
                if field in GROUP_FIELDS:
                    facts[f"distinct_{field}"] = int(len(set(vals)))
        tee(f"  metadata present: {present_meta}", log)
        missing_meta = [f for f in META_FIELDS if not facts[f"has_{f}"]]
        if missing_meta:
            tee(f"  metadata MISSING: {missing_meta}", log)
        for gf in GROUP_FIELDS:
            if facts["values"].get(gf) is not None:
                tee(f"  distinct {gf} = {facts[f'distinct_{gf}']} "
                    f"-> values {sorted(set(facts['values'][gf]))}", log)

        per_split[split] = facts

    # -----------------------------------------------------------------------
    # 2) Group-label feasibility derivations.
    # -----------------------------------------------------------------------
    train = per_split.get("train", {})
    id_val = per_split.get("id_val", {})
    ood_val = per_split.get("ood_val", {})

    train_centers = sorted(set(train.get("values", {}).get("center", []))) \
        if train.get("has_center") else []
    id_centers = sorted(set(id_val.get("values", {}).get("center", []))) \
        if id_val.get("has_center") else []
    ood_centers = sorted(set(ood_val.get("values", {}).get("center", []))) \
        if ood_val.get("has_center") else []

    train_multi_center = len(train_centers) >= 2
    train_has_center = bool(train.get("has_center"))

    gdro_feasible = train_has_center and train_multi_center
    coral_dann_feasible = train_has_center and train_multi_center

    id_val_source_domain_ok = bool(id_val.get("has_center")) and len(id_centers) >= 1
    ood_val_center_identifiable = bool(ood_val.get("has_center")) and len(ood_centers) >= 1

    # ---- Soundness check: is center confounded with label in the train cache? ----
    # DG-by-center needs the group (center) to be reasonably decorrelated from the label;
    # if a center is (near-)single-class, GroupDRO/CORAL/DANN would learn center==label
    # instead of a domain-invariant tumor signal. Quantify per-center tumor fraction and
    # the coverage of the locked source centers {0,3,4}.
    train_center_label = {}  # center -> {0: n0, 1: n1}
    if train.get("has_center") and train.get("labels") is not None:
        tc = train["values"]["center"]
        tl = train["labels"]
        for c, y in zip(tc, tl):
            d = train_center_label.setdefault(int(c), {0: 0, 1: 0})
            d[int(y)] += 1
    train_center_tumor_frac = {
        c: (d[1] / (d[0] + d[1]) if (d[0] + d[1]) else None)
        for c, d in train_center_label.items()
    }
    # A center is "single-class" if its tumor fraction is <=5% or >=95%.
    single_class_centers = [
        c for c, frac in train_center_tumor_frac.items()
        if frac is not None and (frac <= 0.05 or frac >= 0.95)
    ]
    train_center_label_confounded = len(single_class_centers) > 0
    missing_source_centers = sorted(EXPECTED_SOURCE_CENTERS - set(train_centers))
    train_covers_all_source_centers = len(missing_source_centers) == 0

    # DG-by-center is "immediately runnable" (>=2 groups + metadata) but only "sound"
    # if the groups are not label-confounded.
    dg_by_center_runnable = gdro_feasible  # >=2 center groups + metadata
    dg_by_center_sound = dg_by_center_runnable and not train_center_label_confounded

    tee("", log)
    tee("-" * 78, log)
    tee("Group-label feasibility", log)
    tee("-" * 78, log)
    tee(f"  train centers present: {train_has_center}; distinct = {train_centers}", log)
    tee(f"  train multiple source centers (>=2): {train_multi_center}", log)
    tee(f"  train covers all locked source centers {sorted(EXPECTED_SOURCE_CENTERS)}: "
        f"{train_covers_all_source_centers} (missing={missing_source_centers})", log)
    tee(f"  train per-center tumor fraction: "
        f"{ {c: round(v, 3) if v is not None else None for c, v in sorted(train_center_tumor_frac.items())} }", log)
    tee(f"  train center<->label confounded (a center is ~single-class): "
        f"{train_center_label_confounded} (single-class centers={single_class_centers})", log)
    tee(f"  id_val centers: {id_centers} (source-domain validation ok = {id_val_source_domain_ok})", log)
    tee(f"  ood_val centers: {ood_centers} (identifiable = {ood_val_center_identifiable})", log)
    tee(f"  DG-by-center RUNNABLE immediately (>=2 groups + metadata): {dg_by_center_runnable}", log)
    tee(f"  DG-by-center SOUND immediately (groups not label-confounded): {dg_by_center_sound}", log)
    tee(f"  GroupDRO-by-center feasible immediately: {gdro_feasible}", log)
    tee(f"  CORAL/DANN-by-center feasible immediately: {coral_dann_feasible}", log)

    # -----------------------------------------------------------------------
    # 3) Missing-metadata / rebuild reasoning.
    # -----------------------------------------------------------------------
    # The original HF dataset carries these fields (Milestone 1 verification):
    #   image, label, center, image_id, patient, node, x_coord, y_coord, slide.
    hf_has_metadata_fields = list(META_FIELDS)

    metadata_all_present = all(
        per_split.get(s, {}).get(f"has_{gf}", False)
        for s in ("train", "id_val", "ood_val")
        for gf in ("center",)
    )

    # Cache rebuild is needed if either (a) center metadata is absent from any cache, or
    # (b) it is present but train cannot form >=2 sound DG groups -- i.e. single-domain,
    # label-confounded groups, or missing locked source centers.
    rebuild_needed = (
        (not metadata_all_present)
        or (train_has_center and not train_multi_center)
        or train_center_label_confounded
        or (not train_covers_all_source_centers)
    )

    if not metadata_all_present:
        rebuild_reason = (
            "center metadata absent from one or more caches; the original HF dataset "
            "DOES carry it, so a metadata-preserving rebuild is required."
        )
    elif train_has_center and not train_multi_center:
        rebuild_reason = (
            "center metadata present but the train cache is single-domain "
            f"(centers={train_centers}); DG-by-center needs >=2 source domains, so a "
            "stratified rebuild that samples across centers {0,3,4} is required."
        )
    elif train_center_label_confounded or not train_covers_all_source_centers:
        rebuild_reason = (
            "center metadata present and train has >=2 centers, BUT the DG groups are "
            f"UNSOUND for center-based DG: per-center tumor fraction = "
            f"{ {c: round(v, 3) if v is not None else None for c, v in sorted(train_center_tumor_frac.items())} } "
            f"(single-class centers={single_class_centers}), and locked source centers "
            f"missing from train = {missing_source_centers}. The balanced cache was built "
            "by scanning the label-ordered stream to fill per-class quotas, which drew "
            "center 0 as (near-)all non-tumor and center 3 as all-tumor and omitted center "
            "4 -- so center is confounded with label. A center-STRATIFIED rebuild is "
            "required before center-based GroupDRO/CORAL/DANN would be meaningful."
        )
    else:
        rebuild_reason = (
            "center metadata present AND train spans >=2 label-decorrelated source "
            "domains covering {0,3,4}; no rebuild required for DG-by-center."
        )

    # -----------------------------------------------------------------------
    # 4) Write tables.
    # -----------------------------------------------------------------------
    tables = tables_dir()
    logs = logs_dir()

    # Table 1: per-split metadata summary.
    summary_rows = []
    for split, _, expected_n in CACHES:
        f = per_split[split]
        summary_rows.append({
            "split": split,
            "cache_file": f["cache_file"],
            "exists": f["exists"],
            "object_type": f["object_type"],
            "n_examples": f["n_examples"],
            "expected_n": expected_n,
            "n_matches_expected": (f["n_examples"] == expected_n),
            "label_0_count": f["label_0_count"],
            "label_1_count": f["label_1_count"],
            "labels_finite_binary": f["labels_finite"],
            "images_present": f["images_present"],
            "has_center": f["has_center"],
            "has_patient": f["has_patient"],
            "has_slide": f["has_slide"],
            "has_node": f["has_node"],
            "has_image_id": f["has_image_id"],
            "has_x_coord": f["has_x_coord"],
            "has_y_coord": f["has_y_coord"],
            "distinct_center": f["distinct_center"],
            "distinct_patient": f["distinct_patient"],
            "distinct_slide": f["distinct_slide"],
            "load_error": f["load_error"],
        })
    summary_path = tables / "exp07f0_cache_metadata_summary.csv"
    write_csv(summary_path, summary_rows)

    # Table 2: label x group counts (long format) for center and slide where present.
    group_rows = []
    for split, _, _ in CACHES:
        f = per_split[split]
        labels = f.get("labels")
        if labels is None:
            continue
        for gf in ("center", "slide"):
            vals = f.get("values", {}).get(gf)
            if vals is None:
                continue
            pair_counts = Counter(zip(labels, vals))
            for (lab, gval), cnt in sorted(pair_counts.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                group_rows.append({
                    "split": split,
                    "group_field": gf,
                    "group_value": int(gval),
                    "label": int(lab),
                    "count": int(cnt),
                })
    group_path = tables / "exp07f0_group_label_counts.csv"
    if not group_rows:
        # still write a header-only file so downstream/audit expectations hold
        group_rows = [{"split": "", "group_field": "", "group_value": "",
                       "label": "", "count": ""}]
        write_csv(group_path, group_rows)
        group_rows = []
    else:
        write_csv(group_path, group_rows)

    # Table 3: DG method feasibility.
    feas_rows = []
    for method, gf in DG_METHODS:
        train_field_present = bool(train.get(f"has_{gf}"))
        train_distinct = train.get(f"distinct_{gf}")
        runnable = train_field_present and (train_distinct is not None and train_distinct >= 2)
        # "sound" only if groups are also not label-confounded and source coverage complete.
        sound = runnable and dg_by_center_sound and train_covers_all_source_centers
        if not train_field_present:
            note = f"train cache lacks '{gf}' metadata"
        elif train_distinct is not None and train_distinct < 2:
            note = f"train cache single-domain ({gf} distinct={train_distinct}); needs >=2"
        elif not sound:
            note = (
                f"runnable ({train_distinct} {gf} groups) but UNSOUND: center<->label "
                f"confounded (single-class centers={single_class_centers}), "
                f"missing source centers={missing_source_centers}; stratified rebuild advised"
            )
        else:
            note = f"train cache has {train_distinct} sound, label-decorrelated {gf} domains"
        feas_rows.append({
            "method": method,
            "grouping_field": gf,
            "train_has_field": train_field_present,
            "train_distinct_groups": train_distinct,
            "runnable_immediately": runnable,
            "sound_immediately": sound,
            "feasible_immediately": sound,
            "requires": "training (gated, Milestone 7F)",
            "note": note,
        })
    # Extra summary rows for id_val / ood_val roles.
    feas_rows.append({
        "method": "id_val source-domain validation",
        "grouping_field": "center",
        "train_has_field": bool(id_val.get("has_center")),
        "train_distinct_groups": (len(id_centers) if id_val.get("has_center") else None),
        "runnable_immediately": id_val_source_domain_ok,
        "sound_immediately": id_val_source_domain_ok,
        "feasible_immediately": id_val_source_domain_ok,
        "requires": "no training (metadata only)",
        "note": f"id_val centers={id_centers}",
    })
    feas_rows.append({
        "method": "ood_val center identifiable",
        "grouping_field": "center",
        "train_has_field": bool(ood_val.get("has_center")),
        "train_distinct_groups": (len(ood_centers) if ood_val.get("has_center") else None),
        "runnable_immediately": ood_val_center_identifiable,
        "sound_immediately": ood_val_center_identifiable,
        "feasible_immediately": ood_val_center_identifiable,
        "requires": "no training (metadata only)",
        "note": f"ood_val centers={ood_centers}",
    })
    feas_path = tables / "exp07f0_dg_method_feasibility.csv"
    write_csv(feas_path, feas_rows)

    # -----------------------------------------------------------------------
    # 5) Recommendation + report.
    # -----------------------------------------------------------------------
    if dg_by_center_sound and not rebuild_needed:
        recommendation = (
            "B. START GroupDRO/CORAL/DANN training (gated Milestone 7F) - metadata is "
            "present and train spans >=2 label-decorrelated source domains."
        )
    else:
        recommendation = (
            "A. REBUILD metadata-rich, center-STRATIFIED balanced caches before any DG "
            "training. Group metadata IS present and >=2 centers exist, so 7F is not "
            "blocked by missing metadata; however the current train cache is center<->label "
            f"confounded (single-class centers={single_class_centers}) and omits source "
            f"center(s) {missing_source_centers}, so center-based GroupDRO/CORAL/DANN would "
            "learn center==label rather than a domain-invariant tumor signal. A stratified "
            "rebuild (below) is required for a SOUND DG-by-center experiment."
        )

    # Minimal cache-rebuild design (proposal only; no rebuild performed here).
    rebuild_design = [
        "Recommended minimal metadata-rich cache-rebuild design (PROPOSAL ONLY - NOT run here):",
        "  - Keep the locked split mapping and per_class balance unchanged (train pc1000,",
        "    id_val pc250, ood_val pc1000; seeds/order preserved for reproducibility).",
        "  - When collecting the balanced TRAIN subset, stratify sampling across source",
        "    centers {0,3,4} so each center contributes examples of BOTH labels (e.g. fill",
        "    per (center,label) quotas), instead of taking the first per_class of each label",
        "    in raw stream order (which can be single-center). This yields >=2 DG groups.",
        "  - Continue serializing all META_INT_FIELDS (center, patient, slide, node,",
        "    image_id, x_coord, y_coord) - the current serializer already does this, so the",
        "    only change is the SAMPLING policy, not the payload schema.",
        "  - Write to NEW cache filenames (e.g. *_stratcenter_*.pt); do NOT overwrite the",
        "    existing caches (keeps 4A/5A/7A-7E reproducibility intact).",
        "  - This rebuild is itself a separate, explicitly-gated milestone (requires HF",
        "    streaming); it is NOT performed in 7F-0.",
    ]

    report = []
    report.append("=" * 78)
    report.append("Milestone 7F-0 - DG feasibility / group-label metadata audit")
    report.append("Metadata/cache audit only: NO training, NO inference, NO ood_test.")
    report.append("=" * 78)
    report.append("")
    report.append("Inputs audited (read-only; caches NOT modified):")
    for _, rel, _ in CACHES:
        report.append(f"  - {rel}")
    report.append("")
    report.append("-" * 78)
    report.append("1) Cache contents (per split)")
    report.append("-" * 78)
    for split, _, expected_n in CACHES:
        f = per_split[split]
        report.append(f"[{split}]")
        report.append(f"  exists={f['exists']}  object_type={f['object_type']}  "
                      f"keys={f['keys']}")
        report.append(f"  n_examples={f['n_examples']} (expected {expected_n}, "
                      f"match={f['n_examples'] == expected_n})")
        report.append(f"  label_0={f['label_0_count']}  label_1={f['label_1_count']}  "
                      f"labels_finite_binary={f['labels_finite']}")
        report.append(f"  images_present={f['images_present']}")
        present_meta = [m for m in META_FIELDS if f[f'has_{m}']]
        missing_meta = [m for m in META_FIELDS if not f[f'has_{m}']]
        report.append(f"  metadata present: {present_meta}")
        report.append(f"  metadata missing: {missing_meta if missing_meta else '(none)'}")
        for gf in GROUP_FIELDS:
            d = f[f"distinct_{gf}"]
            if d is not None:
                vals = sorted(set(f["values"][gf]))
                shown = vals if len(vals) <= 12 else vals[:12] + ["..."]
                report.append(f"  distinct {gf} = {d}  values={shown}")
        report.append("")
    report.append("-" * 78)
    report.append("2) Label x group counts")
    report.append("-" * 78)
    if group_rows:
        for split, _, _ in CACHES:
            rows = [r for r in group_rows if r["split"] == split]
            for gf in ("center", "slide"):
                sub = [r for r in rows if r["group_field"] == gf]
                if not sub:
                    continue
                report.append(f"[{split}] label x {gf}:")
                for r in sub:
                    report.append(f"    {gf}={r['group_value']:>6}  "
                                  f"label={r['label']}  count={r['count']}")
        report.append("")
    else:
        report.append("  (no center/slide group counts available)")
        report.append("")
    report.append("-" * 78)
    report.append("3) Group-label feasibility")
    report.append("-" * 78)
    report.append(f"  train has center metadata: {train_has_center}")
    report.append(f"  train distinct centers: {train_centers}")
    report.append(f"  train multiple source centers (>=2): {train_multi_center}")
    report.append(f"  train covers all locked source centers {sorted(EXPECTED_SOURCE_CENTERS)}: "
                  f"{train_covers_all_source_centers} (missing={missing_source_centers})")
    report.append(f"  train per-center tumor fraction: "
                  f"{ {c: round(v, 3) if v is not None else None for c, v in sorted(train_center_tumor_frac.items())} }")
    report.append(f"  train center<->label confounded (a center ~single-class): "
                  f"{train_center_label_confounded} (single-class centers={single_class_centers})")
    report.append(f"  id_val centers: {id_centers}  "
                  f"(source-domain validation ok = {id_val_source_domain_ok})")
    report.append(f"  ood_val centers: {ood_centers}  "
                  f"(identifiable = {ood_val_center_identifiable})")
    report.append(f"  DG-by-center RUNNABLE immediately (>=2 groups + metadata): {dg_by_center_runnable}")
    report.append(f"  DG-by-center SOUND immediately (groups not label-confounded): {dg_by_center_sound}")
    report.append(f"  GroupDRO-by-center feasible immediately (sound): {dg_by_center_sound}")
    report.append(f"  CORAL/DANN-by-center feasible immediately (sound): {dg_by_center_sound}")
    report.append("")
    report.append("-" * 78)
    report.append("4) Missing-metadata / rebuild reasoning")
    report.append("-" * 78)
    report.append(f"  Original HF dataset carries (Milestone 1 verification): "
                  f"{hf_has_metadata_fields}")
    report.append(f"  All caches carry 'center' metadata: {metadata_all_present}")
    report.append(f"  Cache rebuild needed: {rebuild_needed}")
    report.append(f"  Reason: {rebuild_reason}")
    report.append("")
    report.extend("  " + line for line in rebuild_design)
    report.append("")
    report.append("-" * 78)
    report.append("5) Conclusion")
    report.append("-" * 78)
    report.append(f"  Group metadata present (all caches):   {metadata_all_present}")
    report.append(f"  DG-by-center RUNNABLE immediately:     {dg_by_center_runnable}")
    report.append(f"  GroupDRO feasible immediately (sound): {dg_by_center_sound}")
    report.append(f"  CORAL/DANN feasible immediately (sound): {dg_by_center_sound}")
    report.append(f"  Cache rebuild needed:                  {rebuild_needed}")
    report.append(f"  RECOMMENDED NEXT: {recommendation}")
    report.append("")
    report.append("Development-stage feasibility audit on capped balanced subsets - NOT a")
    report.append("final result. ood_val inspected for metadata availability only; ood_test")
    report.append("never referenced. No caches, predictions, checkpoints, or metrics modified.")
    report.append("=" * 78)

    report_path = logs / "exp07f0_dg_group_metadata_audit_report.txt"
    write_text(report_path, "\n".join(report) + "\n")

    # Echo the conclusion to console.
    tee("", log)
    tee("\n".join(report[-11:]), log)

    # -----------------------------------------------------------------------
    # Assertions.
    # -----------------------------------------------------------------------
    tee("", log)
    tee("-" * 78, log)
    tee("Assertions", log)
    tee("-" * 78, log)
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        tee(f"  [{'PASS' if ok else 'FAIL'}] {name}", log)

    check("train cache exists", per_split["train"]["exists"])
    check("id_val cache exists", per_split["id_val"]["exists"])
    check("ood_val cache exists", per_split["ood_val"]["exists"])
    check("train_n == 2000", per_split["train"]["n_examples"] == 2000)
    check("id_val_n == 500", per_split["id_val"]["n_examples"] == 500)
    check("ood_val_n == 2000", per_split["ood_val"]["n_examples"] == 2000)
    check("labels present and finite/binary (all splits)",
          all(per_split[s]["labels_finite"] is True for s in ("train", "id_val", "ood_val")))
    check("cache_metadata_summary.csv saved", summary_path.exists())
    check("group_label_counts.csv saved", group_path.exists())
    check("dg_method_feasibility.csv saved", feas_path.exists())
    check("report saved", report_path.exists())
    check("no ood_test usage",
          all("ood_test" not in c["cache_file"] for c in summary_rows))

    all_pass = all(ok for _, ok in checks)

    tee("", log)
    tee("Output paths:", log)
    for p in (summary_path, group_path, feas_path, report_path):
        tee(f"  {p.relative_to(root)}", log)

    tee("", log)
    if warnings:
        tee(f"WARNINGS ({len(warnings)}):", log)
        for w in warnings:
            tee(f"  - {w}", log)
    else:
        tee("WARNINGS: none", log)

    tee("", log)
    tee(f"OVERALL: {'PASS' if all_pass else 'FAIL'}", log)

    # Persist the console log alongside the report for provenance.
    write_text(logs / "exp07f0_run_console.txt", "\n".join(log) + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
