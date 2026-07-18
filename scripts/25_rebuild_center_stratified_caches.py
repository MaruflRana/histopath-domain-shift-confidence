"""Milestone 7F-1 / Script 25: metadata-rich, center-STRATIFIED cache rebuild.

Rebuilds balanced development caches for ``train`` and ``id_val`` so that each locked
source center {0, 3, 4} contributes examples of BOTH labels — removing the
center<->label confound that Milestone 7F-0 identified in the existing caches (built by
filling per-*class* quotas over a label-ordered stream, which drew center 0 as
(near-)all non-tumor, center 3 as all-tumor, and omitted center 4 entirely).

The ONLY thing that changes vs ``src/data/balanced_subset.py`` is the SAMPLING POLICY:
instead of "first ``per_class`` of each label in stream order", we fill a quota for each
``(center, label)`` cell across the required source centers. Everything else is kept
identical:

  * source = the locked HF mirror ``wltjr1007/Camelyon17-WILDS`` via the sanctioned
    streaming loader ``iter_logical_split`` (NO WILDS/CodaLab, NO test/center-2 access),
  * the serialized payload schema is byte-for-byte the same one the existing caches use
    (``_serialize_examples`` -> ``{version, n, images, labels, meta}`` with all seven
    ``META_INT_FIELDS`` preserved), so the new caches load through the same
    ``BalancedSubsetDataset`` deserializer,
  * images are decoded to RGB only for KEPT examples (lazy decode).

Adaptive per-cell quota:
  * Preferred quota — train 300 / id_val 75 per (center, label) cell.
  * Scan availability for each cell. If every required cell reaches the preferred quota,
    use the preferred quota. If any cell has fewer, use the MINIMUM available count over
    all required cells for that split (keeps the cache balanced by construction).
  * If any required (center, label) cell has ZERO examples, DO NOT build a broken DG
    cache for that split — record it and mark DG-by-center not feasible for that split.

STRICT SCOPE (Milestone 7F-1):
  * NO training, NO model inference, NO ``ood_test`` (HF test / center 2) access.
  * Does NOT overwrite existing caches — writes NEW filenames under results/cache.
  * Does NOT modify prediction CSVs, checkpoints, or metrics.
  * ``ood_val`` is not needed here and is not read for cache construction.

Outputs: two new caches (train, id_val) + three tables + one text report.
NOT a final result — a development-stage cache rebuild on capped balanced subsets.
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup BEFORE any datasets import)

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Reuse the EXACT serializer schema + lazy RGB decode from the existing data layer so the
# new caches are format-compatible with balanced_subset.BalancedSubsetDataset. We only
# change the sampling policy; we do not modify src/.
from data.balanced_subset import (  # noqa: E402
    META_INT_FIELDS,
    _CACHE_VERSION,
    _decode_rgb,
    _deserialize_examples,
    _serialize_examples,
)
from data.hf_camelyon17 import iter_logical_split, split_spec  # noqa: E402
from utils.io import (  # noqa: E402
    ensure_dir,
    ensure_results_tree,
    logs_dir,
    project_root,
    tables_dir,
    tee,
    write_csv,
    write_text,
)

# ---------------------------------------------------------------------------
# Configuration. ood_test / center 2 are intentionally absent and never used.
# ---------------------------------------------------------------------------

SEED = 1337
REQUIRED_CENTERS = (0, 3, 4)   # locked source centers for train + id_val
REQUIRED_LABELS = (0, 1)       # 0 = non-tumor, 1 = tumor

# Per (logical split): preferred per-cell quota, a scan cap (bounds a heavily
# label-ordered stream), an output cache filename with an AUTO placeholder for the
# actually-selected per-cell count, and progress cadence.
SPLITS = [
    {
        "split": "train",
        "hf_split_expected": "train",
        "preferred_per_cell": 300,
        # train logical split is centers {0,3,4} over the full HF train stream
        # (~302k). Cap generously so center 4 + both labels are reachable even though
        # the stream is label-ordered; we early-stop once every cell is full.
        "max_scan": 400000,
        "cache_name_tmpl": "center_strat_train_c034_percellAUTO_seed1337_shuf.pt",
        "progress_every": 10000,
    },
    {
        "split": "id_val",
        "hf_split_expected": "validation",
        "preferred_per_cell": 75,
        # id_val logical split is centers {0,3,4} over HF validation (~33.5k after the
        # center-1 exclusion is applied by the loader). Cap above the split size.
        "max_scan": 60000,
        "cache_name_tmpl": "center_strat_id_val_c034_percell75_seed1337_shuf.pt",
        "progress_every": 5000,
    },
]

# Existing caches that MUST NOT be overwritten or modified (integrity check).
EXISTING_CACHES = [
    "results/cache/balanced_train_pc1000_seed1337_shuf.pt",
    "results/cache/balanced_id_val_pc250_seed1337_shuf.pt",
    "results/cache/balanced_ood_val_pc1000_seed1337_noshuf.pt",
    "results/cache/balanced_train_pc128_seed1337_shuf.pt",
    "results/cache/balanced_id_val_pc64_seed1337_shuf.pt",
]


def _file_fingerprint(path: Path):
    """(exists, size, mtime_ns) — used to prove existing caches were not modified."""
    if not path.exists():
        return (False, None, None)
    st = path.stat()
    return (True, st.st_size, st.st_mtime_ns)


def collect_center_stratified(split_cfg, log, warnings):
    """Stream one logical split and fill per-(center,label) buckets.

    Returns a dict with the collected buckets and per-cell availability facts. Only KEPT
    images are decoded (lazy). Early-stops once every required cell reaches the preferred
    quota; otherwise scans until the stream ends or ``max_scan`` is reached.
    """
    split = split_cfg["split"]
    allowed_centers = set(REQUIRED_CENTERS)
    preferred = split_cfg["preferred_per_cell"]
    max_scan = split_cfg["max_scan"]
    progress_every = split_cfg["progress_every"]

    # Guard: this loader path must be a source split, never ood_test / test / center 2.
    spec = split_spec(split)
    assert spec["hf_split"] == split_cfg["hf_split_expected"], (
        f"{split}: unexpected hf_split {spec['hf_split']!r}"
    )
    assert spec["hf_split"] != "test", f"{split}: must not read HF test split"
    center_in = spec.get("center_in")
    assert center_in is not None and 2 not in set(center_in), (
        f"{split}: center 2 / ood_test must never be in the source filter (got {center_in})"
    )

    # buckets[(center,label)] -> list[example dict]; available[(center,label)] -> int seen.
    buckets = {(c, y): [] for c in REQUIRED_CENTERS for y in REQUIRED_LABELS}
    available = {(c, y): 0 for c in REQUIRED_CENTERS for y in REQUIRED_LABELS}
    scanned = 0
    scan_completed = False  # True if the stream ended (availability is exact)

    tee("", log)
    tee(f"[{split}] collecting center-stratified subset "
        f"(preferred {preferred}/cell, centers {sorted(REQUIRED_CENTERS)} x labels "
        f"{list(REQUIRED_LABELS)}, max_scan={max_scan})...", log)

    stream = iter_logical_split(split, decode_images=False, limit=None)
    for ex in stream:
        scanned += 1
        center = int(ex["center"])
        label = int(ex["label"])

        if center in allowed_centers and label in REQUIRED_LABELS:
            cell = (center, label)
            available[cell] += 1
            if len(buckets[cell]) < preferred:
                item = {"image": _decode_rgb(ex["image"]), "label": label}
                for f in META_INT_FIELDS:
                    item[f] = int(ex[f])
                buckets[cell].append(item)

        if progress_every and scanned % progress_every == 0:
            filled = sum(1 for c in buckets if len(buckets[c]) >= preferred)
            tee(f"  [{split}] scanned={scanned}  cells_full={filled}/{len(buckets)}  "
                f"collected={{ {', '.join(f'{k}:{len(v)}' for k, v in buckets.items())} }}",
                log)

        # Early stop: every required cell has reached the preferred quota.
        if all(len(v) >= preferred for v in buckets.values()):
            scan_completed = False  # we stopped early; availability is a LOWER BOUND
            tee(f"  [{split}] all {len(buckets)} cells reached preferred quota "
                f"{preferred} at scanned={scanned}; stopping early.", log)
            break

        if scanned >= max_scan:
            warnings.append(
                f"[{split}] reached max_scan={max_scan} before all cells filled; "
                f"availability counts are exact up to the scan cap."
            )
            tee(f"  [{split}] max_scan={max_scan} reached at scanned={scanned}.", log)
            break
    else:
        # for-loop exhausted the stream naturally -> availability counts are exact.
        scan_completed = True
        tee(f"  [{split}] stream exhausted at scanned={scanned}; availability exact.", log)

    collected_counts = {cell: len(v) for cell, v in buckets.items()}
    tee(f"  [{split}] DONE scan — scanned={scanned}, "
        f"collected per cell={collected_counts}, "
        f"available(>= collected)={available}, scan_completed={scan_completed}.", log)

    return {
        "split": split,
        "buckets": buckets,
        "available": available,
        "collected_counts": collected_counts,
        "scanned": scanned,
        "scan_completed": scan_completed,
        "preferred_per_cell": preferred,
    }


def decide_quota_and_select(collect_result, log, warnings):
    """Apply the adaptive per-cell quota and downsample each cell to it.

    Returns (selected_examples_or_None, decision_facts). If any required cell has zero
    collected examples, returns (None, facts) with feasible=False.
    """
    split = collect_result["split"]
    buckets = collect_result["buckets"]
    collected = collect_result["collected_counts"]
    preferred = collect_result["preferred_per_cell"]

    required_cells = [(c, y) for c in REQUIRED_CENTERS for y in REQUIRED_LABELS]
    min_collected = min(collected[cell] for cell in required_cells)
    zero_cells = [cell for cell in required_cells if collected[cell] == 0]

    facts = {
        "split": split,
        "required_cells": required_cells,
        "collected_counts": collected,
        "min_collected": min_collected,
        "zero_cells": zero_cells,
        "preferred_per_cell": preferred,
    }

    if zero_cells:
        facts["feasible"] = False
        facts["selected_per_cell"] = 0
        warnings.append(
            f"[{split}] required (center,label) cell(s) with ZERO examples: {zero_cells}; "
            f"NOT building a DG cache for this split (would be broken)."
        )
        tee(f"  [{split}] INFEASIBLE — zero-example cells {zero_cells}; no cache written.",
            log)
        return None, facts

    # All cells non-empty. Preferred if every cell reached it, else the min available.
    if all(collected[cell] >= preferred for cell in required_cells):
        selected = preferred
    else:
        selected = min_collected
        warnings.append(
            f"[{split}] some cells below preferred {preferred}; using minimum available "
            f"per-cell quota {selected} (balanced by construction)."
        )
    facts["feasible"] = True
    facts["selected_per_cell"] = selected

    tee(f"  [{split}] selected per-cell quota = {selected} "
        f"(preferred {preferred}, min_collected {min_collected}).", log)

    # Downsample each cell to exactly `selected` with a per-split reproducible RNG.
    rng = random.Random(SEED + (0 if split == "train" else 1))
    chosen = []
    for cell in required_cells:
        pool = buckets[cell]
        if len(pool) > selected:
            idx = sorted(rng.sample(range(len(pool)), selected))
            picked = [pool[i] for i in idx]
        else:
            picked = list(pool)
        chosen.extend(picked)

    # Interleave labels/centers (avoid a block-ordered payload), like shuffle_collected.
    random.Random(SEED).shuffle(chosen)
    facts["final_n"] = len(chosen)
    return chosen, facts


def main() -> int:
    ensure_results_tree()
    root = project_root()
    cache_dir = ensure_dir(root / "results" / "cache")
    log: list[str] = []
    warnings: list[str] = []

    tee("=" * 78, log)
    tee("Milestone 7F-1 / Script 25: metadata-rich center-STRATIFIED cache rebuild", log)
    tee("(NO training, NO inference, NO ood_test, NO overwrite of existing caches)", log)
    tee("=" * 78, log)

    # ------------------------------------------------------------------
    # 0) Fingerprint existing caches so we can prove we did not modify them.
    # ------------------------------------------------------------------
    pre_fp = {rel: _file_fingerprint(root / rel) for rel in EXISTING_CACHES}
    tee("", log)
    tee("Existing caches (pre-run fingerprint; must be unchanged after run):", log)
    for rel, fp in pre_fp.items():
        tee(f"  {rel}: exists={fp[0]} size={fp[1]} mtime_ns={fp[2]}", log)

    # Guard: no output/target path may reference the locked test split.
    for cfg in SPLITS:
        assert "ood_test" not in cfg["cache_name_tmpl"], "output must not name ood_test"
        assert cfg["hf_split_expected"] in ("train", "validation"), (
            f"{cfg['split']}: source must be HF train/validation only"
        )

    # ------------------------------------------------------------------
    # 1) Collect + decide quota + build the new caches.
    # ------------------------------------------------------------------
    per_split = {}
    availability_rows = []      # table 1 (long: split x center x label)
    label_center_rows = []      # table 3 (long: final counts)
    written_caches = {}         # split -> abs path (str) if written

    for cfg in SPLITS:
        split = cfg["split"]
        collect_result = collect_center_stratified(cfg, log, warnings)
        selected, decision = decide_quota_and_select(collect_result, log, warnings)

        # Availability table rows (from the scan).
        for c in REQUIRED_CENTERS:
            for y in REQUIRED_LABELS:
                availability_rows.append({
                    "split": split,
                    "center": c,
                    "label": y,
                    "available_scanned": collect_result["available"][(c, y)],
                    "collected_in_buckets": collect_result["collected_counts"][(c, y)],
                    "scan_completed_exact": collect_result["scan_completed"],
                })

        info = {
            "split": split,
            "collect": collect_result,
            "decision": decision,
            "selected_examples": selected,
            "cache_path": None,
            "final_label_counts": None,
            "final_center_counts": None,
            "final_label_center_counts": None,
            "meta_fields_preserved": None,
            "reload_ok": None,
        }

        if selected is None:
            # Infeasible split: no cache written.
            per_split[split] = info
            continue

        # Compute final distributions BEFORE serialization.
        final_labels = [ex["label"] for ex in selected]
        final_centers = [ex["center"] for ex in selected]
        lc = Counter(final_labels)
        cc = Counter(final_centers)
        lcc = Counter((ex["center"], ex["label"]) for ex in selected)
        info["final_label_counts"] = {int(k): int(v) for k, v in sorted(lc.items())}
        info["final_center_counts"] = {int(k): int(v) for k, v in sorted(cc.items())}
        info["final_label_center_counts"] = {
            f"c{c}_l{y}": int(lcc.get((c, y), 0))
            for c in REQUIRED_CENTERS for y in REQUIRED_LABELS
        }

        for c in REQUIRED_CENTERS:
            for y in REQUIRED_LABELS:
                label_center_rows.append({
                    "split": split,
                    "center": c,
                    "label": y,
                    "count": int(lcc.get((c, y), 0)),
                })

        # Serialize using the EXACT existing schema (no schema change).
        payload = _serialize_examples(selected)
        assert payload["version"] == _CACHE_VERSION
        assert set(payload.keys()) == {"version", "n", "images", "labels", "meta"}
        assert set(payload["meta"].keys()) == set(META_INT_FIELDS)
        info["meta_fields_preserved"] = sorted(payload["meta"].keys())

        # Resolve the cache filename (AUTO -> actual per-cell quota).
        per_cell = decision["selected_per_cell"]
        cache_name = cfg["cache_name_tmpl"].replace("AUTO", f"{per_cell}")
        cache_path = cache_dir / cache_name

        # Safety: never overwrite an existing file (new filenames only).
        assert cache_path.resolve() != (root / EXISTING_CACHES[0]).resolve()
        if cache_path.exists():
            warnings.append(
                f"[{split}] target cache already exists, NOT overwriting: {cache_name}"
            )
            tee(f"  [{split}] target exists — skipping write: {cache_name}", log)
        else:
            torch.save(payload, cache_path)
            tee(f"  [{split}] wrote new cache -> results/cache/{cache_name} "
                f"(n={payload['n']})", log)

        info["cache_path"] = str(cache_path)
        written_caches[split] = str(cache_path)

        # Reload verification: deserialize and re-check counts + metadata (no inference).
        reloaded = _deserialize_examples(torch.load(cache_path, weights_only=False))
        r_labels = [ex["label"] for ex in reloaded]
        r_centers = [ex["center"] for ex in reloaded]
        reload_ok = (
            len(reloaded) == payload["n"]
            and Counter(r_labels) == lc
            and Counter(r_centers) == cc
            and all(all(f in ex for f in META_INT_FIELDS) for ex in reloaded[:5])
        )
        info["reload_ok"] = bool(reload_ok)
        tee(f"  [{split}] reload verification: n={len(reloaded)}, "
            f"label_counts={dict(sorted(Counter(r_labels).items()))}, "
            f"center_counts={dict(sorted(Counter(r_centers).items()))}, ok={reload_ok}", log)

        per_split[split] = info

    # ------------------------------------------------------------------
    # 2) DG feasibility derivations (per split, after rebuild).
    # ------------------------------------------------------------------
    def dg_feasible_for(split):
        info = per_split.get(split, {})
        dec = info.get("decision", {})
        if not dec.get("feasible"):
            return False, "infeasible: a required (center,label) cell had zero examples"
        lcc = info.get("final_label_center_counts", {}) or {}
        centers_present = set()
        cells_ok = True
        for c in REQUIRED_CENTERS:
            has0 = lcc.get(f"c{c}_l0", 0) > 0
            has1 = lcc.get(f"c{c}_l1", 0) > 0
            if has0 or has1:
                centers_present.add(c)
            if not (has0 and has1):
                cells_ok = False
        sound = (len(centers_present) >= 2) and cells_ok
        note = (
            f"centers_present={sorted(centers_present)}, "
            f"every_required_cell_both_labels={cells_ok}"
        )
        return sound, note

    train_dg_sound, train_dg_note = dg_feasible_for("train")
    idval_dg_sound, idval_dg_note = dg_feasible_for("id_val")

    # GroupDRO and CORAL/DANN by center share the same soundness precondition here
    # (>=2 label-decorrelated center groups, each carrying both labels).
    supports_groupdro = train_dg_sound
    supports_coral_dann = train_dg_sound
    avoid_old_confounded = True  # 7F-0 established the old train cache is confounded

    # ------------------------------------------------------------------
    # 3) Write tables.
    # ------------------------------------------------------------------
    tables = tables_dir()
    logs = logs_dir()

    avail_path = tables / "exp07f1_center_stratified_cache_availability.csv"
    write_csv(avail_path, availability_rows)

    # Summary table: one row per split.
    summary_rows = []
    for cfg in SPLITS:
        split = cfg["split"]
        info = per_split[split]
        dec = info["decision"]
        summary_rows.append({
            "split": split,
            "source_hf_split": cfg["hf_split_expected"],
            "centers_included": "|".join(str(c) for c in REQUIRED_CENTERS),
            "labels_included": "|".join(str(y) for y in REQUIRED_LABELS),
            "preferred_per_cell": cfg["preferred_per_cell"],
            "min_collected_per_cell": dec.get("min_collected"),
            "selected_per_cell": dec.get("selected_per_cell"),
            "feasible": dec.get("feasible"),
            "zero_cells": ";".join(f"c{c}_l{y}" for (c, y) in dec.get("zero_cells", [])) or "(none)",
            "final_n": info.get("final_label_counts") and sum(info["final_label_counts"].values()),
            "scanned": info["collect"]["scanned"],
            "scan_completed_exact": info["collect"]["scan_completed"],
            "cache_file": (Path(info["cache_path"]).name if info["cache_path"] else "(not written)"),
            "reload_ok": info.get("reload_ok"),
            "supports_dg_by_center": (train_dg_sound if split == "train" else "(validation)"),
        })
    summary_path = tables / "exp07f1_center_stratified_cache_summary.csv"
    write_csv(summary_path, summary_rows)

    lc_path = tables / "exp07f1_center_stratified_label_center_counts.csv"
    if not label_center_rows:
        label_center_rows = [{"split": "", "center": "", "label": "", "count": ""}]
    write_csv(lc_path, label_center_rows)

    # ------------------------------------------------------------------
    # 4) Report.
    # ------------------------------------------------------------------
    report = []
    report.append("=" * 78)
    report.append("Milestone 7F-1 - metadata-rich center-STRATIFIED cache rebuild")
    report.append("NO training, NO inference, NO ood_test, NO overwrite of existing caches.")
    report.append("=" * 78)
    report.append("")
    report.append("Source: HF mirror 'wltjr1007/Camelyon17-WILDS' via iter_logical_split")
    report.append("        (streaming; NO WILDS/CodaLab; NO HF test / center 2).")
    report.append(f"Required source centers: {sorted(REQUIRED_CENTERS)}")
    report.append(f"Required labels: {list(REQUIRED_LABELS)} (0=non-tumor, 1=tumor)")
    report.append(f"Seed: {SEED}")
    report.append("")
    for cfg in SPLITS:
        split = cfg["split"]
        info = per_split[split]
        dec = info["decision"]
        collect_result = info["collect"]
        report.append("-" * 78)
        report.append(f"[{split}]  source HF split = {cfg['hf_split_expected']}, "
                      f"centers {sorted(REQUIRED_CENTERS)}")
        report.append("-" * 78)
        report.append(f"  scanned streamed records: {collect_result['scanned']} "
                      f"(scan_completed_exact={collect_result['scan_completed']})")
        report.append("  available (scanned) per center x label:")
        for c in REQUIRED_CENTERS:
            row = "    " + "  ".join(
                f"c{c}_l{y}={collect_result['available'][(c, y)]}" for y in REQUIRED_LABELS
            )
            report.append(row)
        report.append("  collected-in-buckets per center x label:")
        for c in REQUIRED_CENTERS:
            row = "    " + "  ".join(
                f"c{c}_l{y}={collect_result['collected_counts'][(c, y)]}" for y in REQUIRED_LABELS
            )
            report.append(row)
        report.append(f"  preferred per-cell quota: {cfg['preferred_per_cell']}")
        report.append(f"  min collected per cell:   {dec.get('min_collected')}")
        report.append(f"  SELECTED per-cell quota:  {dec.get('selected_per_cell')}")
        report.append(f"  feasible (all cells > 0): {dec.get('feasible')} "
                      f"(zero cells: {dec.get('zero_cells')})")
        if info["cache_path"]:
            report.append(f"  final n: {sum(info['final_label_counts'].values())}")
            report.append(f"  final label counts:  {info['final_label_counts']}")
            report.append(f"  final center counts: {info['final_center_counts']}")
            report.append(f"  final label x center counts: {info['final_label_center_counts']}")
            report.append(f"  metadata fields preserved: {info['meta_fields_preserved']}")
            report.append(f"  cache file: results/cache/{Path(info['cache_path']).name}")
            report.append(f"  reload verification ok: {info['reload_ok']}")
        else:
            report.append("  cache file: (NOT written - split infeasible)")
        report.append("")
    report.append("-" * 78)
    report.append("DG feasibility after rebuild")
    report.append("-" * 78)
    report.append(f"  train DG-by-center sound: {train_dg_sound}  ({train_dg_note})")
    report.append(f"  id_val (validation role): {idval_dg_note}")
    report.append(f"  new caches support GroupDRO by center: {supports_groupdro}")
    report.append(f"  new caches support CORAL/DANN by center: {supports_coral_dann}")
    report.append(f"  avoid OLD confounded caches for DG training: {avoid_old_confounded}")
    report.append("    (7F-0: old balanced_train_pc1000 is center<->label confounded -")
    report.append("     center 0 ~all non-tumor, center 3 all tumor, center 4 absent.)")
    report.append("")
    report.append("-" * 78)
    report.append("Cache file paths (new; existing caches untouched)")
    report.append("-" * 78)
    for split, p in written_caches.items():
        report.append(f"  {split}: results/cache/{Path(p).name}")
    report.append("")
    report.append("Existing caches (unchanged - see fingerprints in run console):")
    for rel in EXISTING_CACHES:
        report.append(f"  {rel}")
    report.append("")
    report.append("Development-stage cache rebuild on capped balanced subsets - NOT a final")
    report.append("result. No training, no inference; ood_test never referenced; ood_val not")
    report.append("read for cache construction. GroupDRO/CORAL/DANN not started.")
    report.append("=" * 78)

    report_path = logs / "exp07f1_center_stratified_cache_rebuild_report.txt"
    write_text(report_path, "\n".join(report) + "\n")

    tee("", log)
    tee("\n".join(report), log)

    # ------------------------------------------------------------------
    # 5) Assertions.
    # ------------------------------------------------------------------
    tee("", log)
    tee("-" * 78, log)
    tee("Assertions", log)
    tee("-" * 78, log)
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        tee(f"  [{'PASS' if ok else 'FAIL'}] {name}", log)

    # no ood_test usage anywhere (config + outputs + written paths).
    no_test_ref = (
        all("ood_test" not in cfg["cache_name_tmpl"] for cfg in SPLITS)
        and all("ood_test" not in str(p) and "test" not in Path(p).name for p in written_caches.values())
    )
    check("no ood_test usage", no_test_ref)

    # old cache files still exist and were not modified (fingerprint match).
    post_fp = {rel: _file_fingerprint(root / rel) for rel in EXISTING_CACHES}
    old_unmodified = all(pre_fp[rel] == post_fp[rel] for rel in EXISTING_CACHES)
    check("existing caches present and unmodified (size+mtime)", old_unmodified)

    train_info = per_split.get("train", {})
    idval_info = per_split.get("id_val", {})

    # Feasibility gate: only assert cache existence when the split was feasible.
    train_feasible = train_info.get("decision", {}).get("feasible", False)
    idval_feasible = idval_info.get("decision", {}).get("feasible", False)
    check("train split feasible (all required cells non-empty)", train_feasible)
    check("id_val split feasible (all required cells non-empty)", idval_feasible)

    if train_feasible:
        tp = Path(train_info["cache_path"])
        check("new train cache exists", tp.exists())
    if idval_feasible:
        ip = Path(idval_info["cache_path"])
        check("new id_val cache exists", ip.exists())

    # New caches contain images, labels, meta; metadata fields preserved; labels binary.
    def _payload_checks(info, tag):
        if not info.get("cache_path"):
            return
        payload = torch.load(Path(info["cache_path"]), weights_only=False)
        check(f"{tag} cache has images/labels/meta keys",
              {"images", "labels", "meta"}.issubset(set(payload.keys())))
        check(f"{tag} all required metadata fields preserved",
              set(payload["meta"].keys()) == set(META_INT_FIELDS))
        labs = [int(v) for v in payload["labels"]]
        check(f"{tag} labels binary and finite",
              set(labs).issubset({0, 1}) and all(np.isfinite(labs)))
        centers = [int(v) for v in payload["meta"]["center"]]
        lcc = Counter(zip(centers, labs))
        # each required center appears + each (center,label) cell has >=1 example.
        centers_present = all(any(cc2 == c for (cc2, _l) in lcc) for c in REQUIRED_CENTERS)
        check(f"{tag} each required center present in cache", centers_present)
        all_cells = all(lcc.get((c, y), 0) >= 1 for c in REQUIRED_CENTERS for y in REQUIRED_LABELS)
        check(f"{tag} each required (center,label) cell has >=1 example", all_cells)
        # balanced by construction: all cells equal.
        cell_vals = [lcc.get((c, y), 0) for c in REQUIRED_CENTERS for y in REQUIRED_LABELS]
        check(f"{tag} label x center balanced by construction (all cells equal)",
              len(set(cell_vals)) == 1 and cell_vals[0] >= 1)

    _payload_checks(train_info, "train")
    _payload_checks(idval_info, "id_val")

    check("availability table saved", avail_path.exists())
    check("summary table saved", summary_path.exists())
    check("label_center_counts table saved", lc_path.exists())
    check("report saved", report_path.exists())

    all_pass = all(ok for _, ok in checks)

    tee("", log)
    tee("Output paths:", log)
    for p in (avail_path, summary_path, lc_path, report_path):
        tee(f"  {p.relative_to(root)}", log)
    for split, p in written_caches.items():
        tee(f"  results/cache/{Path(p).name}", log)

    tee("", log)
    if warnings:
        tee(f"WARNINGS ({len(warnings)}):", log)
        for w in warnings:
            tee(f"  - {w}", log)
    else:
        tee("WARNINGS: none", log)

    tee("", log)
    tee(f"OVERALL: {'PASS' if all_pass else 'FAIL'}", log)

    write_text(logs / "exp07f1_run_console.txt", "\n".join(log) + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
