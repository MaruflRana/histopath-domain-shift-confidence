"""Milestone 3C / Script 07: label-order diagnostic for train and id_val.

The HF Camelyon17-WILDS stream is label-ordered enough that naive early batches can
be single-class (Milestone 3B needed to scan 14,102 id_val records to reach label 0).
This script quantifies that ordering WITHOUT decoding images (labels only — fast, no
extra downloads) and records, per split:

  * first-occurrence scan index of each label,
  * running label counts every 1000 scanned samples,
  * whether the first 1000 samples are single-class.

Only ``train`` and ``id_val`` are inspected. ``ood_val`` / ``ood_test`` are untouched.

Outputs:
  results/tables/label_order_diagnostic.csv   (per split, per 1000-sample checkpoint)
  results/logs/label_order_diagnostic.txt     (human-readable report)
"""

import _bootstrap  # noqa: F401  (sys.path + SSL setup before HF import)

import sys

from data.hf_camelyon17 import iter_logical_split
from utils.io import ensure_results_tree, logs_dir, tables_dir, write_csv, write_text

# Split -> capped scan budget (labels only; no image decode).
SCAN_BUDGET = {"train": 5000, "id_val": 20000}
CHECKPOINT_EVERY = 1000
FIRST_WINDOW = 1000


def _diagnose(split_name: str, max_scan: int, emit) -> tuple[list[dict], bool]:
    """Scan one split's labels; return (checkpoint rows, single_class_first_window)."""
    emit("")
    emit(f"### split: {split_name}  (max_scan={max_scan}, labels only)")

    counts = {0: 0, 1: 0}
    first_seen: dict[int, int] = {}
    first_window_counts = {0: 0, 1: 0}
    rows: list[dict] = []
    scanned = 0

    for ex in iter_logical_split(split_name, decode_images=False, limit=max_scan):
        scanned += 1
        label = int(ex["label"])
        counts[label] = counts.get(label, 0) + 1
        if label not in first_seen:
            first_seen[label] = scanned  # 1-based scan index of first occurrence
        if scanned <= FIRST_WINDOW:
            first_window_counts[label] = first_window_counts.get(label, 0) + 1

        if scanned % CHECKPOINT_EVERY == 0:
            rows.append({
                "split": split_name,
                "scanned": scanned,
                "count_label0": counts.get(0, 0),
                "count_label1": counts.get(1, 0),
            })
            emit(
                f"  scanned={scanned:>6}  label0={counts.get(0,0):>6}  "
                f"label1={counts.get(1,0):>6}"
            )

    # final checkpoint if not on a 1000 boundary
    if scanned % CHECKPOINT_EVERY != 0:
        rows.append({
            "split": split_name,
            "scanned": scanned,
            "count_label0": counts.get(0, 0),
            "count_label1": counts.get(1, 0),
        })

    fs0 = first_seen.get(0, None)
    fs1 = first_seen.get(1, None)
    emit(f"  total scanned          : {scanned}")
    emit(f"  final counts           : label0={counts.get(0,0)}  label1={counts.get(1,0)}")
    emit(f"  first label-0 at index : {fs0 if fs0 is not None else 'NOT SEEN'}")
    emit(f"  first label-1 at index : {fs1 if fs1 is not None else 'NOT SEEN'}")
    emit(
        f"  first {FIRST_WINDOW} labels    : "
        f"label0={first_window_counts.get(0,0)}  label1={first_window_counts.get(1,0)}"
    )

    single_class = (first_window_counts.get(0, 0) == 0) or (first_window_counts.get(1, 0) == 0)
    if single_class:
        emit(
            f"  WARNING: first {FIRST_WINDOW} samples of '{split_name}' are SINGLE-CLASS "
            f"— naive early batches would be label-skewed. Use balanced collection."
        )
    else:
        emit(f"  OK: first {FIRST_WINDOW} samples of '{split_name}' contain both classes.")

    return rows, single_class


def main() -> int:
    ensure_results_tree()
    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    emit("=" * 64)
    emit("Milestone 3C — Label-order diagnostic (script 07)")
    emit("Inspecting train + id_val only (labels only, no image decode).")
    emit("=" * 64)

    all_rows: list[dict] = []
    any_warning = False
    for split, budget in SCAN_BUDGET.items():
        rows, single_class = _diagnose(split, budget, emit)
        all_rows.extend(rows)
        any_warning = any_warning or single_class

    csv_path = tables_dir() / "label_order_diagnostic.csv"
    write_csv(
        csv_path,
        all_rows,
        header=["split", "scanned", "count_label0", "count_label1"],
    )

    emit("")
    emit("=" * 64)
    emit(f"SINGLE-CLASS EARLY WINDOW DETECTED: {'YES (see warnings)' if any_warning else 'NO'}")
    emit(f"Saved table  -> {csv_path}")
    emit("=" * 64)

    txt_path = logs_dir() / "label_order_diagnostic.txt"
    write_text(txt_path, "\n".join(lines) + "\n")
    print(f"\nSaved report -> {txt_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
