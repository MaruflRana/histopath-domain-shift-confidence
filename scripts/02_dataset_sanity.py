"""Milestone 1 / Script 02: dataset sanity for the LOCKED logical split mapping.

Logical splits scanned:
    train    = HF train,      center in {0, 3, 4}
    id_val   = HF validation, center in {0, 3, 4}
    ood_val  = HF validation, center == 1
    ood_test = HF test,       center == 2   (final reporting only; scanned here for
                                             counts only, never used for training)

Outputs:
    results/tables/dataset_split_summary.csv
    results/tables/dataset_label_counts.csv
    results/tables/dataset_center_counts.csv
    results/figures/sample_patch_grid.png

Streaming mode; progress prints so long scans do not look stuck.
"""

import _bootstrap  # noqa: F401

import sys

from data.inspect_dataset import save_sample_grid, scan_logical_split
from utils.io import (
    dataset_config,
    ensure_results_tree,
    figures_dir,
    tables_dir,
    write_csv,
)

LABEL_NAMES = {0: "non_tumor", 1: "tumor"}
LOGICAL_SPLITS = ["train", "id_val", "ood_val", "ood_test"]


def main() -> int:
    ensure_results_tree()
    cfg = dataset_config()
    split_cfg = cfg["splits"]

    scans = {}
    for name in LOGICAL_SPLITS:
        scans[name] = scan_logical_split(name)

    # --- split summary (with expected totals + match flag) ------------------
    summary_rows = []
    all_match = True
    for name in LOGICAL_SPLITS:
        r = scans[name]
        expected = split_cfg[name].get("expected_total")
        match = (expected is None) or (r["total"] == expected)
        all_match = all_match and match
        summary_rows.append(
            {
                "logical_split": name,
                "hf_split": split_cfg[name]["hf_split"],
                "center_filter": ";".join(map(str, split_cfg[name].get("center_in") or [])),
                "count": r["total"],
                "expected_total": expected if expected is not None else "",
                "matches_expected": match,
            }
        )
    summary_out = tables_dir() / "dataset_split_summary.csv"
    write_csv(summary_out, summary_rows)

    # --- label counts (long format) -----------------------------------------
    label_rows = []
    for name in LOGICAL_SPLITS:
        lc = scans[name]["label_counts"]
        for label in sorted(lc.keys()):
            label_rows.append(
                {
                    "logical_split": name,
                    "label": label,
                    "label_name": LABEL_NAMES.get(label, str(label)),
                    "count": lc[label],
                }
            )
    label_out = tables_dir() / "dataset_label_counts.csv"
    write_csv(label_out, label_rows)

    # --- center counts (long format) ----------------------------------------
    center_rows = []
    for name in LOGICAL_SPLITS:
        cc = scans[name]["center_counts"]
        for center in sorted(cc.keys()):
            center_rows.append(
                {
                    "logical_split": name,
                    "center": center,
                    "count": cc[center],
                }
            )
    center_out = tables_dir() / "dataset_center_counts.csv"
    write_csv(center_out, center_rows)

    # --- sample grid (decoded RGB, small; id_val to avoid touching ood_test) -
    grid_out = figures_dir() / "sample_patch_grid.png"
    save_sample_grid(grid_out, logical_split="id_val", n=12, ncols=4)

    # --- report --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATASET SANITY SUMMARY")
    print("=" * 60)
    for row in summary_rows:
        flag = "OK" if row["matches_expected"] else "MISMATCH"
        exp = row["expected_total"] if row["expected_total"] != "" else "n/a"
        print(
            f"  {row['logical_split']:9s} "
            f"count={row['count']:>8,}  expected={exp!s:>8}  [{flag}]"
        )
    print("-" * 60)
    print(f"Saved: {summary_out}")
    print(f"Saved: {label_out}")
    print(f"Saved: {center_out}")
    print(f"Saved: {grid_out}")
    print(f"OVERALL COUNTS: {'PASS' if all_match else 'FAIL'}")
    print("=" * 60)

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
