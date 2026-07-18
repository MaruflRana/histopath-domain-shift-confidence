"""Milestone 1 / Script 01: verify the Hugging Face dataset.

Streams the three HF splits (train / validation / test), verifies:
  * required fields are present,
  * center distributions match expectation:
        train      centers {0, 3, 4}
        validation centers {0, 1, 3, 4}
        test       center  {2}
Writes results/tables/hf_split_verification.csv and prints PASS/FAIL per check.

Streaming mode is used so no full download is required.
"""

import _bootstrap  # noqa: F401

import sys

from data.hf_camelyon17 import expected_fields
from data.inspect_dataset import scan_hf_split
from utils.io import dataset_config, ensure_results_tree, tables_dir, write_csv


def main() -> int:
    ensure_results_tree()
    cfg = dataset_config()
    expected_centers = cfg["hf_split_expected_centers"]
    req_fields = set(expected_fields())

    hf_splits = ["train", "validation", "test"]
    results = {}
    for hf_split in hf_splits:
        results[hf_split] = scan_hf_split(hf_split)

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    all_pass = True
    csv_rows = []

    for hf_split in hf_splits:
        r = results[hf_split]
        seen_fields = set(r["fields"])
        seen_centers = sorted(r["center_counts"].keys())
        exp_centers = sorted(expected_centers[hf_split])

        # Field check
        missing = sorted(req_fields - seen_fields)
        fields_ok = not missing

        # Center check
        centers_ok = seen_centers == exp_centers

        split_pass = fields_ok and centers_ok
        all_pass = all_pass and split_pass

        status = "PASS" if split_pass else "FAIL"
        print(f"\n[{hf_split}]  ->  {status}")
        print(f"  total examples : {r['total']:,}")
        print(f"  fields         : {'OK' if fields_ok else 'MISSING ' + str(missing)}")
        print(f"  centers seen   : {seen_centers}")
        print(f"  centers expect : {exp_centers}")
        print(f"  centers match  : {'OK' if centers_ok else 'MISMATCH'}")
        print(f"  center counts  : {r['center_counts']}")
        print(f"  label counts   : {r['label_counts']}")

        csv_rows.append(
            {
                "hf_split": hf_split,
                "total": r["total"],
                "centers_seen": ";".join(map(str, seen_centers)),
                "centers_expected": ";".join(map(str, exp_centers)),
                "centers_match": centers_ok,
                "fields_ok": fields_ok,
                "missing_fields": ";".join(missing),
                "label_counts": str(r["label_counts"]),
                "center_counts": str(r["center_counts"]),
                "status": status,
            }
        )

    out = tables_dir() / "hf_split_verification.csv"
    write_csv(out, csv_rows)
    print("\n" + "=" * 60)
    print(f"Saved split verification -> {out}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
