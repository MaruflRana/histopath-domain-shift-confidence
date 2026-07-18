"""Streaming inspection helpers: field checks, count/label/center scans, sample grid.

These operate on the Hugging Face streaming iterables so that a full dataset
download is not required for verification. Used by scripts 01 and 02.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Optional

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data.hf_camelyon17 import hf_id, to_rgb  # noqa: E402


def scan_hf_split(
    hf_split: str,
    *,
    progress_every: int = 20000,
    label: Optional[str] = None,
) -> dict:
    """Stream a raw HF split (no logical filtering) and tally counts.

    Returns a dict with total, per-center, per-label, and per-(center,label) counts,
    plus the set of fields seen on the first example.
    """
    from datasets import Image, load_dataset

    tag = label or hf_split
    print(f"[scan] HF split '{hf_split}' (streaming, metadata only)...", flush=True)
    ds = load_dataset(hf_id(), split=hf_split, streaming=True)
    ds = ds.cast_column("image", Image(decode=False))

    center_counts: Counter = Counter()
    label_counts: Counter = Counter()
    center_label_counts: Counter = Counter()
    fields: list[str] = []
    total = 0

    for ex in ds:
        if total == 0:
            fields = sorted(ex.keys())
        center_counts[ex["center"]] += 1
        label_counts[ex["label"]] += 1
        center_label_counts[(ex["center"], ex["label"])] += 1
        total += 1
        if total % progress_every == 0:
            print(f"  [{tag}] scanned {total:,} ...", flush=True)

    print(f"  [{tag}] done: {total:,} examples.", flush=True)
    return {
        "hf_split": hf_split,
        "total": total,
        "fields": fields,
        "center_counts": dict(center_counts),
        "label_counts": dict(label_counts),
        "center_label_counts": {f"{c}|{l}": n for (c, l), n in center_label_counts.items()},
    }


def scan_logical_split(
    logical_split: str,
    *,
    progress_every: int = 20000,
) -> dict:
    """Stream a LOGICAL split (with its center filter) and tally counts.

    Returns total, per-label, and per-center counts for the filtered split.
    """
    # Imported here to avoid a hard dependency at module import time.
    from data.hf_camelyon17 import iter_logical_split

    print(f"[scan] logical split '{logical_split}' (streaming, metadata only)...", flush=True)
    center_counts: Counter = Counter()
    label_counts: Counter = Counter()
    total = 0

    for ex in iter_logical_split(logical_split, decode_images=False):
        center_counts[ex["center"]] += 1
        label_counts[ex["label"]] += 1
        total += 1
        if total % progress_every == 0:
            print(f"  [{logical_split}] scanned {total:,} ...", flush=True)

    print(f"  [{logical_split}] done: {total:,} examples.", flush=True)
    return {
        "logical_split": logical_split,
        "total": total,
        "label_counts": dict(label_counts),
        "center_counts": dict(center_counts),
    }


def save_sample_grid(
    out_path: Path,
    *,
    logical_split: str = "id_val",
    n: int = 12,
    ncols: int = 4,
) -> Optional[Path]:
    """Save a small grid of decoded RGB sample patches from a logical split.

    Uses a limited stream (only ``n`` matching examples are decoded), so this does
    not force a full download. Returns the output path, or None if nothing decoded.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data.hf_camelyon17 import iter_logical_split

    print(f"[grid] collecting {n} sample patches from '{logical_split}'...", flush=True)
    samples = []
    for ex in iter_logical_split(logical_split, decode_images=True, limit=n):
        img = to_rgb(ex["image"])
        samples.append((img, ex["label"], ex["center"]))
        print(f"  [grid] {len(samples)}/{n} collected", flush=True)

    if not samples:
        print("  [grid] no samples collected; skipping grid.", flush=True)
        return None

    nrows = (len(samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    label_names = {0: "non-tumor", 1: "tumor"}
    for ax in axes:
        ax.axis("off")
    for ax, (img, label, center) in zip(axes, samples):
        ax.imshow(img)
        ax.set_title(
            f"{label_names.get(label, label)}\ncenter {center}", fontsize=8
        )
        ax.axis("off")

    fig.suptitle(f"Sample patches — {logical_split}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [grid] saved {out_path}", flush=True)
    return out_path
