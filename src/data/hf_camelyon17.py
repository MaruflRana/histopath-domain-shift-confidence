"""Hugging Face loader for Camelyon17-WILDS that honors the LOCKED split mapping.

This is the only sanctioned way to obtain data for the project. It never triggers
the official WILDS / CodaLab download; it uses the Hugging Face ``datasets`` package
with dataset id ``wltjr1007/Camelyon17-WILDS``.

Logical splits (see PROJECT_PROTOCOL.md / configs/dataset/camelyon17_hf.yaml):

    train     -> HF train,      center in {0, 3, 4}
    id_val    -> HF validation, center in {0, 3, 4}
    ood_val   -> HF validation, center == 1
    ood_test  -> HF test,       center == 2   (final reporting ONLY)

Two loading modes:
    * streaming (default): iterable, no full download; used for verification/scans.
    * in-memory: materialized ``Dataset`` with the center filter applied.

Images may be RGBA; use :func:`to_rgb` before any model use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

# Make sibling packages importable when this module is imported from a script.
_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils.io import dataset_config  # noqa: E402

HF_ID = "wltjr1007/Camelyon17-WILDS"

# Logical split -> (hf_split, allowed centers or None for "no filter").
# Kept in sync with configs/dataset/camelyon17_hf.yaml; the config is authoritative.
LOGICAL_SPLITS = ("train", "id_val", "ood_val", "ood_test")


def _config() -> dict:
    return dataset_config()


def split_spec(logical_split: str) -> dict:
    """Return the config entry for a logical split, validating the name."""
    if logical_split not in LOGICAL_SPLITS:
        raise ValueError(
            f"Unknown logical split {logical_split!r}. "
            f"Expected one of {LOGICAL_SPLITS}."
        )
    cfg = _config()
    return cfg["splits"][logical_split]


def hf_id() -> str:
    """The Hugging Face dataset id (from config, with a hard-coded fallback)."""
    return _config().get("dataset", {}).get("hf_id", HF_ID)


def expected_fields() -> list[str]:
    return list(_config()["dataset"]["expected_fields"])


def to_rgb(image):
    """Convert a PIL image to RGB (handles RGBA / P / L). Returns a PIL RGB image."""
    if image is None:
        return None
    if getattr(image, "mode", None) != "RGB":
        image = image.convert("RGB")
    return image


def _center_allowed(center: int, allowed: Optional[Iterable[int]]) -> bool:
    return allowed is None or center in allowed


def iter_logical_split(
    logical_split: str,
    *,
    decode_images: bool = False,
    limit: Optional[int] = None,
) -> Iterator[dict]:
    """Stream examples belonging to a logical split, applying the center filter.

    Parameters
    ----------
    logical_split:
        One of ``train``, ``id_val``, ``ood_val``, ``ood_test``.
    decode_images:
        If False (default) images are NOT decoded — fast for metadata-only scans.
        If True images are decoded (PIL); use :func:`to_rgb` on ``ex["image"]``.
    limit:
        Optional cap on the number of *matching* examples to yield (after filter).

    Yields
    ------
    dict
        The raw HF example dict (with ``image`` decoded per ``decode_images``).
    """
    from datasets import Image, load_dataset

    spec = split_spec(logical_split)
    hf_split = spec["hf_split"]
    allowed = spec.get("center_in")  # list or None
    allowed_set = set(allowed) if allowed is not None else None

    ds = load_dataset(hf_id(), split=hf_split, streaming=True)
    # Avoid decoding image bytes unless explicitly requested (big speedup on scans).
    if not decode_images:
        ds = ds.cast_column("image", Image(decode=False))

    yielded = 0
    for ex in ds:
        if not _center_allowed(ex["center"], allowed_set):
            continue
        if decode_images:
            ex["image"] = to_rgb(ex["image"])
        yield ex
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def load_logical_split(logical_split: str):
    """Return an in-memory HF ``Dataset`` for a logical split with the filter applied.

    This materializes the split (may download/cache shards). Prefer
    :func:`iter_logical_split` for verification scans. Images are left in their
    native mode; convert with :func:`to_rgb` at model time.
    """
    from datasets import load_dataset

    spec = split_spec(logical_split)
    hf_split = spec["hf_split"]
    allowed = spec.get("center_in")

    ds = load_dataset(hf_id(), split=hf_split)
    if allowed is not None:
        allowed_set = set(allowed)
        ds = ds.filter(lambda c: c in allowed_set, input_columns=["center"])
    return ds
