"""Balanced, capped subsets for short controlled training runs (Milestone 3B/3C).

Milestone 3A exposed a problem: the first ``train`` batch (streaming order) held only
label ``1``. Milestone 3C quantified it: the HF stream is heavily label-ordered — the
``train`` label-0 (non-tumor) block does not begin until ~151k into the stream. So we
pre-collect a *balanced* subset — exactly ``per_class`` examples of label 0 and
``per_class`` of label 1 — from a single logical split, then wrap them in a tiny
in-memory ``Dataset`` compatible with the existing transforms pipeline.

Two efficiency measures make this practical:
  * lazy image decode — labels are read WITHOUT decoding images (fast), and only the
    ``2 * per_class`` KEPT images are decoded (not the ~150k scanned-and-skipped ones),
  * a disk cache — the (expensive, download-bound) scan runs ONCE and the resulting
    subset is stored under ``results/cache/``; later runs load it instantly. This
    honors "controlled runtime" and "no duplicate downloads".

Only the requested logical split is read (``train`` for training, ``id_val`` for
validation sanity). ``ood_val`` / ``ood_test`` are never touched here. Images are
converted to RGB at collection time, exactly like :class:`data.torch_dataset`.

Public API:
    collect_balanced_examples(split_name, per_class, max_scan=None) -> list[dict]
    load_or_build_balanced_examples(split_name, per_class, ...)      -> list[dict]
    BalancedSubsetDataset(examples, transform)                       -> map-style Dataset
"""

from __future__ import annotations

import io
import random
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data.hf_camelyon17 import iter_logical_split, to_rgb  # noqa: E402
from utils.io import ensure_dir, project_root  # noqa: E402

# Same integer metadata fields the main dataset returns (kept identical so batches
# collate the same way regardless of which dataset produced them).
META_INT_FIELDS = ("center", "image_id", "patient", "node", "x_coord", "y_coord", "slide")

# Labels we balance across (0 = non-tumor, 1 = tumor).
_LABELS = (0, 1)

# On-disk cache format version (bump if the serialized layout changes).
_CACHE_VERSION = 1


def _decode_rgb(raw_image):
    """Return a fully-loaded PIL RGB image from a decoded PIL image or a bytes dict.

    With ``decode_images=False`` the HF loader yields ``{"bytes": ..., "path": ...}``
    instead of a PIL image; decoding it here (only for kept examples) avoids decoding
    every scanned-and-skipped image. ``.convert("RGB")`` forces a full load so the
    backing ``BytesIO`` can be released immediately.
    """
    if isinstance(raw_image, dict):
        from PIL import Image as PILImage

        data = raw_image.get("bytes")
        if data is not None:
            return PILImage.open(io.BytesIO(data)).convert("RGB")
        path = raw_image.get("path")
        if path:
            return PILImage.open(path).convert("RGB")
        raise ValueError("image dict had neither 'bytes' nor 'path'.")
    # already a PIL image (decode_images=True path)
    return to_rgb(raw_image)


def collect_balanced_examples(
    split_name: str,
    per_class: int,
    max_scan: Optional[int] = None,
    *,
    progress_every: int = 5000,
    seed: int = 1337,
    shuffle_collected: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Stream one logical split and collect a class-balanced subset.

    Collects exactly ``per_class`` examples for label 0 and ``per_class`` for label 1,
    decoding images to RGB. Because the HF stream is label-ordered (see
    ``scripts/07_label_order_diagnostic.py``), collection may scan far past the first
    class before the second appears; ``max_scan`` bounds that work and turns an
    unreachable balance into a clear error instead of an unbounded scan.

    Parameters
    ----------
    split_name:
        Logical split to read from — ``train`` or ``id_val`` only for this milestone.
        (``ood_val`` / ``ood_test`` are permitted by the loader but must NOT be used
        here; the caller is responsible for honoring that rule.)
    per_class:
        Number of examples to collect per class.
    max_scan:
        Cap on how many streamed examples to inspect before giving up. If the cap is
        reached before both classes are filled, a clear error is raised. ``None`` means
        scan until both classes are full (unbounded — use with care).
    progress_every:
        Emit a concise progress line every this many *scanned* examples (not every 64).
        Set <= 0 to silence per-scan progress entirely.
    seed:
        Seed for the shuffle applied when ``shuffle_collected`` is True (reproducible).
    shuffle_collected:
        If True, shuffle the final list so labels are interleaved (avoids a label-0
        block followed by a label-1 block). If False, order is label-0s then label-1s.
    verbose:
        Print start/finish lines. ``progress_every`` independently controls mid-scan
        progress.

    Returns
    -------
    list[dict]
        ``2 * per_class`` example dicts, each
        ``{"image": PIL RGB, "label": int, <meta ints>}`` — the same shape the main
        dataset caches, so :class:`BalancedSubsetDataset` can apply the transforms.

    Raises
    ------
    ValueError
        If ``per_class`` < 1, or if both classes cannot be filled within ``max_scan``
        (with the counts actually collected, for a clear diagnosis).
    """
    if per_class < 1:
        raise ValueError(f"per_class must be >= 1, got {per_class}.")

    buckets: dict[int, list[dict]] = {0: [], 1: []}
    target = per_class
    scanned = 0

    if verbose:
        print(
            f"[balanced:{split_name}] collecting {per_class}/class "
            f"(target {2 * per_class} total, max_scan={max_scan})...",
            flush=True,
        )

    for ex in iter_logical_split(split_name, decode_images=False, limit=None):
        scanned += 1
        label = int(ex["label"])
        if label in buckets and len(buckets[label]) < target:
            # Decode ONLY kept images (not the ~150k scanned-and-skipped ones).
            item = {"image": _decode_rgb(ex["image"]), "label": label}
            for f in META_INT_FIELDS:
                item[f] = int(ex[f])
            buckets[label].append(item)

        if progress_every and progress_every > 0 and scanned % progress_every == 0:
            print(
                f"  [balanced:{split_name}] scanned={scanned}  "
                f"label0={len(buckets[0])}/{target}  label1={len(buckets[1])}/{target}",
                flush=True,
            )

        if len(buckets[0]) >= target and len(buckets[1]) >= target:
            break

        if max_scan is not None and scanned >= max_scan:
            break

    n0, n1 = len(buckets[0]), len(buckets[1])
    if n0 < target or n1 < target:
        raise ValueError(
            f"Could not collect a balanced subset for split '{split_name}': "
            f"needed {target}/class, got label0={n0}, label1={n1} "
            f"after scanning {scanned} examples"
            + (f" (max_scan={max_scan} reached)." if max_scan is not None else ".")
            + " Increase max_scan if the split is heavily label-ordered."
        )

    # label-0 block then label-1 block; optionally interleave via shuffle.
    collected = buckets[0] + buckets[1]
    if shuffle_collected:
        random.Random(seed).shuffle(collected)

    if verbose:
        order = "shuffled" if shuffle_collected else "label-0 then label-1"
        print(
            f"  [balanced:{split_name}] DONE — label0={n0}, label1={n1}, "
            f"total={n0 + n1}, scanned={scanned}, order={order}.",
            flush=True,
        )

    return collected


def default_cache_dir() -> Path:
    """Project-relative cache directory for balanced subsets (``results/cache``)."""
    return project_root() / "results" / "cache"


def _cache_path(
    cache_dir: Path, split_name: str, per_class: int, seed: int, shuffle_collected: bool
) -> Path:
    order = "shuf" if shuffle_collected else "noshuf"
    return cache_dir / f"balanced_{split_name}_pc{per_class}_seed{seed}_{order}.pt"


def _serialize_examples(examples: list[dict]) -> dict:
    """Convert a list of {PIL image, label, meta} dicts to a picklable payload.

    Images are stored as uint8 ``[H, W, 3]`` arrays (portable, no PIL-pickle quirks).
    """
    images = [np.asarray(ex["image"], dtype=np.uint8) for ex in examples]
    labels = [int(ex["label"]) for ex in examples]
    meta = {f: [int(ex[f]) for ex in examples] for f in META_INT_FIELDS}
    return {"version": _CACHE_VERSION, "n": len(examples), "images": images,
            "labels": labels, "meta": meta}


def _deserialize_examples(payload: dict) -> list[dict]:
    """Rebuild the {PIL RGB image, label, meta} dict list from a cache payload."""
    from PIL import Image as PILImage

    n = int(payload["n"])
    examples: list[dict] = []
    for i in range(n):
        arr = np.asarray(payload["images"][i], dtype=np.uint8)
        item = {
            "image": PILImage.fromarray(arr, mode="RGB"),
            "label": int(payload["labels"][i]),
        }
        for f in META_INT_FIELDS:
            item[f] = int(payload["meta"][f][i])
        examples.append(item)
    return examples


def load_or_build_balanced_examples(
    split_name: str,
    per_class: int,
    max_scan: Optional[int] = None,
    *,
    progress_every: int = 5000,
    seed: int = 1337,
    shuffle_collected: bool = True,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Return a balanced subset, loading from disk cache if available.

    On a cache miss (or ``force_rebuild``), runs the one-time — potentially expensive,
    download-bound — :func:`collect_balanced_examples` scan and persists the result to
    ``cache_dir`` (default ``results/cache``). Later calls with the same
    ``(split_name, per_class, seed, shuffle_collected)`` load instantly from disk with
    no HF streaming at all.

    Cache validity is keyed on the filename ``(split, per_class, seed, order)`` plus a
    payload check that the loaded subset is class-balanced with ``per_class`` each.
    """
    cdir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = _cache_path(cdir, split_name, per_class, seed, shuffle_collected)

    if use_cache and not force_rebuild and path.exists():
        if verbose:
            print(f"[balanced:{split_name}] loading cached subset -> {path}", flush=True)
        payload = torch.load(path, weights_only=False)
        examples = _deserialize_examples(payload)
        # sanity: cache must be class-balanced as requested
        counts = {0: 0, 1: 0}
        for ex in examples:
            counts[ex["label"]] = counts.get(ex["label"], 0) + 1
        if counts.get(0, 0) == per_class and counts.get(1, 0) == per_class:
            if verbose:
                print(
                    f"  [balanced:{split_name}] cache HIT — label0={counts[0]}, "
                    f"label1={counts[1]}, total={len(examples)}.",
                    flush=True,
                )
            return examples
        if verbose:
            print(
                f"  [balanced:{split_name}] cache STALE (counts={counts}) — rebuilding.",
                flush=True,
            )

    # cache miss / stale / disabled -> build once and persist.
    examples = collect_balanced_examples(
        split_name,
        per_class,
        max_scan=max_scan,
        progress_every=progress_every,
        seed=seed,
        shuffle_collected=shuffle_collected,
        verbose=verbose,
    )
    if use_cache:
        ensure_dir(cdir)
        torch.save(_serialize_examples(examples), path)
        if verbose:
            print(f"  [balanced:{split_name}] cached subset -> {path}", flush=True)
    return examples


class BalancedSubsetDataset(Dataset):
    """Map-style dataset over a pre-collected list of example dicts.

    Output matches :class:`data.torch_dataset.Camelyon17HFDataset.__getitem__` exactly:
    ``{"image": tensor[3,96,96], "label": long tensor, <meta ints>}``.
    """

    def __init__(self, examples: list[dict], transform: Optional[Callable] = None) -> None:
        self.examples = examples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        raw = self.examples[idx]
        image = raw["image"]
        if self.transform is not None:
            image = self.transform(image)
        out = {
            "image": image,
            "label": torch.tensor(raw["label"], dtype=torch.long),
        }
        for f in META_INT_FIELDS:
            out[f] = int(raw[f])
        return out

    def label_counts(self) -> dict[int, int]:
        """Return a {label: count} dict over the collected examples."""
        counts: dict[int, int] = {}
        for ex in self.examples:
            counts[ex["label"]] = counts.get(ex["label"], 0) + 1
        return counts
