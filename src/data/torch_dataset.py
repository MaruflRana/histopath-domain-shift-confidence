"""Map-style PyTorch dataset wrapping the LOCKED Camelyon17 logical splits.

Data comes from the Hugging Face mirror (``wltjr1007/Camelyon17-WILDS``) via the
loaders in :mod:`data.hf_camelyon17`, which apply the center filters. No random
splits, no WILDS/CodaLab download.

Two materialization modes (both map-style, so DataLoader can index them):
  * ``max_samples`` set  -> stream up to N filtered examples and cache them in RAM.
    Fast, tiny footprint; ideal for smoke tests. Images are decoded once and stored
    as RGB PIL images.
  * ``max_samples=None``  -> materialize the full logical split as an in-memory HF
    ``Dataset`` (downloads/caches shards) and index it lazily per __getitem__.

Each item is a dict:
    image     : float32 tensor [3, 96, 96]  (after transform)
    label     : int64 tensor (scalar)
    center, image_id, patient, node, x_coord, y_coord, slide : python ints

Images may be RGBA from Hugging Face; they are converted to RGB before transform.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data.hf_camelyon17 import (  # noqa: E402
    LOGICAL_SPLITS,
    iter_logical_split,
    load_logical_split,
    split_spec,
    to_rgb,
)

# Integer metadata fields returned alongside the image/label.
META_INT_FIELDS = ("center", "image_id", "patient", "node", "x_coord", "y_coord", "slide")


class Camelyon17HFDataset(Dataset):
    """Map-style dataset for one logical split of Camelyon17-WILDS.

    Parameters
    ----------
    split_name:
        One of ``train``, ``id_val``, ``ood_val``, ``ood_test``.
    transform:
        Callable mapping an RGB PIL image to a tensor. If None, images are returned
        as RGB PIL images (not typical; smoke test always passes a transform).
    max_samples:
        If set, only the first N filtered examples are streamed and cached in RAM.
        If None, the full split is materialized as an in-memory HF Dataset.
    verbose:
        Print progress while caching (streaming can be slow behind a TLS proxy).
    """

    def __init__(
        self,
        split_name: str,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        if split_name not in LOGICAL_SPLITS:
            raise ValueError(
                f"Unknown split_name {split_name!r}; expected one of {LOGICAL_SPLITS}."
            )
        self.split_name = split_name
        self.transform = transform
        self.max_samples = max_samples
        self.spec = split_spec(split_name)

        # Materialization
        self._cached: Optional[list[dict]] = None   # streamed mode (RGB PIL + meta)
        self._hf_ds = None                           # full mode (HF Dataset)

        if max_samples is not None:
            self._cached = self._stream_cache(max_samples, verbose)
            self._length = len(self._cached)
        else:
            if verbose:
                print(
                    f"[dataset] materializing full logical split '{split_name}' "
                    f"(in-memory HF Dataset)...",
                    flush=True,
                )
            self._hf_ds = load_logical_split(split_name)
            self._length = len(self._hf_ds)

    # ------------------------------------------------------------------ #
    def _stream_cache(self, n: int, verbose: bool) -> list[dict]:
        if verbose:
            print(
                f"[dataset] caching up to {n} samples from '{self.split_name}' "
                f"(streaming, RGB decode)...",
                flush=True,
            )
        cached: list[dict] = []
        for ex in iter_logical_split(self.split_name, decode_images=True, limit=n):
            item = {"image": to_rgb(ex["image"]), "label": int(ex["label"])}
            for f in META_INT_FIELDS:
                item[f] = int(ex[f])
            cached.append(item)
            if verbose and len(cached) % 32 == 0:
                print(f"  [dataset:{self.split_name}] cached {len(cached)}/{n}", flush=True)
        if verbose:
            print(f"  [dataset:{self.split_name}] cached {len(cached)} samples.", flush=True)
        return cached

    def _raw_item(self, idx: int) -> dict:
        """Return a normalized raw dict {image(PIL RGB), label, meta...} for idx."""
        if self._cached is not None:
            return self._cached[idx]
        ex = self._hf_ds[idx]
        item = {"image": to_rgb(ex["image"]), "label": int(ex["label"])}
        for f in META_INT_FIELDS:
            item[f] = int(ex[f])
        return item

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        raw = self._raw_item(idx)

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
