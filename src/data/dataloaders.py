"""DataLoader construction for the locked Camelyon17 logical splits.

``shuffle=True`` is used ONLY for ``train``; validation/test splits are never
shuffled (deterministic evaluation order). Windows-safe defaults: ``num_workers=0``
and a seeded generator for reproducible shuffling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data.torch_dataset import Camelyon17HFDataset  # noqa: E402
from data.transforms import eval_transform, train_transform  # noqa: E402

# Splits that get training-style shuffling. Everything else is deterministic.
_SHUFFLE_SPLITS = {"train"}


def transform_for(split_name: str) -> Callable:
    """Return the transform appropriate for a split (train vs eval)."""
    return train_transform if split_name == "train" else eval_transform


def build_dataloader(
    split_name: str,
    batch_size: int,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    *,
    seed: int = 42,
    transform: Optional[Callable] = None,
    verbose: bool = True,
) -> DataLoader:
    """Build a DataLoader for one logical split.

    Parameters
    ----------
    split_name:
        train / id_val / ood_val / ood_test.
    batch_size:
        Batch size.
    num_workers:
        DataLoader workers. Default 0 (Windows-safe). >0 requires the calling script
        to be guarded by ``if __name__ == '__main__':``.
    max_samples:
        Cap the number of cached samples (small = fast smoke tests). None = full split.
    seed:
        Seed for the shuffling generator (train only) for reproducibility.
    transform:
        Override the default per-split transform.
    """
    tfm = transform if transform is not None else transform_for(split_name)
    dataset = Camelyon17HFDataset(
        split_name=split_name,
        transform=tfm,
        max_samples=max_samples,
        verbose=verbose,
    )

    shuffle = split_name in _SHUFFLE_SPLITS
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        generator=generator,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(**loader_kwargs)
