"""Per-example prediction export (Milestone 4B).

Runs a trained model over a DataLoader (no gradients) and returns a tidy pandas
DataFrame with one row per example: the preserved metadata, both logits, both softmax
probabilities, the hard prediction, and whether it was correct. Intended for cached
``train`` / ``id_val`` subsets only — this function does not know about splits and must
never be pointed at ``ood_val`` / ``ood_test`` (the caller enforces that).

Batch format follows :class:`data.balanced_subset.BalancedSubsetDataset` /
:class:`data.torch_dataset.Camelyon17HFDataset`::

    batch["image"]  -> float32 [B, 3, 96, 96]
    batch["label"]  -> int64  [B]
    batch[<meta>]   -> center, image_id, patient, node, x_coord, y_coord, slide (ints)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Metadata fields carried through from the dataset (order matters for the output).
META_FIELDS = ("center", "image_id", "patient", "node", "x_coord", "y_coord", "slide")

OUTPUT_COLUMNS = [
    "split", "index", "label",
    *META_FIELDS,
    "logit_0", "logit_1", "prob_0", "prob_1", "pred_label", "correct",
]


def _to_int_list(value) -> list[int]:
    """Coerce a collated metadata field (tensor or list) to a list of Python ints."""
    if torch.is_tensor(value):
        return [int(v) for v in value.detach().cpu().tolist()]
    return [int(v) for v in value]


def predict_to_dataframe(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
) -> pd.DataFrame:
    """Run inference over ``dataloader`` and return a per-example prediction DataFrame.

    The model is put in eval mode and run under ``torch.no_grad()``; logits are turned
    into probabilities with a softmax over the 2 classes. Row order follows the loader
    (use ``shuffle=False`` so ``index`` maps to dataset position).

    Parameters
    ----------
    model:
        Trained classifier producing logits ``[B, 2]``.
    dataloader:
        Loader over a cached balanced subset (``train`` or ``id_val`` only).
    device:
        Device to run the forward pass on.
    split_name:
        Logical split label written into the ``split`` column (e.g. ``"train"``).

    Returns
    -------
    pandas.DataFrame
        Columns: ``split, index, label, center, image_id, patient, node, x_coord,
        y_coord, slide, logit_0, logit_1, prob_0, prob_1, pred_label, correct``.
    """
    model.eval()

    rows: list[dict] = []
    running_index = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)                       # [B, 2]
            probs = F.softmax(logits, dim=1)             # [B, 2]
            preds = logits.argmax(dim=1)                 # [B]

            logits_cpu = logits.detach().cpu()
            probs_cpu = probs.detach().cpu()
            preds_cpu = preds.detach().cpu()
            labels_cpu = labels.detach().cpu()

            meta = {f: _to_int_list(batch[f]) for f in META_FIELDS}
            bsz = int(labels_cpu.shape[0])

            for i in range(bsz):
                label_i = int(labels_cpu[i].item())
                pred_i = int(preds_cpu[i].item())
                row = {
                    "split": split_name,
                    "index": running_index,
                    "label": label_i,
                }
                for f in META_FIELDS:
                    row[f] = meta[f][i]
                row["logit_0"] = float(logits_cpu[i, 0].item())
                row["logit_1"] = float(logits_cpu[i, 1].item())
                row["prob_0"] = float(probs_cpu[i, 0].item())
                row["prob_1"] = float(probs_cpu[i, 1].item())
                row["pred_label"] = pred_i
                row["correct"] = int(pred_i == label_i)
                rows.append(row)
                running_index += 1

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
