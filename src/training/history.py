"""Incremental per-epoch history logging to CSV (Milestone 4A).

Unlike ``utils.io.write_csv`` (which writes a whole table at once), this appends one
row per call so a partially-completed run still leaves a readable log on disk. The
header is written exactly once, when the file does not yet exist.

Columns (fixed order)::

    epoch, train_loss, train_accuracy, id_val_loss, id_val_accuracy,
    train_n, id_val_n, is_best
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "id_val_loss",
    "id_val_accuracy",
    "train_n",
    "id_val_n",
    "is_best",
]


def append_history_csv(path: str | Path, row: Mapping[str, Any]) -> Path:
    """Append one epoch's ``row`` to the history CSV at ``path``.

    Writes the header row first if the file does not already exist. ``row`` must supply
    every column in :data:`HISTORY_COLUMNS`; extra keys are ignored.

    Parameters
    ----------
    path:
        CSV file to append to (parent dirs are created if needed).
    row:
        Mapping containing the columns in :data:`HISTORY_COLUMNS`.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = [c for c in HISTORY_COLUMNS if c not in row]
    if missing:
        raise KeyError(f"append_history_csv row missing columns: {missing}")

    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({c: row[c] for c in HISTORY_COLUMNS})
    return path
