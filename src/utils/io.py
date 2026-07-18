"""Filesystem helpers for locating the project root, the results tree, and for
writing text / CSV / YAML artifacts in a Windows-safe way.

All output goes under ``results/`` (``tables/``, ``figures/``, ``logs/``).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def project_root() -> Path:
    """Return the project root directory.

    This file lives at ``<root>/src/utils/io.py``; the root is three parents up.
    """
    return Path(__file__).resolve().parents[2]


def results_dir() -> Path:
    return project_root() / "results"


def tables_dir() -> Path:
    return results_dir() / "tables"


def figures_dir() -> Path:
    return results_dir() / "figures"


def logs_dir() -> Path:
    return results_dir() / "logs"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (a directory) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_results_tree() -> None:
    """Make sure results/{tables,figures,logs} all exist."""
    for d in (tables_dir(), figures_dir(), logs_dir()):
        ensure_dir(d)


def load_yaml(path: Path) -> dict:
    """Load a YAML file into a dict."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_config() -> dict:
    """Load the locked dataset config."""
    return load_yaml(project_root() / "configs" / "dataset" / "camelyon17_hf.yaml")


def write_text(path: Path, text: str) -> Path:
    """Write text to ``path`` (UTF-8), creating parent dirs. Returns the path."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]] | Sequence[Sequence[Any]],
    header: Sequence[str] | None = None,
) -> Path:
    """Write rows to a CSV file (Windows-safe: ``newline=""``).

    ``rows`` may be an iterable of dicts (``header`` inferred from the first row
    unless given) or an iterable of sequences (``header`` optional).
    """
    ensure_dir(path.parent)
    rows = list(rows)

    with open(path, "w", encoding="utf-8", newline="") as f:
        if rows and isinstance(rows[0], Mapping):
            fieldnames = list(header) if header is not None else list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
            if header is not None:
                writer.writerow(header)
            writer.writerows(rows)
    return path


def tee(message: str, log_lines: list[str] | None = None) -> None:
    """Print ``message`` (flushed) and optionally append it to ``log_lines``."""
    print(message, flush=True)
    if log_lines is not None:
        log_lines.append(message)


def _add_src_to_path() -> None:
    """Ensure the project ``src`` dir is importable when running scripts directly."""
    src = str(project_root() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
