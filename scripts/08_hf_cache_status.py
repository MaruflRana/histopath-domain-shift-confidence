"""Milestone 3C / Script 08: Hugging Face cache status (read-only, no download).

Reports where HF would cache Camelyon17-WILDS and how much disk is available, WITHOUT
importing ``datasets`` or triggering any download. Helps decide whether a local cache
already exists / is usable before any longer training.

Checks:
  * HF_HOME / HF_DATASETS_CACHE environment variables,
  * huggingface_hub cache constants (if importable),
  * whether a cache folder for ``wltjr1007/Camelyon17-WILDS`` already exists,
  * free disk space on the project drive and the cache drive.

Output:
  results/logs/hf_cache_status.txt
"""

import _bootstrap  # noqa: F401  (sys.path only; no HF import needed)

import os
import shutil
import sys
from pathlib import Path

from utils.io import ensure_results_tree, logs_dir, project_root, write_text

HF_DATASET_ID = "wltjr1007/Camelyon17-WILDS"
# huggingface_hub encodes a repo as "datasets--<org>--<name>" inside the hub cache.
_CACHE_FOLDER_NAME = "datasets--" + HF_DATASET_ID.replace("/", "--")


def _fmt_gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def _disk_free(path: Path):
    """Return (total, used, free) for the volume containing ``path``, or None."""
    try:
        # Walk up to the first existing ancestor (a non-existent subdir still resolves
        # to an existing drive root on Windows).
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = shutil.disk_usage(str(probe))
        return usage.total, usage.used, usage.free
    except Exception:
        return None


def main() -> int:
    ensure_results_tree()
    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    emit("=" * 64)
    emit("Milestone 3C — Hugging Face cache status (script 08, read-only)")
    emit(f"dataset id: {HF_DATASET_ID}")
    emit("=" * 64)

    # --- environment variables ------------------------------------------------ #
    emit("")
    emit("--- environment ---")
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        emit(f"  {var:<20} = {os.environ.get(var, '(not set)')}")

    # --- huggingface_hub constants (no dataset load) -------------------------- #
    emit("")
    emit("--- huggingface_hub cache constants ---")
    hub_cache_dir = None
    try:
        from huggingface_hub import constants as hf_const  # metadata only, no download

        hub_cache_dir = getattr(hf_const, "HF_HUB_CACHE", None) or getattr(
            hf_const, "HUGGINGFACE_HUB_CACHE", None
        )
        hf_home = getattr(hf_const, "HF_HOME", None)
        emit(f"  HF_HOME (resolved)   = {hf_home}")
        emit(f"  HF_HUB_CACHE         = {hub_cache_dir}")
    except Exception as exc:
        emit(f"  huggingface_hub constants unavailable: {type(exc).__name__}: {exc}")

    # --- dataset cache folder existence --------------------------------------- #
    emit("")
    emit("--- dataset cache presence ---")
    cache_status = "unknown"
    candidate_roots: list[Path] = []
    if hub_cache_dir:
        candidate_roots.append(Path(hub_cache_dir))
    # also check the conventional datasets subdir under HF_HOME
    hf_home_env = os.environ.get("HF_HOME")
    if hf_home_env:
        candidate_roots.append(Path(hf_home_env) / "hub")
        candidate_roots.append(Path(hf_home_env) / "datasets")
    default_home = Path.home() / ".cache" / "huggingface"
    candidate_roots.append(default_home / "hub")
    candidate_roots.append(default_home / "datasets")

    found_paths: list[Path] = []
    seen: set[str] = set()
    for root in candidate_roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        exists = root.exists()
        marker = root / _CACHE_FOLDER_NAME
        marker_exists = marker.exists()
        emit(f"  root {root}  exists={exists}")
        if marker_exists:
            emit(f"    -> FOUND dataset cache: {marker}")
            found_paths.append(marker)

    if found_paths:
        # Inspect approximate size / completeness of the first hit.
        target = found_paths[0]
        try:
            total_bytes = sum(
                f.stat().st_size for f in target.rglob("*") if f.is_file()
            )
            n_files = sum(1 for f in target.rglob("*") if f.is_file())
            emit(f"  cached files         = {n_files}")
            emit(f"  cached size (approx) = {_fmt_gb(total_bytes)}")
            # Heuristic: the full dataset is tens of GB; a tiny cache is partial.
            cache_status = "partial (streaming shards only)" if total_bytes < (5 * 1024 ** 3) else "substantial"
        except Exception as exc:
            emit(f"  (could not size cache: {type(exc).__name__}: {exc})")
            cache_status = "present (size unknown)"
    else:
        emit("  no local dataset cache folder found (streaming does not require one).")
        cache_status = "none (streaming mode)"

    # --- disk space ----------------------------------------------------------- #
    emit("")
    emit("--- disk free space ---")
    proj = project_root()
    emit(f"  project root         = {proj}")
    proj_usage = _disk_free(proj)
    if proj_usage:
        total, used, free = proj_usage
        emit(f"  project drive        : free {_fmt_gb(free)} / total {_fmt_gb(total)}")

    if hub_cache_dir:
        cache_usage = _disk_free(Path(hub_cache_dir))
        if cache_usage:
            total, used, free = cache_usage
            emit(f"  cache drive          : free {_fmt_gb(free)} / total {_fmt_gb(total)}  ({hub_cache_dir})")

    emit("")
    emit("=" * 64)
    emit(f"CACHE STATUS: {cache_status}")
    emit("(Streaming is used by the pipeline; a full local cache is NOT required.)")
    emit("=" * 64)

    out = logs_dir() / "hf_cache_status.txt"
    write_text(out, "\n".join(lines) + "\n")
    print(f"\nSaved report -> {out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
