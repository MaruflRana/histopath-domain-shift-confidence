"""Milestone 1 / Script 00: environment check.

Prints Python, PyTorch, CUDA, GPU, and datasets versions, and mirrors the report
to results/logs/env_check.txt. No dataset access.
"""

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

import platform

from utils.io import ensure_results_tree, logs_dir, tee, write_text


def main() -> None:
    ensure_results_tree()
    lines: list[str] = []

    tee("=" * 60, lines)
    tee("Environment check (Milestone 1 / script 00)", lines)
    tee("=" * 60, lines)

    tee(f"Python version    : {platform.python_version()}", lines)
    tee(f"Platform          : {platform.platform()}", lines)

    # PyTorch / CUDA / GPU
    try:
        import torch

        tee(f"PyTorch version   : {torch.__version__}", lines)
        cuda_ok = torch.cuda.is_available()
        tee(f"CUDA available    : {cuda_ok}", lines)
        if cuda_ok:
            tee(f"CUDA version      : {torch.version.cuda}", lines)
            tee(f"GPU count         : {torch.cuda.device_count()}", lines)
            tee(f"GPU name          : {torch.cuda.get_device_name(0)}", lines)
        else:
            tee("GPU name          : (no CUDA device)", lines)
    except ImportError:
        tee("PyTorch version   : NOT INSTALLED", lines)
        tee("CUDA available    : False", lines)
        tee("GPU name          : (torch not installed)", lines)

    # datasets
    try:
        import datasets

        tee(f"datasets version  : {datasets.__version__}", lines)
    except ImportError:
        tee("datasets version  : NOT INSTALLED", lines)

    tee("=" * 60, lines)

    out = logs_dir() / "env_check.txt"
    write_text(out, "\n".join(lines) + "\n")
    print(f"Saved environment report -> {out}", flush=True)


if __name__ == "__main__":
    main()
