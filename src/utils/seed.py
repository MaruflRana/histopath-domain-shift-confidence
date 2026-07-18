"""Global reproducible seeding for Python, NumPy, and PyTorch.

Import and call :func:`set_seed` at the start of every script that has any source
of randomness. Torch is imported lazily so that lightweight scripts (e.g. dataset
verification) do not pay the import cost or require CUDA.
"""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed all known RNGs.

    Parameters
    ----------
    seed:
        The integer seed to apply.
    deterministic:
        If True, request deterministic cuDNN behavior (slower but reproducible).

    Returns
    -------
    int
        The seed that was applied (for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
