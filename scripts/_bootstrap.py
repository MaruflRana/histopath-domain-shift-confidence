"""Shared bootstrap for scripts: put ``src`` on sys.path so ``utils`` / ``data``
packages import cleanly regardless of the current working directory (Windows-safe).

Import this first in every script::

    import _bootstrap  # noqa: F401
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Configure SSL BEFORE any datasets / huggingface_hub import (TLS-intercepting AV).
try:
    from utils.ssl_setup import configure_ssl

    configure_ssl()
except Exception:  # never let SSL setup crash a script; failures surface at request time
    pass
