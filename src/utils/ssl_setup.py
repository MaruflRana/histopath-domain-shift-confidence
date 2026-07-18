"""SSL bootstrap for environments behind a TLS-intercepting proxy / antivirus.

This machine runs Norton, which MITMs HTTPS with its own root CA. Python's default
``certifi`` bundle does not trust that CA, so requests to huggingface.co fail with
``CERTIFICATE_VERIFY_FAILED``. We fix this WITHOUT disabling verification by building
a combined CA bundle (certifi + any extra corporate/AV CAs) and pointing the standard
SSL env vars at it.

:func:`configure_ssl` is idempotent and safe on machines with no extra CA (it then
just re-exports the plain certifi bundle). It MUST run before ``datasets`` /
``huggingface_hub`` are imported, so it is called from ``scripts/_bootstrap.py``.

Extra CA sources (all optional, merged if present):
  * env ``NODE_EXTRA_CA_CERTS``  (Norton sets this)
  * env ``HISTOPATH_EXTRA_CA_CERTS``  (``;`` or ``os.pathsep`` separated list)
  * explicitly configured extra-CA environment variables
"""

from __future__ import annotations

import os
from pathlib import Path

def _bundle_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "configs" / "certs" / "combined_ca_bundle.pem"


def _extra_ca_files() -> list[Path]:
    candidates: list[str] = []
    for var in ("NODE_EXTRA_CA_CERTS", "HISTOPATH_EXTRA_CA_CERTS"):
        val = os.environ.get(var)
        if val:
            candidates.extend(val.split(os.pathsep) if os.pathsep in val else [val])
    seen: set[str] = set()
    out: list[Path] = []
    for c in candidates:
        c = c.strip().strip('"')
        if not c:
            continue
        p = Path(c)
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            out.append(p)
    return out


def configure_ssl(verbose: bool = False) -> str:
    """Build/refresh the combined CA bundle and export SSL env vars.

    Returns the path to the CA bundle now in use. If no extra CA certs are found,
    the plain certifi bundle path is exported unchanged.
    """
    try:
        import certifi
    except ImportError:
        # Nothing we can do without certifi; leave the environment untouched.
        return os.environ.get("SSL_CERT_FILE", "")

    certifi_path = Path(certifi.where())
    extras = _extra_ca_files()

    if not extras:
        bundle = certifi_path
    else:
        bundle = _bundle_path()
        bundle.parent.mkdir(parents=True, exist_ok=True)
        parts = [certifi_path.read_text(encoding="utf-8")]
        for ca in extras:
            try:
                parts.append(ca.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
        bundle.write_text("\n".join(parts), encoding="utf-8")
        if verbose:
            names = ", ".join(str(p) for p in extras)
            print(f"[ssl] combined CA bundle: certifi + [{names}] -> {bundle}", flush=True)

    bundle_str = str(bundle)
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ[var] = bundle_str
    return bundle_str
