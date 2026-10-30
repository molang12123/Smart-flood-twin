"""Portable runtime caches and optional memory telemetry.

No Windows-only or POSIX-only package is required to run the workflow. Missing
memory telemetry is reported as unavailable, never as a measured zero.
"""
from __future__ import annotations

import getpass
import hashlib
import importlib
import math
import os
from pathlib import Path
import sys
import tempfile


def prepare_runtime_cache() -> dict:
    """Use writable OS temporary folders unless the caller configured caches.

    Call before importing matplotlib. Existing explicit environment settings are
    preserved, including their errors if they point at an unusable location.
    """
    try:
        identity = getpass.getuser()
    except (OSError, KeyError):
        identity = str(getattr(os, "getuid", lambda: "current-user")())
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    root = Path(tempfile.gettempdir()) / f"flood-pinn-{suffix}"
    defaults = {"MPLCONFIGDIR": root / "matplotlib", "XDG_CACHE_HOME": root / "cache"}
    paths = {}
    for variable, default in defaults.items():
        value = os.environ.setdefault(variable, str(default))
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ValueError(f"{variable} must refer to a directory")
        paths[variable] = str(path)
    return paths


def process_memory_info() -> dict:
    """Return process peak RSS normalized to KiB, or explicit null telemetry.

    macOS ru_maxrss is bytes; Linux ru_maxrss is KiB. The standard resource
    module is unavailable on Windows, and unsupported platforms must not guess
    the unit. This measures the whole process, not just network inference.
    """
    unavailable = {
        "peak_process_rss_kib": None,
        "rss_scope": "unavailable on this platform; no RSS measurement was made",
        "rss_measurement_status": "unavailable",
        "platform": sys.platform,
    }
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        return unavailable
    try:
        resource = importlib.import_module("resource")
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if not math.isfinite(value) or value < 0:
            return unavailable
    except (ImportError, AttributeError, OSError, ValueError, TypeError):
        return unavailable
    if sys.platform == "darwin":
        value /= 1024.0
    return {
        "peak_process_rss_kib": value,
        "rss_scope": "whole process including loading and plotting; normalized KiB",
        "rss_measurement_status": "measured",
        "platform": sys.platform,
    }
