"""Compatibility shim for the maintained FungiCirno service module.

The application imports ``mushroom_monitor.service`` from ``src/``. This top-level
module used to contain a duplicate implementation and is retained only for older
scripts that import ``service`` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from mushroom_monitor.service import ModelService, add_environment_rows  # noqa: E402,F401

__all__ = ["ModelService", "add_environment_rows"]
