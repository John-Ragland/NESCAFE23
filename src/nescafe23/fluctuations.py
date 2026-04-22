"""
Shim module exposing the n23_fluctuations_ragland paper package as nescafe23.fluctuations.

Install via: uv sync --extra fluctuations
"""

import importlib

try:
    _pkg = importlib.import_module("n23_fluctuations_ragland")
except ImportError as e:
    raise ImportError(
        "n23_fluctuations_ragland is not installed. Run: uv sync --extra fluctuations"
    ) from e


def __getattr__(name: str):
    return getattr(_pkg, name)


def __dir__() -> list[str]:
    return dir(_pkg)
