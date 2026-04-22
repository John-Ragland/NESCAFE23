"""
Shim module exposing the n23_source_localization_hoekstra paper package as
nescafe23.source_localization.

Install via: uv sync --extra source-localization
"""

import importlib

try:
    _pkg = importlib.import_module("n23_source_localization_hoekstra")
except ImportError as e:
    raise ImportError(
        "n23_source_localization_hoekstra is not installed. "
        "Run: uv sync --extra source-localization"
    ) from e


def __getattr__(name: str):
    return getattr(_pkg, name)


def __dir__() -> list[str]:
    return dir(_pkg)
