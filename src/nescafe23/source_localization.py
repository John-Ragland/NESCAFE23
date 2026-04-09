"""
Shim module exposing the n23_source_localization_hoekstra paper package as
nescafe23.source_localization.

Install via: uv sync --extra source-localization
"""

try:
    import n23_source_localization_hoekstra as _pkg  # noqa: F401
    from n23_source_localization_hoekstra import *  # noqa: F401, F403
except ImportError as e:
    raise ImportError(
        "n23_source_localization_hoekstra is not installed. "
        "Run: uv sync --extra source-localization"
    ) from e
