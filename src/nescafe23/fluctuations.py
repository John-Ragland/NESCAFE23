"""
Shim module exposing the n23_fluctuations_ragland paper package as nescafe23.fluctuations.

Install via: uv sync --extra fluctuations
"""

try:
    import n23_fluctuations_ragland as _pkg  # noqa: F401
    from n23_fluctuations_ragland import *  # noqa: F401, F403
except ImportError as e:
    raise ImportError(
        "n23_fluctuations_ragland is not installed. "
        "Run: uv sync --extra fluctuations"
    ) from e
