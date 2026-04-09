"""
Shim module exposing the nescafe23_fluctuations paper package as nescafe23.fluctuations.

Install via: uv sync --extra fluctuations
"""

try:
    import nescafe23_fluctuations as _pkg  # noqa: F401
    from nescafe23_fluctuations import *  # noqa: F401, F403
except ImportError as e:
    raise ImportError(
        "nescafe23_fluctuations is not installed. "
        "Run: uv sync --extra fluctuations"
    ) from e
