from __future__ import annotations

import nescafe23


def test_imports():
    assert hasattr(nescafe23, "convert_to_xarray")
    assert hasattr(nescafe23, "open_receptions")
    assert hasattr(nescafe23, "get_data_dir")
