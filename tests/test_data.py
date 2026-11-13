from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import nescafe23
from nescafe23 import acoustic_fields_from_TS, aog_vsdg


def _stratified_TS():
    depth = np.arange(0.0, 3000.0, 1.0)
    rng = np.array([0.0, 1e3, 2e3])
    t = 2.5 + 15 * np.exp(-depth / 600)
    SP = 34.9 + 0.5 * np.exp(-depth / 600)
    dims = ["depth", "range"]
    coords = {"depth": depth, "range": rng}
    return (
        xr.DataArray(np.tile(t[:, None], (1, 3)), dims=dims, coords=coords),
        xr.DataArray(np.tile(SP[:, None], (1, 3)), dims=dims, coords=coords),
    )


def test_acoustic_fields_from_TS_synthetic():
    t, SP = _stratified_TS()
    out = acoustic_fields_from_TS(t, SP)

    assert set(out.data_vars) == {"t", "SP", "c", "N", "SA", "dcdz_p"}
    assert out["c"].dims == ("depth", "range")

    # stable stratification everywhere
    assert np.all(out["N"].isel(depth=slice(1, -1)) > 0)

    # potential sound speed decreases with depth in a warm thermocline -> positive with z up
    assert np.all(out["dcdz_p"].sel(depth=slice(10, 1500)) > 0)

    # sound speed is Del Grosso at the in-situ pressure
    import gsw

    p = gsw.p_from_z(-t.depth, nescafe23.SION_LATLON[0])
    np.testing.assert_allclose(out["c"], aog_vsdg(p, t, SP))


def test_compute_filtered_fctd_grid_uses_helper():
    try:
        fctd = nescafe23.compute_filtered_fctd_grid(z_smooth=200, r_smooth=15000)
    except (FileNotFoundError, OSError, KeyError, ValueError) as err:
        pytest.skip(f"FCTD data not available: {err}")

    sl = fctd.sel(range=slice(0, 150e3)).transpose("depth", "range")
    rederived = acoustic_fields_from_TS(sl["t"], sl["SP"])
    for var in ["c", "N", "dcdz_p"]:
        xr.testing.assert_allclose(rederived[var], sl[var])
