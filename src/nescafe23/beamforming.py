"""Beamforming utilities for NESCAFE23."""

import numpy as np
import scipy
import xarray as xr


def ULA_conventional_beamform_nostack(da, thetas, c0=1500):
    """
    Conventional beamforming for a uniform linear array without depth stacking.

    Parameters
    ----------
    da : xr.DataArray
        Signal to beamform. Should have dimensions ['depth', 'time'].
    thetas : np.array
        Array of angles (in degrees) to beamform.
    c0 : float
        Reference sound speed in m/s. Default is 1500.

    Returns
    -------
    dmod_bf : xr.DataArray
        Beamformed output with dimensions ['depth', 'time', 'angle'].
    delts : np.ndarray
        Time delays applied during beamforming.
    """
    zs = da.depth.values
    z0 = np.mean(zs)

    dt = float(da.time[1] - da.time[0])
    nt = len(da.time)

    f = scipy.fft.fftfreq(nt, dt)
    ws = 2 * np.pi * f

    w = 2 * np.pi * 250

    delts = -(zs - z0)[:, None] * np.sin(np.radians(thetas[None, :])) / c0

    da = da.transpose("depth", "time")
    dmodf = scipy.fft.fft(da.values, axis=1)
    dmodf_ps = dmodf[:, :, None] * np.exp(
        1j * (ws[None, :, None] + w) * delts[:, None, :]
    )

    dmod_ps_t = scipy.fft.ifft(dmodf_ps, axis=1)
    dmod_bf = xr.DataArray(
        dmod_ps_t,
        dims=["depth", "time", "angle"],
        coords={"depth": da.depth.values, "time": da.time.values, "angle": thetas},
    )

    return dmod_bf, delts
