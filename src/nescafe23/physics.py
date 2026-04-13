"""Oceanographic physics utilities for NESCAFE23."""

import numpy as np
import xarray as xr
import gsw
import xrsignal as xrs
from scipy import io, signal


def aog_vsdg(pr, tem, sal):
    """
    Compute sound speed in seawater using the Del Grosso equation.
    Modified from aog_vsdg.m matlab function.

    Parameters
    ----------
    pr : array_like
        Pressure in units dbar.
    tem : array_like
        In-situ temperature in degrees Celsius.
    sal : array_like
        Salinity in parts per thousand.

    Returns
    -------
    cs : array_like
        Sound speed in meters/second.

    References
    ----------
    Del Grosso, V. A. (1974). New equation for the speed of sound in natural
    waters. J. Acoust. Soc. Am., 56(4).
    """
    pr = pr / 9.80665  # convert pressure to kg/cm**2
    c0 = 1402.392
    vt = (
        0.501109398873e01 - (0.550946843172e-01 - 0.221535969240e-03 * tem) * tem
    ) * tem

    vs = (0.132952290781e01 + 0.128955756844e-03 * sal) * sal

    vp = (0.156059257041e00 + (0.244998688441e-04 - 0.883392332513e-08 * pr) * pr) * pr
    vstp = (
        -0.127562783426e-01 * tem * sal
        + 0.635191613389e-02 * tem * pr
        + 0.265484716608e-07 * tem * tem * pr * pr
        - 0.159349479045e-05 * tem * pr * pr
        + 0.522116437235e-09 * tem * pr * pr * pr
        - 0.438031096213e-06 * tem * tem * tem * pr
        - 0.161674495909e-08 * sal * sal * pr * pr
        + 0.968403156410e-04 * tem * tem * sal
        + 0.485639620015e-05 * tem * sal * sal * pr
        - 0.340597039004e-03 * tem * sal * pr
    )

    cs = c0 + vt + vs + vp + vstp

    return cs


def compute_filtered_fctd_grid(
    fn: str | None = None,
    z_smooth: float | None = 200,
    r_smooth: float | None = 5000,
    return_latlon: bool = False,
) -> xr.Dataset:
    """
    Compute and filter FCTD grid data without slicing or stitching.

    Loads FCTD data, interpolates it to a regular grid, applies Butterworth
    filters in both vertical and horizontal directions, flips the range
    coordinate (centered at 115000 m of original range), and computes
    oceanographic variables using GSW.

    Parameters
    ----------
    fn : str or None
        Path to the FCTD .mat file containing fast CTD data. If None, the path
        is constructed from the ``NESCAFE23_DATADIR`` environment variable as
        ``{NESCAFE23_DATADIR}environment_data/FCTD_SoundSpeed_Colosi_v2.mat``.
    z_smooth : float or None
        Vertical smoothing scale in meters. Default is 200 m.
        If None, no vertical filtering is applied.
    r_smooth : float or None
        Horizontal smoothing scale in meters. Default is 5000 m.
        If None, no horizontal filtering is applied.
    return_latlon : bool
        If True, include latitude and longitude coordinates in the output dataset.

    Returns
    -------
    fctd_output : xarray.Dataset
        Dataset containing filtered FCTD data with oceanographic variables:

        - t: in-situ temperature (deg C)
        - SP: practical salinity (psu)
        - c: sound speed (m/s)
        - N: buoyancy frequency (cph)
        - SA: absolute salinity (g/kg)
        - dcdz_p: adiabatic vertical sound speed gradient (m/s^2)
    """
    if fn is None:
        from nescafe23.data import get_data_dir
        fn = str(get_data_dir() / "environment_data" / "FCTD_SoundSpeed_Colosi_v2.mat")

    fctd_mat = io.loadmat(fn)

    fctds = {}
    fctds_pilot = {}

    deep_value = {"tt": 2.3, "ss": 34.885}

    for var in ["tt", "ss"]:
        fctds[var] = xr.DataArray(
            fctd_mat[var],
            dims=["range", "depth"],
            coords={
                "range": fctd_mat["r"].flatten() * 1000,
                "depth": fctd_mat["z"].flatten(),
                "latitude": ("range", fctd_mat["lat"].flatten()),
                "longitude": ("range", fctd_mat["lon"].flatten()),
            },
        ).sortby("range")

        fctds_pilot[var] = xr.concat(
            (
                fctds[var],
                xr.ones_like(fctds[var][:, -2:-1]).assign_coords({"depth": [10000]})
                * deep_value[var],
            ),
            dim="depth",
        )

    fctds_pilot = xr.Dataset(fctds_pilot)

    fctd_grid = fctds_pilot.interp(
        {
            "range": np.arange(0, fctds_pilot.range[-1], 500),
            "depth": np.arange(0, 6000),
        }
    )

    if z_smooth is not None:
        bz, az = signal.butter(4, 1 / z_smooth, fs=1)
    if r_smooth is not None:
        br, ar = signal.butter(4, 1 / r_smooth, fs=1 / 500)

    fctd_filtered = {}
    for var in ["tt", "ss"]:
        fctd_smooth = fctd_grid[var]

        if r_smooth is not None:
            fctd_smooth = xrs.filtfilt(fctd_smooth, dim="range", b=br, a=ar)

        if z_smooth is not None:
            fctd_smooth = xrs.filtfilt(fctd_smooth, dim="depth", b=bz, a=az)

        fctd_filtered[var] = fctd_smooth

    fctd_filtered = xr.Dataset(fctd_filtered)

    if not return_latlon:
        fctd_filtered = fctd_filtered.drop_vars(["latitude", "longitude"])

    fctd_filtered["tt"].attrs["units"] = "deg C"
    fctd_filtered["tt"].name = "in-situ temperature"
    fctd_filtered["ss"].attrs["units"] = "psu"
    fctd_filtered["ss"].name = "practical salinity"

    coords = {
        "SION": [39 + 37.194 / 60, -63 - 39.427 / 60],
    }

    fctd_transposed = fctd_filtered.transpose("depth", "range")

    p = gsw.p_from_z(-fctd_transposed.depth, coords["SION"][0])

    SA = gsw.SA_from_SP(fctd_transposed["ss"], p, coords["SION"][1], coords["SION"][0])
    SA.attrs["units"] = "g/kg"
    SA.name = "Absolute Salinity"

    CT = gsw.CT_from_t(SA, fctd_transposed["tt"], p)
    CT.attrs["units"] = "deg C"
    CT.name = "Conservative Temperature"

    c = aog_vsdg(p, fctd_transposed["tt"], fctd_transposed["ss"])
    c.attrs["units"] = "m/s"
    c.name = "Sound Speed"

    N2, pmid = gsw.Nsquared(SA, CT, p.values[:, None], lat=coords["SION"][0], axis=0)
    z_mid = gsw.z_from_p(pmid, coords["SION"][0])[:, 0]
    N2x = xr.DataArray(
        N2,
        dims=["depth", "range"],
        coords={"depth": -z_mid, "range": fctd_transposed.range},
    )
    N = (
        np.sqrt(
            N2x.interp(
                {"depth": fctd_transposed.depth},
                kwargs={"bounds_error": False, "fill_value": "extrapolate"},
            )
        )
        / (2 * np.pi)
        * 3600
    )
    N.attrs["units"] = "cph"
    N.name = "Buoyancy Frequency"

    PT = gsw.pt_from_t(SA, fctd_transposed["tt"], p, 0)
    PT.attrs["units"] = "deg C"
    PT.name = "potential temperature"

    C_adiabatic = aog_vsdg(0, PT, fctd_transposed["ss"])
    C_adiabatic.attrs["units"] = "deg C"
    C_adiabatic.name = "adiabatic sound speed"

    dcdz_p = -C_adiabatic.differentiate("depth")
    dcdz_p.attrs["units"] = "$m/s^2$"
    dcdz_p.name = "adiabatic vertical sound speed gradient"

    fctd_output = xr.Dataset(
        {
            "t": fctd_transposed["tt"],
            "SP": fctd_transposed["ss"],
            "c": c,
            "N": N,
            "SA": SA,
            "dcdz_p": dcdz_p,
        }
    )

    fctd_centered = fctd_output.assign_coords(
        {"range": 115000 - fctd_output.range.values}
    ).sortby("range")

    fctd_centered = xr.concat(
        (
            fctd_centered,
            fctd_centered.isel({"range": -1}).assign_coords({"range": 300000}),
        ),
        dim="range",
    )
    fctd_centered = xr.concat(
        (
            fctd_centered.isel({"range": 0}).assign_coords({"range": -300000}),
            fctd_centered,
        ),
        dim="range",
    )

    return fctd_centered
