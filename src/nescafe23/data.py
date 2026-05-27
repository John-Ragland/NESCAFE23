"""Data path utilities for NESCAFE23."""

import os
import pickle
from pathlib import Path

import fsspec
import gsw
import h5py
import numpy as np
import scipy.io as sio
import xarray as xr
import xrsignal as xrs
from dotenv import find_dotenv, load_dotenv
from scipy import signal
from tqdm import tqdm

load_dotenv(find_dotenv())


def get_data_dir() -> Path:
    """
    Return the NESCAFE23 data directory from the environment.

    Set ``NESCAFE23_DATADIR`` in your ``.env`` file (or environment) to the
    root of the experiment data tree.

    Returns
    -------
    Path
        Resolved path to the data directory.

    Raises
    ------
    EnvironmentError
        If ``NESCAFE23_DATADIR`` is not set.
    """
    path = os.environ.get("NESCAFE23_DATADIR")
    if not path:
        raise OSError(
            "Set NESCAFE23_DATADIR in your .env file to the data directory path. This can be set with nescafe23 CLI by typing `nescafe23` in the terminal."
        )
    return Path(path)


def _load_ctd(path: Path | None, filename: str, suffix: str) -> xr.Dataset:
    if path is None:
        path = get_data_dir() / "environment_data" / filename
    mat = sio.loadmat(str(path))
    return xr.Dataset(
        {
            "z": xr.DataArray(
                mat[f"z{suffix}"].squeeze(), dims="depth", attrs={"units": "m"}
            ),
            "t": xr.DataArray(
                mat[f"t{suffix}"].squeeze(),
                dims="depth",
                attrs={"units": "deg C", "long_name": "in-situ temperature"},
            ),
            "pt": xr.DataArray(
                mat[f"pt{suffix}"].squeeze(),
                dims="depth",
                attrs={"units": "deg C", "long_name": "potential temperature"},
            ),
            "s": xr.DataArray(
                mat[f"s{suffix}"].squeeze(),
                dims="depth",
                attrs={"units": "psu", "long_name": "practical salinity"},
            ),
            "c": xr.DataArray(
                mat[f"c{suffix}"].squeeze(),
                dims="depth",
                attrs={"units": "m/s", "long_name": "sound speed"},
            ),
        }
    )


def open_sion_ctd(path: Path | None = None) -> xr.Dataset:
    """
    Load the SION CTD cast.

    Parameters
    ----------
    path:
        Path to ``SION_CTD.mat``. If None, resolved from ``NESCAFE23_DATADIR``.

    Returns
    -------
    xr.Dataset
        Variables: ``z``, ``t``, ``pt``, ``s``, ``c`` along a ``depth`` dimension.
    """
    return _load_ctd(path, "SION_CTD.mat", "5")


def open_sioe_ctd(path: Path | None = None) -> xr.Dataset:
    """
    Load the SIOE CTD cast.

    Parameters
    ----------
    path:
        Path to ``SIOE_CTD.mat``. If None, resolved from ``NESCAFE23_DATADIR``.

    Returns
    -------
    xr.Dataset
        Variables: ``z``, ``t``, ``pt``, ``s``, ``c`` along a ``depth`` dimension.
    """
    return _load_ctd(path, "SIOE_CTD.mat", "8")


def open_glider(path: Path | None = None) -> xr.Dataset:
    """
    Open the Gulf Stream glider dataset.

    Parameters
    ----------
    path:
        Path to the NetCDF file. If None, resolved from ``GLIDER_DATA``.

    Returns
    -------
    xr.Dataset
        Gulf Stream glider dataset.
    """
    if path is None:
        p = os.environ.get("GLIDER_DATA")
        if not p:
            raise OSError("Set GLIDER_DATA in your .env file.")
        path = Path(p)
    return xr.open_dataset(path)


def open_argo(path: Path | None = None) -> xr.Dataset:
    """
    Open the NESMA-region Argo float profiles dataset.

    Parameters
    ----------
    path:
        Path to ``nesma_region_2005_2025.nc``. If None, resolved from
        ``ARGO_DATA_DIR``.

    Returns
    -------
    xr.Dataset
        Argo float profiles for the NESMA region.
    """
    if path is None:
        argo_dir = os.environ.get("ARGO_DATA_DIR")
        if not argo_dir:
            raise OSError("Set ARGO_DATA_DIR in your .env file.")
        path = Path(argo_dir) / "nesma_region_2005_2025.nc"
    return xr.open_dataset(path)


def open_ssh(path: Path | None = None) -> xr.Dataset:
    """
    Open the Copernicus CMEMS satellite altimetry SSH dataset.

    Parameters
    ----------
    path:
        Path to the NetCDF file. If None, resolved from
        ``COPERNICUS_SATELLITE_ALTIMETRY_PATH``.

    Returns
    -------
    xr.Dataset
        Copernicus SEALEVEL_GLO_PHY_L4_MY_008_047 altimetry dataset.
    """
    if path is None:
        p = os.environ.get("COPERNICUS_SATELLITE_ALTIMETRY_PATH")
        if not p:
            raise OSError("Set COPERNICUS_SATELLITE_ALTIMETRY_PATH in your .env file.")
        path = Path(p)
    return xr.open_dataset(path)


def open_woa(depth_grid: np.ndarray, path: Path | None = None) -> xr.Dataset:
    """
    Open and process World Ocean Atlas 2023 T/S/N² data for the NESMA region.

    Parameters
    ----------
    depth_grid:
        Target depth grid for interpolation (metres).
    path:
        Directory containing WOA23 NetCDF files. If None, resolved from
        ``WOA2023_PATH``.

    Returns
    -------
    xr.Dataset
        Variables: ``T``, ``S``, ``T_std``, ``N2`` interpolated onto
        ``depth_grid``.
    """
    if path is None:
        p = os.environ.get("WOA2023_PATH")
        if not p:
            raise OSError("Set WOA2023_PATH in your .env file.")
        path = Path(p)

    woa_T_ds = xr.open_dataset(path / "woa23_decav_t00_04.nc", decode_times=False)
    woa_S_ds = xr.open_dataset(path / "woa23_decav_s00_04.nc", decode_times=False)

    woa_T = (
        woa_T_ds["t_an"]
        .sel({"lon": slice(-66, -59), "lat": slice(36, 41)})[0]
        .mean(["lat", "lon"])
    )
    woa_S = (
        woa_S_ds["s_an"]
        .sel({"lon": slice(-66, -59), "lat": slice(36, 41)})[0]
        .mean(["lat", "lon"])
    )
    woa_T_std = (
        woa_T_ds["t_sd"]
        .sel({"lon": slice(-66, -59), "lat": slice(36, 41)})[0]
        .mean(["lat", "lon"])
    )

    pressure = gsw.p_from_z(-woa_T_ds["depth"], 38.5)
    SA = gsw.SA_from_SP(woa_S, pressure, -62.5, 38.5)
    CT = gsw.CT_from_t(SA, woa_T, pressure)
    N2, pmid = gsw.Nsquared(SA, CT, pressure, lat=38.5)
    N2_interp = np.interp(pressure, pmid, N2)

    woa = xr.Dataset(
        {"T": woa_T, "S": woa_S, "T_std": woa_T_std, "N2": (("depth",), N2_interp)},
        coords={"depth": woa_T_ds["depth"]},
    )
    woa = woa.dropna("depth")

    return woa.interp(
        {"depth": depth_grid},
        kwargs={"bounds_error": False, "fill_value": "extrapolate"},
    )


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
    from nescafe23.physics import aog_vsdg

    if fn is None:
        fn = str(get_data_dir() / "environment_data" / "FCTD_SoundSpeed_Colosi_v2.mat")

    fctd_mat = sio.loadmat(fn)

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


def convert_to_xarray(fn):
    """
    Convert a NESCAFE23 demodulated .mat / HDF5 file to an xarray Dataset.

    Specifically coded to parse the ``dmod`` .mat file structure for the
    processed NESCAFE 2023 data.

    Parameters
    ----------
    fn : str
        File path to the demodulated data file.

    Returns
    -------
    xarray.Dataset
        Dataset containing the pilot data.
    """
    ds = xr.open_dataset(fn, group="/dm", engine="h5netcdf", phony_dims="sort")
    actual_depth = h5py.File(fn)["hmdepact"][:].flatten()

    ds = ds.squeeze().rename_dims(
        {
            "phony_dim_5": "hydrophone",
            "phony_dim_6": "time",
            "phony_dim_7": "jds_dim",
            "phony_dim_8": "string",
        }
    )

    ds = ds.assign_coords(
        {
            "time": ds["tax"],
            "nominal_depth": ("hydrophone", ds["mrcvz"].values),
            "actual_depth": ("hydrophone", actual_depth),
        }
    ).drop_vars(["tax", "mrcvz"])

    dmod_c = xr.DataArray(
        ds["dmod"].values.view(dtype=np.complex128), dims=["hydrophone", "time"]
    )
    ds["dmod"] = dmod_c

    attrs_vars = [
        "Gs",
        "SL",
        "Tswp",
        "dmrms",
        "dmsnr",
        "fc",
        "fhi",
        "flo",
        "gdla",
        "gdr",
        "rcv",
        "rreflat",
        "rreflon",
        "src",
        "sreflat",
        "sreflon",
        "tofs",
    ]
    attrs = {}
    for var in attrs_vars:
        attrs[var] = ds[var].values

    ds.attrs = attrs
    ds = ds.drop_vars(attrs_vars)
    return ds


def open_receptions(time_slice=(90, 120)):
    """
    Open all receptions from the NESCAFE23 pilot study and remove tidal time shift.

    Requires ``NESCAFE23_DATADIR`` to be set in the environment or ``.env`` file.

    Parameters
    ----------
    time_slice : tuple of float
        (start, end) time window in seconds. Default is (90, 120).

    Returns
    -------
    list of xarray.DataArray
        One DataArray per reception file, with tidal time shift removed.
    """
    data_dir = get_data_dir()
    fs = fsspec.filesystem("")
    fns = fs.glob(f"{data_dir}/dmod/dm*.mat")

    T_t_path = data_dir / "dataproducts" / "T_t_estimated.pkl"
    try:
        with T_t_path.open("rb") as f:
            dT_t = pickle.load(f)["dT_t"]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"T_t_estimated.pkl not found under {data_dir}/dataproducts/.\n"
            "Generate it by running 'nescafe23' and selecting "
            "'compute arrival time offset'."
        ) from None

    dss = []
    for k, fn in enumerate(tqdm(fns)):
        ds = convert_to_xarray(fn).sel({"time": slice(time_slice[0], time_slice[1])})
        ds = ds["dmod"].assign_coords({"time": ds.time.values + dT_t[k]})
        dss.append(ds)

    return dss


def open_peak_locations():
    """
    Open observed peak locations.

    The file is produced by running::

        nescafe23

    and selecting **estimate peak locations**.

    Returns
    -------
    peak_locations : dict
        Dictionary containing observed peak locations with keys
        ``times``, ``angles``, and ``snrs``.

    Raises
    ------
    FileNotFoundError
        If ``dataproducts/peak_locations.pkl`` does not exist under the
        data directory.
    """
    path = get_data_dir() / "dataproducts" / "peak_locations.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Peak locations file not found at {path}.\n"
            "Generate it by running 'nescafe23' and selecting "
            "'estimate peak locations'."
        )
    with path.open("rb") as f:
        return pickle.load(f)
