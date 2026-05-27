"""Data path utilities for NESCAFE23."""

import os
from pathlib import Path

import gsw
import numpy as np
import scipy.io as sio
import xarray as xr
from dotenv import find_dotenv, load_dotenv

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
