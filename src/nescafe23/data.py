"""Data path utilities for NESCAFE23."""

import os
from pathlib import Path

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
