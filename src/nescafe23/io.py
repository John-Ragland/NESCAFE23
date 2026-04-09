"""Data I/O utilities for NESCAFE23."""

import os
import pickle

import fsspec
import h5py
import numpy as np
import xarray as xr
from tqdm import tqdm

from nescafe23.data import get_data_dir


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
    data_dir = str(get_data_dir())
    fs = fsspec.filesystem("")
    fns = fs.glob(f"{data_dir}dmod/dm*.mat")

    with open(f"{data_dir}/dataproducts/T_t_estimated.pkl", "rb") as f:
        dT_t = pickle.load(f)["dT_t"]

    dss = []
    for k, fn in enumerate(tqdm(fns)):
        ds = convert_to_xarray(fn).sel({"time": slice(time_slice[0], time_slice[1])})
        ds = ds["dmod"].assign_coords({"time": ds.time.values + dT_t[k]})
        dss.append(ds)

    return dss
