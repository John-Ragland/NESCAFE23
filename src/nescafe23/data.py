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

from nescafe23.processing import aog_vsdg

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


def _load_ctd(
    path: Path | None,
    filename: str,
    suffix: str,
    lat: float,
    lon: float,
    z_smooth: float | None = None,
    interpolate: bool = True,
    varmap: dict[str, str] | None = None,
) -> xr.Dataset:
    """
    Parameters
    ----------
    path : Path or None
        Directory containing the CTD .mat file. If None, resolved from ``NESCAFE23_DATADIR``.
    filename : str
        Name of the CTD .mat file to load.
    suffix : str
        Suffix used in the .mat file variable names (e.g. "5" for SION, "8" for SIOE).
        Ignored when ``varmap`` is provided.
    lat : float
        Latitude of the cast in decimal degrees (used for gsw pressure/salinity conversions).
    lon : float
        Longitude of the cast in decimal degrees (used for gsw absolute salinity conversion).
    z_smooth : float or None
        Vertical smoothing scale in meters. If None, no vertical filtering is applied
    interpolate : bool
        Whether to interpolate CTD data to regular 1 m depth grid. Default is True. If False and z_smooth is True, the CTD is smoothed anyways with a warning.
    varmap : dict or None
        Explicit mapping from canonical names (``z``, ``t``, ``s``) to the
        actual variable names in the .mat file. When None, names are derived as
        ``{var}{suffix}``.
    Returns
    -------
    ctd : xr.Dataset
        Variables: ``t``, ``s``, ``c``, ``pt`` along a ``depth`` dimension.
        ``c`` and ``pt`` are computed from ``t`` and ``s`` via gsw + aog_vsdg.
    """
    if path is None:
        path = get_data_dir() / "environment_data" / filename
    mat = sio.loadmat(str(path))

    def _key(canonical: str) -> str:
        return varmap[canonical] if varmap is not None else f"{canonical}{suffix}"

    depth_vals = mat[_key("z")].squeeze()

    t = xr.DataArray(
        mat[_key("t")].squeeze(),
        dims="depth",
        coords={"depth": depth_vals},
        attrs={"units": "deg C", "long_name": "in-situ temperature"},
    )
    s = xr.DataArray(
        mat[_key("s")].squeeze(),
        dims="depth",
        coords={"depth": depth_vals},
        attrs={"units": "psu", "long_name": "practical salinity"},
    )
    # extend t and s to 0 m by repeating the first value (if not already present)
    if float(t.depth[0]) != 0.0:
        t = xr.concat(
            (xr.ones_like(t[0:1]).assign_coords({"depth": [0]}) * float(t[0]), t),
            dim="depth",
        )
        s = xr.concat(
            (xr.ones_like(s[0:1]).assign_coords({"depth": [0]}) * float(s[0]), s),
            dim="depth",
        )
    # extend t and s to 10000 m by repeating the last value
    t = xr.concat(
        (t, xr.ones_like(t[-2:-1]).assign_coords({"depth": [10000]}) * float(t[-1])),
        dim="depth",
    )
    s = xr.concat(
        (s, xr.ones_like(s[-2:-1]).assign_coords({"depth": [10000]}) * float(s[-1])),
        dim="depth",
    )

    ctd = xr.Dataset({"t": t, "s": s})

    # interpolate to 1 m vertical grid
    if interpolate:
        ctd_grid = ctd.interp({"depth": np.arange(0, 6000, 1)})
    else:
        if z_smooth is not None:
            print(
                "Warning: z_smooth is set but interpolate is False. Smoothing will be applied to the original CTD depth grid, which may produce unexpected results. Consider setting interpolate to True for better results."
            )
    if z_smooth is not None:
        bz, az = signal.butter(4, 1 / z_smooth, fs=1)
        target = ctd_grid if interpolate else ctd
        dim_vars = set(target.dims)
        # filtfilt propagates NaN, so drop NaN rows and dimension-named variables first
        target_valid = target.drop_vars(
            [v for v in target.data_vars if v in dim_vars]
        ).dropna("depth")
        filtered_vars = {}
        for var in target_valid.data_vars:
            filtered_vals = xrs.filtfilt(
                target_valid[var], dim="depth", b=bz, a=az
            ).values
            filtered_vars[var] = xr.DataArray(
                filtered_vals,
                dims=["depth"],
                coords={"depth": target_valid.depth.values},
                attrs=target_valid[var].attrs,
            )
        result = xr.Dataset(filtered_vars)
    else:
        result = ctd if not interpolate else ctd_grid

    # compute derived variables from t and s
    p = gsw.p_from_z(-result.depth, lat)
    SA = gsw.SA_from_SP(result["s"], p, lon, lat)
    c = aog_vsdg(p, result["t"], result["s"])
    c.attrs["units"] = "m/s"
    c.attrs["long_name"] = "sound speed"
    pt = gsw.pt_from_t(SA, result["t"], p, 0)
    pt.attrs["units"] = "deg C"
    pt.attrs["long_name"] = "potential temperature"
    result = result.assign({"c": c, "pt": pt})

    return result


def open_sion_ctd(
    path: Path | None = None,
    z_smooth: float | None = None,
    interpolate: bool = True,
    lat: float = 39 + 37.194 / 60,
    lon: float = -63 - 39.427 / 60,
) -> xr.Dataset:
    """
    Load the SION CTD cast.

    Parameters
    ----------
    path:
        Path to ``SION_CTD.mat``. If None, resolved from ``NESCAFE23_DATADIR``.
    z_smooth : float or None
        Vertical smoothing scale in meters. If None, no vertical filtering is applied
    interpolate : bool
        Whether to interpolate CTD data to regular 1 m depth grid. Default is True. If False and z_smooth is True, the CTD is smoothed anyways with a warning.
    lat : float
        Latitude of the cast in decimal degrees. Default is the SION mooring position.
    lon : float
        Longitude of the cast in decimal degrees. Default is the SION mooring position.

    Returns
    -------
    xr.Dataset
        Variables: ``t``, ``pt``, ``s``, ``c`` along a ``depth`` dimension.
        ``c`` is computed from ``t`` and ``s`` via gsw + aog_vsdg.
    """
    return _load_ctd(
        path,
        "SION_CTD_new.mat",
        "5",
        lat=lat,
        lon=lon,
        z_smooth=z_smooth,
        interpolate=interpolate,
        varmap={"z": "zi", "t": "temp", "s": "sal"},
    )


def open_sioe_ctd(
    path: Path | None = None,
    z_smooth: float | None = None,
    interpolate: bool = True,
    lat: float = 38.76510,
    lon: float = -62.26043,
) -> xr.Dataset:
    """
    Load the SIOE CTD cast.

    Parameters
    ----------
    path:
        Path to ``SIOE_CTD.mat``. If None, resolved from ``NESCAFE23_DATADIR``.
    z_smooth : float or None
        Vertical smoothing scale in meters. If None, no vertical filtering is applied
    interpolate : bool
        Whether to interpolate CTD data to regular 1 m depth grid. Default is True. If False and z_smooth is True, the CTD is smoothed anyways with a warning.
    lat : float
        Latitude of the cast in decimal degrees. Default is the SIOE mooring position.
    lon : float
        Longitude of the cast in decimal degrees. Default is the SIOE mooring position.

    Returns
    -------
    xr.Dataset
        Variables: ``t``, ``pt``, ``s``, ``c`` along a ``depth`` dimension.
        ``c`` is computed from ``t`` and ``s`` via gsw + aog_vsdg.
    """
    return _load_ctd(
        path,
        "SIOE_CTD.mat",
        "8",
        lat=lat,
        lon=lon,
        z_smooth=z_smooth,
        interpolate=interpolate,
    )


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


def open_T_t():
    """
    Open the estimated arrival time offset data.

    Returns
    -------
    dict
        Dictionary containing the estimated arrival time offset data.
    """
    data_dir = get_data_dir()
    T_t_path = data_dir / "dataproducts" / "T_t_estimated.pkl"
    with T_t_path.open("rb") as f:
        return pickle.load(f)


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


def _matlab_datenum_to_datetime64(datenum: np.ndarray) -> np.ndarray:
    """Convert MATLAB datenum (days since year 0) to numpy datetime64[us]."""
    days = np.asarray(datenum, dtype="float64") - 719529.0  # offset to 1970-01-01
    return np.datetime64("1970-01-01") + (days * 86_400e6).astype("int64").astype(
        "timedelta64[us]"
    )


_FCTD_UNITS = {
    "depth": "m",
    "pressure": "dbar",
    "temperature": "deg C",
    "conductivity": "S/m",
    "salinity": "psu",
    "density": "kg/m^3",
    "longitude": "deg E",
    "latitude": "deg N",
}


def _fctd_struct_to_dataset(struct, depth: np.ndarray | None = None) -> xr.Dataset:
    """
    Convert one FCTDgrid/FCTDdown/FCTDup/tGrid mat_struct to an xr.Dataset.

    Parameters
    ----------
    depth : np.ndarray or None
        Depth coordinate to use. If None, taken from ``struct.depth``.
        ``tGrid.depth`` is sometimes a bogus all-NaN ``(depth, time)``
        array, so callers should pass the parent struct's 1D depth array
        when converting a ``tGrid``.
    """
    if depth is None:
        depth = np.asarray(struct.depth)
    time = _matlab_datenum_to_datetime64(struct.time)

    data_vars = {}
    for field in struct._fieldnames:
        if field in ("depth", "time", "tGrid"):
            continue
        arr = np.asarray(getattr(struct, field))
        da = xr.DataArray(arr, dims=("depth", "time"))
        if field in _FCTD_UNITS:
            da.attrs["units"] = _FCTD_UNITS[field]
        data_vars[field] = da

    return xr.Dataset(data_vars, coords={"depth": depth, "time": time})


def open_fctd_tracks(
    path: Path | None = None,
    pattern: str = "FCTDgrid_*.mat",
) -> dict[str, dict[str, xr.Dataset]]:
    """
    Open FCTD track files as xarray Datasets.

    Parameters
    ----------
    path : Path or None
        Root FCTD directory containing ``FCTD_2024/`` and
        ``FCTD_epsi_2023/``. If None, resolved from ``FCTD_DATA_DIR``.
    pattern : str
        Glob pattern applied to filenames within each subdirectory
        (default ``"FCTDgrid_*.mat"``, matching files with the expected
        ``FCTDgrid``/``FCTDdown``/``FCTDup`` struct layout). E.g.
        ``"FCTDgrid_d3*"`` to load dive 3 from both years. Note that
        ``FCTD_epsi_2023/`` also contains ``griddedProfiles_*.mat``
        files with an unrelated ``GRID`` struct layout, which are
        excluded by the default pattern.

    Returns
    -------
    dict of dict
        Keyed by filename stem (without extension, e.g.
        ``"FCTDgrid_d3_A2along"``). Each value is a dict with keys
        ``"FCTDgrid"``, ``"FCTDdown"``, ``"FCTDdown_tGrid"``,
        ``"FCTDup"``, ``"FCTDup_tGrid"``, each mapping to an
        ``xr.Dataset`` with dims ``(depth, time)``. Each dataset's
        ``attrs["source_file"]`` records the originating file path.
        Files cannot be combined into a single gridded Dataset because
        time lengths and variable sets differ across files/years.
    """
    if path is None:
        fctd_dir = os.environ.get("FCTD_DATA_DIR")
        if not fctd_dir:
            raise OSError("Set FCTD_DATA_DIR in your .env file.")
        path = Path(fctd_dir)

    fs = fsspec.filesystem("")
    fns = sorted(
        fs.glob(f"{path}/FCTD_2024/{pattern}")
        + fs.glob(f"{path}/FCTD_epsi_2023/{pattern}")
    )

    results = {}
    for fn in tqdm(fns):
        mat = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)
        entry = {}
        for key in ("FCTDgrid", "FCTDdown", "FCTDup"):
            struct = mat[key]
            ds = _fctd_struct_to_dataset(struct)
            ds.attrs["source_file"] = fn
            entry[key] = ds
            if hasattr(struct, "tGrid"):
                tds = _fctd_struct_to_dataset(
                    struct.tGrid, depth=np.asarray(struct.depth)
                )
                tds.attrs["source_file"] = fn
                entry[f"{key}_tGrid"] = tds
        results[Path(fn).stem] = entry

    return results


def open_fctd_archive():
    """
    open FCTD between SION and SIOE, flip range coordinate to reflect Source > Receiver

    Returns
    -------
    fctd : xarray.DataArray
        FCTD sound speed data with dimensions (range, depth) and coordinates in meters.
    """

    fn = str(get_data_dir() / "environment_data" / "FCTD_SoundSpeed_Colosi_v2.mat")

    fctd_mat = sio.loadmat(fn)

    fctd = xr.DataArray(
        np.flip(fctd_mat["cc"], axis=0),
        dims=["range", "depth"],
        coords={
            "range": fctd_mat["r"].flatten() * 1000,
            "depth": fctd_mat["z"].flatten(),
            #'latitude':('range', fctd_mat['lat'].flatten()),
            #'longitude':('range', fctd_mat['lon'].flatten())
        },
    ).sortby("range")

    # fctd_pilot = fctd.assign_coords({'range':fctd.range.values - (201222-153500)})
    fctd_pilot = fctd.assign_coords({"range": fctd.range.values})

    # Add 10000 meter
    fctd_pilot = xr.concat(
        (
            fctd_pilot,
            xr.ones_like(fctd_pilot[:, -2:-1]).assign_coords({"depth": [10000]}) * 1640,
        ),
        dim="depth",
    )

    # fctd_grid = fctd_pilot.interp({'range':np.arange(0,201222,500), 'depth':np.arange(0,6000)})
    fctd_grid = fctd_pilot.interp(
        {"range": np.arange(0, 250431, 500), "depth": np.arange(0, 6000)}
    )
    # fctd_grid = fctd_grid.interpolate_na('depth')
    fctd_grid = fctd_grid[:, 5:]

    bz, az = signal.butter(4, 1 / 100, fs=1)
    # br, ar = signal.butter(4, 1 / 10e3, fs=1 / 500)

    # fctd_smooth = xrs.filtfilt(xrs.filtfilt(fctd_grid, dim='range', b=br, a=ar), dim='depth', b=bz, a=az)
    fctd_smooth = xrs.filtfilt(fctd_grid, dim="depth", b=bz, a=az)

    return fctd_smooth
