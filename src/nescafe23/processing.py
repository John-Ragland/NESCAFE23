import numpy as np
import scipy
import xarray as xr
from tqdm import tqdm

# Angle / time bounding boxes for each arrival (chosen by hand).
# Each entry is either:
#   - a tuple (t1, t2, th1, th2) — same bounds for all transmissions
#   - a dict {Tx_num: (t1, t2, th1, th2)} — per-transmission bounds
PEAK_BOXES_LOWER = {
    "+7": (101, 101.1, -13, -4),
    "+6": {
        0: (100.855, 100.885, 7.5, 10),
        1: (100.85, 100.885, 8, 10),
        2: (100.85, 100.885, 8, 11),
        3: (100.84, 100.87, 8, 10.5),
        4: (100.84, 100.885, 8.5, 11),
        5: (100.85, 100.88, 7.5, 11),
        6: (100.84, 100.885, 6, 10),
        7: (100.82, 100.88, 6, 9.5),
        8: (100.82, 100.878, 6, 10),
        9: (100.825, 100.88, 5.5, 9),
        10: (100.82, 100.88, 4, 9),
        11: (100.83, 100.885, 4.5, 8),
        12: (100.83, 100.88, 5.5, 8.5),
        13: (100.83, 100.87, 6.5, 9),
        14: (100.82, 100.88, 8, 10.5),
        15: (100.83, 100.875, 7, 10.5),
        16: (100.83, 100.88, 8.5, 11),
        17: (100.83, 100.89, 8, 10.5),
        18: (100.84, 100.88, 8, 10.5),
        19: (100.83, 100.89, 8, 11),
        20: (100.84, 100.89, 8, 10.5),
        21: (100.84, 100.89, 6, 10.5),
        22: (100.84, 100.87, 6, 9),
    },
    "-7": {
        0: (100.89, 100.93, 5.5, 8.5),
        1: (100.89, 100.93, 6.5, 8.5),
        2: (100.89, 100.93, 7, 10),
        3: (100.88, 100.91, 7, 9.5),
        4: (100.89, 100.93, 6.5, 10),
        5: (100.89, 100.93, 5.5, 10),
        6: (100.89, 100.94, 6.5, 9),
        7: (100.88, 100.92, 5, 8.5),
        8: (100.89, 100.94, 4.5, 9),
        9: (100.89, 100.925, 4, 7),
        10: (100.88, 100.94, 2, 8),
        11: (100.89, 100.94, 3, 7),
        12: (100.88, 100.93, 4, 7),
        13: (100.89, 100.93, 4, 7),
        14: (100.86, 100.93, 6, 8),
        15: (100.88, 100.92, 7, 9.5),
        16: (100.89, 100.94, 7, 9.5),
        17: (100.88, 100.95, 6, 9),
        18: (100.9, 100.94, 6.5, 9),
        19: (100.9, 100.94, 5, 8.5),
        20: (100.89, 100.94, 6, 8),
        21: (100.89, 100.94, 5, 10),
        22: (100.87, 100.91, 4.5, 9),
    },
    "-6": (100.65, 100.8, -16, -8),
    "-5b": (100.4, 100.6, 7, 18),
    "-6b": (100.8, 100.975, -22, -14),
    "+6b": (101.2, 101.4, 12, 20),
    "+7b": (101.6, 101.75, -23, -13),
    "-7b": (101.95, 102.1, 13, 21),
    "-8b": (102.35, 102.55, -22, -15),
    "+8b": (102.75, 103, 14, 22),
    "+9b": (103.25, 103.4, -26, -16),
    "-9b": (103.70, 103.85, 14, 26),
    "-10b": (104.2, 104.4, -27, -15),
    "+10b": (104.7, 104.9, 16, 26),
    "+11b": (105.2, 105.4, -27, -17),
    "-11b": (105.8, 106, 18, 27),
    "-12b": (106.4, 106.6, -28, -20),
    "+12b": (106.9, 107.2, 20, 28),
    "+13b": (107.6, 107.8, -30, -21),
    "-13b": (108.3, 108.4, 22, 30),
    "-14b": (108.9, 109.1, -33, -24),
    "+14b": (109.5, 109.85, 22, 30),
    "+15b": (110.3, 110.5, -34, -26),
    "-15b": (110, 111.4, 20, 30),
}

PEAK_BOXES_UPPER = {
    "+6b": (101.3, 101.7, 12, 19),
    "+7b": (101.25, 101.75, -20, -12),
    "-7b": (102, 102.1, 13, 20),
    "-8b": (102, 102.5, -22, -12),
    "+8b": (102.8, 103.4, 13, 22),
    "+9b": (102.75, 103.5, -22, -12),
    "-9b": (103.8, 104.3, 16, 23),
    "-10b": (103.75, 104.5, -24, -14),
    "+10b": (104.8, 105.4, 18, 22),
    "+11b": (104.75, 105.5, -26, -15),
    "-11b": (106, 106.5, 18, 26),
    "-12b": (106, 106.25, -28, -18),
    "+12b": (107, 107.8, 20, 26),
    "+13b": (107, 108, -28, -20),
    "-13b": (108.25, 109.5, 22, 30),
    "-14b": (108.5, 109.5, -28, -22),
    "+14b": (109.5, 110.5, 22, 32),
    "+15b": (109.75, 110.5, -32, -22),
}


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


def ULA_conventional_beamform(da, thetas, c0=1500):
    """

    Parameters
    ----------
    da : xr.DataArray
        signal to beamform. should have dimensions ['depth', 'time']
    thetas : np.array
        array of angles (in degrees) to beamform
    """

    zs = da.depth.values
    len(zs)
    z0 = np.mean(zs)

    # FFT stuff
    dt = float(da.time[1] - da.time[0])
    nt = len(da.time)
    2 * np.pi / (nt * dt)

    # Nyquist
    2 * np.pi / (2 * dt)

    # create frequency vector
    f = scipy.fft.fftfreq(nt, dt)
    ws = 2 * np.pi * f

    w = 2 * np.pi * 250

    delts = -(zs - z0)[:, None] * np.sin(np.radians(thetas[None, :])) / (c0)

    da = da.transpose("depth", "time")
    dmodf = scipy.fft.fft(da.values, axis=1)
    dmodf_ps = dmodf[:, :, None] * np.exp(
        1j * (ws[None, :, None] + w) * delts[:, None, :]
    )

    dmod_ps_t = scipy.fft.ifft(dmodf_ps, axis=1)
    dmod_bf_np = np.sum(dmod_ps_t, axis=0)
    dmod_bf = xr.DataArray(
        dmod_bf_np,
        dims=["time", "angle"],
        coords=({"time": da.time.values, "angle": thetas}),
    )

    # flip angles so that positive is down (to match ray code)
    # dmod_bf = dmod_bf.assign_coords({'angle':np.flip(dmod_bf.angle.values)}).sortby('angle')

    return dmod_bf


def get_peak_location(dss, Tx_idxs=None):
    """
    Estimate peak arrival locations using hand-derived bounding boxes.

    For each transmission, beamforms the lower receiver array and finds the
    time and angle of maximum intensity within each ray's bounding box
    (defined in :data:`PEAK_BOXES_LOWER`).

    Parameters
    ----------
    dss : list of xarray.DataArray
        List of receptions, one per transmission. Created with
        :func:`nescafe23.open_receptions`.
    Tx_idxs : array-like
        Transmission indices to process. Default is all 23.

    Returns
    -------
    arrival_times : dict
        Dict mapping ray_id → array of arrival times (shape: 23).
    arrival_angles : dict
        Dict mapping ray_id → array of arrival angles (shape: 23).
    arrival_snr : dict
        Dict mapping ray_id → array of SNR values (shape: 23).
    """
    if Tx_idxs is None:
        Tx_idxs = np.arange(23)

    arrival_times = {}
    arrival_angles = {}
    arrival_snr = {}

    arrival_times_upper = {}
    arrival_angles_upper = {}
    arrival_snr_upper = {}

    for ray_id in PEAK_BOXES_LOWER:
        arrival_times[ray_id] = np.ones(23) * np.nan
        arrival_angles[ray_id] = np.ones(23) * np.nan
        arrival_snr[ray_id] = np.ones(23) * np.nan

    for ray_id in PEAK_BOXES_UPPER:
        arrival_times_upper[ray_id] = np.ones(23) * np.nan
        arrival_angles_upper[ray_id] = np.ones(23) * np.nan
        arrival_snr_upper[ray_id] = np.ones(23) * np.nan

    for Tx_num in tqdm(Tx_idxs, desc="Tx num"):
        thetas = np.linspace(-35, 35, 200)
        lower_Rx = (
            dss[Tx_num]
            .swap_dims({"hydrophone": "nominal_depth"})
            .rename({"nominal_depth": "depth"})[32:]
        )

        upper_Rx = (
            dss[Tx_num]
            .swap_dims({"hydrophone": "nominal_depth"})
            .rename({"nominal_depth": "depth"})[:32]
        )

        bf_unstacked, _ = ULA_conventional_beamform_nostack(lower_Rx, thetas)
        bf = bf_unstacked.sum("depth")

        bf_unstacked_upper, _ = ULA_conventional_beamform_nostack(upper_Rx, thetas)
        bf_upper = bf_unstacked_upper.sum("depth")

        bf_noise = 20 * np.log10(np.abs(bf.sel({"time": slice(90, 95)}).std("time")))
        bf_noise_upper = 20 * np.log10(
            np.abs(bf_upper.sel({"time": slice(90, 95)}).std("time"))
        )

        for ray_id in PEAK_BOXES_LOWER:
            if isinstance(PEAK_BOXES_LOWER[ray_id], tuple):
                t1, t2, th1, th2 = PEAK_BOXES_LOWER[ray_id]
            elif isinstance(PEAK_BOXES_LOWER[ray_id], dict):
                t1, t2, th1, th2 = PEAK_BOXES_LOWER[ray_id][Tx_num]
            else:
                raise ValueError("PEAK_BOXES_LOWER entry must be tuple or dict")

            env_slice = 20 * np.log10(
                np.abs(bf.sel({"time": slice(t1, t2), "angle": slice(th1, th2)}))
            )
            flat_idx = np.argmax(env_slice.data)
            tmax_idx, thmax_idx = np.unravel_index(flat_idx, env_slice.shape)
            tmax = env_slice.time[tmax_idx].values
            thmax = env_slice.angle[thmax_idx].values

            arrival_times[ray_id][Tx_num] = tmax
            arrival_angles[ray_id][Tx_num] = thmax
            arrival_snr[ray_id][Tx_num] = env_slice.max() - bf_noise[thmax_idx]

        for ray_id in PEAK_BOXES_UPPER:
            t1, t2, th1, th2 = PEAK_BOXES_UPPER[ray_id]

            env_slice = 20 * np.log10(
                np.abs(bf_upper.sel({"time": slice(t1, t2), "angle": slice(th1, th2)}))
            )
            flat_idx = np.argmax(env_slice.data)
            tmax_idx, thmax_idx = np.unravel_index(flat_idx, env_slice.shape)
            tmax = env_slice.time[tmax_idx].values
            thmax = env_slice.angle[thmax_idx].values

            arrival_times_upper[ray_id][Tx_num] = tmax
            arrival_angles_upper[ray_id][Tx_num] = thmax
            arrival_snr_upper[ray_id][Tx_num] = (
                env_slice.max() - bf_noise_upper[thmax_idx]
            )

    return (
        arrival_times,
        arrival_angles,
        arrival_snr,
        arrival_times_upper,
        arrival_angles_upper,
        arrival_snr_upper,
    )


def parse_ray_id(ray_id: str) -> tuple:
    """
    Convert ray_id into pygenray compatible format.

    Parameters
    ----------
    ray_id : str
        Ray identifier string (e.g., '-7', '+10b').

    Returns
    -------
    ray_id_pr : str
        Converted ray identifier in pygenray format (e.g., '-7.0', '+10.0b').
    (sign, number, suffix) : tuple
        Parsed components of the ray identifier.
    """

    # remove upper array suffix if present
    if ray_id.endswith("u"):
        ray_id = ray_id[:-1]

    sign = ray_id[0]  # '+' or '-'
    if ray_id.endswith("b"):
        number_str = ray_id[1:-1]  # digits between sign and 'b'
        suffix = "b"
    else:
        number_str = ray_id[1:]  # digits after sign
        suffix = ""

    number = int(number_str)

    ray_id_pr = f"-{number}.0{suffix}" if sign == "-" else f"{number}.0{suffix}"

    return ray_id_pr, (sign, number, suffix)


def n_botts_ray_id(ray_ids: list) -> dict:
    """
    Compute the number of bottom bounces for each ray ID.

    Parameters
    ----------
    ray_ids : list of str
        Ray identifier strings (e.g., ['-7', '+10b']).

    Returns
    -------
    n_botts : dict
        Dictionary with keys of ray_ids and values are the number of bottom
        bounces for each ray_id.
    """
    n_botts = []
    for ray_id in ray_ids:
        _, (ray_id_sign, ray_id_num, _) = parse_ray_id(ray_id)

        if ray_id_num == 0:
            n_botts.append(0)
        elif ray_id_num % 2 == 0:
            n_botts.append(ray_id_num / 2)
        else:
            if ray_id_sign == "+":
                n_botts.append((ray_id_num - 1) / 2)
            else:
                n_botts.append((ray_id_num + 1) / 2)

    return dict(zip(ray_ids, n_botts, strict=False))


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
