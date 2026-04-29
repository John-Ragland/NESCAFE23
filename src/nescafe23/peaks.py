"""Peak location estimation for NESCAFE23 ray arrivals."""

import numpy as np
from tqdm import tqdm

from nescafe23.beamforming import ULA_conventional_beamform_nostack

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
