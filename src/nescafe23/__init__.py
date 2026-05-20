"""NESCAFE23 — shared utilities for the 2023 New England Seamounts Calibrated Acoustic Fluctuation Experiment."""

from nescafe23.data import get_data_dir, open_sioe_ctd, open_sion_ctd
from nescafe23.io import convert_to_xarray, open_peak_locations, open_receptions
from nescafe23.physics import aog_vsdg, compute_filtered_fctd_grid
from nescafe23.ray import n_botts_ray_id, parse_ray_id
from nescafe23.visualization import collect_upper_array_bboxs, plot_marginal_gaussian

__all__ = [
    "aog_vsdg",
    "collect_upper_array_bboxs",
    "compute_filtered_fctd_grid",
    "convert_to_xarray",
    "get_data_dir",
    "n_botts_ray_id",
    "open_peak_locations",
    "open_receptions",
    "open_sioe_ctd",
    "open_sion_ctd",
    "parse_ray_id",
    "plot_marginal_gaussian",
]
