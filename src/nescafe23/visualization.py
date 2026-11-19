"""Visualization utilities for NESCAFE23."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import cmps
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm

import nescafe23.data as data
from nescafe23.processing import (
    PEAK_BOXES_LOWER,
    PEAK_BOXES_UPPER,
    ULA_conventional_beamform,
)


def plot_marginal_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    indices: tuple[int, int],
    n_grid: int = 200,
    n_std: float = 3.5,
    cmap: str = "viridis",
    plot_type: str = "image",
    n_contours: int = 6,
    ax: plt.Axes = None,
    add_colorbar: bool = True,
) -> plt.Figure:
    """
    Extract the 2D marginal of a multivariate Gaussian and plot it.

    Parameters
    ----------
    mu : (n,) ndarray
        Mean vector of the full distribution.
    sigma : (n, n) ndarray
        Covariance matrix of the full distribution.
    indices : tuple of int
        (i, j) pair of dimension indices to marginalise onto.
    n_grid : int
        Number of grid points along each axis. Default 200.
    n_std : float
        How many standard deviations to show on each axis. Default 3.5.
    cmap : str
        Matplotlib colormap name. Default "viridis".
    plot_type : str
        ``"image"`` renders a heatmap with white contour overlays (default);
        ``"contour"`` renders unfilled contour lines at 1, 2, and 3 sigma.
    n_contours : int
        Number of contour levels in ``"image"`` mode. Default 6.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into; a new figure is created if None.
    add_colorbar : bool
        Whether to add a colorbar to the plot. Default True.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    i, j = indices

    mu_marg = mu[[i, j]]
    sigma_marg = sigma[np.ix_([i, j], [i, j])]

    std = np.sqrt(np.diag(sigma_marg))
    x_range = np.linspace(
        mu_marg[0] - n_std * std[0], mu_marg[0] + n_std * std[0], n_grid
    )
    y_range = np.linspace(
        mu_marg[1] - n_std * std[1], mu_marg[1] + n_std * std[1], n_grid
    )
    X, Y = np.meshgrid(x_range, y_range)

    pos = np.stack([X, Y], axis=-1)
    diff = pos - mu_marg

    sigma_inv = np.linalg.inv(sigma_marg)
    exponent = np.einsum("...i,ij,...j->...", diff, sigma_inv, diff)
    det = np.linalg.det(sigma_marg)
    Z = np.exp(-0.5 * exponent) / (2 * np.pi * np.sqrt(det))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    if plot_type == "image":
        img = ax.imshow(
            Z,
            extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
            origin="lower",
            cmap=cmap,
            aspect="auto",
        )
        ax.contour(
            X, Y, Z, levels=n_contours, colors="white", linewidths=0.6, alpha=0.5
        )
        fig.colorbar(img, ax=ax, label="p(x)")
    elif plot_type == "contour":
        sigma_levels = [1, 2, 3]
        levels = sorted(
            [np.exp(-0.5 * k**2) / (2 * np.pi * np.sqrt(det)) for k in sigma_levels]
        )
        ax.contour(X, Y, Z, levels=levels, cmap=cmap, linewidths=1.5)
    else:
        raise ValueError(f"plot_type must be 'image' or 'contour', got {plot_type!r}")

    ax.set_xlabel(f"$x_{{{i}}}$", fontsize=13)
    ax.set_ylabel(f"$x_{{{j}}}$", fontsize=13)
    ax.set_title(
        f"Marginal $p(x_{{{i}}}, x_{{{j}}})$\n"
        rf"$\mu = ({mu_marg[0]:.2f},\, {mu_marg[1]:.2f})$, "
        rf"$\sigma = ({std[0]:.2f},\, {std[1]:.2f})$",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def _figures_dir() -> Path:
    """Return the repo-root ``figures/`` directory, creating it if needed."""
    repo_root = Path(__file__).resolve().parents[2]
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _load_array_qc_data(show_colosi_picks: bool) -> dict:
    """Load all data needed to render array QC figures, once for reuse across frames."""
    return {
        "dss": data.open_receptions(),
        "peak_locations": data.open_peak_locations(),
        "T_t": data.open_T_t()["dT_t"],
        "colosi_peak_locations": (
            data.open_colosi_peak_locations() if show_colosi_picks else None
        ),
    }


def _render_array_qc_figure(
    Tx_idx: int,
    array: Literal["upper", "lower"],
    bundle: dict,
    *,
    time_windows: Sequence[tuple[float, float]] = ((100, 104), (104, 108), (108, 112)),
    thetas: np.ndarray = None,
    vmin: float = 110,
    figsize: tuple[float, float] = (13, 18),
) -> plt.Figure:
    """Render the 3-pane peak-picking QC figure for one transmission and array."""
    if thetas is None:
        thetas = np.linspace(-35, 35, 200)

    ds = (
        bundle["dss"][Tx_idx]
        .swap_dims({"hydrophone": "nominal_depth"})
        .rename({"nominal_depth": "depth"})
    )
    ds = ds[:32, :] if array == "upper" else ds[32:, :]

    t_lo = min(w[0] for w in time_windows) - 2
    t_hi = max(w[1] for w in time_windows) + 2
    ds = ds.sel({"time": slice(t_lo, t_hi)})
    bf = ULA_conventional_beamform(ds, thetas)

    peak_locations = bundle["peak_locations"]
    if array == "upper":
        peak_boxes = PEAK_BOXES_UPPER
        times_key, angles_key, snrs_key = "upper_times", "upper_angles", "upper_snrs"
    else:
        peak_boxes = PEAK_BOXES_LOWER
        times_key, angles_key, snrs_key = "times", "angles", "snrs"

    colosi_peak_locations = bundle["colosi_peak_locations"]
    T_t = bundle["T_t"]

    fig, axes = plt.subplots(len(time_windows), 1, figsize=figsize)
    axes = np.atleast_1d(axes)

    for pane, xlim in enumerate(time_windows):
        ax = axes[pane]
        (20 * np.log10(np.abs(bf))).plot(
            x="time", cmap=cmps.timefront(), rasterized=True, vmin=vmin, ax=ax
        )

        artists = []
        for peak, box in peak_boxes.items():
            if isinstance(box, tuple):
                t1, t2, th1, th2 = box
            elif isinstance(box, dict):
                t1, t2, th1, th2 = box[Tx_idx]
            else:
                continue

            rect = Rectangle(
                (t1, th1),
                t2 - t1,
                th2 - th1,
                linewidth=1,
                edgecolor="k",
                facecolor="none",
            )
            ax.add_patch(rect)

            ax.scatter(
                peak_locations[times_key][peak][Tx_idx],
                peak_locations[angles_key][peak][Tx_idx],
                lw=1,
                edgecolor="k",
                facecolors=(1, 1, 1, 0.3),
            )

            if array == "upper" and colosi_peak_locations is not None:
                colosi_tx = colosi_peak_locations[Tx_idx]
                ax.scatter(
                    colosi_tx["eig_time"] + T_t[Tx_idx],
                    colosi_tx["eig_ang"],
                    lw=1,
                    edgecolor="#228B22",
                    facecolors=(1, 1, 1, 0.3),
                )
                for i in range(len(colosi_tx["eig_id"][0])):
                    t_id = colosi_tx["eig_time"][0, i] + T_t[Tx_idx]
                    if xlim[0] <= t_id <= xlim[1]:
                        ax.text(
                            t_id,
                            colosi_tx["eig_ang"][0, i],
                            colosi_tx["eig_id"][0, i],
                            color="#228B22",
                        )

            box_properties = {"facecolor": "white", "edgecolor": "white", "alpha": 0.5}
            txt = ax.text(
                t1,
                th2,
                f"{peak}\n{peak_locations[snrs_key][peak][Tx_idx]:0.2f}",
                color="k",
                fontsize=8,
                va="bottom",
                bbox=box_properties,
            )
            artists.append((t1, t2, th1, th2, rect, txt))

        def update_visibility(ax=ax, artists=artists):
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            for t1, t2, th1, th2, rect, txt in artists:
                visible = t1 < x1 and t2 > x0 and th1 < y1 and th2 > y0
                rect.set_visible(visible)
                txt.set_visible(visible)
            fig.canvas.draw_idle()

        ax.callbacks.connect("xlim_changed", update_visibility)
        ax.callbacks.connect("ylim_changed", update_visibility)
        update_visibility()

        ax.set_title(f"Transmission {Tx_idx}")
        ax.set_xlim(xlim)

    fig.tight_layout()
    return fig


def plot_array_qc(
    array: Literal["upper", "lower"],
    tx_idx: int | Sequence[int] | None = None,
    *,
    mode: Literal["figure", "gif"] = "figure",
    show_colosi_picks: bool = True,
    fps: float = 2.0,
    out_dir: Path | str | None = None,
) -> Path | list[Path]:
    """
    Render peak-picking QC figures for the upper or lower hydrophone array.

    Reproduces the manual QC plots from ``notebooks/peak_picking_QC.ipynb``:
    a beamformed reception (3 stacked time-window panes) with peak bounding
    boxes, picked-peak scatter points, and SNR labels overlaid.

    Parameters
    ----------
    array : {"upper", "lower"}
        Which hydrophone array to plot.
    tx_idx : int, sequence of int, or None
        Transmission index/indices to render. None renders all available
        transmissions.
    mode : {"figure", "gif"}
        ``"figure"`` saves one PNG per transmission and returns its path (or
        a list of paths for multiple transmissions). ``"gif"`` renders every
        transmission in `tx_idx` as a frame of a single animated GIF (with a
        progress bar, since this is slow) and returns the GIF path.
    show_colosi_picks : bool
        Overlay Colosi's hand-picked peaks. Only applies to the upper array
        (ignored for "lower", since no such comparison data exists).
        Requires ``NESCAFE23_COLOSI_PICKS_DIR`` to be set.
    fps : float
        Frames per second for ``mode="gif"``.
    out_dir : Path, str, or None
        Output directory. Defaults to the repo-root ``figures/`` directory.

    Returns
    -------
    Path or list of Path
        Path(s) to the saved PNG(s) (``mode="figure"``) or the saved GIF
        (``mode="gif"``).
    """
    show_colosi_picks = show_colosi_picks and array == "upper"
    out_dir = Path(out_dir) if out_dir is not None else _figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = _load_array_qc_data(show_colosi_picks)

    scalar_input = isinstance(tx_idx, int)
    if tx_idx is None:
        tx_idxs = list(range(len(bundle["dss"])))
    elif scalar_input:
        tx_idxs = [tx_idx]
    else:
        tx_idxs = list(tx_idx)

    if mode == "figure":
        paths = []
        for idx in tx_idxs:
            fig = _render_array_qc_figure(idx, array, bundle)
            path = out_dir / f"{array}_array_qc_tx{idx:02d}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        return paths[0] if scalar_input else paths

    if mode == "gif":
        frames = []
        for idx in tqdm(tx_idxs, desc=f"Rendering {array} array QC GIF"):
            fig = _render_array_qc_figure(idx, array, bundle)
            fig.canvas.draw()
            frame = Image.frombuffer(
                "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
            ).convert("RGB")
            frames.append(frame)
            plt.close(fig)

        path = out_dir / f"{array}_array_qc.gif"
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 / fps,
            loop=0,
        )
        return path

    raise ValueError(f"mode must be 'figure' or 'gif', got {mode!r}")
