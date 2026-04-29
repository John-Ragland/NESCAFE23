"""Visualization utilities for NESCAFE23."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.widgets import Button, RectangleSelector


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


def collect_upper_array_bboxs(
    bf: xr.DataArray,
    *,
    vmin: float = 80,
    vmax: float | None = None,
    figsize: tuple[float, float] = (14, 6),
) -> dict[str, tuple[float, float, float, float]]:
    """
    Interactively collect bounding boxes for each ray in PEAK_BBOXS_UPPER.

    Displays the beamformed envelope (depth-averaged, dB scale) and steps
    through each key of PEAK_BBOXS_UPPER in order.  For each ray the view
    auto-zooms to the existing bbox ±30 % padding.  Draw a rectangle then
    click Accept to save it, or Skip to keep the existing PEAK_BBOXS_UPPER
    value unchanged.  Requires an interactive matplotlib backend
    (e.g. ``%matplotlib widget`` in Jupyter).

    Parameters
    ----------
    bf      : xr.DataArray
        Beamformed output with at least ``time`` and ``angle`` dimensions.
        If a ``depth`` dimension is present it is averaged out.
    vmin    : float
        Colour-scale minimum in dB. Default 80.
    vmax    : float or None
        Colour-scale maximum in dB. None → data maximum.
    figsize : tuple of float
        Figure size in inches. Default (14, 6).

    Returns
    -------
    dict[str, tuple[float, float, float, float]]
        ``{ray_id: (t1, t2, th1, th2)}`` for every key in
        :data:`nescafe23.peaks.PEAK_BBOXS_UPPER`.
    """
    from nescafe23.peaks import PEAK_BBOXS_UPPER

    if "depth" in bf.dims:
        bf = bf.mean("depth")

    bf_db: xr.DataArray = 20 * np.log10(np.abs(bf))
    _vmax = float(bf_db.max()) if vmax is None else vmax

    keys = list(PEAK_BBOXS_UPPER)
    n_keys = len(keys)

    state: dict = {
        "idx": 0,
        "rect": None,
        "coords": None,
        "results": {},
        "draw_mode": False,
    }

    # ── figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")

    ax_im = fig.add_axes([0.07, 0.18, 0.74, 0.74])
    ax_cbar = fig.add_axes([0.83, 0.18, 0.02, 0.74])
    ax_draw = fig.add_axes([0.07, 0.04, 0.18, 0.08])
    ax_accept = fig.add_axes([0.27, 0.04, 0.18, 0.08])
    ax_skip = fig.add_axes([0.47, 0.04, 0.18, 0.08])

    # ── plot full image ────────────────────────────────────────────────────
    bf_db.plot(
        ax=ax_im,
        x="time",
        vmin=vmin,
        vmax=_vmax,
        cmap="viridis",
        add_colorbar=True,
        cbar_ax=ax_cbar,
    )
    ax_im.set_facecolor("#0d1117")
    for spine in ax_im.spines.values():
        spine.set_edgecolor("#30363d")
    ax_im.tick_params(colors="#c9d1d9")
    ax_im.xaxis.label.set_color("#c9d1d9")
    ax_im.yaxis.label.set_color("#c9d1d9")
    ax_cbar.tick_params(colors="#c9d1d9")
    ax_cbar.yaxis.label.set_color("#c9d1d9")

    fig.text(
        0.07,
        0.135,
        "Use toolbar to zoom/pan  |  Draw Box to draw a selection  |  Accept saves & advances  |  Skip keeps existing value",
        color="#8b949e",
        fontsize=8,
        va="bottom",
    )

    # ── buttons ────────────────────────────────────────────────────────────
    btn_draw = Button(ax_draw, "Draw Box", color="#21262d", hovercolor="#30363d")
    btn_accept = Button(ax_accept, "Accept", color="#1f6feb", hovercolor="#388bfd")
    btn_skip = Button(ax_skip, "Skip", color="#21262d", hovercolor="#30363d")
    for btn in (btn_draw, btn_accept, btn_skip):
        btn.label.set_color("white")
        btn.label.set_fontsize(9)

    # ── rectangle selector (starts inactive) ──────────────────────────────
    def _on_select(eclick, erelease):
        if None in (eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata):
            return
        t0, t1 = sorted([eclick.xdata, erelease.xdata])
        th0, th1 = sorted([eclick.ydata, erelease.ydata])
        state["coords"] = (t0, t1, th0, th1)

        if state["rect"] is not None:
            state["rect"].remove()

        rect = mpatches.Rectangle(
            (t0, th0),
            t1 - t0,
            th1 - th0,
            linewidth=1.5,
            edgecolor="#f78166",
            facecolor="#f7816622",
            linestyle="--",
            zorder=5,
        )
        ax_im.add_patch(rect)
        state["rect"] = rect
        fig.canvas.draw()

    rs = RectangleSelector(
        ax_im,
        _on_select,
        useblit=False,
        button=[1],
        minspanx=0,
        minspany=0,
        spancoords="data",
        interactive=False,
    )
    rs.set_active(False)

    # ── draw-box toggle ────────────────────────────────────────────────────
    toolbar = fig.canvas.toolbar

    def _set_draw_mode(active: bool):
        state["draw_mode"] = active
        rs.set_active(active)
        if active:
            btn_draw.color = "#238636"
            btn_draw.hovercolor = "#2ea043"
            btn_draw.label.set_text("Drawing...")
        else:
            btn_draw.color = "#21262d"
            btn_draw.hovercolor = "#30363d"
            btn_draw.label.set_text("Draw Box")
        fig.canvas.draw()

    def _on_draw_toggle(_event):
        _set_draw_mode(not state.get("draw_mode", False))

    # ── zoom helper ────────────────────────────────────────────────────────
    def _zoom_to(ray_id: str):
        t1, t2, th1, th2 = PEAK_BBOXS_UPPER[ray_id]
        t_pad = max((t2 - t1) * 1.5, 0.05)
        th_pad = max((th2 - th1) * 1.5, 2.0)
        ax_im.set_xlim(t1 - t_pad, t2 + t_pad)
        ax_im.set_ylim(th1 - th_pad, th2 + th_pad)

    def _update_title():
        idx = state["idx"]
        key = keys[idx]
        ax_im.set_title(
            f"Ray  {key}   [{idx + 1} / {n_keys}]   — draw box, then Accept",
            color="#c9d1d9",
            pad=6,
            fontsize=11,
        )

    # ── advance to next ray ────────────────────────────────────────────────
    def _advance():
        _set_draw_mode(False)
        if state["rect"] is not None:
            state["rect"].remove()
            state["rect"] = None
        state["coords"] = None

        state["idx"] += 1
        if state["idx"] >= n_keys:
            plt.close(fig)
            return

        _zoom_to(keys[state["idx"]])
        _update_title()
        fig.canvas.draw()

    # ── button callbacks ───────────────────────────────────────────────────
    def _on_accept(_event):
        # deactivate RS immediately so it can't interfere with what follows
        _set_draw_mode(False)
        coords = state["coords"]
        key = keys[state["idx"]]

        if coords is None:
            ax_im.set_title(
                f"Ray  {key}  — draw a box first!",
                color="#f85149",
                pad=6,
                fontsize=11,
            )
            fig.canvas.draw()
            return

        try:
            state["results"][key] = coords
            perm = mpatches.Rectangle(
                (coords[0], coords[2]),
                coords[1] - coords[0],
                coords[3] - coords[2],
                linewidth=1.5,
                edgecolor="#3fb950",
                facecolor="#3fb95018",
                zorder=4,
            )
            ax_im.add_patch(perm)
            ax_im.text(
                (coords[0] + coords[1]) / 2,
                (coords[2] + coords[3]) / 2,
                key,
                ha="center",
                va="center",
                fontsize=7,
                color="#3fb950",
                fontweight="bold",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "#0d1117",
                    "ec": "#3fb950",
                    "alpha": 0.7,
                },
            )
            _advance()
        except Exception as exc:
            ax_im.set_title(f"⚠ {exc}", color="#f85149", pad=6, fontsize=9)
            fig.canvas.draw()

    def _on_skip(_event):
        try:
            key = keys[state["idx"]]
            state["results"][key] = PEAK_BBOXS_UPPER[key]
            _advance()
        except Exception as exc:
            ax_im.set_title(f"⚠ {exc}", color="#f85149", pad=6, fontsize=9)
            fig.canvas.draw()

    btn_draw.on_clicked(_on_draw_toggle)
    btn_accept.on_clicked(_on_accept)
    btn_skip.on_clicked(_on_skip)

    # ── initialise view ────────────────────────────────────────────────────
    # Draw once at full extent so the toolbar records it as the "home" view,
    # then zoom into the first ray.
    fig.canvas.draw()
    if toolbar is not None:
        toolbar.push_current()
    _zoom_to(keys[0])
    _update_title()

    plt.show(block=True)

    return state["results"]
