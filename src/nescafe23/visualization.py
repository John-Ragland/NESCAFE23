"""Visualization utilities for NESCAFE23."""

import matplotlib.pyplot as plt
import numpy as np


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
