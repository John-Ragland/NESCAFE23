"""nescafe23 unified CLI."""

import pickle
from functools import wraps
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import questionary
from scipy import signal

from nescafe23.data import get_data_dir
from nescafe23.io import convert_to_xarray


def check_output_exists(*output_paths):
    """
    Decorator to check if output file(s) exist and handle overwrite logic.

    Parameters
    ----------
    *output_paths : str or callable
        One or more paths to output files to check, or callables that return paths.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, force_overwrite=False, **kwargs):
            filepaths = []
            for output_path in output_paths:
                if callable(output_path):
                    filepath = output_path(*args, **kwargs)
                else:
                    filepath = output_path
                filepaths.append(Path(filepath))

            existing_files = [fp for fp in filepaths if fp.exists()]

            if not force_overwrite and existing_files:
                files_list = "\n".join(str(fp) for fp in existing_files)
                prompt_message = (
                    f"Output file(s) already exist:\n{files_list}\n"
                    "Do you want to overwrite them?"
                )
                should_proceed = questionary.confirm(
                    prompt_message, default=False
                ).ask()

                if not should_proceed:
                    print(f"Skipping {func.__name__}. Output already exists.")
                    return None

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _workspace_root() -> Path:
    return Path(__file__).parent.parent.parent


def _ensure_data_dir():
    """Prompt for NESCAFE23_DATADIR if not already set in the .env file."""
    from dotenv import find_dotenv, load_dotenv

    env_path = _workspace_root() / ".env"

    existing_path = None
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("NESCAFE23_DATADIR"):
                    existing_path = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if existing_path:
        return

    print("\nNESCAFE23_DATADIR is not set.")
    _set_data_dir(env_path)
    # Reload so get_data_dir() sees the newly written value in this process.
    load_dotenv(find_dotenv(), override=True)


def _set_data_dir(env_path=None):
    """Interactively set the NESCAFE23 data directory."""
    if env_path is None:
        env_path = _workspace_root() / ".env"

    existing_path = None
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("NESCAFE23_DATADIR"):
                    existing_path = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if existing_path:
        print(f"\nCurrent NESCAFE23 data directory: {existing_path}")
        change = questionary.confirm(
            "Do you want to change this path?", default=False
        ).ask()
        if not change:
            print("Keeping existing path.")
            return

    new_path = questionary.path(
        "Enter the absolute path to the NESCAFE23 data directory:",
        only_directories=True,
    ).ask()

    if new_path is None:
        print("Cancelled.")
        return

    new_path = new_path.rstrip("/") + "/"

    with open(env_path, "w") as f:
        f.write(f'NESCAFE23_DATADIR="{new_path}"\n')

    print(f"\n✓ Set NESCAFE23_DATADIR to: {new_path}")
    print(f"✓ Saved to: {env_path}")


@check_output_exists(lambda: get_data_dir() / "dataproducts" / "T_t_estimated.pkl")
def compute_arrival_time_offset():
    """Compute arrival time offset between transmissions using cross-correlations."""
    print("Estimating arrival time offset using cross-correlations...")

    data_dir = get_data_dir()
    T_t_path = data_dir / "dataproducts" / "T_t_estimated.pkl"

    fs = fsspec.filesystem("")
    fns = fs.glob(str(data_dir / "dmod" / "dm*.mat"))

    dss = []
    for fn in fns:
        dss.append(convert_to_xarray(fn))

    depth_grid_up = np.linspace(125, 540, 50)
    depth_grid_down = np.linspace(885, 1315, 50)

    dss_interp_up = []
    dss_interp_down = []

    for ds in dss:
        dss_interp_up.append(
            ds["dmod"][:32]
            .swap_dims({"hydrophone": "actual_depth"})
            .drop_vars(["nominal_depth"])
            .interp(
                {"actual_depth": depth_grid_up},
                kwargs={"bounds_error": False, "fill_value": 0},
            )
            .sel({"time": slice(100, 115)})
        )
        dss_interp_down.append(
            ds["dmod"][32:]
            .swap_dims({"hydrophone": "actual_depth"})
            .drop_vars(["nominal_depth"])
            .interp(
                {"actual_depth": depth_grid_down},
                kwargs={"bounds_error": False, "fill_value": 0},
            )
            .sel({"time": slice(100, 115)})
        )

    Ts = float(dss_interp_up[0].time.diff("time")[0])
    Rxx_taus = np.arange(-15 / 2, 15 / 2, Ts)[1:]

    Rxxs = []
    for k in range(23):
        interp1 = dss_interp_up[0].values - dss_interp_up[0].mean().values
        interp2 = dss_interp_up[k].values - dss_interp_up[k].mean().values

        interp1[np.isnan(interp1)] = 0
        interp2[np.isnan(interp2)] = 0

        Rxx = signal.fftconvolve(
            np.abs(interp1), np.flip(np.abs(interp2), axis=1), mode="same", axes=1
        )
        Rxxs.append(Rxx)

    dT_t_upper = []
    for k, Rxx in enumerate(Rxxs):
        Rxx_sum = np.sum(Rxx, axis=0)
        idx = np.argmax(Rxx_sum)
        dT_t_upper.append(Rxx_taus[idx])
    dT_t_upper = np.array(dT_t_upper)

    Rxxs = []
    for k in range(23):
        interp1 = dss_interp_down[0].values - dss_interp_down[0].mean().values
        interp2 = dss_interp_down[k].values - dss_interp_down[k].mean().values

        interp1[np.isnan(interp1)] = 0
        interp2[np.isnan(interp2)] = 0

        Rxx = signal.fftconvolve(
            np.abs(interp1), np.flip(np.abs(interp2), axis=1), mode="same", axes=1
        )
        Rxxs.append(Rxx)

    dT_t_lower = []
    for Rxx in Rxxs:
        Rxx_sum = np.sum(Rxx, axis=0)
        idx = np.argmax(Rxx_sum)
        dT_t_lower.append(Rxx_taus[idx])
    dT_t_lower = np.array(dT_t_lower)

    dT_avg = (dT_t_upper + dT_t_lower) / 2

    plt.rcParams.update({"font.size": 8})
    fig = plt.figure(figsize=(3.37, 2))
    plt.plot(np.arange(1, 24), dT_t_upper, label="upper array", c="C0", lw=1)
    plt.plot(np.arange(1, 24), dT_t_lower, label="lower array", c="C1", lw=1)
    plt.plot(np.arange(1, 24), dT_avg, c="k", lw=1, label="average")
    plt.ylabel("$\\Delta \\tau$ [s]")
    plt.legend()
    plt.xlabel("transmission number")
    plt.tight_layout()

    figures_dir = data_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "common_arrival_time_shift_estimation.pdf", bbox_inches="tight")

    with open(T_t_path, "wb") as f:
        pickle.dump(
            {
                "dT_t": dT_avg,
                "notes": "tidal arrival time offset estimated with cross-correlations in time of the reception envelope",
            },
            f,
        )

    print(f"✓ Saved to {T_t_path}")


@check_output_exists(lambda: get_data_dir() / "dataproducts" / "peak_locations.pkl")
def estimate_peak_locations():
    """
    Estimate peak arrival locations in time and angle for each transmission.

    Using the location of maximum intensity within hand-chosen angle/time
    bounding boxes, finds the arrival time and angle for each ray and
    transmission.

    Requires ``T_t_estimated.pkl`` — run 'compute arrival time offset' first.
    """
    from nescafe23.io import open_receptions
    from nescafe23.peaks import get_peak_location

    data_dir = get_data_dir()

    if not (data_dir / "dataproducts" / "T_t_estimated.pkl").exists():
        raise FileNotFoundError(
            "T_t_estimated.pkl not found. Run 'nescafe23' and select "
            "'compute arrival time offset' first."
        )

    dss = open_receptions()
    times, angles, snrs = get_peak_location(dss)
    peak_locations = {"times": times, "angles": angles, "snrs": snrs}

    out_path = data_dir / "dataproducts" / "peak_locations.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(peak_locations, f)

    print(f"✓ Saved to {out_path}")


def main():
    """Unified NESCAFE23 CLI."""
    _ensure_data_dir()

    try:
        get_data_dir()
    except OSError:
        print("\nNESCAFE23_DATADIR is still not set. Exiting.")
        return

    choice = questionary.select(
        "What would you like to do?",
        choices=[
            "compute arrival time offset",
            "estimate peak locations",
            "set data directory",
            "exit",
        ],
    ).ask()

    if choice == "compute arrival time offset":
        compute_arrival_time_offset()
    elif choice == "estimate peak locations":
        estimate_peak_locations()
    elif choice == "set data directory":
        _set_data_dir()
    elif choice == "exit":
        print("Goodbye!")
