"""Data path utilities for NESCAFE23."""

import os
from pathlib import Path

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
