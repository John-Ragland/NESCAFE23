"""nescafe23 initialization CLI."""

from pathlib import Path

import questionary


def _workspace_root() -> Path:
    return Path(__file__).parent.parent.parent


def main():
    """Set the NESCAFE23 data directory path in the workspace .env file."""
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
        print("Initialization cancelled.")
        return

    new_path = new_path.rstrip("/") + "/"

    with open(env_path, "w") as f:
        f.write(f'NESCAFE23_DATADIR="{new_path}"\n')

    print(f"\n✓ Set NESCAFE23_DATADIR to: {new_path}")
    print(f"✓ Saved to: {env_path}")
