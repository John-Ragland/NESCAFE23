# NESCAFE 2023

Analysis and research codebase for the 2023 pilot New England Seamounts Calibrated Acoustic
Fluctuation Experiment (NESCAFE23).

In late April 2023, two acoustic transceivers (SION and SIOE) were moored about 150 km apart
across the Gulf Stream, near the New England Seamount chain, and exchanged broadband 250 Hz
transmissions. The 23 receptions were recorded on two vertical sub-arrays (64 hydrophones in
all). This repository is a [uv](https://docs.astral.sh/uv/) workspace with:

- **`nescafe23`** (`src/nescafe23/`): the shared base library. It reads the reception and
  environmental data, applies the tidal arrival-time correction, holds the hand-picked arrival
  peaks and their QC flags, and has the ray-ID, beamforming, sound-speed and plotting helpers
  every project uses. It also provides the `nescafe23` command-line tool.
- **Packages and projects** (`packages/`, `projects/`): each is its own git repository,
  included here as a git submodule, that depends on `nescafe23`.

## Released components

Submodules are released **incrementally**, usually when the paper they support is submitted.
Until then a submodule's repository stays private, and its folder in this workspace stays empty
in a public clone.

| Component | Path | Description | Status |
|---|---|---|---|
| `nescafe23` | `src/nescafe23` | Base library and CLI | Public |
| [`n23_fluctuations_ragland`](https://github.com/John-Ragland/n23_fluctuations_ragland) | `projects/n23_fluctuations_ragland` | Acoustic fluctuations: phase and log-amplitude spectra, coherence, internal-wave theory | Public |
| `acoustic_inversion` | `packages/acoustic_inversion` | Ray-based acoustic inversion | Not yet released |
| `nesma_param` | `packages/nesma_param` | Ocean-state parameterization of the NESMA region | Not yet released |
| `n23_inversion_ragland` | `projects/n23_inversion_ragland` | Source localization and ocean acoustic inversion | Not yet released |
| `n23_source_localization_hoekstra` | `projects/n23_source_localization_hoekstra` | Source localization from boundary-reflected arrivals | Not yet released |
| `n23_PE_boundary_decourcy` | `projects/n23_PE_boundary_decourcy` | Parabolic-equation modelling | Not yet released |
| `n23_backprop_colosi` | `projects/n23_backprop_colosi` | Acoustic backpropagation | Not yet released |

## Installation

Install [uv](https://docs.astral.sh/uv/) first. Then clone the workspace and check out the
public submodules you want:

```bash
git clone https://github.com/John-Ragland/NESCAFE23.git
cd NESCAFE23
git submodule update --init projects/n23_fluctuations_ragland
uv sync --all-packages --all-groups
```

**Don't clone with `--recurse-submodules`** and don't run a bare `git submodule update --init`.
Git would fail, or ask for GitHub credentials, on the submodules that are not released yet.
Their folders stay empty, which `uv` ignores. Because they are missing, `uv sync` updates
`uv.lock` to drop them. This is expected, and `uv sync --locked` will fail.

`uv sync --all-packages --all-groups` creates one virtual environment (`.venv` at the workspace
root). It installs `nescafe23` and every checked-out submodule in editable mode, plus the test,
dev and docs dependency groups. (A bare `uv sync` installs only `nescafe23` and removes the
other workspace members from the environment.) The workspace pins Python 3.13
(`.python-version`), and `uv` downloads it if needed. The dependencies that are not on PyPI
([pygenray](https://github.com/John-Ragland/pygenray) and
[cmps](https://github.com/John-Ragland/cmps)) are installed from GitHub automatically.

Check the installation with

```bash
uv run pytest tests                                    # nescafe23
uv run pytest projects/n23_fluctuations_ragland/tests  # each checked-out submodule
```

The tests don't need the experiment data. Each submodule's README describes how to run its own
analysis.

**Development.** Notebook outputs are stripped by `nbstripout` and ruff runs as a pre-commit
hook. To enable both, run at the workspace root:

```bash
uv run nbstripout --install
uv run pre-commit install
```

Each submodule is its own git repository, so commit changes to a submodule from within its
folder.

## Data

The experiment data are kept in one directory with a fixed layout. Its location is set by
`NESCAFE23_DATADIR` in a `.env` file at the workspace root (see [`.env.example`](.env.example),
which also lists the optional paths to external datasets that some readers use):

```bash
NESCAFE23_DATADIR="/path/to/data/directory/"
```

Running `uv run nescafe23` asks for the directory if it is not set and saves it to `.env`. The
same CLI computes the tidal arrival-time offsets and estimates the peak locations, which are
written to `dataproducts/`. The expected layout is:

```
NESCAFE23_DATADIR/
├── dmod/dm*.mat                                     # demodulated receptions, one per transmission
├── environment_data/FCTD_SoundSpeed_Colosi_v2.mat   # fast CTD sound speed
└── dataproducts/
    ├── T_t_estimated.pkl                            # tidal arrival-time offsets (from the CLI)
    └── peak_locations.pkl                           # estimated peak locations (from the CLI)
```

To access the experiment data, please contact John Ragland (john.ragland@whoi.edu).

## Citation

Each released project has its own paper and `CITATION.cff`; please cite the paper for the
analysis you use. The first paper from this workspace is

> Ragland, J., Colosi, J., Hoekstra, K., Dzieciuch, M., Alford, M., and Bellerjeau, C. (2026).
> Analysis of acoustic fluctuations for 150-km, low frequency transmissions across the Gulf
> Stream in the vicinity of the New England Seamount chain. Preprint, ESS Open Archive,
> [doi:10.22541/essoar.15006016/v1](https://doi.org/10.22541/essoar.15006016/v1). Submitted to
> the Journal of the Acoustical Society of America.

The citation for this repository is in [`CITATION.cff`](CITATION.cff), and GitHub's "Cite this
repository" button exports it as APA or BibTeX.

## License

BSD-3-Clause; see [`LICENSE`](LICENSE).
