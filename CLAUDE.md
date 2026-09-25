# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NESCAFE23 is the analysis and research codebase for the 2023 pilot New England Seamounts Calibrated Acoustic
Fluctuation Experiment (NESCAFE23). It is a `uv` workspace with a base library plus six **git submodules**
(each its own GitHub repo, cloned under `packages/` and `projects/`):

- **`src/nescafe23/`** — the base shared library (data I/O, physics utilities, CLI, visualization). Every
  other package/project depends on this; it depends on nothing else in the workspace.
- **`packages/acoustic_inversion/`** (submodule) — ray-based underwater acoustic inversion math (Jacobians,
  MAP updates, forward models via `pygenray`). Depends on `nescafe23` and `nesma_param`.
- **`packages/nesma_param/`** (submodule) — parameterization of NESMA-region oceanography (Argo-based EOF
  models, JAX/Equinox generative ocean models). Depends on `nescafe23`.
- **`projects/n23_inversion_ragland/`** (submodule) — simultaneous source localization + ocean acoustic
  inversion. Depends on `nescafe23`, `acoustic_inversion`, `nesma_param`.
- **`projects/n23_fluctuations_ragland/`** (submodule) — acoustic fluctuation (GM internal-wave theory,
  phase/log-amplitude spectra) analysis. Depends on `nescafe23` only.
- **`projects/n23_source_localization_hoekstra/`** (submodule) — source localization via boundary-reflected
  arrivals. Depends on `nescafe23`, `acoustic_inversion`.
- **`projects/n23_PE_boundary_decourcy/`** (submodule) — parabolic-equation (PE) acoustic propagation model
  output compared against ray tracing and observed receptions. Depends on `nescafe23`. Early-stage/notebook-only;
  not previously listed here.

Each submodule has its own `CLAUDE.md` with details specific to it — read the relevant one before working
inside `packages/*` or `projects/*`. Because these are git submodules, edits inside them are commits in a
*separate* repository; `git status` at the workspace root will show the submodule as having a dirty/detached
working tree, which is expected.

## Dependency graph

```
nescafe23 (src/, base library)
├── nesma_param
│   └── acoustic_inversion
│       ├── n23_inversion_ragland
│       └── n23_source_localization_hoekstra
├── n23_fluctuations_ragland
└── n23_PE_boundary_decourcy
```

No circular dependencies exist. `nesma_param`'s `pyproject.toml` does not explicitly declare `nescafe23` as a
dependency even though it imports it (`import nescafe23 as n23`) — this works only because of workspace-wide
resolution; be aware of it if `nesma_param` is ever used outside this workspace.

## Commands

**Install everything (editable, workspace-wide):**
```bash
uv sync --all-packages --all-groups
```
A bare `uv sync` at the workspace root installs **only** the root `nescafe23` project and *uninstalls every
other workspace member* — after one, `import nesma_param` / `import acoustic_inversion` fail and most of the
test suite errors during collection. Always pass `--all-packages`; `--all-groups` additionally keeps the
`dev`/`test`/`docs` dependency groups (pytest, ipykernel, sphinx, ...) installed.

Each package's version is derived from git by `hatch-vcs` and written to a gitignored
`src/<pkg>/_version.py`, while the installed `dist-info` records the version from the last build. New commits
make the two disagree, which `tests/test_package.py::test_version` then reports as a failure. Fix it by
forcing a rebuild — note that `--reinstall-package` alone may be served from uv's cache and leave the stale
`dist-info` in place, so combine it with a full sync:
```bash
uv sync --all-packages --all-groups --reinstall-package <pkg>
```

**Run tests (workspace root or within a package/project):**
```bash
uv run pytest
# single test:
uv run pytest tests/test_foo.py::test_bar
```
Root `pyproject.toml` lists `testpaths` explicitly for `tests`, `packages/acoustic_inversion/tests`,
`packages/nesma_param/tests`, and all `projects/*/tests`, with `ignore::pytest.PytestConfigWarning` so a
partial checkout (submodules not initialized) doesn't hard-fail collection.

**Lint (within a package or project that has a noxfile):**
```bash
uv run noxfile.py -s lint     # ruff via prek (pre-commit) on all files
uv run noxfile.py -s pylint   # pylint
```
Most submodules exclude `notebooks/`, `dev/`, and `*.ipynb` from lint — the real research code often lives in
notebooks and is not style-checked.

**Build docs:**
```bash
uv run noxfile.py -s docs
```

**Run the CLI:**
```bash
uv run nescafe23
```
The CLI is interactive (uses `questionary`) and handles: setting `NESCAFE23_DATADIR`, computing arrival time
offsets, and estimating peak locations. Several downstream projects have their own CLI entry points too (e.g.
`nescafe23-processing` in `n23_fluctuations_ragland`, `n23-inversion` in `n23_inversion_ragland`) that wrap
their own processing pipelines with the same `questionary` pattern.

## Data directory

All base-library data access goes through `nescafe23.get_data_dir()`, which reads `NESCAFE23_DATADIR` from a
`.env` file at the workspace root (or the environment). Running `nescafe23` in the terminal will prompt to set
it if missing.

Expected data structure under `NESCAFE23_DATADIR`:
- `dmod/dm*.mat` — demodulated reception files (HDF5/MAT format)
- `environment_data/FCTD_SoundSpeed_Colosi_v2.mat` — fast CTD data
- `dataproducts/T_t_estimated.pkl` — tidal arrival time offsets (generated by CLI)
- `dataproducts/peak_locations.pkl` — estimated peak arrival locations (generated by CLI)

**Downstream projects define their own, separate `.env` variables** for their own derived-data directories —
e.g. `n23_inversion_ragland` uses `N23_INVERSION_DATADIR`, `n23_PE_boundary_decourcy` uses `PE_DATADIR`. Don't
assume `NESCAFE23_DATADIR` is the only data root in play; check each submodule's own CLAUDE.md/`.env.example`.
`nescafe23.data` also resolves several other env vars for external datasets with no CLI setup helper:
`GLIDER_DATA`, `ARGO_DATA_DIR`, `COPERNICUS_SATELLITE_ALTIMETRY_PATH`, `WOA2023_PATH`, `FCTD_DATA_DIR`,
`NESCAFE23_COLOSI_PICKS_DIR`, `NESCAFE23_BATHY_25M_PATH`.

**Bathymetry — two grids, opposite sign conventions.** `nescafe23.open_bathymetry_25m(lat_bounds,
lon_bounds)` / `sample_bathymetry_25m(lats, lons)` read the RR2409 25 m multibeam grid
(`NESCAFE23_BATHY_25M_PATH`, a MATLAB **v7.3** file so `h5py` not `scipy.io.loadmat`; datasets `lar`,
`lor`, `zr(lat, lon)` which h5py reports transposed), returning **positive-down depth in metres**. It
is 1.37 GB whole, so only the requested lat/lon window is ever read. `nesma_param.open_data.
open_bathymetry()` is a different dataset: GEBCO 2024 at ~450 m, variable `elevation`, positive
**up**, used for map figures only. `n23_inversion_ragland` traces the 25 m grid; the others still
build flat or ramped bottoms inline.  Provenance per the PI (2026-09-22): the 25 m
grid comes from **GMRT**; a defensible vertical uncertainty for it has not been established, and
the inversion is sensitive to the bottom datum at the sub-metre level (1 m of rigid offset = 4.0 ms
rms in `n23_inversion_ragland`'s differenced travel times). See that project's `CLAUDE.md` for why
its runs smooth the profile at 100 km and why `receiver_range` legitimately exceeds the
anchor-to-anchor geodesic.

## Architecture

### Data flow
Raw `.mat` files → `convert_to_xarray()` → xarray Datasets with `hydrophone`, `time` dims and `dmod`
(complex128) variable → tidal-shift-corrected receptions via `open_receptions()` (a **list** of `xr.DataArray`,
one per transmission — not a single combined Dataset) → peak location estimation (`open_peak_locations`,
requires the CLI's "estimate peak locations" step to have run first) → ray-based Jacobians/forward models
(`acoustic_inversion`) using ocean-state parameterizations (`nesma_param`) → MAP inversion (per-project).

### Shared conventions (used across nearly every submodule)
- **Ray IDs — two namespaces, not interchangeable.**
  - *NESCAFE23 ids*: `"+7"`, `"-6b"`, `"-5u"` — leading sign = ray branch direction, number = **arrival
    index** (which expected arrival this is, in the fixed experiment geometry), optional `b`/`u` suffix =
    lower/upper receiver array. Parsed via `nescafe23.parse_ray_id`; bounce counts are derived *from the
    label* by `n_botts_ray_id`. `acoustic_inversion._disambiguate_arrivals_n23` hard-codes NESCAFE23
    disambiguation rules around these — explicitly experiment-specific, not general-purpose.
  - *Generic ids*: `"+6_b3s2"` — the ray's **actually-traced signature**: signed turning-point count plus
    real bottom/surface bounce counts. Built/parsed by `acoustic_inversion.build_ray_id` /
    `parse_generic_ray_id`, resolved by `disambiguate_arrivals_generic`, and meaningful in any
    environment. This is the namespace the synthetic/OSSE path speaks in.
  - `acoustic_inversion.is_generic_ray_id` is the predicate that tells them apart, and code that must
    branch on the namespace (e.g. where a ray's bounce count comes from) uses it. Do not feed a
    NESCAFE23 id to the generic parser or vice versa: `parse_generic_ray_id("-5b")` raises, and
    `parse_ray_id("+6_b3s2")` raises a bare `ValueError` from an `int()` conversion.
  - **A bridge between the two now exists**: `nescafe23.processing.PEAK_LOWER_RAY_ID_LABELS` maps
    each hand-picked NESCAFE23 arrival to its generic id, read via
    `nescafe23.ray_id_labels(array)` / `generic_ray_id(ray_id, array)`. It is purely additive —
    `peak_locations.pkl` and both QC tables keep their NESCAFE23 keys — and it is what lets
    `n23_inversion_ragland.observations` invert the real data in the generic namespace, using
    `disambiguate_arrivals_generic` instead of the experiment-specific
    `_disambiguate_arrivals_n23` rules. The values were derived by tracing, not from the labels:
    a NESCAFE23 name carries no surface-reflection count at all. The upper array is deliberately
    still unlabelled. See that project's CLAUDE.md for what the relabelling does and does not change.
  - `acoustic_inversion.disambiguate_arrivals_n23` is a public alias for the private
    NESCAFE23 disambiguator, added so the labelling above can resolve an id the old way and read
    the chosen eigenray's real signature back out.
- **Hydrophone array**: 64 hydrophones total, indices 0–31 = upper array, 32–63 = lower array.
- **23 transmissions** total, indexed 0–22, each with its own `dm*.mat` file.
- **Peak QC flags** (`PEAK_LOWER_QC`/`PEAK_UPPER_QC` in `nescafe23.processing`): integer codes 1–5 per ray id
  per transmission; downstream code conventionally treats flags `(1, 5)` as "good" and drops everything else.
  Note `PEAK_UPPER_QC` has two **empty** rows (`-6b`, `-15b`), so indexing one by transmission raises —
  treat a short row as "unreviewed, hence unusable" rather than assuming 23 entries.
- **Sound speed**: computed via the Del Grosso equation, `nescafe23.processing.aog_vsdg` — note `nesma_param`
  has its own duplicate `tools.aog_vsdg`, but `nesma_param.gen_models` actually calls the `nescafe23` version
  (`n23.aog_vsdg`), so the local duplicate is effectively dead/inconsistent.
- **Ray tracing**: all ray work goes through `pygenray` (installed from GitHub, not PyPI) —
  `pr.OceanEnvironment2D`, `pr.shoot_rays`, `pr.find_eigenrays`.
- **The bottom.** `pr.OceanEnvironment2D`'s `bathymetry` is a 1-D `xr.DataArray` over a `range` dim,
  positive-down metres, on its own range grid (it need not match the sound speed's). Three traps:
  mutating `.bathymetry` after construction updates neither `bathymetry_fe` nor
  `bottom_angle`/`bottom_angle_interp`, so **always rebuild the environment**; the bottom *depth*
  lookup silently clamps past the end of the profile while the bottom *slope* interpolation
  (`interp1d(kind="cubic")`, needing ≥4 points) **raises**, so an over-range trace lies or dies
  depending on ray luck; and a single NaN depth is traced without complaint, because
  `abs(nan) > 90` never fires the backward-ray termination.

### Known cross-cutting issues (verified against current code, not just docs)
- `src/nescafe23/visualization.py:collect_upper_array_bboxs` imports from a `nescafe23.peaks` module that no
  longer exists (leftover from a module refactor) — calling it raises `ImportError`.
- `packages/nesma_param/CLAUDE.md` (prior to this rewrite) described a stale early-scaffold state of the
  package; it undercounted the modules and referenced a dependency (`pygem`) no longer in use.
- `packages/acoustic_inversion/src/acoustic_inversion/acoustic_inversion.py:ray_forward2` is an incomplete
  stub (TODO, no return) — do not treat it as working.

### Key dependencies
- **`pygenray`** — ray tracing (installed from GitHub, not PyPI); used in `acoustic_inversion` and directly in
  several projects (`n23_fluctuations_ragland`, `n23_PE_boundary_decourcy`, `n23_source_localization_hoekstra`)
  for `pr.shoot_rays`, `pr.find_eigs`/`find_eigenrays`, `pr.OceanEnvironment2D`
- **`xrsignal`** — xarray-native signal processing (Butterworth filter application via `xrs.filtfilt`)
- **`jax` / `equinox`** — used in `nesma_param` for differentiable ocean models, and by `acoustic_inversion`
  for JAX-autodiff ocean-model Jacobian columns
- **`gsw`** — TEOS-10 seawater thermodynamics
- **`cmps`** — plotting/colormap helper package, installed from GitHub, used in several projects' panel apps
  and figure notebooks

### Typing and linting
- All packages use `src/` layout with `py.typed` markers (fully typed)
- mypy strict mode on `src/` code; relaxed for tests
- ruff with an extensive rule set (see any `pyproject.toml`); `T20` (print) ignored in tests/noxfile
- `PLR09` (too many args/branches) and `PLR2004` (magic values) are globally ignored
