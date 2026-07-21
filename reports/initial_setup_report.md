# Initial Setup Report

Date/time in UTC: 2026-07-21T02:40:27Z

## Repository state before setup

- Path: `/Users/daniel/Documents/tau-sc-symbolic-regression`
- Initial classification: empty repository directory, no Git repository before setup.
- Existing files before scaffold: none detected.
- `python`: command not found.
- `python3`: Python 3.14.6.
- `pytest`, `ruff`, `uv`, `conda`, and `python3.11`: not detected before setup.

## Files created

- Root governance and project files: `AGENTS.md`, `README.md`, `PROJECT_STATUS.md`, `CHANGELOG.md`, `ACCESS_NOTICE.md`, `CONTRIBUTING.md`, `CITATION.cff`, `.gitignore`, `.gitattributes`, `.editorconfig`, `.env.example`, `Makefile`, and `pyproject.toml`.
- Environment files under `environment/`.
- YAML configs under `configs/`.
- Data placeholder directories under `data/`.
- Metadata templates under `metadata/`.
- Scientific, governance, QC, Laplacian, splitting, and reproducibility docs under `docs/`.
- Python package under `src/tau_sr/`.
- Thin workflow scripts under `scripts/`.
- Synthetic-only tests under `tests/`.
- Placeholder directories under `notebooks/`, `logs/`, `results/`, `figures/`, and `models/`.
- Report templates and this setup report under `reports/`.

## Existing files preserved

- None detected.

## Existing files modified

- None pre-existing; newly generated scaffold files were formatted and repaired during verification.

## Environment detected

- Target runtime: Python >=3.11, with Python 3.11 preferred.
- Runtime used for setup: Python 3.14.6 in local virtual environment `.venv`.
- Git repository initialized locally.

## Dependencies installed

Installed into `.venv` with `python -m pip install -e ".[dev]"`:

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- PyYAML
- pytest
- ruff
- mypy

## Dependencies intentionally not installed

- PyTorch.
- torchdiffeq.
- PySR.
- Julia.
- MRtrix3.
- FreeSurfer.
- Other heavy neuroimaging or symbolic-regression systems.

PySR may require a compatible Julia runtime and remains intentionally deferred.

## Commands run

- `pwd`
- `git status --short`
- `find . -maxdepth 3 -type f | sort`
- `python --version`
- `python3 --version`
- `python3 -m venv .venv`
- `.venv/bin/python -m pip install -e ".[dev]"`
- `.venv/bin/python -m ruff format .`
- `.venv/bin/python -m ruff check .`
- `.venv/bin/python -m pytest`
- `.venv/bin/python -m tau_sr smoke-test --output-report reports/smoke_tests/synthetic_smoke_report.json`
- `.venv/bin/python -m mypy src/tau_sr`
- `make verify PYTHON=.venv/bin/python`
- `.venv/bin/python -m tau_sr verify-environment --config-dir configs --output reports/environment_check.json`
- `.venv/bin/python -m tau_sr initialize-data-dirs --root data --output-record reports/data_directory_initialization.json --dry-run`
- CLI help smoke check for all `tau-sr` subcommands.
- `git status --short --ignored`

## Test results

- Ruff lint: passed.
- Pytest: 35 passed.
- Make verification: passed with `PYTHON=.venv/bin/python`.
- CLI help: all subcommands returned exit code 0.
- Mypy: non-blocking third-party typing issue. NumPy's installed Python 3.14 stubs use syntax that conflicts with the configured Python 3.11 mypy target before project files are checked.

## Smoke-test results

- Synthetic smoke test passed.
- It generated 148 synthetic ROI labels, synthetic tau visits, count/FA/length matrices, inventory/checksum records, QC reports, three network representations, both Laplacian types, Laplacian validation, the transport-direction check, and subject-level splits.
- Temporary restricted-style artifacts were deleted on normal completion.

## Security controls created

- `.env` ignored and `.env.example` committed.
- Restricted data directories and generated subject-level outputs ignored.
- NIfTI, DICOM, imaging archives, NumPy arrays, Parquet files, checkpoints, PyTorch artifacts, credentials, local overrides, and PySR/Julia state ignored.
- Incoming-data directory warnings documented and implemented.
- SHA-256 checksum tooling implemented.
- Identifier flagging reports paths, field names, and reasons without storing discovered values.
- Scripts are non-destructive by default and require explicit output paths.
- Synthetic fixtures use generated identifiers such as `sub-syn001`.

## Scientific assumptions encoded

- Primary model:

```text
dy_{s,i}/dt
    = -rho_m (L_s^(m) y_s)_i
      + r_{theta,m}(y_{s,i})
```

- Subject-specific DWI-derived structural connectivity only.
- Approved initial network representations: streamline count, FA-weighted, and tract-length-adjusted.
- No functional connectivity, group-average networks, learned transport functions, or region-specific reaction networks were introduced.
- Symbolic regression is restricted to input `y` and target `r_theta(y)` unless the protocol is formally changed.

## All UNSET scientific decisions

See `docs/unresolved_decisions.md`, including tau tracer, SUVR reference region, PVC status, DWI pipeline, adjacency scaling, primary Laplacian convention, timing thresholds, cohort size, missing-data thresholds, approved storage/computing environment, and final split strategy.

## Recommended next actions when data arrive

- Confirm approved computing/storage environment and set `TAU_SR_DATA_ROOT`.
- Resolve all `UNSET` scientific and governance decisions through the decision register.
- Run intake inventory and checksums against authorized immutable incoming files.
- Build a draft manifest from approved mappings.
- Validate ROI alignment before any tau/connectome pairing.
- Run tau-PET and structural-connectivity QC.
- Freeze subject-level splits only after cohort eligibility is approved.

## Failures or incomplete items

- Python 3.11 is not installed on this machine; setup used Python 3.14.6.
- Mypy did not complete because of third-party NumPy stub syntax under the configured Python 3.11 target.
- No real-data intake, real QC, neural ODE training, symbolic regression, or biological interpretation has been performed.
