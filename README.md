# Subject-Specific Structural Connectome Representation and Symbolic Discovery of Tau Reaction Laws

This repository contains pre-data infrastructure for a longitudinal Alzheimer's tau-progression study. It is ready to receive authorized research data, but it does not contain real subject data.

## Scientific Model

For subject `s`, brain region `i`, and structural-network representation `m`:

```text
dy_{s,i}/dt
    = -rho_m (L_s^(m) y_s)_i
      + r_{theta,m}(y_{s,i})
```

`y` is the regional tau-PET state, `L_s^(m)` is a subject-specific DWI-derived graph Laplacian, `rho_m` is a nonnegative transport coefficient, and `r_theta,m` is a compact neural representation of the local net reaction function. Later symbolic regression fits `y -> r_theta(y)` and does not target the transport term in the primary analysis.

## Plain-Language Purpose

The project asks whether different ways of representing each person's structural connectome change held-out tau-PET prediction accuracy or the symbolic form of the local reaction law inferred after neural modeling.

## Current Stage

Repository initialization and pre-data infrastructure. No real-data intake, neural ODE training, symbolic regression, or biological interpretation has been performed.

## Security Warning

Treat all future real data as restricted research data. Never commit raw or processed subject-level data, subject identifiers, credentials, checkpoints, or generated subject outputs. Incoming files are immutable and must be read from `TAU_SR_DATA_ROOT`.

## Repository Structure

- `configs/`: documented YAML configuration with unresolved decisions marked `UNSET`.
- `data/`: ignored local data staging directories with placeholder README files only.
- `docs/`: governance, scientific, QC, and reproducibility protocols.
- `environment/`: environment and dependency specifications.
- `metadata/`: CSV templates for manifests, data dictionaries, logs, and decisions.
- `src/tau_sr/`: Python package for intake, validation, QC, graph utilities, splitting, and CLI.
- `scripts/`: thin command-line wrappers for the setup workflow.
- `tests/`: synthetic-only test suite.
- `reports/`: report templates and setup report.

## Installation

Use Python 3.11 or newer. Python 3.11 is the target version for reproducible runs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Heavy modeling and symbolic packages are intentionally separated:

```bash
python -m pip install -r environment/requirements-ml.txt
python -m pip install -r environment/requirements-symbolic.txt
```

Do not install those heavy packages during repository initialization unless the analysis lead approves. PySR may require a compatible Julia runtime.

## Environment Setup

Copy `.env.example` to a local `.env` only on an approved machine. `.env` is ignored by Git.

Required environment variable:

```bash
export TAU_SR_DATA_ROOT=/approved/restricted/storage/root
```

Do not hardcode personal paths, institutional network shares, usernames, or machine names in committed files.

## Configuration Workflow

Review every file in `configs/`. Fields marked `UNSET` must be approved before real-data modeling. In particular, `configs/laplacian.yaml` intentionally leaves the primary Laplacian type unset pending formal approval.

## Initial Command Sequence

```bash
git status --short
python --version || python3 --version
python -m pip install -e ".[dev]"
tau-sr verify-environment --config-dir configs --output reports/environment_check.json
tau-sr initialize-data-dirs --root data --output-record reports/data_directory_initialization.json --dry-run
tau-sr smoke-test --output-report reports/smoke_tests/synthetic_smoke_report.json
make verify
```

## Testing

```bash
make lint
make test
make smoke-test
```

`make verify` runs linting, tests, and the synthetic smoke test.

## Data-Intake Workflow

1. Place authorized files outside the repository under the directory named by `TAU_SR_DATA_ROOT`.
2. Keep `data/00_incoming_readonly` immutable and untracked.
3. Run inventory and checksums into explicit output directories.
4. Build a draft manifest using externally approved mappings.
5. Run ROI, tau, and connectome QC.
6. Freeze subject-level splits only after cohort eligibility is approved.

## Synthetic Versus Real Data

Synthetic examples use identifiers such as `sub-syn001` and are generated deterministically for tests. They are not patient-like records and must not be replaced with real data.

## Known Unresolved Decisions

See `docs/unresolved_decisions.md`. These include tau tracer, SUVR reference region, partial-volume correction, DWI pipeline, adjacency scaling, Laplacian convention, timing thresholds, cohort size, missing-data thresholds, and final split strategy.

## Contributing Rules

Read `AGENTS.md` before coding. Keep changes deterministic, non-destructive, documented, and synthetic-testable. Never commit real data or secrets.

## Citation

See `CITATION.cff`. Citation metadata are placeholders until project ownership and release details are approved.

## Contact and Ownership

Project owner: `UNSET`

Data steward: `UNSET`

Analysis lead: `UNSET`
