# Subject-Specific Structural Connectome Representation and Symbolic Discovery of Tau Reaction Laws (better title soon)

## Scientific Model

For subject `s`, brain region `i`, and structural-network representation `m`:

```text
dy_{s,i}/dt
    = -rho_m (L_s^(m) y_s)_i
      + r_{theta,m}(y_{s,i})
```

`y` is the regional tau-PET burden, `L_s^(m)` is a subject-specific DWI-derived graph Laplacian, `rho_m` is a transport coefficient, and `r_theta,m` is a compact neural representation of the local net reaction function. Symbolic regression fits `y -> r_theta(y)`

## Repo Structure

- `configs/`: documented YAML configuration with unresolved decisions marked `UNSET`.
- `data/`: ignored local data staging directories with placeholder README files only.
- `docs/`: governance, scientific, QC, and reproducibility protocols.
- `environment/`: environment and dependency specifications.
- `metadata/`: CSV templates for manifests, data dictionaries, logs, and decisions.
- `src/tau_sr/`: Python package for intake, validation, QC, graph utilities, splitting, and CLI.
- `scripts/`: thin command-line wrappers for the setup workflow.
- `tests/`: synthetic-only test suite.
- `reports/`: report templates and setup report.
