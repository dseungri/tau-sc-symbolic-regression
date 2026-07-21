# Subject-Specific Structural Connectome Representation and Symbolic Regression of Tau Reaction Laws (better title soon...)

Model
For subject `s`, brain region `i`, and structural-network representation `m`:

`y` is the regional tau-PET burden, `L_s^(m)` is a subject-specific DWI-derived graph Laplacian, `rho_m` is a transport coefficient, and `r_theta,m` is a compact neural representation of the local net reaction function. Symbolic regression fits `y -> r_theta(y)`

Repo Structure

- `data/`: ignored local data staging directories with placeholder README files only.
- `metadata/`: CSV templates for manifests, data dictionaries, logs, and decisions.
- `src/tau_sr/`: Python package for intake, validation, QC, graph utilities, splitting, and CLI.
- `scripts/`: thin command-line wrappers for the setup workflow.
