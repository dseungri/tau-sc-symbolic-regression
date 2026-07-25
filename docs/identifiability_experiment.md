### Neural Identifiability Experiment

This note records the first controlled noise-free identifiability attempt after
adding zero-anchored and oracle diagnostic models. The machine-readable artifact
tracked by Git is:

```text
experiments/results/identifiability_experiment.json
```

It is a byte-for-byte copy of the completed runtime artifact originally written
to `results/identifiability_experiment.json`. Both files have SHA-256 checksum
`9b3c663eafb85b12cf581d84d63e2ca8b439b8e1071f565efb758209824aa939`.
The `results/` copy remains an ignored regenerated-output location.

The run used the established synthetic recovery dataset: 24 subjects, 148 ROIs,
visits at 0.0, 0.5, 1.0, 1.5, and 2.0 years, seed 42, symmetric-normalized count
Laplacians, true rho 0.1, and true reaction `0.5*y - 0.5*y**2`. The subject
split was fixed at 16 train, 4 validation, and 4 test subjects. Hyperparameter
selection for the oracle-rho and zero-anchored joint experiments used validation
loss only.

### Neural Results

| Experiment | Selected candidate | Test trajectory RMSE | Learned rho | Rho abs. error | Reaction RMSE | Reaction MAE | r(0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original baseline | baseline | 0.008244739 | 0.095316271 | 0.004683729 | 0.014340430 | 0.010442975 | 0.035498285 |
| Fixed-rho zero-anchored | fixed_rho_0.1 | 0.000223523 | 0.100000004 | 0.000000004 | 0.000545209 | 0.000311496 | 0.0 |
| Oracle reaction | rho_lr_0.01 | 0.000117904 | 0.099325654 | 0.000674346 | 0.0 | 0.0 | 0.0 |
| Zero-anchored joint | rho_lr_0.01 | 0.000300047 | 0.102968123 | 0.002968123 | 0.000905949 | 0.000600262 | 0.0 |

The zero-anchored joint model materially improved both reaction recovery and rho
recovery relative to the original baseline, so the script ran real PySR on the
validation-selected zero-anchored model.

### Symbolic Results

PySR used 500 evenly spaced reaction samples from the training-state domain,
restricted operators `["+", "-", "*"]`, no unary operators, serial deterministic
settings, `model_selection="best"`, and seed 42.

| Quantity | Value |
|---|---:|
| Selected equation | `((y * -0.48615044) + 0.49511972) * y` |
| Expanded equation | `-0.48615044*y**2 + 0.49511972*y` |
| Complexity | 7 |
| PySR loss | 2.5297688e-08 |
| Fit domain | [0.050014085, 0.626900642] |
| Evaluation domain | [0.050302529, 0.626612198] |
| PySR runtime | 22.710398 s |
| PySR version | 1.5.10 |
| Julia version | 1.12.6 |

Independent-grid reaction metrics:

| Comparison | RMSE | MAE |
|---|---:|---:|
| Neural versus true | 0.000895686 | 0.000594984 |
| Symbolic versus neural | 0.000158076 | 0.000129469 |
| Symbolic versus true | 0.000883148 | 0.000649828 |

Polynomial coefficients:

| Coefficient | True | Recovered | Error |
|---|---:|---:|---:|
| Linear | 0.5 | 0.49511972 | -0.00488028 |
| Quadratic | -0.5 | -0.48615044 | 0.01384956 |

The symbolic distillation error was much smaller than the neural recovery error
(`0.000158` versus `0.000896` RMSE on the dense grid). The remaining symbolic
versus true error therefore appears to originate primarily from imperfect neural
reaction recovery, with a smaller contribution from symbolic distillation. The
symbolic equation slightly changed the error profile relative to the neural
curve, so the final error is not a pure copy of the neural error, but neural
mechanistic recovery is the dominant source.

### Run Notes

The completed JSON was written after a total suite runtime of about 811.03 s.
The command session did not return control cleanly after writing the artifact,
apparently during PySR/Julia cleanup, so the lingering process was interrupted
after confirming that the JSON contained the completed PySR result.

Evidence from the original run showed that `scripts/run_identifiability_experiment.py`
had already written a complete, parseable JSON file while the parent Python
process still did not return control. The parent script had no process pool,
executor, or owned subprocess for PySR at that point; it ran `fit_symbolic_reaction`
and imported `juliacall` directly in the main process. The exact upstream Julia
shutdown state was not proven, so this is treated as an observed embedded
PySR/Julia shutdown-lifecycle issue.

The experiment script now isolates only the PySR stage in
`scripts/run_pysr_worker.py`. The parent writes worker requests and final
artifacts atomically, validates parseability after replacement, and never
overwrites the final JSON with a partial result. If a worker reports a completed
valid result but remains alive past the shutdown grace period, the parent
terminates only that owned worker process group and records whether cleanup was
required. Temporary PySR working directories and regenerated `results/` outputs
remain ignored.
