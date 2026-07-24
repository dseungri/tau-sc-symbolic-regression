### Current interfaces

Data interfaces used in project. New modules should keep formats

### Tau table

Defined in `schemas.py`

One row represents one subject, one visit, one brain region

Required columns:

| Column | Meaning |
|---|---|
| `subject_id` | Synthetic or deidentified subject identifier |
| `visit_id` | Visit identifier |
| `time_years` | Time from baseline in years |
| `roi_index` | Region index |
| `roi_name` | Region label |
| `tau_value` | Regional tau state |

### Connectome bundle

Let `R = expected_n_rois`.

```python
{
    "subject_id": str,
    "roi_labels": tuple[str, ...],  # length: R
    "count": np.ndarray,            # shape: (R, R)
    "fa": np.ndarray,               # shape: (R, R)
    "length": np.ndarray,           # shape: (R, R)
}
```

### Public validation APIs

```python
validate_tau_table(table: pd.DataFrame, expected_n_rois: int = 148, require_complete_visits: bool = False) -> pd.DataFrame

validate_connectome_bundle(bundle: ConnectomeBundle | Mapping, expected_n_rois: int = 148) -> ConnectomeBundle
```

### Synthetic connectome representations

```python
build_connectome_representations(
    count_matrix,
    fa_matrix,
    length_matrix,
    *,
    length_epsilon=1e-8,
) -> dict[str, np.ndarray]
```

Let `R = expected_n_rois`. Each input and output matrix uses identical ROI ordering and
shape `(R, R)`

Return keys, in order:

```python
[
    "count",
    "fa_weighted",
    "length_adjusted",
]
```

Formulas:

```python
W_count = count
W_fa = count * fa
scaled_length = length / subject_median_positive_length
W_length = count / (scaled_length + length_epsilon)
```

`scaled_length` and `W_length` are calculated on count support only All representations
preserve count support, are symmetric, and have zero diagonals, and length normalization is
performed separately in each subject

### Graph laplacians

```python
build_graph_laplacian(
    adjacency,
    kind="unnormalized",
) -> np.ndarray
```

Accepted `kind` values:

```python
[
    "unnormalized",
    "symmetric_normalized",
]
```

Let `R = expected_n_rois`. The weighted adjacency `W` has shape `(R, R)`, and the
returned Laplacian `L` has shape `(R, R)`. `W` must be finite, nonnegative,
symmetric, zero-diagonal, and every node must currently have finite, positive
weighted degree. ROI ordering is unchanged

Formulas:

```python
D[i, i] = sum_j W[i, j]
L = D - W
L_sym = I - D^(-1/2) W D^(-1/2)
```

A separate Laplacian is constructed for each subject and network representation
The transport term for a state vector `y` with shape `(R,)` is `-rho * (L @ y)`
This function returns NumPy arrays and does not
convert them to PyTorch tensors

### Synthetic baseline tau

```python
generate_baseline_tau(
    n_regions,
    rng,
    *,
    tau_min=0.05,
    tau_max=0.40,
) -> np.ndarray
```

Let `R = n_regions`. The output is a newly allocated one-dimensional floating NumPy
array with shape `(R,)`. Each value is sampled independently from the half-open
interval `[tau_min, tau_max)` using the caller-supplied `np.random.Generator`
Independent generators initialized with the same seed reproduce the same output, and
calls with the same generator advance that generator normally

Validation requires `n_regions >= 2`, `rng` to be a NumPy generator, and finite
non-Boolean numeric bounds satisfying `0 <= tau_min < tau_max`. Invalid inputs raise
`SchemaError` before random values are consumed

Alignment rule:

```python
baseline_tau[i] <-> roi_labels[i] <-> row/column i of adjacency and Laplacian matrices
```

### Generic NumPy RK4 integration

Defined in `ode.py`

```python
rk4_step(
    rhs,
    t: float,
    y: np.ndarray,
    step_size: float,
) -> np.ndarray

integrate_rk4(
    rhs,
    y0: np.ndarray,
    times: np.ndarray,
    *,
    max_step=0.05,
) -> np.ndarray
```

The right-hand side contract is:

```python
rhs(t: float, y: np.ndarray) -> np.ndarray
```

State and derivative arrays must be one-dimensional real numeric NumPy arrays with
matching shape `(R,)`, nonempty and finite. `rk4_step` returns one newly allocated
floating state vector with shape `(R,)`

`integrate_rk4` requires `times` to be a one-dimensional, finite, strictly increasing
NumPy array. It returns a newly allocated floating trajectory with shape
`(len(times), R)`, where the first row is a copy of `y0`. Forward integration only is
supported. `max_step` is the maximum RK4 substep size, not a step count, and defaults
to `0.05`

For each requested interval, the solver uses:

```python
n_steps = ceil((times[i + 1] - times[i]) / max_step)
h = (times[i + 1] - times[i]) / n_steps
```



### NumPy Reaction-Diffusion RHS

Defined in `dynamics.py`

```python
make_reaction_diffusion_rhs(
    laplacian: np.ndarray,
    rho: float,
    reaction_fn,
) -> Callable[[float, np.ndarray], np.ndarray]
```

The returned callable has the RK4-compatible signature:

```python
rhs(t: float, y: np.ndarray) -> np.ndarray
```

For one subject and one connectome representation, it computes:

```python
transport = -rho * (laplacian @ y)
reaction = reaction_fn(y)
dy_dt = transport + reaction
```

`laplacian` must be a finite, real numeric, square NumPy array with shape `(R, R)`
`rho` must be finite, non-Boolean, and nonnegative `reaction_fn` must be callable
and return a finite, real numeric NumPy vector with shape `(R,)` when passed a state
vector with shape `(R,)`

Dynamic states `y` must be finite, real numeric, one-dimensional NumPy arrays with
shape `(R,)`. The output is a newly allocated floating NumPy array with shape `(R,)`
The equation is autonomous, so `t` is validated for RK4 compatibility but is not used
in the calculation

The Laplacian and state must use the same ROI order:

```python
y[i] <-> roi_labels[i] <-> row/column i of laplacian
```

The factory stores an internal copy of the Laplacian
 The returned RHS passes a fresh copy of `y` to `reaction_fn`. Invalid factory inputs/invalid dynamic inputs/invalid reaction outputs/nonfinite computed
transport/derivatives raise `SchemaError`

This helper is compatible with `rk4_step` and `integrate_rk4`. 

### Single-subject reaction-diffusion trajectory

Defined in `simulation.py`

```python
simulate_reaction_diffusion_trajectory(
    y0: np.ndarray,
    times: np.ndarray,
    laplacian: np.ndarray,
    rho: float,
    reaction_fn,
    *,
    max_step=0.05,
) -> np.ndarray
```

One call simulates one subject with one Laplacian and one caller-supplied reaction
law by composing `make_reaction_diffusion_rhs` and `integrate_rk4`

For state vector `y` with shape `(R,)`, the integrated equation is:

```python
dy_dt = -rho * (laplacian @ y) + reaction_fn(y)
```

Inputs use:

```python
y0.shape == (R,)
times.shape == (T,)
laplacian.shape == (R, R)
```

The returned trajectory is a newly allocated floating NumPy array with shape
`(T, R)`. Row `trajectory[0]` is a copy of `y0`, and rows are returned only for
the exact requested observation times in `times`; internal RK4 substeps are not
included. `times` may be irregularly spaced, may start from a nonzero or negative
absolute time, and may contain a single valid time.

`max_step` is forwarded unchanged to `integrate_rk4` as the maximum RK4 substep
size. It is not a step count.

ROI ordering is unchanged:

```python
trajectory[t_index, i] <-> y0[i] <-> roi_labels[i] <-> row/column i of laplacian
```

Validation is
delegated to `make_reaction_diffusion_rhs` and `integrate_rk4`; invalid inputs,
invalid reaction outputs, or nonfinite computed derivatives raise `SchemaError`.

### Noise-free reaction-diffusion cohort

Defined in `simulation.py`

```python
simulate_reaction_diffusion_cohort(
    baseline_states: np.ndarray,
    times_by_subject: Sequence[np.ndarray],
    laplacians: np.ndarray,
    rho: float,
    reaction_fn,
    *,
    max_step=0.05,
) -> tuple[np.ndarray, ...]
```

One call simulates a cohort of `S` subjects with `R` regions. Inputs use:

```python
baseline_states.shape == (S, R)
laplacians.shape == (S, R, R)
len(times_by_subject) == S
times_by_subject[s].shape == (T_s,)
```

The return value is an ordered tuple because subjects may have different numbers
of visits:

```python
trajectories[s].shape == (T_s, R)
```

Subject index `s` is positional and must refer to the same subject across
`baseline_states[s]`, `laplacians[s]`, `times_by_subject[s]`, and
`trajectories[s]`. ROI index `i` is also positional:

```python
baseline_states[s, i]
    <-> laplacians[s, i, :]
    <-> laplacians[s, :, i]
    <-> trajectories[s][:, i]
    <-> roi_labels[i]
```

The function requires one Laplacian per subject and does not broadcast a single
two-dimensional Laplacian or use a group-average network. `rho` and
`reaction_fn` are shared across the cohort, while baseline states, Laplacians,
and observation-time arrays are subject-specific.

Each time array must follow the single-subject time contract: one-dimensional,
nonempty, finite, real numeric, and strictly increasing when it has multiple
entries. Subjects may have irregular visits, different baseline absolute times,
and different numbers of visits. A two-dimensional NumPy `times_by_subject`
array is accepted as a fixed-visit convenience, where each row is one subject's
time array.

The cohort simulator reuses `simulate_reaction_diffusion_trajectory` once per
subject. It does not reimplement RK4 and does not reimplement the
reaction-diffusion RHS. It validates cohort-level shapes and all time arrays
before beginning simulation when the invalidity is knowable from outer inputs.
If a lower-level per-subject simulation raises `SchemaError`, the cohort error
message includes the subject index and preserves the original error as the
exception cause.

The function is deterministic and noise-free, uses no random state, preserves
caller-owned baseline, Laplacian, and time arrays, and returns newly allocated
trajectory arrays. It does not generate baselines or connectomes, choose a
ground-truth reaction law, add observation noise, split subjects into training
and test sets, or apply clipping, nonnegativity projection, or biological bounds.
It can later accept validated arrays derived from real data.

### Synthetic Linear-Quadratic Reaction Law

Defined in `reactions.py`

```python
make_linear_quadratic_reaction(
    growth_rate: float,
    saturation_rate: float,
) -> Callable[[np.ndarray], np.ndarray]
```

This factory constructs the controlled synthetic ground-truth reaction law for
pipeline-recovery validation:

```python
r(y) = growth_rate * y - saturation_rate * y**2
```

`growth_rate` is the positive linear growth coefficient. `saturation_rate` is
the nonnegative quadratic saturation coefficient. When `saturation_rate == 0`,
the returned law is the linear-control case:

```python
r(y) = growth_rate * y
```

When `saturation_rate > 0`, the positive equilibrium is:

```python
K = growth_rate / saturation_rate
```

The returned callable has the standard reaction contract:

```python
reaction_fn(y: np.ndarray) -> np.ndarray
```

The state `y` must be a one-dimensional, nonempty, finite, real numeric NumPy
array with shape `(R,)`. The output is a newly allocated floating NumPy array
with the same shape `(R,)`. The callable does not mutate `y`, does not share
memory with `y`, uses no randomness, and validates that computed reaction values
remain finite.

Both coefficients are validated before construction. `growth_rate` must be
finite, real numeric, non-Boolean, and strictly positive. `saturation_rate` must
be finite, real numeric, non-Boolean, and nonnegative. Invalid coefficients,
invalid states, or nonfinite computed reaction values raise `SchemaError`.

The polynomial form is deliberately compatible with the restricted symbolic
operator set:

```python
[
    "+",
    "-",
    "*",
]
```

This reaction law does not include diffusion, integrate trajectories, generate
baselines or connectomes, add observation noise, estimate parameters, run
symbolic regression, or apply clipping or biological bounds. The coefficients
are software-controlled synthetic parameters, not biological estimates or
patient-specific measurements. Recovery experiments should choose coefficients
so curvature is visible in the simulated state range, trajectories remain
finite and numerically stable, and reaction and diffusion terms have comparable,
inspectable magnitudes.
