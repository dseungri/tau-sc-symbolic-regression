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
weighted degree. ROI ordering is unchanged.

Formulas:

```python
D[i, i] = sum_j W[i, j]
L = D - W
L_sym = I - D^(-1/2) W D^(-1/2)
```

A separate Laplacian is constructed for each subject and network representation.
The transport term for a state vector `y` with shape `(R,)` is `-rho * (L @ y)`.
This function returns NumPy arrays and does not
convert them to PyTorch tensors.

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
interval `[tau_min, tau_max)` using the caller-supplied `np.random.Generator`.
Independent generators initialized with the same seed reproduce the same output, and
calls with the same generator advance that generator normally.

Validation requires `n_regions >= 2`, `rng` to be a NumPy generator, and finite
non-Boolean numeric bounds satisfying `0 <= tau_min < tau_max`. Invalid inputs raise
`SchemaError` before random values are consumed.

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
floating state vector with shape `(R,)`.

`integrate_rk4` requires `times` to be a one-dimensional, finite, strictly increasing
NumPy array. It returns a newly allocated floating trajectory with shape
`(len(times), R)`, where the first row is a copy of `y0`. Forward integration only is
supported. `max_step` is the maximum RK4 substep size, not a step count, and defaults
to `0.05`.

For each requested interval, the solver uses:

```python
n_steps = ceil((times[i + 1] - times[i]) / max_step)
h = (times[i + 1] - times[i]) / n_steps
```

Each requested interval is subdivided into equal substeps. The returned trajectory
contains only the requested observation times, not internal substep times.

Invalid public inputs and invalid RHS return values raise `SchemaError`. Caller-owned
arrays are not mutated, and every RHS call receives a fresh copy of the current RK4
stage state so a mutating RHS cannot corrupt solver state or previously recorded
trajectory rows. Integer states are integrated in floating point.

This module is a generic NumPy ODE utility. It contains no tau-specific equation, uses
no connectome or graph Laplacian, applies no clipping or nonnegativity constraint,
adds no observation noise, performs no symbolic regression, and does not use adaptive
time stepping.
