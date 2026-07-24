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

### Noise-free synthetic recovery dataset

Defined in `recovery.py`

```python
@dataclass(frozen=True)
class SyntheticRecoveryDataset:
    subject_ids: tuple[str, ...]
    roi_labels: tuple[str, ...]
    baseline_states: np.ndarray
    adjacencies: np.ndarray
    laplacians: np.ndarray
    times_by_subject: tuple[np.ndarray, ...]
    trajectories: tuple[np.ndarray, ...]
    representation: str
    laplacian_kind: str
    rho: float
    growth_rate: float
    saturation_rate: float
    seed: int
```

```python
build_noise_free_recovery_dataset(
    config: SyntheticCohortConfig,
    times_by_subject: Sequence[np.ndarray],
    *,
    representation: str,
    laplacian_kind: str,
    rho: float,
    growth_rate: float,
    saturation_rate: float,
    max_step=0.05,
) -> SyntheticRecoveryDataset
```

The builder assembles controlled synthetic data for recovery validation only.
It contains no patient observations, performs no model training, performs no
symbolic regression, and performs no file serialization. The coefficients are
software-controlled ground-truth parameters, not biological estimates.

For `S = config.n_subjects` and `R = config.n_regions`, output shapes are:

```python
baseline_states.shape == (S, R)
adjacencies.shape == (S, R, R)
laplacians.shape == (S, R, R)
len(times_by_subject) == S
trajectories[s].shape == (T_s, R)
len(subject_ids) == S
len(roi_labels) == R
```

Subject IDs are stable synthetic identifiers:

```python
synthetic_subject_0000
synthetic_subject_0001
```

ROI labels come from `generate_synthetic_roi_labels`. Subject index `s` and ROI
index `i` are positional and must stay aligned:

```python
baseline_states[s, i]
    <-> adjacencies[s, i, :]
    <-> laplacians[s, i, :]
    <-> trajectories[s][:, i]
    <-> roi_labels[i]
```

The builder uses exactly one local `np.random.default_rng(config.seed)` and lets
that generator advance through the existing tested helpers. Each subject gets an
independently generated count connectome, aligned FA and length attributes, the
explicitly requested adjacency representation, an independently constructed
Laplacian of the explicitly requested kind, and a subject-specific baseline tau
state.

The shared ground-truth dynamics are:

```python
dy_s_dt = -rho * (laplacians[s] @ y_s) + growth_rate * y_s - saturation_rate * y_s**2
```

One shared nonnegative `rho` and one shared linear-quadratic reaction law are
used across the cohort. The stored metadata preserves `representation`,
`laplacian_kind`, `rho`, `growth_rate`, `saturation_rate`, and `seed` exactly
after validation.

Observation-time arrays are caller supplied and subject-specific. The builder
requires one valid one-dimensional NumPy time array per subject, copies those
arrays before simulation and storage, and supports irregular schedules and
different visit counts. The dataset is noise-free; configurations requesting
observation noise are rejected with `SchemaError`.

The returned dataclass is frozen by attribute, but NumPy array contents are not
deeply immutable. Returned arrays are independently owned by the dataset and do
not alias caller-provided time arrays. This assembly layer reuses existing
connectome, graph, baseline, reaction, and cohort-simulation modules rather than
duplicating their formulas.

### Canonical PyTorch neural reaction-diffusion model

Defined in `neural.py`

```python
class NeuralReactionDiffusion(nn.Module):
    def __init__(
        self,
        *,
        hidden_width: int = 16,
        initial_rho: float = 0.1,
    ) -> None:
        ...

    @property
    def rho(self) -> torch.Tensor:
        ...

    def reaction(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def forward(
        self,
        y: torch.Tensor,
        laplacian: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def integrate(
        self,
        y0: torch.Tensor,
        laplacian: torch.Tensor,
        times: torch.Tensor,
        *,
        max_step=0.05,
    ) -> torch.Tensor:
        ...
```

`NeuralReactionDiffusion` is the canonical reusable neural model. `sr_model_1.py`
is the historical architectural reference, and `sr_model_2.py` is not the
canonical implementation.

The neural reaction law is one shared scalar MLP applied independently to every
ROI:

```python
Linear(1, hidden_width)
Tanh()
Linear(hidden_width, hidden_width)
Tanh()
Linear(hidden_width, 1)
```


Represents:

```python
dy_s_dt = -rho * L_s @ y_s + r_theta(y_s)
```

`raw_rho` is an unconstrained learned scalar parameter, while `rho` is exposed
as `softplus(raw_rho)` and is therefore positive. `initial_rho` is validated as
a finite positive scalar and converted through inverse-softplus so the physical
initial value matches the requested coefficient.

Supported state and Laplacian combinations are:

```python
y.shape == (R,)
laplacian.shape == (R, R)
output.shape == (R,)

y.shape == (B, R)
laplacian.shape == (R, R)
output.shape == (B, R)

y.shape == (B, R)
laplacian.shape == (B, R, R)
output.shape == (B, R)
```

Subject-specific Laplacians are supplied externally and fixed by the caller
The model contains no global graph, no fallback chain graph, and no group-average
network. Unsupported shape combinations, non-floating tensors, nonfinite values
and device or dtype mismatches raise `SchemaError`. Inputs are not silently
moved, cast, symmetrized, normalized, clipped, or projected to be nonnegative

`integrate` uses differentiable classical RK4 implemented with PyTorch
operations only. NumPy RK4 remains the ground-truth synthetic-data solver;
PyTorch RK4 is required for later neural training because it preserves gradient
flow to `raw_rho`, all reaction-network parameters, and `y0` when `y0` requires
gradients. Gradients with respect to observation times are not a supported
objective.

For requested times:

```python
y0.shape == (R,)    -> trajectory.shape == (T, R)
y0.shape == (B, R) -> trajectory.shape == (T, B, R)
```

The first trajectory row is a clone of `y0`, and internal RK4 substeps are not
returned. `times` must be a one-dimensional finite floating tensor on the same
device and with the same dtype as `y0`; values must be strictly increasing when
more than one time is supplied. `max_step` is a finite positive scalar used as
the maximum RK4 substep size.

### Neural synthetic-recovery training

Defined in `training.py`

```python
@dataclass(frozen=True)
class SubjectSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

make_subject_split(
    n_subjects: int,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> SubjectSplit
```

Splits are subject-level only. Visits and ROIs from one subject are never split
across partitions. The deterministic splitter uses `np.random.default_rng(seed)`,
does not touch NumPy's legacy global RNG, returns nonempty train, validation, and
test sets, and requires the three sets to be disjoint, complete, and in range.

```python
@dataclass(frozen=True)
class NeuralTrainingConfig:
    hidden_width: int = 16
    initial_rho: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 500
    patience: int = 50
    min_delta: float = 1e-7
    max_step: float = 0.05
    seed: int = 42
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
```

Configuration validation requires positive integer width, epoch, and patience
values; positive finite initial `rho`, learning rate, and `max_step`;
nonnegative finite weight decay and `min_delta`; a non-Boolean integer seed; a
real floating PyTorch dtype; and an available requested device. There is no
silent fallback from unavailable accelerators.

```python
@dataclass(frozen=True)
class NeuralTrainingResult:
    model: NeuralReactionDiffusion
    split: SubjectSplit
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    epochs_run: int

fit_noise_free_recovery_model(
    dataset: SyntheticRecoveryDataset,
    split: SubjectSplit,
    config: NeuralTrainingConfig,
) -> NeuralTrainingResult
```

`fit_noise_free_recovery_model` constructs a fresh canonical
`NeuralReactionDiffusion` from `config.hidden_width` and `config.initial_rho`.
Initialization is scoped with PyTorch RNG forking and `config.seed`, so fitting
does not permanently change the caller's PyTorch RNG state. The model is moved
explicitly to the configured device and dtype.

Training uses only synthetic trajectories from `SyntheticRecoveryDataset`.
For each train or validation subject, the tensors supplied to the model are:

```python
baseline_states[s]  -> y0
laplacians[s]       -> L_s
times_by_subject[s] -> times
trajectories[s]     -> target
```

Adjacency matrices, subject IDs, ROI labels, true `rho`, `growth_rate`, and
`saturation_rate` are not learned inputs and are not part of the optimization
loss. The two real pilot subjects are not used by this interface.

The trajectory loss excludes the baseline row because that row is supplied as
the initial condition:

```python
subject_loss = mean((predicted[1:] - target[1:])**2)
training_loss = mean(subject_loss over training subjects)
```

This subject-balanced aggregation prevents subjects with more visits from
automatically dominating training. Train and validation subjects must have at
least two visits. Test-subject trajectories are not used for optimization,
validation, early stopping, hyperparameter selection, or model-state selection.

Each epoch uses Adam, verifies finite training loss, finite gradients, finite
model parameters after the optimizer step, and finite validation loss. The best
validation state is cloned in memory when validation improves by more than
`min_delta`; training stops after `patience` consecutive non-improving epochs.
The returned model is restored to the best validation-loss state. No checkpoint
files are written.

```python
@dataclass(frozen=True)
class NeuralRecoveryMetrics:
    subject_indices: tuple[int, ...]
    trajectory_mse: float
    trajectory_rmse: float
    trajectory_mae: float
    estimated_rho: float
    true_rho: float
    rho_absolute_error: float
    rho_relative_error: float
    reaction_curve_rmse: float
    reaction_curve_mae: float

evaluate_neural_recovery(
    model: NeuralReactionDiffusion,
    dataset: SyntheticRecoveryDataset,
    subject_indices: Sequence[int],
    reaction_grid: np.ndarray,
    *,
    max_step: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> NeuralRecoveryMetrics
```

Evaluation sets the model to eval mode under `torch.no_grad()` and restores the
prior training/eval mode when finished. Trajectory metrics are subject-balanced
over the requested subject indices and also exclude baseline rows. The true
synthetic parameters are used only after fitting to calculate recovery metrics:
`estimated_rho = model.rho`, `true_rho = dataset.rho`, absolute rho error,
relative rho error for positive true `rho`, and neural reaction-curve RMSE/MAE
against the known linear-quadratic law.

The caller supplies `reaction_grid`. For strict recovery reporting, derive that
grid from training-subject state values only, then reuse it unchanged for train,
validation, and test metrics. The grid must be one-dimensional, nonempty, real
floating, finite, and strictly increasing, and it is converted freshly to the
model's requested dtype and device. Symbolic regression is not run in this phase,
and neither `sr_model_1.py` nor `sr_model_2.py` is imported by the training
workflow.

### Symbolic neural-reaction distillation

Defined in `symbolic.py`

```python
@dataclass(frozen=True)
class ReactionSamples:
    input_values: np.ndarray              # shape: (N, 1)
    neural_reaction_values: np.ndarray    # shape: (N,)
    domain_min: float
    domain_max: float

@dataclass(frozen=True)
class SymbolicRegressionConfig:
    niterations: int = 100
    populations: int = 15
    population_size: int = 33
    maxsize: int = 15
    seed: int = 42
    model_selection: str = "best"
    output_directory: str | None = None
```

The fitting domain is derived from training-subject trajectories only:

```python
training_state_domain(dataset, split) -> tuple[float, float]
```

The split object must provide `train_indices`, or the caller may pass a direct
sequence of training indices. Validation and test subjects do not determine the
symbolic fitting domain.

Neural reaction sampling uses:

```python
sample_neural_reaction(
    model,
    *,
    domain_min,
    domain_max,
    n_samples=500,
    device="cpu",
    dtype=torch.float64,
) -> ReactionSamples
```

Sampling evaluates only:

```python
model.reaction(y_samples)
```

It does not call graph diffusion, `forward`, trajectory integration, the
ground-truth reaction function, or PySR. The model is temporarily put in eval
mode and restored to its prior training/eval mode afterwards. The requested
device and dtype must match the model parameters; inputs are not silently moved
or cast to hide caller errors.

PySR receives only:

```python
X = ReactionSamples.input_values
y = ReactionSamples.neural_reaction_values
```

It does not receive graph data, trajectories, subject metadata, visit times,
`rho_true`, `growth_rate`, `saturation_rate`, or true reaction values. Ground
truth is evaluation-only.

The operator set is fixed:

```python
binary_operators = ["+", "-", "*"]
unary_operators = []
model_selection = "best"
```

Division, exponentiation, logarithms, trigonometric functions, powers as
separate operators, conditionals, and custom operators are not configurable in
this interface. Numeric constants are allowed. PySR is imported lazily inside
`fit_symbolic_reaction`, so importing `tau_sr.symbolic` does not start Julia or
run a search.

```python
fit_symbolic_reaction(
    samples: ReactionSamples,
    config: SymbolicRegressionConfig,
) -> SymbolicFitResult
```

The fit result stores the selected equation text, complexity, PySR training
loss, exact operator metadata, seed, selected equation index when available,
SymPy expression text when available, PySR version metadata, number of evaluated
equations when available, and the PySR artifact directory. If no
`output_directory` is supplied, a temporary directory named with the
`tau_sr_pysr_` prefix is created. Generated PySR search artifacts are not source
files and should not be committed.

Evaluation happens only after symbolic fitting:

```python
evaluate_symbolic_recovery(
    fit_result,
    neural_model,
    dataset,
    evaluation_grid,
    *,
    device="cpu",
    dtype=torch.float64,
) -> SymbolicRecoveryMetrics
```

Use an independent evaluation grid over the same training-derived domain, for
example:

```python
make_midpoint_evaluation_grid(domain_min, domain_max, n_points=1000)
```

The evaluation decomposition is:

```python
neural approximation error:      r_theta(y) versus r_true(y)
symbolic distillation error:     r_symbolic(y) versus r_theta(y)
end-to-end symbolic recovery:    r_symbolic(y) versus r_true(y)
```

Metrics include RMSE and MAE for all three comparisons. Normalized RMSE values
are reported only when the true reaction range is not numerically negligible.

Polynomial interpretation is optional and post-fit:

```python
interpret_symbolic_polynomial(
    fit_result,
    *,
    true_growth_rate=None,
    true_saturation_rate=None,
) -> PolynomialInterpretation
```

When the selected expression can be interpreted as a polynomial in `y`, the
helper extracts constant, linear, quadratic, and highest-degree information.
For the controlled synthetic target `a*y - b*y**2`, the expected signs are:

```python
constant ~= 0
linear ~= a
quadratic ~= -b
```

Coefficient errors are calculated only when requested true coefficients are
provided and extraction succeeds. The selected PySR equation is not rewritten,
manually replaced, or chosen because it resembles the known target.

This symbolic layer uses no patient data, does not train the neural model, does
not evaluate trajectories, does not import legacy SR files, does not broaden the
operator set, and does not run robustness or multi-seed experiments.
