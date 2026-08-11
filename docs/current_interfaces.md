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

Two functions everything else calls to check data before it gets used

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

Let `R = expected_n_rois`. Each input and output matrix uses same ROI ordering and
shape `(R, R)`

Return keys, in order

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

`scaled_length` and `W_length` are only calclated when there's actual count support All 3 representations keep the same count support, are symmetric, and have zero diagonals. Length normalization is
done separately in each subject

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
returned Laplacian `L` has shape `(R, R)`. `W` has be finite, nonnegative,
symmetric, zero-diagonal, and every node needs finite, positive
weighted degree. ROI order doesn't change

Formulas:

```python
D[i, i] = sum_j W[i, j]
L = D - W
L_sym = I - D^(-1/2) W D^(-1/2)
```

A separate Laplacian is constructed for each subject and network representation
The transport term for a state vector `y` with shape `(R,)` is `-rho * (L @ y)`
Returns NumPy arrays

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

Let `R = n_regions`. Output is a newly allocated ID float NumPy 
array with shape `(R,)`. Each value is sampled independently from the half-open
interval `[tau_min, tau_max)` using whatever caller-supplied `np.random.Generator`
Independent generators initialized with the same seed reproduce the same output, and
calls with the same generator advance that generator normally

Validation needs `n_regions >= 2`, `rng` to be a NumPy generator, and finite
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

State and derivative arrays have to be one-dimensional real numeric NumPy arrays with
matching shape `(R,)`, nonempty and finite. `rk4_step` returns one newly allocated
floating state vector with shape `(R,)`

`integrate_rk4` needs `times` to be a one-dimensional, finite, strictly increasing
NumPy array. It returns a newly allocated floating trajectory with shape
`(len(times), R)`, where the first row is a copy of `y0`. Forward integration only is
supported. `max_step` is the max RK4 substep size, not a step count, and defaults
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

`laplacian` has to be finite, real, square NumPy array with shape `(R, R)`
`rho` has to be finite, non-Boolean, and nonnegative `reaction_fn` has to be callable
and return a finite, real numeric NumPy vector with shape `(R,)` when passed a state
vector with shape `(R,)`.

Dynamic states `y` has to be finite, real numeric, one-dimensional NumPy arrays with
shape `(R,)`. The output is a newly allocated floating NumPy array with shape `(R,)`
The equation is autonomous, so `t` is validated for RK4 compatibility but not used in the math.

The Laplacian and state have to use the same ROI order:

```python
y[i] <-> roi_labels[i] <-> row/column i of laplacian
```

The factory keeps an internal copy of the Laplacian
 Whatever RHS it returns passes a fresh copy of `y` to `reaction_fn`. Invalid factory inputs/invalid dynamic inputs/invalid reaction outputs/nonfinite computed
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
`(T, R)`. Row `trajectory[0]` is a copy of `y0`, and rows come back only for
the exact times you requested. `times` can be irregularly spaced, or start from a nonzero or negative
absolute time, can contain a single valid time.

`max_step` is passed straight to `integrate_rk4` as the max RK4 substep
size. it'sn't a step count.

ROI ordering is the same

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

The return value is an ordered tuple because subjects might have different numbers
of visits:

```python
trajectories[s].shape == (T_s, R)
```

Subject index `s` is positional and has to refer to the same subject across
`baseline_states[s]`, `laplacians[s]`, `times_by_subject[s]`, and
`trajectories[s]`. ROI index `i` is also positional:

```python
baseline_states[s, i]
    <-> laplacians[s, i, :]
    <-> laplacians[s, :, i]
    <-> trajectories[s][:, i]
    <-> roi_labels[i]
```

The function needs one Laplacian per subject and doesn't broadcast a single
two-dimensional Laplacian or use a group-average network. `rho` and
`reaction_fn` are shared across the cohort, while baseline states, Laplacians,
and observation-time arrays are subject-specific.

Each time array has to follow the single-subject time contract: one-dimensional,
nonempty, finite, real numeric, and strictly increasing when it has multiple
entries. Subjects might have irregular visits, different baseline absolute times,
and different numbers of visits. A two-dimensional NumPy `times_by_subject`
array is accepted as a fixed-visit convenience, where each row is one subject's
time array.

The cohort simulator reuses `simulate_reaction_diffusion_trajectory` once per
subject. It doesn't reimplement RK4 and doesn't reimplement the
reaction-diffusion RHS. It validates cohort-level shapes and all time arrays
before beginning simulation when the invalidity is knowable from outer inputs.
If a lower-level per-subject simulation raises `SchemaError`, the cohort error
message includes the subject index and preserves the original error as the
exception cause.

The function is deterministic and noise-free, uses no random state, preserves
caller-owned baseline, Laplacian, and time arrays, and returns newly allocated
trajectory arrays. It doesn't generate baselines or connectomes, choose a
ground-truth reaction law, add observation noise, split subjects into training
and test sets, or apply clipping, nonnegativity projection, or biological bounds.
It can later accept validated arrays derived from real data.

### Pilot longitudinal adapter

Defined in `pilot_adapter.py`

```python
@dataclass(frozen=True)
class LongitudinalTrajectoryDataset:
    subject_ids: tuple[str, ...]
    roi_labels: tuple[str, ...]
    adjacencies: np.ndarray
    laplacians: np.ndarray
    times_by_subject: tuple[np.ndarray, ...]
    trajectories: tuple[np.ndarray, ...]
    atlas: str = "unknown"
    roi_alignment: str = "unknown"
    tau_measurement: str = "unknown"
    tau_reference: str = "unknown"
    sc_weight_type: str = "unknown"
    sc_static_over_time: bool = False
    laplacian_kind: str = "unknown"
    data_alignment_ready: bool = False
    formal_evaluation_ready: bool = False

    @property
    def baseline_states(self) -> np.ndarray:
        ...
```

`LongitudinalTrajectoryDataset` is the real-data-facing model-input contract.
It contains neutral subject labels only, no source identifiers, no scan IDs, no
exact dates, no demographics, no raw paths, and no synthetic truth fields such as
true rho or reaction coefficients. Arrays are copied and stored read-only.
`baseline_states` is derived from the first trajectory row instead of stored as
duplicated input. Direct construction uses conservative `"unknown"` provenance
and readiness defaults; loaders with verified provenance, such as
`load_pilot_dataset`, must supply explicit metadata.

```python
load_pilot_dataset(
    tau_workbook_path,
    connectome_paths,
    *,
    roi_labels=None,
    tau_reference="pending_exact_confirmation",
    laplacian_kind="symmetric_normalized",
) -> LongitudinalTrajectoryDataset
```

The pilot loader expects one local tau workbook, two local headerless numeric SC
CSV matrices, and optionally an explicit ordered 160-label ROI sequence.
Advisor-confirmed direct index alignment means external labels are not required
for numerical alignment: tau `Node i` maps directly to SC row `i` and column `i`
for `i=1..160`. If `roi_labels` is `None`, the adapter uses deterministic
non-anatomical labels in the original order:

```text
LH_Destrieux_01 ... LH_Destrieux_74
RH_Destrieux_01 ... RH_Destrieux_74
Subcortical_01 ... Subcortical_12
```

Supplying custom labels changes names only and never reorders tau or SC arrays.
Supplied labels must be exactly 160 unique nonempty strings. The separate
333-parcel functional-atlas workbook is not used for this 160-node structural
pilot; there is no 333-to-160 mapping, aggregation, or reordering step.

The tau workbook is read locally. The loader uses source subject IDs, scan IDs,
and exact exam dates only transiently for grouping, uniqueness checks, matching,
sorting, and conversion to elapsed years. Relative time is computed as days from
each subject's baseline divided by `365.25`. Diagnostic and demographic columns
are ignored and are not passed to the model.

The pilot tau values are SUVR. The exact reference-region wording is pending
confirmation, so the default metadata is
`tau_reference="pending_exact_confirmation"`.

Each connectome is interpreted as static subject-specific normalized fiber-count
weights. Values are not renormalized. SC preprocessing is:

```python
W = 0.5 * (W_raw + W_raw.T)
np.fill_diagonal(W, 0.0)
```

The adapter validates finite nonnegative weights, zero raw diagonal, exact
post-symmetry, zero post-symmetry diagonal, and strictly positive weighted
degree, then delegates Laplacian construction to `build_graph_laplacian`.

The pilot has 160 nodes in the confirmed structural order: Nodes 1-74 are
left-hemisphere Destrieux cortical regions, Nodes 75-148 are right-hemisphere
Destrieux cortical regions, and Nodes 149-160 are 12 subcortical regions. The
previous 16-subcortical assumption was incorrect. The data alignment status is
`data_alignment_ready=True`; formal model evaluation remains
`formal_evaluation_ready=False` because two subjects cannot support an honest
subject-level train/validation/test evaluation. No real-data mechanistic or
generalization claims should be made at this phase.

```python
inspect_pilot_inputs(
    tau_workbook_path,
    connectome_paths,
) -> dict[str, object]
```

This structural inspection helper returns only privacy-safe aggregate metadata:
neutral subject counts, neutral visit counts, relative elapsed times, node count,
`missing_tau_values`, adjacency and Laplacian shapes, symmetrization validity,
minimum degree, direct-alignment status, and readiness flags.
`missing_tau_values` counts blank, missing, or non-finite tau node cells
normalized to `NaN` during workbook inspection; finite node cells are counted
separately internally. The helper does not return rows, arrays, identifiers,
dates, filenames, paths, demographics, or source-to-neutral mappings.

### Full-cohort two-timepoint adapter

Defined in `cohort_adapter.py`

```python
load_two_timepoint_cohort(
    structure_dir,
    baseline_tau_dir,
    followup_tau_dir,
    *,
    laplacian_kind="symmetric_normalized",
) -> LongitudinalTrajectoryDataset

inspect_two_timepoint_cohort(
    structure_dir,
    baseline_tau_dir,
    followup_tau_dir,
) -> dict[str, object]
```

The full-cohort adapter loads one structural-connectivity CSV, one measured
baseline tau CSV, and one measured one-year follow-up tau CSV per subject. The
external naming can be misleading; internally these roles are always called
`structural_connectivity`, `baseline_tau`, and `followup_tau`.

Each structural-connectivity matrix must have shape `(160, 160)`. Each tau file
must have shape `(160,)`, `(160, 1)`, or `(1, 160)`. Tau values are interpreted
as SUVR referenced to brainstem, and the two time points are fixed at
`[0.0, 1.0]` years. The adapter does not threshold, renormalize, impute, or
reorder arrays.

The full-cohort structural inputs are static whole-brain-normalized
connectivity weights. SC preprocessing is:

```python
W = 0.5 * (W_raw + W_raw.T)
np.fill_diagonal(W, 0.0)
```

The adapter rejects non-finite values, negative values, invalid shapes, and
processed matrices with non-finite or non-positive weighted degree. It delegates
symmetric-normalized Laplacian construction to `build_graph_laplacian` by
default.

The ROI order is the same 160-node order as the pilot adapter:

```text
LH_Destrieux_01 ... LH_Destrieux_74
RH_Destrieux_01 ... RH_Destrieux_74
Subcortical_01 ... Subcortical_12
```

CSV triplets are matched one-to-one by a narrow terminal numeric subject token.
Hidden metadata files are ignored. Duplicate tokens, malformed CSV names, and
missing counterparts raise `SchemaError`. Public summaries and artifacts report
aggregate counts only and never include source identifiers, filenames, private
paths, raw rows, full vectors, exact dates, or source-to-neutral mappings.

```python
make_frozen_cohort_split(
    subject_ids,
    *,
    seed=42,
    train_fraction=0.70,
    validation_fraction=0.10,
    test_fraction=0.20,
) -> SubjectSplit
```

The frozen split is subject-level, deterministic, and does not mutate NumPy's
global random state. For 160 subjects the split is 112 train, 16 validation, and
32 test. For 150 subjects the split is 105 train, 15 validation, and 30 test.
The private split manifest stores only neutral subject labels.

```text
scripts/prepare_real_cohort.py
```

The preparation CLI requires explicit input directories, an explicit private
output directory, and an optional split seed. It runs aggregate QC, stops on
validation failure, loads the dataset, creates the frozen split, and writes
privacy-safe JSON artifacts with restrictive permissions. It does not run
neural training, symbolic regression, Julia, PySR, tuning, model comparison, or
figure generation.

### Frozen Full-Cohort Audit

Defined in `cohort_audit.py`

```python
@dataclass(frozen=True)
class AuditInputs:
    frozen_artifact_dir: str | Path
    audit_output_dir: str | Path
    structure_dir: str | Path | None = None
    baseline_tau_dir: str | Path | None = None
    followup_tau_dir: str | Path | None = None
    preparation_artifact_dir: str | Path | None = None
    experiment_commit: str | None = None

audit_frozen_experiment(inputs: AuditInputs) -> dict[str, object]
```

The audit reads an already completed private full-cohort experiment and writes a
separate private aggregate audit artifact. It refuses to write inside the
repository, inside the frozen experiment directory, or over the frozen
artifacts. Audit output directories are created with mode `0o700`, and JSON
audit files are written atomically with mode `0o600`.

The audit is read-only with respect to the frozen experiment. It does not train
models, regenerate predictions, rerun held-out test evaluation, run PySR,
initialize Julia, change the split, or select new model settings. It verifies
artifact integrity, split/firewall checks, training semantics, aggregate metric
reproduction from saved subject-level metrics, subject-paired bootstrap
comparisons, and aggregate split characterization when the private cohort input
directories are supplied.

```text
scripts/audit_cohort_experiment.py
```

The CLI requires explicit frozen-artifact and private audit-output directories.
Optional cohort input directories enable aggregate split characterization. CLI
output is aggregate-only and must not print source identifiers, filenames,
private paths, split membership lists, subject-level metrics, vectors, matrices,
predictions, or residual arrays.

### Development-Only Full-Cohort Robustness CV

Defined in `cohort_robustness.py`

```python
@dataclass(frozen=True)
class RobustnessConfig:
    outer_folds: int = 5
    master_seed: int = 42
    inner_validation_fraction: float = 0.20
    ridge_alpha_grid: tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
    )
    model_config: CohortExperimentConfig = CohortExperimentConfig()

run_development_cross_validation(
    structure_dir,
    baseline_tau_dir,
    followup_tau_dir,
    preparation_artifact_dir,
    output_dir,
    *,
    config: RobustnessConfig | None = None,
) -> dict[str, object]
```

This runner performs deterministic five-fold nested cross-validation on the
development-only portion of the full cohort. It uses the frozen split manifest
only to identify and exclude the original held-out test subjects before any
numerical subject files are opened. The original frozen test subjects must not
contribute to fold construction, training, inner validation, seed selection,
ridge-alpha selection, PySR fitting, metrics, figures, or aggregate summaries.

Outer folds are subject-level and deterministic from `master_seed`. Each outer
evaluation fold is held out from all fitting decisions. Inner training and
validation splits are deterministic with seed formula:

```text
master_seed + 1009 * (fold_index + 1)
```

Models are prespecified as persistence, multivariate ridge, diffusion-only,
linear reaction-diffusion, linear-quadratic reaction-diffusion, zero-anchored
neural reaction-diffusion, and symbolic distillation of the refitted neural
reaction. Ridge uses training-only centering and standardization and selects
alpha by inner-validation RMSE, breaking exact ties toward larger alpha.

Private outputs include checkpoints, out-of-fold predictions, and residual
arrays so future audits can reproduce raw-vector metrics. Output directories
must be outside the repository and use restrictive permissions. Console output
must be aggregate-only and must not print identifiers, filenames, private paths,
fold membership, subject-level results, vectors, matrices, predictions, or
residual arrays.

```text
scripts/run_cohort_robustness.py
```

The CLI requires explicit input, preparation-artifact, and private output
directories. It runs models sequentially, uses safe checkpoint/resume
fingerprints, and prints only aggregate progress and completion counts.

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

The state `y` has to be a one-dimensional, nonempty, finite, real numeric NumPy
array with shape `(R,)`. The output is a newly allocated floating NumPy array
with the same shape `(R,)`. The callable doesn't mutate `y`, doesn't share
memory with `y`, uses no randomness, and validates that computed reaction values
remain finite.

Both coefficients are validated before construction. `growth_rate` has to be
finite, real numeric, non-Boolean, and strictly positive. `saturation_rate` has to
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

This reaction law doesn't include diffusion, integrate trajectories, generate
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
index `i` are positional and have to stay aligned:

```python
baseline_states[s, i]
    <-> adjacencies[s, i, :]
    <-> laplacians[s, i, :]
    <-> trajectories[s][:, i]
    <-> roi_labels[i]
```

The builder uses one local `np.random.default_rng(config.seed)` and lets
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
`laplacian_kind`, `rho`, `growth_rate`, `saturation_rate`, and `seed`
after validation.

Observation-time arrays are caller supplied and subject-specific. The builder
needs one valid one-dimensional NumPy time array per subject, copies those
arrays before simulation and storage, and supports irregular schedules and
different visit counts. The dataset is noise-free; configurations requesting
observation noise are rejected with `SchemaError`.

The returned dataclass is frozen by attribute, but NumPy array contents aren't
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
is the historical architectural reference, and `sr_model_2.py` isn't the
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
and device or dtype mismatches raise `SchemaError`. Inputs aren't moved, cast, 
symmetrized, normalized, clipped, or projected to be nonnegative

`integrate` uses differentiable classical RK4 implemented with PyTorch
operations only. NumPy RK4 remains the ground-truth synthetic-data solver;
PyTorch RK4 is required for later neural training because it preserves gradient
flow to `raw_rho`, all reaction-network parameters, and `y0` when `y0` needs
gradients. Gradients with respect to observation times aren't a supported
objective.

For requested times:

```python
y0.shape == (R,)    -> trajectory.shape == (T, R)
y0.shape == (B, R) -> trajectory.shape == (T, B, R)
```

The first trajectory row is a clone of `y0`, and internal RK4 substeps aren't
returned. `times` has to be a one-dimensional finite floating tensor on the same
device and with the same dtype as `y0`; values have to be strictly increasing when
more than one time is supplied. `max_step` is a finite positive scalar used as
the max RK4 substep size.

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
doesn't touch NumPy's legacy global RNG, returns nonempty train, validation, and
test sets, and needs the three sets to be disjoint, complete, and in range.

```python
@dataclass(frozen=True)
class NeuralTrainingConfig:
    hidden_width: int = 16
    initial_rho: float = 0.05
    learning_rate: float = 1e-3
    rho_learning_rate: float | None = None
    weight_decay: float = 0.0
    max_epochs: int = 500
    patience: int = 50
    min_delta: float = 1e-7
    max_step: float = 0.05
    seed: int = 42
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
```

Configuration validation needs positive integer width, epoch, and patience
values; positive finite initial `rho`, learning rate, optional rho learning
rate, and `max_step`; nonnegative finite weight decay and `min_delta`; a
non-Boolean integer seed; a real floating PyTorch dtype; and an available
requested device. There is no silent fallback from unavailable accelerators.

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
    *,
    model: NeuralReactionDiffusion | None = None,
) -> NeuralTrainingResult
```

`fit_noise_free_recovery_model` constructs a fresh canonical
`NeuralReactionDiffusion` from `config.hidden_width` and `config.initial_rho`
when no model is supplied. Initialization is scoped with PyTorch RNG forking and
`config.seed`, so fitting doesn't permanently change the caller's PyTorch RNG
state. A supplied model has to be compatible with the same forward, integrate, and
reaction interfaces. The model is moved explicitly to the configured device and
dtype.

When `rho_learning_rate` is omitted, Adam uses the existing single optimizer
parameter group with `learning_rate` for all trainable parameters. When
`rho_learning_rate` is provided, trainable reaction parameters use
`learning_rate` and the trainable `raw_rho` parameter uses `rho_learning_rate`.
Fixed-rho diagnostic models that expose `raw_rho` as a buffer simply omit the
rho optimizer group.

Training uses only synthetic trajectories from `SyntheticRecoveryDataset`.
For each train or validation subject, the tensors supplied to the model are:

```python
baseline_states[s]  -> y0
laplacians[s]       -> L_s
times_by_subject[s] -> times
trajectories[s]     -> target
```

Adjacency matrices, subject IDs, ROI labels, true `rho`, `growth_rate`, and
`saturation_rate` aren't learned inputs and aren't part of the optimization
loss. The two real pilot subjects aren't used by this interface.

The trajectory loss excludes the baseline row because that row is supplied as
the initial condition:

```python
subject_loss = mean((predicted[1:] - target[1:])**2)
training_loss = mean(subject_loss over training subjects)
```

This subject-balanced aggregation prevents subjects with more visits from
automatically dominating training. Train and validation subjects have to have at
least two visits. Test-subject trajectories aren't used for optimization,
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
    rho_relative_error: float | None
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
against the known linear-quadratic law. When `true_rho == 0`, relative rho error
is undefined and reported as `None`; absolute rho error remains available.

The caller supplies `reaction_grid`. For strict recovery reporting, derive that
grid from training-subject state values only, then reuse it unchanged for train,
validation, and test metrics. The grid has to be one-dimensional, nonempty, real
floating, finite, and strictly increasing, and it's converted freshly to the
model's requested dtype and device. Symbolic regression isn't run in this phase,
and neither `sr_model_1.py` nor `sr_model_2.py` is imported by the training
workflow.

### Neural identifiability diagnostics

Defined in `identifiability.py`

```python
class ZeroAnchoredNeuralReactionDiffusion(NeuralReactionDiffusion)
class FixedRhoZeroAnchoredReactionDiffusion(ZeroAnchoredNeuralReactionDiffusion)
class OracleReactionDiffusion(NeuralReactionDiffusion)
```

These models are controlled synthetic diagnostics, not real-data fitting
defaults. `ZeroAnchoredNeuralReactionDiffusion` parameterizes the scalar
reaction as `r_theta(y) = y * g_theta(y)`, so `r_theta(0) == 0`.
`FixedRhoZeroAnchoredReactionDiffusion` fixes rho to a known synthetic value
while training the zero-anchored reaction. `OracleReactionDiffusion` fixes the
known synthetic reaction law and trains only rho.

```python
build_controlled_recovery_dataset() -> SyntheticRecoveryDataset
controlled_subject_split() -> SubjectSplit
controlled_reaction_grid(dataset, split, *, n_points=200) -> np.ndarray
fit_identifiability_model(dataset, split, config, model) -> tuple[NeuralTrainingResult, float]
summarize_identifiability_result(...) -> IdentifiabilityExperimentResult
```

The controlled dataset and split recreate the 24-subject noise-free recovery
setup used for mechanistic checks. Reaction grids are derived from training
trajectories only. The runnable experiment entry point is
`scripts/run_identifiability_experiment.py`, and the first controlled result is
summarized in `docs/identifiability_experiment.md`.

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

The split object has to provide `train_indices`, or the caller might pass a direct
sequence of training indices. Validation and test subjects don't determine the
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

It doesn't call graph diffusion, `forward`, trajectory integration, the
ground-truth reaction function, or PySR. The model is temporarily put in eval
mode and restored to its prior training/eval mode afterwards. The requested
device and dtype have to match the model parameters; inputs aren't moved
or cast to hide caller errors.

PySR receives only:

```python
X = ReactionSamples.input_values
y = ReactionSamples.neural_reaction_values
```

It doesn't receive graph data, trajectories, subject metadata, visit times,
`rho_true`, `growth_rate`, `saturation_rate`, or true reaction values. Ground
truth is evaluation-only.

The operator set is fixed:

```python
binary_operators = ["+", "-", "*"]
unary_operators = []
model_selection = "best"
```

Division, exponentiation, logarithms, trigonometric functions, powers as
separate operators, conditionals, and custom operators aren't configurable in
this interface. Numeric constants are allowed. PySR is imported lazily inside
`fit_symbolic_reaction`, so importing `tau_sr.symbolic` doesn't start Julia or
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
`tau_sr_pysr_` prefix is created. Generated PySR search artifacts aren't source
files and shouldn't be committed.

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
are reported only when the true reaction range isn't numerically negligible.

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
helper extracts constant, linear, quadratic, and highest-degree information,
For the controlled synthetic target `a*y - b*y**2`, the expected signs are

```python
constant ~= 0
linear ~= a
quadratic ~= -b
```

Coefficient errors are calculated only when requested true coefficients are
there and extraction succeeds
