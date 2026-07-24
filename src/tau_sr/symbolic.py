"""Symbolic distillation of learned neural reaction functions."""

from __future__ import annotations

import importlib.metadata
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import torch

from tau_sr.neural import NeuralReactionDiffusion
from tau_sr.reactions import make_linear_quadratic_reaction
from tau_sr.recovery import SyntheticRecoveryDataset
from tau_sr.schemas import SchemaError

SYMBOLIC_BINARY_OPERATORS: tuple[str, ...] = ("+", "-", "*")
SYMBOLIC_UNARY_OPERATORS: tuple[str, ...] = ()
VALID_MODEL_SELECTIONS: tuple[str, ...] = ("best", "accuracy", "score")
_NEGLIGIBLE_RANGE = 1e-12

__all__ = (
    "PolynomialInterpretation",
    "ReactionSamples",
    "SYMBOLIC_BINARY_OPERATORS",
    "SYMBOLIC_UNARY_OPERATORS",
    "SymbolicFitResult",
    "SymbolicRecoveryMetrics",
    "SymbolicRegressionConfig",
    "evaluate_symbolic_recovery",
    "fit_symbolic_reaction",
    "interpret_symbolic_polynomial",
    "make_midpoint_evaluation_grid",
    "sample_neural_reaction",
    "training_state_domain",
)


@dataclass(frozen=True)
class ReactionSamples:
    input_values: np.ndarray
    neural_reaction_values: np.ndarray
    domain_min: float
    domain_max: float

    def __post_init__(self) -> None:
        inputs = _validate_sample_inputs(self.input_values)
        outputs = _validate_vector(self.neural_reaction_values, "neural_reaction_values")
        if outputs.shape != (inputs.shape[0],):
            raise SchemaError(
                "neural_reaction_values must have shape "
                f"({inputs.shape[0]},); got {outputs.shape}."
            )
        domain_min = _require_real_scalar(self.domain_min, "domain_min")
        domain_max = _require_real_scalar(self.domain_max, "domain_max")
        if domain_min >= domain_max:
            raise SchemaError("domain_min must be less than domain_max.")
        if not np.all(np.diff(inputs[:, 0]) > 0.0):
            raise SchemaError("input_values must be strictly increasing.")
        if not math.isclose(float(inputs[0, 0]), domain_min, rel_tol=0.0, abs_tol=1e-12):
            raise SchemaError("domain_min must match the first input value.")
        if not math.isclose(float(inputs[-1, 0]), domain_max, rel_tol=0.0, abs_tol=1e-12):
            raise SchemaError("domain_max must match the last input value.")

        object.__setattr__(self, "input_values", inputs.copy())
        object.__setattr__(self, "neural_reaction_values", outputs.copy())
        object.__setattr__(self, "domain_min", domain_min)
        object.__setattr__(self, "domain_max", domain_max)


@dataclass(frozen=True)
class SymbolicRegressionConfig:
    niterations: int = 100
    populations: int = 15
    population_size: int = 33
    maxsize: int = 15
    seed: int = 42
    model_selection: str = "best"
    output_directory: str | None = None

    def __post_init__(self) -> None:
        _require_int(self.niterations, "niterations", minimum=1)
        _require_int(self.populations, "populations", minimum=1)
        _require_int(self.population_size, "population_size", minimum=1)
        _require_int(self.maxsize, "maxsize", minimum=1)
        _require_int(self.seed, "seed", minimum=0)
        if self.model_selection not in VALID_MODEL_SELECTIONS:
            raise SchemaError("model_selection must be one of: best, accuracy, score.")
        if self.output_directory is not None and (
            not isinstance(self.output_directory, str) or not self.output_directory.strip()
        ):
            raise SchemaError("output_directory must be a nonempty string when provided.")


@dataclass(frozen=True)
class SymbolicFitResult:
    regressor: object
    equation_string: str
    complexity: int
    training_loss: float
    operator_set: tuple[str, ...]
    unary_operator_set: tuple[str, ...]
    seed: int
    selected_equation_index: int | None
    sympy_expression_string: str | None
    pysr_version: str
    n_equations: int | None
    artifact_directory: str

    def __post_init__(self) -> None:
        if not isinstance(self.equation_string, str) or not self.equation_string.strip():
            raise SchemaError("equation_string must be nonempty.")
        _require_int(self.complexity, "complexity", minimum=0)
        training_loss = _require_real_scalar(self.training_loss, "training_loss")
        if training_loss < 0.0:
            raise SchemaError("training_loss must be nonnegative.")
        if tuple(self.operator_set) != SYMBOLIC_BINARY_OPERATORS:
            raise SchemaError("operator_set must be exactly ('+', '-', '*').")
        if tuple(self.unary_operator_set) != SYMBOLIC_UNARY_OPERATORS:
            raise SchemaError("unary_operator_set must be empty.")
        _require_int(self.seed, "seed", minimum=0)
        if self.selected_equation_index is not None:
            _require_int(self.selected_equation_index, "selected_equation_index", minimum=0)
        if self.sympy_expression_string is not None and not isinstance(
            self.sympy_expression_string, str
        ):
            raise SchemaError("sympy_expression_string must be a string when provided.")
        if not isinstance(self.pysr_version, str) or not self.pysr_version.strip():
            raise SchemaError("pysr_version must be nonempty.")
        if self.n_equations is not None:
            _require_int(self.n_equations, "n_equations", minimum=0)
        if not isinstance(self.artifact_directory, str) or not self.artifact_directory.strip():
            raise SchemaError("artifact_directory must be nonempty.")

        object.__setattr__(self, "equation_string", self.equation_string.strip())
        object.__setattr__(self, "training_loss", training_loss)
        object.__setattr__(self, "operator_set", tuple(self.operator_set))
        object.__setattr__(self, "unary_operator_set", tuple(self.unary_operator_set))


@dataclass(frozen=True)
class SymbolicRecoveryMetrics:
    evaluation_domain_min: float
    evaluation_domain_max: float
    n_evaluation_points: int
    neural_vs_true_rmse: float
    neural_vs_true_mae: float
    symbolic_vs_neural_rmse: float
    symbolic_vs_neural_mae: float
    symbolic_vs_true_rmse: float
    symbolic_vs_true_mae: float
    true_reaction_range: float
    neural_vs_true_normalized_rmse: float | None
    symbolic_vs_neural_normalized_rmse: float | None
    symbolic_vs_true_normalized_rmse: float | None
    symbolic_complexity: int

    def __post_init__(self) -> None:
        domain_min = _require_real_scalar(self.evaluation_domain_min, "evaluation_domain_min")
        domain_max = _require_real_scalar(self.evaluation_domain_max, "evaluation_domain_max")
        if domain_min > domain_max:
            raise SchemaError("evaluation_domain_min cannot exceed evaluation_domain_max.")
        _require_int(self.n_evaluation_points, "n_evaluation_points", minimum=1)
        for field_name in (
            "neural_vs_true_rmse",
            "neural_vs_true_mae",
            "symbolic_vs_neural_rmse",
            "symbolic_vs_neural_mae",
            "symbolic_vs_true_rmse",
            "symbolic_vs_true_mae",
            "true_reaction_range",
        ):
            value = _require_real_scalar(getattr(self, field_name), field_name)
            if value < 0.0:
                raise SchemaError(f"{field_name} must be nonnegative.")
        for field_name in (
            "neural_vs_true_normalized_rmse",
            "symbolic_vs_neural_normalized_rmse",
            "symbolic_vs_true_normalized_rmse",
        ):
            value = getattr(self, field_name)
            if value is not None:
                checked = _require_real_scalar(value, field_name)
                if checked < 0.0:
                    raise SchemaError(f"{field_name} must be nonnegative when available.")
        _require_int(self.symbolic_complexity, "symbolic_complexity", minimum=0)


@dataclass(frozen=True)
class PolynomialInterpretation:
    is_polynomial: bool
    expression_string: str
    expanded_expression_string: str | None
    constant: float | None
    linear: float | None
    quadratic: float | None
    highest_degree: int | None
    constant_error: float | None = None
    linear_error: float | None = None
    quadratic_error: float | None = None


def training_state_domain(
    dataset: SyntheticRecoveryDataset,
    split: object,
) -> tuple[float, float]:
    """Derive a symbolic fitting domain from training-subject trajectories only."""

    if not isinstance(dataset, SyntheticRecoveryDataset):
        raise SchemaError("dataset must be a SyntheticRecoveryDataset.")
    train_indices = _training_indices_from_split(split)
    if not train_indices:
        raise SchemaError("training indices must be nonempty.")

    subject_count = len(dataset.trajectories)
    values: list[np.ndarray] = []
    for index in train_indices:
        if index < 0 or index >= subject_count:
            raise SchemaError(f"training index {index} is outside the dataset range.")
        trajectory = _validate_matrix(dataset.trajectories[index], f"trajectories[{index}]")
        values.append(trajectory.ravel())

    combined = np.concatenate(values)
    if combined.size == 0:
        raise SchemaError("training trajectories must contain at least one state value.")
    return float(combined.min()), float(combined.max())


def sample_neural_reaction(
    model: NeuralReactionDiffusion,
    *,
    domain_min: float,
    domain_max: float,
    n_samples: int = 500,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ReactionSamples:
    """Sample y -> r_theta(y) on a closed interval using model.reaction only."""

    if not isinstance(model, NeuralReactionDiffusion):
        raise SchemaError("model must be a NeuralReactionDiffusion.")
    domain_min_value = _require_real_scalar(domain_min, "domain_min")
    domain_max_value = _require_real_scalar(domain_max, "domain_max")
    if domain_min_value >= domain_max_value:
        raise SchemaError("domain_min must be less than domain_max.")
    _require_int(n_samples, "n_samples", minimum=2)
    torch_device = _validate_torch_device(device)
    torch_dtype = _validate_torch_dtype(dtype)
    _validate_model_device_dtype(model, device=torch_device, dtype=torch_dtype)

    input_values = np.linspace(domain_min_value, domain_max_value, n_samples, dtype=float)
    y_samples = torch.as_tensor(
        input_values.reshape(-1, 1),
        device=torch_device,
        dtype=torch_dtype,
    )

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            reaction_values = model.reaction(y_samples)
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    if reaction_values.shape != y_samples.shape:
        raise SchemaError(
            "model.reaction output has shape "
            f"{tuple(reaction_values.shape)}; expected {tuple(y_samples.shape)}."
        )
    if not torch.isfinite(reaction_values).all().item():
        raise SchemaError("neural reaction samples must contain only finite values.")

    return ReactionSamples(
        input_values=input_values.reshape(-1, 1).copy(),
        neural_reaction_values=reaction_values.detach().cpu().numpy().reshape(-1).copy(),
        domain_min=domain_min_value,
        domain_max=domain_max_value,
    )


def fit_symbolic_reaction(
    samples: ReactionSamples,
    config: SymbolicRegressionConfig,
) -> SymbolicFitResult:
    """Fit PySR to sampled neural reaction values only."""

    checked_samples = _validate_reaction_samples(samples)
    checked_config = _validate_symbolic_config(config)
    PySRRegressor = _load_pysr_regressor()
    artifact_directory = _artifact_directory(checked_config)
    regressor = PySRRegressor(
        model_selection=checked_config.model_selection,
        binary_operators=list(SYMBOLIC_BINARY_OPERATORS),
        unary_operators=list(SYMBOLIC_UNARY_OPERATORS),
        niterations=checked_config.niterations,
        populations=checked_config.populations,
        population_size=checked_config.population_size,
        maxsize=checked_config.maxsize,
        random_state=checked_config.seed,
        deterministic=True,
        parallelism="serial",
        verbosity=0,
        progress=False,
        output_directory=artifact_directory,
        temp_equation_file=False,
        update=False,
    )
    regressor.fit(
        checked_samples.input_values,
        checked_samples.neural_reaction_values,
        variable_names=["y"],
    )
    return extract_symbolic_fit_result(
        regressor,
        checked_config,
        artifact_directory=artifact_directory,
    )


def extract_symbolic_fit_result(
    regressor: object,
    config: SymbolicRegressionConfig,
    *,
    artifact_directory: str,
) -> SymbolicFitResult:
    """Extract the selected PySR equation using the installed result interface."""

    checked_config = _validate_symbolic_config(config)
    best = _selected_equation(regressor, checked_config.model_selection)
    equation_string = _row_value(best, "equation")
    complexity = int(_row_value(best, "complexity"))
    training_loss = float(_row_value(best, "loss"))
    selected_index = _row_index(best)
    sympy_expression = _safe_sympy_string(regressor)
    n_equations = _equation_count(regressor)
    pysr_version = _pysr_version()

    return SymbolicFitResult(
        regressor=regressor,
        equation_string=str(equation_string),
        complexity=complexity,
        training_loss=training_loss,
        operator_set=SYMBOLIC_BINARY_OPERATORS,
        unary_operator_set=SYMBOLIC_UNARY_OPERATORS,
        seed=checked_config.seed,
        selected_equation_index=selected_index,
        sympy_expression_string=sympy_expression,
        pysr_version=pysr_version,
        n_equations=n_equations,
        artifact_directory=artifact_directory,
    )


def make_midpoint_evaluation_grid(
    domain_min: float,
    domain_max: float,
    *,
    n_points: int = 1000,
) -> np.ndarray:
    """Build an independent midpoint-shifted evaluation grid."""

    domain_min_value = _require_real_scalar(domain_min, "domain_min")
    domain_max_value = _require_real_scalar(domain_max, "domain_max")
    if domain_min_value >= domain_max_value:
        raise SchemaError("domain_min must be less than domain_max.")
    _require_int(n_points, "n_points", minimum=1)
    step = (domain_max_value - domain_min_value) / n_points
    return domain_min_value + (np.arange(n_points, dtype=float) + 0.5) * step


def evaluate_symbolic_recovery(
    fit_result: SymbolicFitResult,
    neural_model: NeuralReactionDiffusion,
    dataset: SyntheticRecoveryDataset,
    evaluation_grid: np.ndarray,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> SymbolicRecoveryMetrics:
    """Evaluate neural, symbolic, and true reaction curves on an independent grid."""

    if not isinstance(fit_result, SymbolicFitResult):
        raise SchemaError("fit_result must be a SymbolicFitResult.")
    if not isinstance(dataset, SyntheticRecoveryDataset):
        raise SchemaError("dataset must be a SyntheticRecoveryDataset.")
    grid = _validate_evaluation_grid(evaluation_grid)
    neural_prediction = _evaluate_neural_reaction_on_grid(
        neural_model,
        grid,
        device=device,
        dtype=dtype,
    )
    symbolic_prediction = _evaluate_symbolic_regressor(fit_result.regressor, grid)
    true_reaction = make_linear_quadratic_reaction(
        dataset.growth_rate,
        dataset.saturation_rate,
    )
    true_prediction = _validate_prediction(
        true_reaction(grid.copy()),
        "true reaction prediction",
        expected_shape=grid.shape,
    )

    neural_vs_true = _comparison_metrics(neural_prediction, true_prediction)
    symbolic_vs_neural = _comparison_metrics(symbolic_prediction, neural_prediction)
    symbolic_vs_true = _comparison_metrics(symbolic_prediction, true_prediction)
    true_range = float(np.ptp(true_prediction))
    denominator = true_range if true_range > _NEGLIGIBLE_RANGE else None

    return SymbolicRecoveryMetrics(
        evaluation_domain_min=float(grid.min()),
        evaluation_domain_max=float(grid.max()),
        n_evaluation_points=int(grid.shape[0]),
        neural_vs_true_rmse=neural_vs_true[0],
        neural_vs_true_mae=neural_vs_true[1],
        symbolic_vs_neural_rmse=symbolic_vs_neural[0],
        symbolic_vs_neural_mae=symbolic_vs_neural[1],
        symbolic_vs_true_rmse=symbolic_vs_true[0],
        symbolic_vs_true_mae=symbolic_vs_true[1],
        true_reaction_range=true_range,
        neural_vs_true_normalized_rmse=(
            neural_vs_true[0] / denominator if denominator is not None else None
        ),
        symbolic_vs_neural_normalized_rmse=(
            symbolic_vs_neural[0] / denominator if denominator is not None else None
        ),
        symbolic_vs_true_normalized_rmse=(
            symbolic_vs_true[0] / denominator if denominator is not None else None
        ),
        symbolic_complexity=fit_result.complexity,
    )


def interpret_symbolic_polynomial(
    fit_result: SymbolicFitResult,
    *,
    true_growth_rate: float | None = None,
    true_saturation_rate: float | None = None,
    variable_name: str = "y",
) -> PolynomialInterpretation:
    """Extract polynomial coefficients from the selected symbolic expression when possible."""

    if not isinstance(fit_result, SymbolicFitResult):
        raise SchemaError("fit_result must be a SymbolicFitResult.")
    if not isinstance(variable_name, str) or not variable_name.strip():
        raise SchemaError("variable_name must be a nonempty string.")
    true_growth = (
        None
        if true_growth_rate is None
        else _require_real_scalar(true_growth_rate, "true_growth_rate")
    )
    true_saturation = (
        None
        if true_saturation_rate is None
        else _require_real_scalar(true_saturation_rate, "true_saturation_rate")
    )

    expression_text = fit_result.sympy_expression_string or fit_result.equation_string
    try:
        import sympy as sp

        symbol = sp.Symbol(variable_name)
        expression = sp.sympify(expression_text, locals={variable_name: symbol})
        expanded = sp.expand(expression)
        polynomial = sp.Poly(expanded, symbol)
    except Exception:
        return PolynomialInterpretation(
            is_polynomial=False,
            expression_string=expression_text,
            expanded_expression_string=None,
            constant=None,
            linear=None,
            quadratic=None,
            highest_degree=None,
        )

    constant = float(polynomial.coeff_monomial(symbol**0))
    linear = float(polynomial.coeff_monomial(symbol))
    quadratic = float(polynomial.coeff_monomial(symbol**2))
    highest_degree = int(polynomial.degree())
    constant_error = constant
    linear_error = None if true_growth is None else linear - true_growth
    quadratic_error = None if true_saturation is None else quadratic + true_saturation

    return PolynomialInterpretation(
        is_polynomial=True,
        expression_string=expression_text,
        expanded_expression_string=str(expanded),
        constant=constant,
        linear=linear,
        quadratic=quadratic,
        highest_degree=highest_degree,
        constant_error=constant_error,
        linear_error=linear_error,
        quadratic_error=quadratic_error,
    )


def _training_indices_from_split(split: object) -> tuple[int, ...]:
    if isinstance(split, Mapping):
        raw_indices = split.get("train_indices")
    elif hasattr(split, "train_indices"):
        raw_indices = split.train_indices
    else:
        raw_indices = split

    if isinstance(raw_indices, np.ndarray):
        indices = raw_indices.tolist()
    elif isinstance(raw_indices, (str, bytes)) or not isinstance(raw_indices, Sequence):
        raise SchemaError("split must provide a sequence of training indices.")
    else:
        indices = list(raw_indices)

    checked: list[int] = []
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError("training indices must be integers.")
        checked.append(value)
    return tuple(checked)


def _validate_reaction_samples(samples: object) -> ReactionSamples:
    if not isinstance(samples, ReactionSamples):
        raise SchemaError("samples must be ReactionSamples.")
    return ReactionSamples(
        input_values=samples.input_values,
        neural_reaction_values=samples.neural_reaction_values,
        domain_min=samples.domain_min,
        domain_max=samples.domain_max,
    )


def _validate_symbolic_config(config: object) -> SymbolicRegressionConfig:
    if not isinstance(config, SymbolicRegressionConfig):
        raise SchemaError("config must be SymbolicRegressionConfig.")
    return SymbolicRegressionConfig(
        niterations=config.niterations,
        populations=config.populations,
        population_size=config.population_size,
        maxsize=config.maxsize,
        seed=config.seed,
        model_selection=config.model_selection,
        output_directory=config.output_directory,
    )


def _load_pysr_regressor() -> type:
    from pysr import PySRRegressor

    return PySRRegressor


def _pysr_version() -> str:
    try:
        return importlib.metadata.version("pysr")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _artifact_directory(config: SymbolicRegressionConfig) -> str:
    if config.output_directory is not None:
        path = Path(config.output_directory)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    return tempfile.mkdtemp(prefix="tau_sr_pysr_")


def _selected_equation(regressor: object, model_selection: str) -> object:
    if hasattr(regressor, "get_best"):
        return regressor.get_best()

    equations = getattr(regressor, "equations_", None)
    if equations is None:
        raise SchemaError("regressor does not expose equations_.")
    if isinstance(equations, list):
        raise SchemaError("multi-output symbolic equations are not supported.")

    if model_selection == "accuracy" or "score" not in equations:
        selected_index = equations["loss"].idxmin()
    elif model_selection == "score":
        selected_index = equations["score"].idxmax()
    elif model_selection == "best":
        threshold = 1.5 * equations["loss"].min()
        selected_index = equations.loc[equations["loss"] <= threshold, "score"].idxmax()
    else:
        raise SchemaError("unsupported model_selection.")
    return equations.loc[selected_index]


def _row_value(row: object, key: str) -> object:
    if isinstance(row, Mapping):
        if key not in row:
            raise SchemaError(f"selected symbolic equation is missing {key!r}.")
        return row[key]
    try:
        return row[key]  # type: ignore[index]
    except Exception as exc:
        raise SchemaError(f"selected symbolic equation is missing {key!r}.") from exc


def _row_index(row: object) -> int | None:
    index = getattr(row, "name", None)
    if index is None:
        return None
    if isinstance(index, bool):
        return None
    if isinstance(index, int):
        return index
    try:
        return int(index)
    except (TypeError, ValueError):
        return None


def _safe_sympy_string(regressor: object) -> str | None:
    if not hasattr(regressor, "sympy"):
        return None
    try:
        expression = regressor.sympy()
    except Exception:
        return None
    if isinstance(expression, list):
        return None
    return str(expression)


def _equation_count(regressor: object) -> int | None:
    equations = getattr(regressor, "equations_", None)
    if equations is None:
        return None
    if isinstance(equations, list):
        return sum(len(equation_table) for equation_table in equations)
    try:
        return len(equations)
    except TypeError:
        return None


def _evaluate_neural_reaction_on_grid(
    model: NeuralReactionDiffusion,
    grid: np.ndarray,
    *,
    device: str,
    dtype: torch.dtype,
) -> np.ndarray:
    if not isinstance(model, NeuralReactionDiffusion):
        raise SchemaError("neural_model must be a NeuralReactionDiffusion.")
    torch_device = _validate_torch_device(device)
    torch_dtype = _validate_torch_dtype(dtype)
    _validate_model_device_dtype(model, device=torch_device, dtype=torch_dtype)
    y_values = torch.as_tensor(grid.reshape(-1, 1), device=torch_device, dtype=torch_dtype)

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            prediction = model.reaction(y_values)
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    if prediction.shape != y_values.shape:
        raise SchemaError(
            "neural reaction prediction has shape "
            f"{tuple(prediction.shape)}; expected {tuple(y_values.shape)}."
        )
    return _validate_prediction(
        prediction.detach().cpu().numpy().reshape(-1),
        "neural reaction prediction",
        expected_shape=grid.shape,
    )


def _evaluate_symbolic_regressor(regressor: object, grid: np.ndarray) -> np.ndarray:
    if not hasattr(regressor, "predict"):
        raise SchemaError("symbolic regressor must provide predict().")
    prediction = regressor.predict(grid.reshape(-1, 1))
    return _validate_prediction(
        prediction,
        "symbolic prediction",
        expected_shape=grid.shape,
    )


def _comparison_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    difference = left - right
    return float(np.sqrt(np.mean(difference**2))), float(np.mean(np.abs(difference)))


def _validate_prediction(
    prediction: object,
    field_name: str,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(prediction)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    values = _validate_vector(values, field_name)
    if values.shape != expected_shape:
        raise SchemaError(f"{field_name} has shape {values.shape}; expected {expected_shape}.")
    return values.copy()


def _validate_evaluation_grid(grid: object) -> np.ndarray:
    values = _validate_vector(grid, "evaluation_grid")
    if values.size == 0:
        raise SchemaError("evaluation_grid must be nonempty.")
    return values.copy()


def _validate_sample_inputs(values: object) -> np.ndarray:
    inputs = np.asarray(values)
    if inputs.ndim != 2 or inputs.shape[1] != 1:
        raise SchemaError("input_values must have shape (N, 1).")
    if inputs.shape[0] < 2:
        raise SchemaError("input_values must contain at least two samples.")
    if not _is_real_numeric_dtype(inputs):
        raise SchemaError("input_values must have a real numeric dtype.")
    if not np.isfinite(inputs).all():
        raise SchemaError("input_values must contain only finite values.")
    return inputs.astype(float, copy=True)


def _validate_vector(values: object, field_name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise SchemaError(f"{field_name} must be one-dimensional.")
    if array.size == 0:
        raise SchemaError(f"{field_name} must be nonempty.")
    if not _is_real_numeric_dtype(array):
        raise SchemaError(f"{field_name} must have a real numeric dtype.")
    if not np.isfinite(array).all():
        raise SchemaError(f"{field_name} must contain only finite values.")
    return array.astype(float, copy=True)


def _validate_matrix(values: object, field_name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise SchemaError(f"{field_name} must be two-dimensional.")
    if array.size == 0:
        raise SchemaError(f"{field_name} must be nonempty.")
    if not _is_real_numeric_dtype(array):
        raise SchemaError(f"{field_name} must have a real numeric dtype.")
    if not np.isfinite(array).all():
        raise SchemaError(f"{field_name} must contain only finite values.")
    return array.astype(float, copy=True)


def _is_real_numeric_dtype(array: np.ndarray) -> bool:
    return bool(np.issubdtype(array.dtype, np.floating))


def _require_real_scalar(value: object, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise SchemaError(f"{field_name} must be a real numeric scalar.")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise SchemaError(f"{field_name} must be finite.")
    return value_float


def _require_int(value: object, field_name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field_name} must be an integer.")
    if value < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")


def _validate_torch_device(device: object) -> torch.device:
    try:
        return torch.device(device)  # type: ignore[arg-type]
    except (TypeError, RuntimeError) as exc:
        raise SchemaError("device must identify a valid torch device.") from exc


def _validate_torch_dtype(dtype: object) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise SchemaError("dtype must be a torch.dtype.")
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        raise SchemaError("dtype must be a real floating torch dtype.")
    return dtype


def _validate_model_device_dtype(
    model: NeuralReactionDiffusion,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise SchemaError("model must have parameters.") from exc
    if parameter.device != device:
        raise SchemaError("model parameters must be on the requested device.")
    if parameter.dtype != dtype:
        raise SchemaError("model parameter dtype must match the requested dtype.")
