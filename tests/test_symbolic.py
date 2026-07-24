from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

import tau_sr.symbolic as symbolic
from tau_sr.neural import NeuralReactionDiffusion
from tau_sr.recovery import SyntheticRecoveryDataset
from tau_sr.schemas import SchemaError
from tau_sr.symbolic import (
    SYMBOLIC_BINARY_OPERATORS,
    SYMBOLIC_UNARY_OPERATORS,
    ReactionSamples,
    SymbolicFitResult,
    SymbolicRegressionConfig,
    evaluate_symbolic_recovery,
    extract_symbolic_fit_result,
    fit_symbolic_reaction,
    interpret_symbolic_polynomial,
    make_midpoint_evaluation_grid,
    sample_neural_reaction,
    training_state_domain,
)


@dataclass(frozen=True)
class FakeSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...] = ()
    test_indices: tuple[int, ...] = ()


def _dataset_with_trajectories(
    trajectories: tuple[np.ndarray, ...],
    *,
    growth_rate: float = 0.5,
    saturation_rate: float = 0.5,
) -> SyntheticRecoveryDataset:
    n_subjects = len(trajectories)
    n_regions = trajectories[0].shape[1]
    return SyntheticRecoveryDataset(
        subject_ids=tuple(f"synthetic_subject_{index:04d}" for index in range(n_subjects)),
        roi_labels=tuple(f"syn_roi_{index + 1:03d}" for index in range(n_regions)),
        baseline_states=np.stack([trajectory[0] for trajectory in trajectories]),
        adjacencies=np.zeros((n_subjects, n_regions, n_regions), dtype=float),
        laplacians=np.zeros((n_subjects, n_regions, n_regions), dtype=float),
        times_by_subject=tuple(
            np.arange(trajectory.shape[0], dtype=float) for trajectory in trajectories
        ),
        trajectories=tuple(trajectory.copy() for trajectory in trajectories),
        representation="count",
        laplacian_kind="unnormalized",
        rho=0.1,
        growth_rate=growth_rate,
        saturation_rate=saturation_rate,
        seed=42,
    )


def _small_dataset() -> SyntheticRecoveryDataset:
    return _dataset_with_trajectories(
        (
            np.array([[0.2, 0.4], [0.3, 0.5]]),
            np.array([[10.0, 11.0], [12.0, 13.0]]),
            np.array([[-5.0, -4.0], [-3.0, -2.0]]),
        )
    )


class SpyReactionModel(NeuralReactionDiffusion):
    def __init__(self) -> None:
        super().__init__(hidden_width=1)
        self.reaction_calls = 0
        self.forward_calls = 0
        self.integrate_calls = 0
        self.saw_grad_enabled: bool | None = None
        self.saw_training: bool | None = None

    def reaction(self, y: torch.Tensor) -> torch.Tensor:
        self.reaction_calls += 1
        self.saw_grad_enabled = torch.is_grad_enabled()
        self.saw_training = self.training
        return 2.0 * y

    def forward(self, y: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        raise AssertionError("sampling must not call forward")

    def integrate(
        self,
        y0: torch.Tensor,
        laplacian: torch.Tensor,
        times: torch.Tensor,
        *,
        max_step: float = 0.05,
    ) -> torch.Tensor:
        self.integrate_calls += 1
        raise AssertionError("sampling must not call integrate")


class NonfiniteReactionModel(NeuralReactionDiffusion):
    def reaction(self, y: torch.Tensor) -> torch.Tensor:
        return torch.full_like(y, math.inf)


class LinearReactionModel(NeuralReactionDiffusion):
    def __init__(self, slope: float) -> None:
        super().__init__(hidden_width=1)
        self.slope = slope

    def reaction(self, y: torch.Tensor) -> torch.Tensor:
        return self.slope * y


class FakeRow(dict[str, object]):
    def __init__(self, values: dict[str, object], *, name: int | None = None) -> None:
        super().__init__(values)
        self.name = name


class FakePySRRegressor:
    instances: list[FakePySRRegressor] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.fit_variable_names: list[str] | None = None
        self.equations_ = [
            {"equation": "0.5*y", "loss": 0.5, "complexity": 2},
            {"equation": "0.5*y - 0.5*y**2", "loss": 0.01, "complexity": 5},
        ]
        FakePySRRegressor.instances.append(self)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        variable_names: list[str] | None = None,
    ) -> FakePySRRegressor:
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        self.fit_variable_names = variable_names
        return self

    def get_best(self) -> FakeRow:
        return FakeRow(
            {"equation": "0.5*y - 0.5*y**2", "loss": 0.01, "complexity": 5},
            name=1,
        )

    def sympy(self) -> str:
        return "0.5*y - 0.5*y**2"

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = X[:, 0]
        return 0.5 * y - 0.5 * y**2


class LinearSymbolicRegressor:
    def __init__(self, slope: float) -> None:
        self.slope = slope

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.slope * X[:, 0]


class BadShapeRegressor:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], 2), dtype=float)


class NonfiniteRegressor:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], math.inf)


def _samples() -> ReactionSamples:
    X = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    return ReactionSamples(X, 0.5 * X[:, 0], 0.0, 1.0)


def _fit_result(
    regressor: object | None = None,
    *,
    equation: str = "0.5*y - 0.5*y**2",
    sympy_expression: str | None = "0.5*y - 0.5*y**2",
    complexity: int = 5,
    training_loss: float = 0.01,
) -> SymbolicFitResult:
    return SymbolicFitResult(
        regressor=LinearSymbolicRegressor(1.0) if regressor is None else regressor,
        equation_string=equation,
        complexity=complexity,
        training_loss=training_loss,
        operator_set=SYMBOLIC_BINARY_OPERATORS,
        unary_operator_set=SYMBOLIC_UNARY_OPERATORS,
        seed=42,
        selected_equation_index=1,
        sympy_expression_string=sympy_expression,
        pysr_version="1.5.10",
        n_equations=2,
        artifact_directory="/tmp/tau_sr_pysr_fake",
    )


def test_training_domain_uses_training_trajectories_only() -> None:
    dataset = _small_dataset()

    domain = training_state_domain(
        dataset,
        FakeSplit(train_indices=(0,), validation_indices=(1,), test_indices=(2,)),
    )

    assert domain == (0.2, 0.5)


def test_training_domain_accepts_mapping_and_direct_indices() -> None:
    dataset = _small_dataset()

    assert training_state_domain(dataset, {"train_indices": (2,)}) == (-5.0, -2.0)
    assert training_state_domain(dataset, (0, 2)) == (-5.0, 0.5)


@pytest.mark.parametrize(
    "split",
    [
        FakeSplit(train_indices=()),
        FakeSplit(train_indices=(3,)),
        FakeSplit(train_indices=(-1,)),
        FakeSplit(train_indices=(True,)),  # type: ignore[arg-type]
        object(),
    ],
)
def test_training_domain_rejects_invalid_or_empty_training_indices(split: object) -> None:
    with pytest.raises(SchemaError):
        training_state_domain(_small_dataset(), split)


def test_training_domain_rejects_nonfinite_trajectory_values() -> None:
    dataset = _dataset_with_trajectories((np.array([[0.0, math.inf]]),))

    with pytest.raises(SchemaError, match="finite"):
        training_state_domain(dataset, FakeSplit(train_indices=(0,)))


def test_sample_neural_reaction_shapes_order_domain_and_no_forward() -> None:
    model = SpyReactionModel().double()
    model.train()
    state_before = {name: value.clone() for name, value in model.state_dict().items()}

    samples = sample_neural_reaction(
        model,
        domain_min=0.0,
        domain_max=1.0,
        n_samples=5,
        dtype=torch.float64,
    )

    assert samples.input_values.shape == (5, 1)
    assert samples.neural_reaction_values.shape == (5,)
    np.testing.assert_allclose(samples.input_values[:, 0], np.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(samples.neural_reaction_values, 2.0 * samples.input_values[:, 0])
    assert model.reaction_calls == 1
    assert model.forward_calls == 0
    assert model.integrate_calls == 0
    assert model.saw_grad_enabled is False
    assert model.saw_training is False
    assert model.training is True
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name])


def test_sample_neural_reaction_does_not_alias_returned_arrays() -> None:
    model = SpyReactionModel().double()

    samples = sample_neural_reaction(
        model,
        domain_min=0.0,
        domain_max=1.0,
        n_samples=3,
        dtype=torch.float64,
    )

    assert samples.input_values.flags.owndata
    assert samples.neural_reaction_values.flags.owndata


@pytest.mark.parametrize(
    "kwargs",
    [
        {"domain_min": 1.0, "domain_max": 1.0},
        {"domain_min": math.nan, "domain_max": 1.0},
        {"domain_min": 0.0, "domain_max": math.inf},
        {"n_samples": 1},
        {"n_samples": True},
        {"dtype": torch.int64},
        {"dtype": "float64"},
        {"device": "not-a-real-device"},
    ],
)
def test_sample_neural_reaction_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    parameters = {"domain_min": 0.0, "domain_max": 1.0, "n_samples": 5, "dtype": torch.float64}
    parameters.update(kwargs)

    with pytest.raises(SchemaError):
        sample_neural_reaction(SpyReactionModel().double(), **parameters)  # type: ignore[arg-type]


def test_sample_neural_reaction_rejects_model_dtype_mismatch() -> None:
    with pytest.raises(SchemaError, match="dtype"):
        sample_neural_reaction(
            SpyReactionModel(),
            domain_min=0.0,
            domain_max=1.0,
            dtype=torch.float64,
        )


def test_sample_neural_reaction_rejects_nonfinite_outputs() -> None:
    with pytest.raises(SchemaError, match="finite"):
        sample_neural_reaction(
            NonfiniteReactionModel().double(),
            domain_min=0.0,
            domain_max=1.0,
            dtype=torch.float64,
        )


def test_reaction_samples_validate_shapes_order_and_domain() -> None:
    with pytest.raises(SchemaError, match="shape"):
        ReactionSamples(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.0, 1.0)
    with pytest.raises(SchemaError, match="increasing"):
        ReactionSamples(np.array([[1.0], [0.0]]), np.array([1.0, 0.0]), 0.0, 1.0)
    with pytest.raises(SchemaError, match="first"):
        ReactionSamples(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), -0.1, 1.0)


def test_symbolic_operator_constants_are_restricted() -> None:
    assert SYMBOLIC_BINARY_OPERATORS == ("+", "-", "*")
    assert SYMBOLIC_UNARY_OPERATORS == ()
    assert "/" not in SYMBOLIC_BINARY_OPERATORS
    assert "^" not in SYMBOLIC_BINARY_OPERATORS
    assert "exp" not in SYMBOLIC_UNARY_OPERATORS
    assert "binary_operators" not in SymbolicRegressionConfig.__dataclass_fields__
    assert "unary_operators" not in SymbolicRegressionConfig.__dataclass_fields__


@pytest.mark.parametrize(
    "kwargs",
    [
        {"niterations": 0},
        {"populations": 0},
        {"population_size": 0},
        {"maxsize": 0},
        {"seed": -1},
        {"seed": True},
        {"model_selection": "unknown"},
        {"output_directory": ""},
    ],
)
def test_symbolic_regression_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(SchemaError):
        SymbolicRegressionConfig(**kwargs)  # type: ignore[arg-type]


def test_fit_symbolic_reaction_passes_only_samples_and_restricted_pysr_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakePySRRegressor.instances.clear()
    monkeypatch.setattr(symbolic, "_load_pysr_regressor", lambda: FakePySRRegressor)
    samples = _samples()
    config = SymbolicRegressionConfig(
        niterations=7,
        populations=3,
        population_size=5,
        maxsize=9,
        seed=123,
        output_directory=str(tmp_path),
    )

    result = fit_symbolic_reaction(samples, config)

    fake = FakePySRRegressor.instances[0]
    assert fake.kwargs["binary_operators"] == ["+", "-", "*"]
    assert fake.kwargs["unary_operators"] == []
    assert fake.kwargs["random_state"] == 123
    assert fake.kwargs["deterministic"] is True
    assert fake.kwargs["parallelism"] == "serial"
    assert fake.kwargs["model_selection"] == "best"
    assert fake.kwargs["output_directory"] == str(tmp_path)
    assert fake.kwargs["update"] is False
    np.testing.assert_array_equal(fake.fit_X, samples.input_values)
    np.testing.assert_array_equal(fake.fit_y, samples.neural_reaction_values)
    assert fake.fit_variable_names == ["y"]
    assert result.equation_string == "0.5*y - 0.5*y**2"
    assert result.complexity == 5
    assert result.training_loss == 0.01
    assert result.seed == 123
    assert result.operator_set == ("+", "-", "*")
    assert result.unary_operator_set == ()
    assert result.artifact_directory == str(tmp_path)


def test_fit_symbolic_reaction_rejects_invalid_samples_and_config() -> None:
    with pytest.raises(SchemaError, match="samples"):
        fit_symbolic_reaction(object(), SymbolicRegressionConfig())  # type: ignore[arg-type]
    with pytest.raises(SchemaError, match="config"):
        fit_symbolic_reaction(_samples(), object())  # type: ignore[arg-type]


def test_ground_truth_metadata_is_not_passed_to_fitting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakePySRRegressor.instances.clear()
    monkeypatch.setattr(symbolic, "_load_pysr_regressor", lambda: FakePySRRegressor)
    dataset = _small_dataset()

    fit_symbolic_reaction(
        _samples(),
        SymbolicRegressionConfig(output_directory=str(tmp_path)),
    )

    fake = FakePySRRegressor.instances[0]
    captured_values = list(fake.kwargs.values()) + [fake.fit_X, fake.fit_y, fake.fit_variable_names]
    assert "growth_rate" not in fake.kwargs
    assert "saturation_rate" not in fake.kwargs
    assert "laplacians" not in fake.kwargs
    assert all(value is not dataset for value in captured_values)
    assert all(value is not dataset.trajectories for value in captured_values)
    assert all(value is not dataset.laplacians for value in captured_values)
    assert all(value is not dataset.subject_ids for value in captured_values)


def test_extract_symbolic_fit_result_validates_selected_equation() -> None:
    config = SymbolicRegressionConfig(seed=3)

    class EmptyEquationRegressor(FakePySRRegressor):
        def get_best(self) -> FakeRow:
            return FakeRow({"equation": "", "loss": 0.1, "complexity": 1})

    class NonfiniteLossRegressor(FakePySRRegressor):
        def get_best(self) -> FakeRow:
            return FakeRow({"equation": "y", "loss": math.inf, "complexity": 1})

    with pytest.raises(SchemaError, match="equation"):
        extract_symbolic_fit_result(
            EmptyEquationRegressor(),
            config,
            artifact_directory="/tmp/tau_sr_pysr_fake",
        )
    with pytest.raises(SchemaError, match="training_loss"):
        extract_symbolic_fit_result(
            NonfiniteLossRegressor(),
            config,
            artifact_directory="/tmp/tau_sr_pysr_fake",
        )


def test_make_midpoint_evaluation_grid_is_independent_from_fit_grid() -> None:
    fit_grid = np.linspace(0.0, 1.0, 5)
    evaluation_grid = make_midpoint_evaluation_grid(0.0, 1.0, n_points=5)

    np.testing.assert_allclose(evaluation_grid, np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
    assert not np.array_equal(fit_grid, evaluation_grid)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"domain_min": 1.0, "domain_max": 1.0},
        {"domain_min": math.nan, "domain_max": 1.0},
        {"n_points": 0},
        {"n_points": True},
    ],
)
def test_make_midpoint_evaluation_grid_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    parameters = {"domain_min": 0.0, "domain_max": 1.0, "n_points": 10}
    parameters.update(kwargs)

    with pytest.raises(SchemaError):
        make_midpoint_evaluation_grid(**parameters)  # type: ignore[arg-type]


def test_evaluate_symbolic_recovery_computes_distinct_error_decomposition() -> None:
    dataset = _dataset_with_trajectories(
        (np.array([[0.0, 1.0]]),),
        growth_rate=1.0,
        saturation_rate=0.0,
    )
    neural_model = LinearReactionModel(2.0).double()
    fit_result = _fit_result(
        LinearSymbolicRegressor(3.0),
        equation="3*y",
        sympy_expression="3*y",
        complexity=3,
    )
    grid = np.array([0.0, 0.5, 1.0])

    metrics = evaluate_symbolic_recovery(
        fit_result,
        neural_model,
        dataset,
        grid,
        dtype=torch.float64,
    )

    expected_one_grid = math.sqrt(np.mean(grid**2))
    assert metrics.neural_vs_true_rmse == pytest.approx(expected_one_grid)
    assert metrics.symbolic_vs_neural_rmse == pytest.approx(expected_one_grid)
    assert metrics.symbolic_vs_true_rmse == pytest.approx(2.0 * expected_one_grid)
    assert metrics.neural_vs_true_mae == pytest.approx(np.mean(grid))
    assert metrics.symbolic_vs_neural_mae == pytest.approx(np.mean(grid))
    assert metrics.symbolic_vs_true_mae == pytest.approx(2.0 * np.mean(grid))
    assert metrics.true_reaction_range == pytest.approx(1.0)
    assert metrics.symbolic_complexity == 3


@pytest.mark.parametrize(
    "grid",
    [
        np.array([[0.0], [1.0]]),
        np.array([]),
        np.array([0.0, math.inf]),
        np.array([0.0, math.nan]),
        np.array([0, 1]),
    ],
)
def test_evaluate_symbolic_recovery_rejects_invalid_grids(grid: np.ndarray) -> None:
    with pytest.raises(SchemaError):
        evaluate_symbolic_recovery(
            _fit_result(),
            LinearReactionModel(1.0).double(),
            _small_dataset(),
            grid,
            dtype=torch.float64,
        )


def test_evaluate_symbolic_recovery_rejects_bad_prediction_shapes_and_values() -> None:
    grid = np.array([0.0, 1.0])
    model = LinearReactionModel(1.0).double()
    dataset = _small_dataset()

    with pytest.raises(SchemaError, match="symbolic prediction"):
        evaluate_symbolic_recovery(
            _fit_result(BadShapeRegressor()),
            model,
            dataset,
            grid,
            dtype=torch.float64,
        )
    with pytest.raises(SchemaError, match="finite"):
        evaluate_symbolic_recovery(
            _fit_result(NonfiniteRegressor()),
            model,
            dataset,
            grid,
            dtype=torch.float64,
        )


def test_evaluate_symbolic_recovery_normalized_errors_unavailable_for_zero_true_range() -> None:
    dataset = _dataset_with_trajectories(
        (np.array([[0.0, 1.0]]),),
        growth_rate=0.5,
        saturation_rate=0.5,
    )
    metrics = evaluate_symbolic_recovery(
        _fit_result(LinearSymbolicRegressor(1.0)),
        LinearReactionModel(1.0).double(),
        dataset,
        np.array([1.0, 1.0]),
        dtype=torch.float64,
    )

    assert metrics.true_reaction_range == 0.0
    assert metrics.neural_vs_true_normalized_rmse is None
    assert metrics.symbolic_vs_neural_normalized_rmse is None
    assert metrics.symbolic_vs_true_normalized_rmse is None


def test_interpret_symbolic_polynomial_extracts_coefficients_and_errors() -> None:
    interpretation = interpret_symbolic_polynomial(
        _fit_result(sympy_expression="0.5*y - 0.5*y**2"),
        true_growth_rate=0.5,
        true_saturation_rate=0.5,
    )

    assert interpretation.is_polynomial is True
    assert interpretation.constant == pytest.approx(0.0)
    assert interpretation.linear == pytest.approx(0.5)
    assert interpretation.quadratic == pytest.approx(-0.5)
    assert interpretation.highest_degree == 2
    assert interpretation.constant_error == pytest.approx(0.0)
    assert interpretation.linear_error == pytest.approx(0.0)
    assert interpretation.quadratic_error == pytest.approx(0.0)


def test_interpret_symbolic_polynomial_handles_reordered_terms_and_small_constant() -> None:
    interpretation = interpret_symbolic_polynomial(
        _fit_result(sympy_expression="-0.5*y**2 + 0.5*y + 1e-6"),
        true_growth_rate=0.5,
        true_saturation_rate=0.5,
    )

    assert interpretation.is_polynomial is True
    assert interpretation.constant == pytest.approx(1e-6)
    assert interpretation.linear == pytest.approx(0.5)
    assert interpretation.quadratic == pytest.approx(-0.5)


def test_interpret_symbolic_polynomial_reports_non_polynomial_without_crashing() -> None:
    interpretation = interpret_symbolic_polynomial(_fit_result(sympy_expression="sin(y)"))

    assert interpretation.is_polynomial is False
    assert interpretation.constant is None
    assert interpretation.linear is None
    assert interpretation.quadratic is None
    assert interpretation.highest_degree is None


def test_import_has_no_pysr_julia_training_or_legacy_side_effects() -> None:
    code = (
        "import sys\n"
        "import tau_sr.symbolic\n"
        "assert 'pysr' not in sys.modules\n"
        "assert 'tau_sr.sr_model_1' not in sys.modules\n"
        "assert 'tau_sr.sr_model_2' not in sys.modules\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_symbolic_module_exports_expected_public_api() -> None:
    assert "fit_symbolic_reaction" in symbolic.__all__
    assert "sample_neural_reaction" in symbolic.__all__
    assert "evaluate_symbolic_recovery" in symbolic.__all__
