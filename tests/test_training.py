from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest
import torch

import tau_sr.training as training
from tau_sr.neural import NeuralReactionDiffusion
from tau_sr.recovery import SyntheticRecoveryDataset
from tau_sr.schemas import SchemaError
from tau_sr.training import (
    NeuralRecoveryMetrics,
    NeuralTrainingConfig,
    NeuralTrainingResult,
    SubjectSplit,
    evaluate_neural_recovery,
    fit_noise_free_recovery_model,
    make_subject_split,
)


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


def _set_rho(model: NeuralReactionDiffusion, value: float) -> None:
    with torch.no_grad():
        model.raw_rho.copy_(
            torch.tensor(
                _inverse_softplus(value),
                dtype=model.raw_rho.dtype,
                device=model.raw_rho.device,
            )
        )


def _base_dataset(
    *,
    trajectories: tuple[np.ndarray, ...] | None = None,
    laplacians: np.ndarray | None = None,
    rho: float = 0.1,
    growth_rate: float = 0.5,
    saturation_rate: float = 0.5,
) -> SyntheticRecoveryDataset:
    times_by_subject = (
        np.array([0.0, 0.10], dtype=float),
        np.array([0.0, 0.10, 0.20], dtype=float),
        np.array([0.0, 0.15], dtype=float),
        np.array([0.0, 0.05, 0.10], dtype=float),
    )
    baseline_states = np.array(
        [
            [0.15, 0.30],
            [0.20, 0.35],
            [0.25, 0.40],
            [0.30, 0.45],
        ],
        dtype=float,
    )
    if trajectories is None:
        trajectories = tuple(
            baseline_states[index][None, :]
            + times[:, None] * (0.12 + 0.04 * baseline_states[index][None, :])
            for index, times in enumerate(times_by_subject)
        )
    if laplacians is None:
        laplacians = np.stack(
            [
                np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float) * (index + 1) / 10.0
                for index in range(4)
            ]
        )
    return SyntheticRecoveryDataset(
        subject_ids=tuple(f"synthetic_subject_{index:04d}" for index in range(4)),
        roi_labels=("syn_roi_001", "syn_roi_002"),
        baseline_states=baseline_states.copy(),
        adjacencies=np.zeros((4, 2, 2), dtype=float),
        laplacians=laplacians.copy(),
        times_by_subject=tuple(times.copy() for times in times_by_subject),
        trajectories=tuple(trajectory.copy() for trajectory in trajectories),
        representation="count",
        laplacian_kind="unnormalized",
        rho=rho,
        growth_rate=growth_rate,
        saturation_rate=saturation_rate,
        seed=42,
    )


def _small_split() -> SubjectSplit:
    return SubjectSplit(train_indices=(0, 1), validation_indices=(2,), test_indices=(3,))


def _small_config(**overrides: object) -> NeuralTrainingConfig:
    values = {
        "hidden_width": 4,
        "initial_rho": 0.05,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "max_epochs": 4,
        "patience": 4,
        "min_delta": 0.0,
        "max_step": 0.05,
        "seed": 7,
        "dtype": torch.float64,
        "device": "cpu",
    }
    values.update(overrides)
    return NeuralTrainingConfig(**values)  # type: ignore[arg-type]


def _copy_dataset_arrays(dataset: SyntheticRecoveryDataset) -> dict[str, object]:
    return {
        "baseline_states": dataset.baseline_states.copy(),
        "adjacencies": dataset.adjacencies.copy(),
        "laplacians": dataset.laplacians.copy(),
        "times_by_subject": tuple(times.copy() for times in dataset.times_by_subject),
        "trajectories": tuple(trajectory.copy() for trajectory in dataset.trajectories),
    }


def _assert_dataset_arrays_equal(
    dataset: SyntheticRecoveryDataset,
    snapshot: dict[str, object],
) -> None:
    np.testing.assert_array_equal(dataset.baseline_states, snapshot["baseline_states"])
    np.testing.assert_array_equal(dataset.adjacencies, snapshot["adjacencies"])
    np.testing.assert_array_equal(dataset.laplacians, snapshot["laplacians"])
    for observed, expected in zip(
        dataset.times_by_subject,
        snapshot["times_by_subject"],
        strict=True,
    ):
        np.testing.assert_array_equal(observed, expected)
    for observed, expected in zip(
        dataset.trajectories,
        snapshot["trajectories"],
        strict=True,
    ):
        np.testing.assert_array_equal(observed, expected)


def _state_dict_for_seed(config: NeuralTrainingConfig) -> dict[str, torch.Tensor]:
    device = torch.device(config.device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = NeuralReactionDiffusion(
            hidden_width=config.hidden_width,
            initial_rho=config.initial_rho,
        ).to(device=device, dtype=config.dtype)
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


class ZeroIntegratorModel(NeuralReactionDiffusion):
    def __init__(self) -> None:
        super().__init__(hidden_width=1, initial_rho=0.1)

    def integrate(
        self,
        y0: torch.Tensor,
        laplacian: torch.Tensor,
        times: torch.Tensor,
        *,
        max_step: float = 0.05,
    ) -> torch.Tensor:
        del laplacian, max_step
        return torch.zeros((times.shape[0], y0.shape[0]), dtype=y0.dtype, device=y0.device)


class StoredTrajectoryModel(NeuralReactionDiffusion):
    def __init__(
        self,
        trajectory: np.ndarray,
        *,
        rho: float,
        growth_rate: float,
        saturation_rate: float,
        baseline_offset: float = 0.0,
    ) -> None:
        super().__init__(hidden_width=1, initial_rho=rho)
        self.trajectory = trajectory.copy()
        self.growth_rate = growth_rate
        self.saturation_rate = saturation_rate
        self.baseline_offset = baseline_offset
        _set_rho(self, rho)

    def integrate(
        self,
        y0: torch.Tensor,
        laplacian: torch.Tensor,
        times: torch.Tensor,
        *,
        max_step: float = 0.05,
    ) -> torch.Tensor:
        del y0, laplacian, max_step
        trajectory = torch.tensor(
            self.trajectory,
            dtype=times.dtype,
            device=times.device,
        )
        trajectory[0] = trajectory[0] + self.baseline_offset
        return trajectory

    def reaction(self, y: torch.Tensor) -> torch.Tensor:
        return self.growth_rate * y - self.saturation_rate * y**2


class LinearReactionStoredTrajectoryModel(StoredTrajectoryModel):
    def reaction(self, y: torch.Tensor) -> torch.Tensor:
        return 2.0 * y


class NonfinitePredictionModel(StoredTrajectoryModel):
    def integrate(
        self,
        y0: torch.Tensor,
        laplacian: torch.Tensor,
        times: torch.Tensor,
        *,
        max_step: float = 0.05,
    ) -> torch.Tensor:
        prediction = super().integrate(y0, laplacian, times, max_step=max_step)
        prediction[-1, 0] = math.inf
        return prediction


def test_make_subject_split_is_deterministic_complete_and_subject_level() -> None:
    first = make_subject_split(12, validation_fraction=0.25, test_fraction=0.25, seed=123)
    second = make_subject_split(12, validation_fraction=0.25, test_fraction=0.25, seed=123)
    different = make_subject_split(12, validation_fraction=0.25, test_fraction=0.25, seed=124)

    assert first == second
    assert first != different
    assert len(first.train_indices) == 6
    assert len(first.validation_indices) == 3
    assert len(first.test_indices) == 3

    all_indices = first.train_indices + first.validation_indices + first.test_indices
    assert sorted(all_indices) == list(range(12))
    assert len(set(first.train_indices).intersection(first.validation_indices)) == 0
    assert len(set(first.train_indices).intersection(first.test_indices)) == 0
    assert len(set(first.validation_indices).intersection(first.test_indices)) == 0
    assert max(all_indices) < 12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_subjects": 2},
        {"n_subjects": True},
        {"n_subjects": 4.5},
        {"validation_fraction": 0.0},
        {"validation_fraction": -0.1},
        {"validation_fraction": math.nan},
        {"test_fraction": 0.0},
        {"test_fraction": math.inf},
        {"validation_fraction": 0.6, "test_fraction": 0.4},
        {"seed": True},
        {"seed": -1},
    ],
)
def test_make_subject_split_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    parameters = {
        "n_subjects": 5,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "seed": 42,
    }
    parameters.update(kwargs)

    with pytest.raises(SchemaError):
        make_subject_split(**parameters)  # type: ignore[arg-type]


def test_subject_split_rejects_empty_duplicate_and_overlapping_indices() -> None:
    with pytest.raises(SchemaError, match="nonempty"):
        SubjectSplit(train_indices=(), validation_indices=(1,), test_indices=(2,))
    with pytest.raises(SchemaError, match="duplicate"):
        SubjectSplit(train_indices=(0, 0), validation_indices=(1,), test_indices=(2,))
    with pytest.raises(SchemaError, match="overlap"):
        SubjectSplit(train_indices=(0,), validation_indices=(0,), test_indices=(2,))


def test_make_subject_split_does_not_change_numpy_global_rng_state() -> None:
    np.random.seed(123)
    expected_draws = np.random.random(5)
    np.random.seed(123)

    make_subject_split(12, validation_fraction=0.25, test_fraction=0.25, seed=999)

    np.testing.assert_array_equal(np.random.random(5), expected_draws)


def test_neural_training_config_default_constructs_successfully() -> None:
    config = NeuralTrainingConfig()

    assert config.hidden_width == 16
    assert config.dtype is torch.float64
    assert config.device == "cpu"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_width": 0},
        {"hidden_width": True},
        {"initial_rho": 0.0},
        {"initial_rho": math.inf},
        {"learning_rate": 0.0},
        {"learning_rate": math.nan},
        {"weight_decay": -1e-3},
        {"max_epochs": 0},
        {"max_epochs": 1.2},
        {"patience": 0},
        {"patience": True},
        {"min_delta": -1e-7},
        {"max_step": 0.0},
        {"max_step": math.inf},
        {"seed": True},
        {"seed": -1},
        {"dtype": torch.int64},
        {"dtype": "float64"},
        {"device": "not-a-real-device"},
    ],
)
def test_neural_training_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(SchemaError):
        NeuralTrainingConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_neural_training_config_rejects_unavailable_cuda() -> None:
    with pytest.raises(SchemaError, match="CUDA"):
        NeuralTrainingConfig(device="cuda")


def test_training_returns_canonical_model_and_valid_histories() -> None:
    result = fit_noise_free_recovery_model(_base_dataset(), _small_split(), _small_config())

    assert isinstance(result, NeuralTrainingResult)
    assert isinstance(result.model, NeuralReactionDiffusion)
    assert result.split == _small_split()
    assert result.epochs_run == len(result.training_losses) == len(result.validation_losses)
    assert 1 <= result.epochs_run <= 4
    assert 0 <= result.best_epoch < result.epochs_run
    assert all(math.isfinite(value) and value >= 0.0 for value in result.training_losses)
    assert all(math.isfinite(value) and value >= 0.0 for value in result.validation_losses)


def test_training_takes_optimizer_step_that_changes_model_parameters() -> None:
    config = _small_config(max_epochs=2, patience=2, seed=11)
    initial_state = _state_dict_for_seed(config)

    result = fit_noise_free_recovery_model(_base_dataset(), _small_split(), config)

    assert any(
        not torch.allclose(result.model.state_dict()[name], initial_value)
        for name, initial_value in initial_state.items()
    )


def test_returned_model_matches_best_validation_epoch() -> None:
    dataset = _base_dataset()
    split = _small_split()
    config = _small_config(max_epochs=5, patience=5, min_delta=0.0)

    result = fit_noise_free_recovery_model(dataset, split, config)

    with torch.no_grad():
        restored_validation_loss = training._split_trajectory_loss(
            result.model,
            dataset,
            split.validation_indices,
            max_step=config.max_step,
            device=torch.device(config.device),
            dtype=config.dtype,
        )

    assert result.best_epoch == min(
        range(result.epochs_run),
        key=lambda index: result.validation_losses[index],
    )
    assert restored_validation_loss.item() == pytest.approx(min(result.validation_losses))


def test_early_stopping_activates_when_validation_does_not_improve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoOpAdam:
        def __init__(self, params: object, **_kwargs: object) -> None:
            self.params = list(params)

        def zero_grad(self) -> None:
            for parameter in self.params:
                parameter.grad = None

        def step(self) -> None:
            return None

    monkeypatch.setattr(training.torch.optim, "Adam", NoOpAdam)

    result = fit_noise_free_recovery_model(
        _base_dataset(),
        _small_split(),
        _small_config(max_epochs=5, patience=2),
    )

    assert result.best_epoch == 0
    assert result.epochs_run == 3
    assert result.validation_losses[0] == pytest.approx(result.validation_losses[1])
    assert result.validation_losses[1] == pytest.approx(result.validation_losses[2])


def test_training_does_not_access_test_subject_data() -> None:
    trajectories = list(_base_dataset().trajectories)
    poisoned_test = trajectories[3].copy()
    poisoned_test[-1, 0] = math.inf
    trajectories[3] = poisoned_test
    dataset = _base_dataset(trajectories=tuple(trajectories))

    result = fit_noise_free_recovery_model(dataset, _small_split(), _small_config(max_epochs=2))

    assert result.epochs_run == 2


def test_private_loss_excludes_baseline_and_weights_subjects_equally() -> None:
    trajectories = (
        np.array([[1000.0, -1000.0], [2.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
    )
    dataset = _base_dataset(trajectories=trajectories)
    model = ZeroIntegratorModel().double()

    loss = training._split_trajectory_loss(
        model,
        dataset,
        (0, 1),
        max_step=0.05,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert loss.item() == pytest.approx((2.0 + 0.5) / 2.0)


def test_training_does_not_mutate_recovery_dataset() -> None:
    dataset = _base_dataset()
    snapshot = _copy_dataset_arrays(dataset)

    fit_noise_free_recovery_model(dataset, _small_split(), _small_config(max_epochs=2))

    _assert_dataset_arrays_equal(dataset, snapshot)


def test_training_does_not_use_ground_truth_parameters_in_loss() -> None:
    dataset = _base_dataset(rho=0.1, growth_rate=0.5, saturation_rate=0.5)
    changed_metadata = _base_dataset(rho=9.0, growth_rate=3.0, saturation_rate=2.0)
    split = _small_split()
    config = _small_config(max_epochs=3, seed=21)

    first = fit_noise_free_recovery_model(dataset, split, config)
    second = fit_noise_free_recovery_model(changed_metadata, split, config)

    assert first.training_losses == pytest.approx(second.training_losses)
    assert first.validation_losses == pytest.approx(second.validation_losses)
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name])


def test_training_import_has_no_pysr_or_legacy_side_effects() -> None:
    code = (
        "import sys\n"
        "import tau_sr.training\n"
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


def test_training_gradients_and_parameters_are_finite() -> None:
    dataset = _base_dataset()
    config = _small_config(max_epochs=2)
    model = training._new_model(
        config,
        device=torch.device(config.device),
        dtype=config.dtype,
    )

    loss = training._split_trajectory_loss(
        model,
        dataset,
        _small_split().train_indices,
        max_step=config.max_step,
        device=torch.device(config.device),
        dtype=config.dtype,
    )
    loss.backward()

    assert model.raw_rho.grad is not None
    assert torch.isfinite(model.raw_rho.grad).all()
    for parameter in model.reaction_eq.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    result = fit_noise_free_recovery_model(dataset, _small_split(), config)
    for parameter in result.model.parameters():
        assert torch.isfinite(parameter).all()


def test_nonfinite_training_subject_raises_clear_error() -> None:
    trajectories = list(_base_dataset().trajectories)
    poisoned_train = trajectories[0].copy()
    poisoned_train[-1, 0] = math.inf
    trajectories[0] = poisoned_train
    dataset = _base_dataset(trajectories=tuple(trajectories))

    with pytest.raises(SchemaError, match="finite"):
        fit_noise_free_recovery_model(dataset, _small_split(), _small_config(max_epochs=1))


def test_training_is_reproducible_for_same_seed_and_changes_for_different_seed() -> None:
    dataset = _base_dataset()
    split = _small_split()
    first = fit_noise_free_recovery_model(dataset, split, _small_config(seed=101))
    second = fit_noise_free_recovery_model(dataset, split, _small_config(seed=101))
    different = fit_noise_free_recovery_model(dataset, split, _small_config(seed=102))

    assert first.training_losses == pytest.approx(second.training_losses, abs=1e-15)
    assert first.validation_losses == pytest.approx(second.validation_losses, abs=1e-15)
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name], rtol=1e-12, atol=1e-12)

    assert any(
        not torch.allclose(value, different.model.state_dict()[name])
        for name, value in first.model.state_dict().items()
    )


def test_training_preserves_global_numpy_and_torch_rng_state() -> None:
    np.random.seed(123)
    expected_numpy = np.random.random(5)
    np.random.seed(123)
    torch.manual_seed(456)
    expected_torch = torch.rand(5)
    torch.manual_seed(456)

    fit_noise_free_recovery_model(_base_dataset(), _small_split(), _small_config(max_epochs=2))

    np.testing.assert_array_equal(np.random.random(5), expected_numpy)
    torch.testing.assert_close(torch.rand(5), expected_torch)


def test_evaluate_neural_recovery_perfect_prediction_has_zero_errors() -> None:
    dataset = _base_dataset()
    model = StoredTrajectoryModel(
        dataset.trajectories[3],
        rho=dataset.rho,
        growth_rate=dataset.growth_rate,
        saturation_rate=dataset.saturation_rate,
    ).double()
    grid = np.linspace(0.0, 1.0, 5)

    metrics = evaluate_neural_recovery(
        model,
        dataset,
        (3,),
        grid,
        max_step=0.05,
        dtype=torch.float64,
    )

    assert isinstance(metrics, NeuralRecoveryMetrics)
    assert metrics.subject_indices == (3,)
    assert metrics.trajectory_mse == pytest.approx(0.0)
    assert metrics.trajectory_rmse == pytest.approx(0.0)
    assert metrics.trajectory_mae == pytest.approx(0.0)
    assert metrics.estimated_rho == pytest.approx(dataset.rho)
    assert metrics.true_rho == pytest.approx(dataset.rho)
    assert metrics.rho_absolute_error == pytest.approx(0.0, abs=1e-8)
    assert metrics.rho_relative_error == pytest.approx(0.0, abs=1e-7)
    assert metrics.reaction_curve_rmse == pytest.approx(0.0)
    assert metrics.reaction_curve_mae == pytest.approx(0.0)


def test_evaluation_excludes_baseline_error_and_restores_model_mode() -> None:
    dataset = _base_dataset()
    snapshot = _copy_dataset_arrays(dataset)
    model = StoredTrajectoryModel(
        dataset.trajectories[0],
        rho=dataset.rho,
        growth_rate=dataset.growth_rate,
        saturation_rate=dataset.saturation_rate,
        baseline_offset=999.0,
    ).double()
    model.train()

    metrics = evaluate_neural_recovery(
        model,
        dataset,
        (0,),
        np.linspace(0.0, 1.0, 5),
        max_step=0.05,
        dtype=torch.float64,
    )

    assert metrics.trajectory_mse == pytest.approx(0.0)
    assert model.training is True
    _assert_dataset_arrays_equal(dataset, snapshot)


def test_evaluate_neural_recovery_reports_rho_and_reaction_errors() -> None:
    dataset = _base_dataset(rho=0.1, growth_rate=1.0, saturation_rate=0.0)
    model = LinearReactionStoredTrajectoryModel(
        dataset.trajectories[0],
        rho=0.2,
        growth_rate=dataset.growth_rate,
        saturation_rate=dataset.saturation_rate,
    ).double()
    grid = np.array([0.0, 0.5, 1.0], dtype=float)

    metrics = evaluate_neural_recovery(
        model,
        dataset,
        (0,),
        grid,
        max_step=0.05,
        dtype=torch.float64,
    )

    assert metrics.trajectory_mse == pytest.approx(0.0)
    assert metrics.estimated_rho == pytest.approx(0.2)
    assert metrics.true_rho == pytest.approx(0.1)
    assert metrics.rho_absolute_error == pytest.approx(0.1)
    assert metrics.rho_relative_error == pytest.approx(1.0)
    assert metrics.reaction_curve_rmse == pytest.approx(math.sqrt(np.mean(grid**2)))
    assert metrics.reaction_curve_mae == pytest.approx(np.mean(np.abs(grid)))


@pytest.mark.parametrize(
    "grid",
    [
        np.array([[0.0], [1.0]], dtype=float),
        np.array([], dtype=float),
        np.array([0.0, math.inf], dtype=float),
        np.array([0.0, math.nan], dtype=float),
        np.array([1.0, 0.0], dtype=float),
        np.array([0, 1]),
    ],
)
def test_evaluate_neural_recovery_rejects_invalid_reaction_grid(grid: np.ndarray) -> None:
    dataset = _base_dataset()
    model = StoredTrajectoryModel(
        dataset.trajectories[0],
        rho=dataset.rho,
        growth_rate=dataset.growth_rate,
        saturation_rate=dataset.saturation_rate,
    ).double()

    with pytest.raises(SchemaError):
        evaluate_neural_recovery(
            model,
            dataset,
            (0,),
            grid,
            max_step=0.05,
            dtype=torch.float64,
        )


def test_evaluate_neural_recovery_rejects_nonfinite_predictions() -> None:
    dataset = _base_dataset()
    model = NonfinitePredictionModel(
        dataset.trajectories[0],
        rho=dataset.rho,
        growth_rate=dataset.growth_rate,
        saturation_rate=dataset.saturation_rate,
    ).double()

    with pytest.raises(SchemaError, match="finite"):
        evaluate_neural_recovery(
            model,
            dataset,
            (0,),
            np.linspace(0.0, 1.0, 5),
            max_step=0.05,
            dtype=torch.float64,
        )


def test_evaluate_neural_recovery_rejects_invalid_subjects_and_zero_true_rho() -> None:
    dataset = _base_dataset()
    model = StoredTrajectoryModel(
        dataset.trajectories[0],
        rho=dataset.rho,
        growth_rate=dataset.growth_rate,
        saturation_rate=dataset.saturation_rate,
    ).double()

    with pytest.raises(SchemaError, match="outside"):
        evaluate_neural_recovery(
            model,
            dataset,
            (4,),
            np.linspace(0.0, 1.0, 5),
            max_step=0.05,
            dtype=torch.float64,
        )
    with pytest.raises(SchemaError, match="dataset.rho"):
        evaluate_neural_recovery(
            model,
            _base_dataset(rho=0.0),
            (0,),
            np.linspace(0.0, 1.0, 5),
            max_step=0.05,
            dtype=torch.float64,
        )
