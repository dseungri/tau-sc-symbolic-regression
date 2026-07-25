"""Controlled neural identifiability diagnostics for synthetic recovery experiments."""

from __future__ import annotations

import importlib.metadata
import math
import platform
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import torch

from tau_sr.neural import NeuralReactionDiffusion
from tau_sr.recovery import SyntheticRecoveryDataset, build_noise_free_recovery_dataset
from tau_sr.schemas import SchemaError
from tau_sr.synthetic import SyntheticCohortConfig
from tau_sr.training import (
    NeuralTrainingConfig,
    NeuralTrainingResult,
    SubjectSplit,
    evaluate_neural_recovery,
    fit_noise_free_recovery_model,
    make_subject_split,
)

__all__ = (
    "BASELINE_TEST_REACTION_RMSE",
    "BASELINE_TEST_RHO",
    "BASELINE_TEST_TRAJECTORY_RMSE",
    "CONTROLLED_TIMES",
    "CONTROLLED_TRUE_GROWTH_RATE",
    "CONTROLLED_TRUE_RHO",
    "CONTROLLED_TRUE_SATURATION_RATE",
    "FixedRhoZeroAnchoredReactionDiffusion",
    "IdentifiabilityExperimentResult",
    "OracleReactionDiffusion",
    "ZeroAnchoredNeuralReactionDiffusion",
    "build_controlled_recovery_dataset",
    "controlled_reaction_grid",
    "controlled_subject_split",
    "environment_versions",
    "fit_identifiability_model",
    "make_baseline_training_config",
    "make_oracle_training_config",
    "make_zero_anchored_training_config",
    "summarize_identifiability_result",
)

CONTROLLED_N_SUBJECTS = 24
CONTROLLED_N_REGIONS = 148
CONTROLLED_TIMES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
CONTROLLED_TRUE_RHO = 0.1
CONTROLLED_TRUE_GROWTH_RATE = 0.5
CONTROLLED_TRUE_SATURATION_RATE = 0.5
CONTROLLED_SEED = 42
BASELINE_TEST_TRAJECTORY_RMSE = 0.00852985777384891
BASELINE_TEST_RHO = 0.06185591017770647
BASELINE_TEST_REACTION_RMSE = 0.012004275151955537


class ZeroAnchoredNeuralReactionDiffusion(NeuralReactionDiffusion):
    """Neural reaction-diffusion model with r_theta(0) fixed at zero.

    The parameterization is r_theta(y) = y * g_theta(y). It preserves a flexible
    nonlinear scalar law while enforcing the physical zero-tau anchor.
    """

    def reaction(self, y: torch.Tensor) -> torch.Tensor:
        state = self._validate_state(y)
        gain = self.reaction_eq(state.unsqueeze(-1)).squeeze(-1)
        reaction = state * gain
        return _validate_torch_tensor(
            reaction,
            "zero-anchored reaction",
            expected_shape=state.shape,
        )


class FixedRhoZeroAnchoredReactionDiffusion(ZeroAnchoredNeuralReactionDiffusion):
    """Diagnostic oracle with fixed diffusion and trainable zero-anchored reaction.

    This model uses the known synthetic rho and is not available for real-data
    fitting, where the diffusion coefficient is unknown.
    """

    def __init__(
        self,
        *,
        fixed_rho: float = CONTROLLED_TRUE_RHO,
        hidden_width: int = 16,
    ) -> None:
        fixed_rho_value = _require_real_scalar(fixed_rho, "fixed_rho", positive=True)
        super().__init__(hidden_width=hidden_width, initial_rho=fixed_rho_value)
        raw_rho = self.raw_rho.detach().clone()
        del self._parameters["raw_rho"]
        self.register_buffer("raw_rho", raw_rho)


class OracleReactionDiffusion(NeuralReactionDiffusion):
    """Diagnostic oracle with exact synthetic reaction and trainable rho.

    This model answers whether rho can be recovered when the reaction law is
    known. It is not a real-data fitting model.
    """

    def __init__(
        self,
        *,
        growth_rate: float = CONTROLLED_TRUE_GROWTH_RATE,
        saturation_rate: float = CONTROLLED_TRUE_SATURATION_RATE,
        initial_rho: float = 0.05,
    ) -> None:
        growth_rate_value = _require_real_scalar(
            growth_rate,
            "growth_rate",
            positive=True,
        )
        saturation_rate_value = _require_real_scalar(
            saturation_rate,
            "saturation_rate",
            minimum=0.0,
        )
        super().__init__(hidden_width=1, initial_rho=initial_rho)
        del self.reaction_eq
        self.growth_rate = growth_rate_value
        self.saturation_rate = saturation_rate_value

    def reaction(self, y: torch.Tensor) -> torch.Tensor:
        state = self._validate_state(y)
        reaction = self.growth_rate * state - self.saturation_rate * state**2
        return _validate_torch_tensor(
            reaction,
            "oracle reaction",
            expected_shape=state.shape,
        )


@dataclass(frozen=True)
class IdentifiabilityExperimentResult:
    experiment_name: str
    candidate_name: str
    model_type: str
    seed: int
    training_config: dict[str, Any]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    epochs_run: int | None
    best_epoch: int | None
    best_validation_loss: float | None
    final_training_loss: float | None
    final_validation_loss: float | None
    validation_trajectory_mse: float | None
    validation_trajectory_rmse: float | None
    validation_trajectory_mae: float | None
    test_trajectory_mse: float | None
    test_trajectory_rmse: float | None
    test_trajectory_mae: float | None
    learned_rho: float | None
    rho_absolute_error: float | None
    rho_relative_error: float | None
    reaction_curve_rmse: float | None
    reaction_curve_mae: float | None
    reaction_at_zero: float | None
    numerical_instability: bool
    runtime_seconds: float
    validation_selected: bool
    test_evaluated: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "candidate_name": self.candidate_name,
            "model_type": self.model_type,
            "seed": self.seed,
            "training_config": dict(self.training_config),
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "test_indices": list(self.test_indices),
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "final_training_loss": self.final_training_loss,
            "final_validation_loss": self.final_validation_loss,
            "validation_trajectory_mse": self.validation_trajectory_mse,
            "validation_trajectory_rmse": self.validation_trajectory_rmse,
            "validation_trajectory_mae": self.validation_trajectory_mae,
            "test_trajectory_mse": self.test_trajectory_mse,
            "test_trajectory_rmse": self.test_trajectory_rmse,
            "test_trajectory_mae": self.test_trajectory_mae,
            "learned_rho": self.learned_rho,
            "rho_absolute_error": self.rho_absolute_error,
            "rho_relative_error": self.rho_relative_error,
            "reaction_curve_rmse": self.reaction_curve_rmse,
            "reaction_curve_mae": self.reaction_curve_mae,
            "reaction_at_zero": self.reaction_at_zero,
            "numerical_instability": self.numerical_instability,
            "runtime_seconds": self.runtime_seconds,
            "validation_selected": self.validation_selected,
            "test_evaluated": self.test_evaluated,
            "error": self.error,
        }


def build_controlled_recovery_dataset() -> SyntheticRecoveryDataset:
    """Recreate the deterministic 24-subject noise-free recovery dataset."""

    times_by_subject = tuple(
        np.array(CONTROLLED_TIMES, dtype=float) for _ in range(CONTROLLED_N_SUBJECTS)
    )
    return build_noise_free_recovery_dataset(
        SyntheticCohortConfig(
            n_subjects=CONTROLLED_N_SUBJECTS,
            n_regions=CONTROLLED_N_REGIONS,
            visits_per_subject=len(CONTROLLED_TIMES),
            seed=CONTROLLED_SEED,
            edge_probability=0.08,
            baseline_tau_min=0.05,
            baseline_tau_max=0.40,
            add_observation_noise=False,
            observation_noise_std=0.0,
        ),
        times_by_subject,
        representation="count",
        laplacian_kind="symmetric_normalized",
        rho=CONTROLLED_TRUE_RHO,
        growth_rate=CONTROLLED_TRUE_GROWTH_RATE,
        saturation_rate=CONTROLLED_TRUE_SATURATION_RATE,
        max_step=0.05,
    )


def controlled_subject_split() -> SubjectSplit:
    """Recreate the fixed 16/4/4 subject split."""

    return make_subject_split(
        CONTROLLED_N_SUBJECTS,
        validation_fraction=4 / CONTROLLED_N_SUBJECTS,
        test_fraction=4 / CONTROLLED_N_SUBJECTS,
        seed=CONTROLLED_SEED,
    )


def controlled_reaction_grid(
    dataset: SyntheticRecoveryDataset,
    split: SubjectSplit,
    *,
    n_points: int = 200,
) -> np.ndarray:
    """Build a reaction grid from training trajectories only."""

    _require_int(n_points, "n_points", minimum=2)
    values = np.concatenate(
        [dataset.trajectories[index].ravel() for index in split.train_indices]
    )
    if not np.isfinite(values).all():
        raise SchemaError("training trajectories must contain only finite values.")
    return np.linspace(float(values.min()), float(values.max()), n_points, dtype=float)


def make_baseline_training_config() -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        hidden_width=16,
        initial_rho=0.05,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=500,
        patience=50,
        min_delta=1e-7,
        max_step=0.05,
        seed=CONTROLLED_SEED,
        dtype=torch.float64,
        device="cpu",
    )


def make_oracle_training_config(
    *,
    max_epochs: int = 1000,
    patience: int = 100,
    rho_learning_rate: float | None = None,
) -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        hidden_width=16,
        initial_rho=0.05,
        learning_rate=1e-3,
        rho_learning_rate=rho_learning_rate,
        weight_decay=0.0,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=1e-7,
        max_step=0.05,
        seed=CONTROLLED_SEED,
        dtype=torch.float64,
        device="cpu",
    )


def make_zero_anchored_training_config(
    *,
    rho_learning_rate: float,
    max_epochs: int = 1000,
    patience: int = 100,
) -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        hidden_width=16,
        initial_rho=0.05,
        learning_rate=1e-3,
        rho_learning_rate=rho_learning_rate,
        weight_decay=0.0,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=1e-7,
        max_step=0.05,
        seed=CONTROLLED_SEED,
        dtype=torch.float64,
        device="cpu",
    )


def fit_identifiability_model(
    dataset: SyntheticRecoveryDataset,
    split: SubjectSplit,
    config: NeuralTrainingConfig,
    model: NeuralReactionDiffusion,
) -> tuple[NeuralTrainingResult, float]:
    """Fit one supplied identifiability model and return runtime seconds."""

    start = time.perf_counter()
    result = fit_noise_free_recovery_model(dataset, split, config, model=model)
    return result, time.perf_counter() - start


def summarize_identifiability_result(
    *,
    experiment_name: str,
    candidate_name: str,
    training_result: NeuralTrainingResult,
    runtime_seconds: float,
    dataset: SyntheticRecoveryDataset,
    split: SubjectSplit,
    config: NeuralTrainingConfig,
    reaction_grid: np.ndarray,
    validation_selected: bool,
    evaluate_test: bool,
) -> IdentifiabilityExperimentResult:
    """Create a machine-readable summary after validation-based model selection."""

    validation_metrics = evaluate_neural_recovery(
        training_result.model,
        dataset,
        split.validation_indices,
        reaction_grid,
        max_step=config.max_step,
        device=config.device,
        dtype=config.dtype,
    )
    test_metrics = (
        evaluate_neural_recovery(
            training_result.model,
            dataset,
            split.test_indices,
            reaction_grid,
            max_step=config.max_step,
            device=config.device,
            dtype=config.dtype,
        )
        if evaluate_test
        else None
    )
    mechanistic_metrics = test_metrics if test_metrics is not None else validation_metrics
    reaction_at_zero = _reaction_at_zero(
        training_result.model,
        device=torch.device(config.device),
        dtype=config.dtype,
    )

    return IdentifiabilityExperimentResult(
        experiment_name=experiment_name,
        candidate_name=candidate_name,
        model_type=type(training_result.model).__name__,
        seed=config.seed,
        training_config=_config_dict(config),
        train_indices=split.train_indices,
        validation_indices=split.validation_indices,
        test_indices=split.test_indices,
        epochs_run=training_result.epochs_run,
        best_epoch=training_result.best_epoch,
        best_validation_loss=min(training_result.validation_losses),
        final_training_loss=training_result.training_losses[-1],
        final_validation_loss=training_result.validation_losses[-1],
        validation_trajectory_mse=validation_metrics.trajectory_mse,
        validation_trajectory_rmse=validation_metrics.trajectory_rmse,
        validation_trajectory_mae=validation_metrics.trajectory_mae,
        test_trajectory_mse=None if test_metrics is None else test_metrics.trajectory_mse,
        test_trajectory_rmse=None if test_metrics is None else test_metrics.trajectory_rmse,
        test_trajectory_mae=None if test_metrics is None else test_metrics.trajectory_mae,
        learned_rho=mechanistic_metrics.estimated_rho,
        rho_absolute_error=mechanistic_metrics.rho_absolute_error,
        rho_relative_error=mechanistic_metrics.rho_relative_error,
        reaction_curve_rmse=mechanistic_metrics.reaction_curve_rmse,
        reaction_curve_mae=mechanistic_metrics.reaction_curve_mae,
        reaction_at_zero=reaction_at_zero,
        numerical_instability=False,
        runtime_seconds=runtime_seconds,
        validation_selected=validation_selected,
        test_evaluated=evaluate_test,
        error=None,
    )


def environment_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    for package in ("pysr", "juliacall", "juliapkg"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _reaction_at_zero(
    model: NeuralReactionDiffusion,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    with torch.no_grad():
        zero = torch.zeros((1, 1), device=device, dtype=dtype)
        reaction = model.reaction(zero)
    _validate_torch_tensor(reaction, "reaction_at_zero", expected_shape=zero.shape)
    return float(reaction.detach().cpu().reshape(-1)[0].item())


def _config_dict(config: NeuralTrainingConfig) -> dict[str, Any]:
    return {
        "hidden_width": config.hidden_width,
        "initial_rho": config.initial_rho,
        "learning_rate": config.learning_rate,
        "rho_learning_rate": config.rho_learning_rate,
        "weight_decay": config.weight_decay,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "min_delta": config.min_delta,
        "max_step": config.max_step,
        "seed": config.seed,
        "dtype": str(config.dtype),
        "device": config.device,
    }


def _validate_torch_tensor(
    value: torch.Tensor,
    field_name: str,
    *,
    expected_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    if tuple(value.shape) != tuple(expected_shape):
        raise SchemaError(
            f"{field_name} has shape {tuple(value.shape)}; expected {tuple(expected_shape)}."
        )
    if not torch.isfinite(value).all().item():
        raise SchemaError(f"{field_name} must contain only finite values.")
    return value


def _require_int(value: object, field_name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field_name} must be an integer.")
    if value < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")


def _require_real_scalar(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise SchemaError(f"{field_name} must be a real numeric scalar.")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise SchemaError(f"{field_name} must be finite.")
    if positive and value_float <= 0.0:
        raise SchemaError(f"{field_name} must be positive.")
    if minimum is not None and value_float < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")
    return value_float
