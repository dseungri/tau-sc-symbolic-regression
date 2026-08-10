"""Full-cohort model comparison with a frozen subject-level split."""

from __future__ import annotations

import importlib.metadata
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from tau_sr.cohort_adapter import (
    _cohort_fingerprint,
    _json_safe,
    inspect_two_timepoint_cohort,
    load_two_timepoint_cohort,
    make_frozen_cohort_split,
    split_subject_labels,
)
from tau_sr.identifiability import ZeroAnchoredNeuralReactionDiffusion, environment_versions
from tau_sr.neural import NeuralReactionDiffusion
from tau_sr.ode import integrate_rk4
from tau_sr.pilot_adapter import LongitudinalTrajectoryDataset
from tau_sr.pysr_lifecycle import (
    PySRWorkerError,
    parse_json_file,
    run_json_worker,
    write_json_atomically,
)
from tau_sr.schemas import SchemaError
from tau_sr.symbolic import (
    ReactionSamples,
    SymbolicRegressionConfig,
    make_midpoint_evaluation_grid,
    sample_neural_reaction,
)
from tau_sr.training import (
    NeuralTrainingConfig,
    NeuralTrainingResult,
    SubjectSplit,
    fit_recovery_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SUBJECT_COUNT = 163
EXPECTED_TRAIN_COUNT = 114
EXPECTED_VALIDATION_COUNT = 16
EXPECTED_TEST_COUNT = 33
FROZEN_SPLIT_SEED = 42
DEFAULT_MAX_STEP = 0.05
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_PYSR_N_SAMPLES = 500
DEFAULT_PYSR_NITERATIONS = 100
DEFAULT_PYSR_POPULATIONS = 15
DEFAULT_PYSR_POPULATION_SIZE = 33
DEFAULT_PYSR_MAXSIZE = 15
DEFAULT_PYSR_TIMEOUT_SECONDS = 1_200.0
DEFAULT_PYSR_SHUTDOWN_GRACE_SECONDS = 10.0

ParametricKind = Literal["diffusion", "linear", "quadratic"]

__all__ = (
    "CohortExperimentConfig",
    "FrozenCohort",
    "ParametricFitResult",
    "bootstrap_paired_rmse",
    "cohort_training_state_domain",
    "evaluate_model_forecasts",
    "fit_parametric_model",
    "forecast_persistence",
    "forecast_parametric",
    "forecast_symbolic",
    "load_completed_checkpoint",
    "load_verified_frozen_cohort",
    "run_full_cohort_experiment",
    "select_neural_result",
    "symbolic_reaction_from_expression",
    "validate_private_output_directory",
    "write_private_json",
)


@dataclass(frozen=True)
class CohortExperimentConfig:
    hidden_width: int = 16
    initial_rho: float = 0.05
    learning_rate: float = 1e-3
    rho_learning_rate: float | None = None
    parametric_learning_rate: float = 1e-2
    max_epochs: int = 1000
    patience: int = 100
    min_delta: float = 1e-7
    max_step: float = DEFAULT_MAX_STEP
    neural_seeds: tuple[int, ...] = (41, 42, 43)
    parametric_seed: int = 42
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = 42
    pysr_seed: int = 42
    pysr_n_samples: int = DEFAULT_PYSR_N_SAMPLES
    pysr_niterations: int = DEFAULT_PYSR_NITERATIONS
    pysr_populations: int = DEFAULT_PYSR_POPULATIONS
    pysr_population_size: int = DEFAULT_PYSR_POPULATION_SIZE
    pysr_maxsize: int = DEFAULT_PYSR_MAXSIZE

    def __post_init__(self) -> None:
        _positive_int(self.hidden_width, "hidden_width")
        _positive_float(self.initial_rho, "initial_rho")
        _positive_float(self.learning_rate, "learning_rate")
        if self.rho_learning_rate is not None:
            _positive_float(self.rho_learning_rate, "rho_learning_rate")
        _positive_float(self.parametric_learning_rate, "parametric_learning_rate")
        _positive_int(self.max_epochs, "max_epochs")
        _positive_int(self.patience, "patience")
        _nonnegative_float(self.min_delta, "min_delta")
        _positive_float(self.max_step, "max_step")
        _positive_int(self.bootstrap_resamples, "bootstrap_resamples")
        _nonnegative_int(self.bootstrap_seed, "bootstrap_seed")
        _nonnegative_int(self.pysr_seed, "pysr_seed")
        _positive_int(self.pysr_n_samples, "pysr_n_samples")
        _positive_int(self.pysr_niterations, "pysr_niterations")
        _positive_int(self.pysr_populations, "pysr_populations")
        _positive_int(self.pysr_population_size, "pysr_population_size")
        _positive_int(self.pysr_maxsize, "pysr_maxsize")
        seeds = tuple(_nonnegative_int(seed, "neural_seeds") for seed in self.neural_seeds)
        if not seeds:
            raise SchemaError("neural_seeds must be nonempty.")
        if str(torch.device(self.device)) != "cpu":
            raise SchemaError("full-cohort experiment is prespecified for CPU.")
        if self.dtype is not torch.float64:
            raise SchemaError("full-cohort experiment is prespecified for torch.float64.")
        object.__setattr__(self, "neural_seeds", seeds)


@dataclass(frozen=True)
class FrozenCohort:
    dataset: LongitudinalTrajectoryDataset
    split: SubjectSplit
    qc_summary: dict[str, Any]
    split_manifest: dict[str, Any]
    cohort_fingerprint: str


@dataclass(frozen=True)
class ParametricFitResult:
    model_name: str
    rho: float
    linear_coefficient: float
    quadratic_coefficient: float
    best_epoch: int
    epochs_run: int
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    numerical_instability: bool
    runtime_seconds: float


class _TorchParametricReactionDiffusion(torch.nn.Module):
    def __init__(
        self,
        *,
        kind: ParametricKind,
        initial_rho: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if kind not in ("diffusion", "linear", "quadratic"):
            raise SchemaError("parametric model kind is unsupported.")
        self.kind = kind
        self.raw_rho = torch.nn.Parameter(
            torch.tensor(_inverse_softplus(initial_rho), dtype=dtype, device=device)
        )
        self.linear_coefficient = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device))
        self.quadratic_coefficient = torch.nn.Parameter(
            torch.tensor(0.0, dtype=dtype, device=device)
        )
        if kind == "diffusion":
            self.linear_coefficient.requires_grad_(False)
            self.quadratic_coefficient.requires_grad_(False)
        elif kind == "linear":
            self.quadratic_coefficient.requires_grad_(False)

    @property
    def rho(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_rho)

    def reaction(self, state: torch.Tensor) -> torch.Tensor:
        reaction = torch.zeros_like(state)
        if self.kind in ("linear", "quadratic"):
            reaction = reaction + self.linear_coefficient * state
        if self.kind == "quadratic":
            reaction = reaction + self.quadratic_coefficient * state**2
        return reaction

    def forward(self, state: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        transport = -self.rho * torch.einsum("bij,bj->bi", laplacian, state)
        derivative = transport + self.reaction(state)
        if not torch.isfinite(derivative).all().item():
            raise SchemaError("parametric derivative must contain finite values.")
        return derivative

    def integrate(
        self,
        y0: torch.Tensor,
        laplacian: torch.Tensor,
        *,
        max_step: float,
    ) -> torch.Tensor:
        y_current = y0
        n_steps = math.ceil(1.0 / max_step)
        h = torch.tensor(1.0 / n_steps, dtype=y0.dtype, device=y0.device)
        for _ in range(n_steps):
            k1 = self.forward(y_current, laplacian)
            k2 = self.forward(y_current + 0.5 * h * k1, laplacian)
            k3 = self.forward(y_current + 0.5 * h * k2, laplacian)
            k4 = self.forward(y_current + h * k3, laplacian)
            y_current = y_current + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not torch.isfinite(y_current).all().item():
                raise SchemaError("parametric integrated state must contain finite values.")
        return y_current


def validate_private_output_directory(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    if _is_relative_to(output_dir, REPO_ROOT):
        try:
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", str(output_dir)],
                cwd=REPO_ROOT,
                check=False,
            )
        except OSError as exc:
            raise SchemaError("could not verify repository-contained output privacy.") from exc
        if ignored.returncode != 0:
            raise SchemaError("private output directory inside the repository must be ignored.")
    if output_dir.exists() and not output_dir.is_dir():
        raise SchemaError("private output location must be a directory.")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    return output_dir


def write_private_json(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    write_json_atomically(destination, _json_safe(payload))
    os.chmod(destination, 0o600)
    parsed = parse_json_file(destination)
    if parsed is None:
        raise SchemaError("private JSON artifact is empty.")
    return destination


def load_completed_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        payload = parse_json_file(path)
    except Exception as exc:
        raise SchemaError("resume checkpoint is unavailable or invalid.") from exc
    if not isinstance(payload, dict):
        raise SchemaError("resume checkpoint must contain a JSON object.")
    if payload.get("completion_status") != "complete":
        raise SchemaError("resume checkpoint is not a completed experiment.")
    return payload


def load_verified_frozen_cohort(
    structure_dir: str | Path,
    baseline_tau_dir: str | Path,
    followup_tau_dir: str | Path,
    preparation_artifact_dir: str | Path,
) -> FrozenCohort:
    prep_dir = Path(preparation_artifact_dir).expanduser()
    qc_saved = _read_private_json(prep_dir / "cohort_qc_summary.json", "QC")
    split_manifest = _read_private_json(prep_dir / "cohort_split_seed_42.json", "split")
    qc_recomputed = _json_safe(
        inspect_two_timepoint_cohort(structure_dir, baseline_tau_dir, followup_tau_dir)
    )
    if qc_saved != qc_recomputed:
        raise SchemaError("cohort QC artifact does not match recomputed QC.")

    dataset = load_two_timepoint_cohort(structure_dir, baseline_tau_dir, followup_tau_dir)
    split = _split_from_manifest(dataset, split_manifest)
    recomputed_split = make_frozen_cohort_split(dataset.subject_ids, seed=FROZEN_SPLIT_SEED)
    recomputed_split_labels = split_subject_labels(dataset.subject_ids, recomputed_split)
    recomputed_labels = {name: list(values) for name, values in recomputed_split_labels.items()}
    if split_manifest.get("subject_labels") != recomputed_labels:
        raise SchemaError("saved split differs from the recomputed frozen split.")

    fingerprint = _cohort_fingerprint(dataset)
    if split_manifest.get("cohort_fingerprint") != fingerprint:
        raise SchemaError("cohort fingerprint changed after preparation.")
    _validate_frozen_prerequisites(qc_saved, split_manifest, dataset, split)
    return FrozenCohort(
        dataset=dataset,
        split=split,
        qc_summary=qc_saved,
        split_manifest=split_manifest,
        cohort_fingerprint=fingerprint,
    )


def forecast_persistence(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
) -> dict[str, np.ndarray]:
    checked = _validate_indices(dataset, indices)
    return {
        dataset.subject_ids[index]: dataset.trajectories[index][0].astype(float, copy=True)
        for index in checked
    }


def fit_parametric_model(
    dataset: LongitudinalTrajectoryDataset,
    split: SubjectSplit,
    *,
    kind: ParametricKind,
    config: CohortExperimentConfig,
) -> ParametricFitResult:
    _validate_dataset_and_split(dataset, split)
    checked_config = CohortExperimentConfig(**config.__dict__)
    device = torch.device(checked_config.device)
    dtype = checked_config.dtype
    train_batch = _torch_batch(dataset, split.train_indices, device=device, dtype=dtype)
    validation_batch = _torch_batch(dataset, split.validation_indices, device=device, dtype=dtype)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(checked_config.parametric_seed)
        model = _TorchParametricReactionDiffusion(
            kind=kind,
            initial_rho=checked_config.initial_rho,
            dtype=dtype,
            device=device,
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise SchemaError("parametric model exposes no trainable parameters.")
    optimizer = torch.optim.Adam(parameters, lr=checked_config.parametric_learning_rate)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation = math.inf
    epochs_without_improvement = 0
    training_losses: list[float] = []
    validation_losses: list[float] = []
    numerical_instability = False
    start = time.perf_counter()

    for epoch in range(checked_config.max_epochs):
        model.train()
        optimizer.zero_grad()
        try:
            training_loss = _parametric_loss(model, train_batch, max_step=checked_config.max_step)
            training_loss.backward()
            _validate_torch_finite(training_loss, "parametric training loss")
            _validate_gradients(parameters)
            optimizer.step()
            _validate_parameters(parameters)
            model.eval()
            with torch.no_grad():
                validation_loss = _parametric_loss(
                    model,
                    validation_batch,
                    max_step=checked_config.max_step,
                )
            _validate_torch_finite(validation_loss, "parametric validation loss")
        except SchemaError:
            numerical_instability = True
            break

        training_value = float(training_loss.detach().cpu().item())
        validation_value = float(validation_loss.detach().cpu().item())
        training_losses.append(training_value)
        validation_losses.append(validation_value)
        if validation_value < best_validation - checked_config.min_delta:
            best_validation = validation_value
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= checked_config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif not numerical_instability:
        raise SchemaError("parametric training did not produce a validation-best state.")
    runtime_seconds = time.perf_counter() - start
    model.eval()
    return ParametricFitResult(
        model_name=kind,
        rho=float(model.rho.detach().cpu().item()),
        linear_coefficient=float(model.linear_coefficient.detach().cpu().item()),
        quadratic_coefficient=float(model.quadratic_coefficient.detach().cpu().item()),
        best_epoch=best_epoch,
        epochs_run=len(training_losses),
        training_losses=tuple(training_losses),
        validation_losses=tuple(validation_losses),
        numerical_instability=numerical_instability,
        runtime_seconds=runtime_seconds,
    )


def evaluate_model_forecasts(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
    forecasts: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    checked = _validate_indices(dataset, indices)
    residuals: list[np.ndarray] = []
    subject_rmse: list[float] = []
    subject_mae: list[float] = []
    subject_metrics: dict[str, dict[str, float]] = {}
    numerical_instability_count = 0
    for index in checked:
        label = dataset.subject_ids[index]
        if label not in forecasts:
            raise SchemaError("forecast mapping is missing a requested subject.")
        predicted = np.asarray(forecasts[label], dtype=float)
        observed = dataset.trajectories[index][1].astype(float, copy=True)
        if predicted.ndim != 1 or predicted.shape != observed.shape:
            raise SchemaError("forecast vector shape does not match observed follow-up.")
        if not np.isfinite(predicted).all():
            numerical_instability_count += 1
            continue
        residual = predicted - observed
        residuals.append(residual)
        rmse = float(np.sqrt(np.mean(residual**2)))
        mae = float(np.mean(np.abs(residual)))
        subject_rmse.append(rmse)
        subject_mae.append(mae)
        subject_metrics[label] = {
            "rmse": rmse,
            "mae": mae,
            "mean_prediction_error": float(np.mean(residual)),
        }
    if not residuals:
        raise SchemaError("no finite forecasts were available for evaluation.")
    all_residuals = np.concatenate(residuals)
    mse = float(np.mean(all_residuals**2))
    return {
        "pooled_rmse": math.sqrt(mse),
        "pooled_mae": float(np.mean(np.abs(all_residuals))),
        "mean_prediction_error": float(np.mean(all_residuals)),
        "mean_subject_rmse": float(np.mean(subject_rmse)),
        "median_subject_rmse": float(np.median(subject_rmse)),
        "mean_subject_mae": float(np.mean(subject_mae)),
        "median_subject_mae": float(np.median(subject_mae)),
        "numerical_instability_count": numerical_instability_count,
        "subject_metrics": subject_metrics,
    }


def bootstrap_paired_rmse(
    first_residuals: Sequence[np.ndarray],
    second_residuals: Sequence[np.ndarray],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    first = tuple(_vector(residual, "first residual") for residual in first_residuals)
    second = tuple(_vector(residual, "second residual") for residual in second_residuals)
    if len(first) != len(second) or not first:
        raise SchemaError("paired residual collections must have the same nonzero length.")
    _positive_int(n_resamples, "n_resamples")
    seed_value = _nonnegative_int(seed, "seed")
    subject_count = len(first)
    first_subject_rmse = np.asarray([math.sqrt(float(np.mean(values**2))) for values in first])
    second_subject_rmse = np.asarray([math.sqrt(float(np.mean(values**2))) for values in second])
    observed_difference = _pooled_rmse(first, np.arange(subject_count)) - _pooled_rmse(
        second,
        np.arange(subject_count),
    )
    rng = np.random.default_rng(seed_value)
    differences = np.empty(n_resamples, dtype=float)
    for sample_index in range(n_resamples):
        sampled = rng.integers(0, subject_count, size=subject_count)
        differences[sample_index] = _pooled_rmse(first, sampled) - _pooled_rmse(second, sampled)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "rmse_difference_first_minus_second": float(observed_difference),
        "ci95": [float(lower), float(upper)],
        "percent_subjects_improved": float(
            np.mean(first_subject_rmse < second_subject_rmse) * 100.0
        ),
        "direction": _direction(observed_difference),
        "n_resamples": n_resamples,
        "seed": seed_value,
    }


def cohort_training_state_domain(
    dataset: LongitudinalTrajectoryDataset,
    subject_indices: Sequence[int],
) -> tuple[float, float]:
    checked = _validate_indices(dataset, subject_indices)
    values = np.concatenate([dataset.trajectories[index].ravel() for index in checked])
    if values.size == 0 or not np.isfinite(values).all():
        raise SchemaError("training state domain requires finite training values.")
    domain_min = float(values.min())
    domain_max = float(values.max())
    if domain_min >= domain_max:
        raise SchemaError("training state domain is degenerate.")
    return domain_min, domain_max


def select_neural_result(results: Mapping[int, dict[str, Any]]) -> int:
    if not results:
        raise SchemaError("neural results must be nonempty.")
    best_seed: int | None = None
    best_value = math.inf
    for seed, result in results.items():
        validation = result.get("validation_metrics")
        if not isinstance(validation, Mapping):
            raise SchemaError("neural result is missing validation_metrics.")
        value = _finite_float(validation.get("pooled_rmse"), "validation pooled_rmse")
        if value < best_value:
            best_seed = int(seed)
            best_value = value
    if best_seed is None:
        raise SchemaError("no neural seed could be selected.")
    return best_seed


def run_full_cohort_experiment(
    structure_dir: str | Path,
    baseline_tau_dir: str | Path,
    followup_tau_dir: str | Path,
    preparation_artifact_dir: str | Path,
    experiment_output_dir: str | Path,
    *,
    config: CohortExperimentConfig | None = None,
    resume_checkpoint: str | Path | None = None,
    pysr_worker_timeout_seconds: float = DEFAULT_PYSR_TIMEOUT_SECONDS,
    pysr_worker_shutdown_grace_seconds: float = DEFAULT_PYSR_SHUTDOWN_GRACE_SECONDS,
) -> dict[str, Any]:
    if resume_checkpoint is not None:
        return load_completed_checkpoint(resume_checkpoint)
    checked_config = (
        CohortExperimentConfig()
        if config is None
        else CohortExperimentConfig(**config.__dict__)
    )
    output_dir = validate_private_output_directory(experiment_output_dir)
    suite_start = time.perf_counter()
    frozen = load_verified_frozen_cohort(
        structure_dir,
        baseline_tau_dir,
        followup_tau_dir,
        preparation_artifact_dir,
    )
    _write_checkpoint(
        output_dir,
        {"stage": "loaded_verified_cohort", "completion_status": "incomplete"},
    )

    smoke = training_stability_smoke_test(frozen.dataset, frozen.split, checked_config)
    if not smoke["raw_suvr_stable"]:
        raise SchemaError("raw SUVR training smoke test was numerically unstable.")

    model_configs = _model_config_summary(checked_config)
    validation: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    start = time.perf_counter()
    validation["persistence"] = _validation_record(
        frozen.dataset,
        frozen.split.validation_indices,
        forecast_persistence(frozen.dataset, frozen.split.validation_indices),
        runtime_seconds=time.perf_counter() - start,
    )
    _write_checkpoint(
        output_dir,
        {"stage": "persistence_validated", "completion_status": "incomplete"},
    )

    parametric_results: dict[str, ParametricFitResult] = {}
    for kind in ("diffusion", "linear", "quadratic"):
        result = fit_parametric_model(
            frozen.dataset,
            frozen.split,
            kind=kind,
            config=checked_config,
        )
        if result.numerical_instability:
            raise SchemaError(f"{kind} training was numerically unstable.")
        parametric_results[kind] = result
        forecasts = forecast_parametric(
            frozen.dataset,
            frozen.split.validation_indices,
            rho=result.rho,
            linear_coefficient=result.linear_coefficient,
            quadratic_coefficient=result.quadratic_coefficient,
            max_step=checked_config.max_step,
        )
        validation[kind] = _validation_record(
            frozen.dataset,
            frozen.split.validation_indices,
            forecasts,
            runtime_seconds=result.runtime_seconds,
        )
        fitted[kind] = _parametric_result_payload(result)
        _write_checkpoint(output_dir, {"stage": f"{kind}_fit", "completion_status": "incomplete"})

    neural_payloads: dict[int, dict[str, Any]] = {}
    neural_models: dict[int, ZeroAnchoredNeuralReactionDiffusion] = {}
    for seed in checked_config.neural_seeds:
        neural_result, runtime_seconds = _fit_neural_seed(
            frozen.dataset,
            frozen.split,
            checked_config,
            seed,
        )
        forecasts = forecast_neural(
            neural_result.model,
            frozen.dataset,
            frozen.split.validation_indices,
            config=checked_config,
        )
        metrics = evaluate_model_forecasts(
            frozen.dataset,
            frozen.split.validation_indices,
            forecasts,
        )
        payload = {
            "seed": seed,
            "best_epoch": neural_result.best_epoch,
            "epochs_run": neural_result.epochs_run,
            "training_loss_at_best": neural_result.training_losses[neural_result.best_epoch],
            "validation_loss_at_best": neural_result.validation_losses[neural_result.best_epoch],
            "learned_rho": float(neural_result.model.rho.detach().cpu().item()),
            "runtime_seconds": runtime_seconds,
            "validation_metrics": _aggregate_metrics_only(metrics),
        }
        neural_payloads[seed] = payload
        neural_models[seed] = neural_result.model
        _write_checkpoint(
            output_dir,
            {"stage": f"neural_seed_{seed}_fit", "completion_status": "incomplete"},
        )

    selected_seed = select_neural_result(neural_payloads)
    selected_neural_model = neural_models[selected_seed]
    validation["neural"] = neural_payloads[selected_seed]["validation_metrics"]
    fitted["neural"] = {
        "selected_seed": selected_seed,
        "seed_sensitivity": neural_payloads,
    }

    symbolic_summary = run_symbolic_distillation(
        selected_neural_model,
        frozen.dataset,
        frozen.split.train_indices,
        config=checked_config,
        output_dir=output_dir,
        timeout_seconds=pysr_worker_timeout_seconds,
        shutdown_grace_seconds=pysr_worker_shutdown_grace_seconds,
    )
    if not symbolic_summary.get("completed"):
        freeze = _analysis_freeze_payload(
            frozen,
            checked_config,
            model_configs,
            fitted,
            validation,
            symbolic_summary,
            smoke,
            suite_start=suite_start,
            test_not_inspected=True,
        )
        write_private_json(output_dir / "analysis_freeze.json", freeze)
        _write_checkpoint(
            output_dir,
            {
                "stage": "symbolic_blocked_before_test",
                "completion_status": "blocked",
                "reason": symbolic_summary.get("error", "symbolic distillation failed"),
            },
        )
        return {
            **freeze,
            "completion_status": "blocked_before_test",
            "reason": "symbolic_distillation_blocked",
        }

    symbolic_reaction = symbolic_reaction_from_expression(symbolic_summary)
    symbolic_validation_forecasts = forecast_symbolic(
        frozen.dataset,
        frozen.split.validation_indices,
        rho=float(selected_neural_model.rho.detach().cpu().item()),
        reaction_fn=symbolic_reaction,
        max_step=checked_config.max_step,
    )
    validation["symbolic"] = _aggregate_metrics_only(
        evaluate_model_forecasts(
            frozen.dataset,
            frozen.split.validation_indices,
            symbolic_validation_forecasts,
        )
    )

    freeze_payload = _analysis_freeze_payload(
        frozen,
        checked_config,
        model_configs,
        fitted,
        validation,
        symbolic_summary,
        smoke,
        suite_start=suite_start,
        test_not_inspected=True,
    )
    freeze_path = write_private_json(output_dir / "analysis_freeze.json", freeze_payload)
    if not freeze_path.exists():
        raise SchemaError("analysis freeze record was not written.")

    test_results = evaluate_frozen_test_stage(
        frozen.dataset,
        frozen.split,
        parametric_results,
        selected_neural_model,
        symbolic_reaction,
        config=checked_config,
        analysis_freeze_written=True,
    )
    summary = {
        **freeze_payload,
        "completion_status": "complete",
        "test_results": test_results["aggregate_metrics"],
        "subject_level_metrics": test_results["subject_metrics"],
        "bootstrap_comparisons": test_results["bootstrap_comparisons"],
        "linear_versus_nonlinear_conclusion": _nonlinearity_conclusion(test_results),
        "software_versions": {
            **environment_versions(),
            "scipy": _package_version("scipy"),
        },
        "runtime_seconds": time.perf_counter() - suite_start,
    }
    write_private_json(output_dir / "full_cohort_experiment_summary.json", summary)
    _write_checkpoint(output_dir, summary)
    return summary


def training_stability_smoke_test(
    dataset: LongitudinalTrajectoryDataset,
    split: SubjectSplit,
    config: CohortExperimentConfig,
) -> dict[str, Any]:
    start = time.perf_counter()
    batch = _torch_batch(
        dataset,
        split.train_indices[: min(8, len(split.train_indices))],
        device=torch.device(config.device),
        dtype=config.dtype,
    )
    model = _TorchParametricReactionDiffusion(
        kind="linear",
        initial_rho=config.initial_rho,
        dtype=config.dtype,
        device=torch.device(config.device),
    )
    with torch.no_grad():
        loss = _parametric_loss(model, batch, max_step=config.max_step)
    stable = bool(torch.isfinite(loss).item())
    return {
        "raw_suvr_stable": stable,
        "training_subjects_used": int(batch["y0"].shape[0]),
        "runtime_seconds": time.perf_counter() - start,
    }


def forecast_parametric(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
    *,
    rho: float,
    linear_coefficient: float = 0.0,
    quadratic_coefficient: float = 0.0,
    max_step: float = DEFAULT_MAX_STEP,
) -> dict[str, np.ndarray]:
    checked = _validate_indices(dataset, indices)
    return {
        dataset.subject_ids[index]: _integrate_numpy_polynomial(
            dataset.trajectories[index][0],
            dataset.laplacians[index],
            rho=rho,
            linear_coefficient=linear_coefficient,
            quadratic_coefficient=quadratic_coefficient,
            max_step=max_step,
        )
        for index in checked
    }


def forecast_neural(
    model: NeuralReactionDiffusion,
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
    *,
    config: CohortExperimentConfig,
) -> dict[str, np.ndarray]:
    checked = _validate_indices(dataset, indices)
    was_training = model.training
    forecasts: dict[str, np.ndarray] = {}
    try:
        model.eval()
        with torch.no_grad():
            for group in _subject_groups_by_time(dataset, checked):
                y0 = torch.as_tensor(
                    np.stack([dataset.trajectories[index][0] for index in group]),
                    device=torch.device(config.device),
                    dtype=config.dtype,
                )
                laplacian = torch.as_tensor(
                    np.stack([dataset.laplacians[index] for index in group]),
                    device=torch.device(config.device),
                    dtype=config.dtype,
                )
                times = torch.as_tensor(
                    np.asarray(dataset.times_by_subject[group[0]], dtype=float).copy(),
                    device=torch.device(config.device),
                    dtype=config.dtype,
                )
                prediction = model.integrate(y0, laplacian, times, max_step=config.max_step)
                followup = prediction[-1].detach().cpu().numpy()
                for row, index in enumerate(group):
                    forecasts[dataset.subject_ids[index]] = followup[row].astype(float, copy=True)
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
    return forecasts


def forecast_symbolic(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
    *,
    rho: float,
    reaction_fn,
    max_step: float = DEFAULT_MAX_STEP,
) -> dict[str, np.ndarray]:
    checked = _validate_indices(dataset, indices)
    forecasts: dict[str, np.ndarray] = {}
    for index in checked:
        y0 = dataset.trajectories[index][0]
        laplacian = dataset.laplacians[index]

        def rhs(_t: float, state: np.ndarray, *, graph: np.ndarray = laplacian) -> np.ndarray:
            reaction = np.asarray(reaction_fn(state.copy()), dtype=float)
            return -rho * (graph @ state) + reaction

        trajectory = integrate_rk4(rhs, y0, np.asarray([0.0, 1.0], dtype=float), max_step=max_step)
        forecasts[dataset.subject_ids[index]] = trajectory[-1].astype(float, copy=True)
    return forecasts


def run_symbolic_distillation(
    model: ZeroAnchoredNeuralReactionDiffusion,
    dataset: LongitudinalTrajectoryDataset,
    train_indices: Sequence[int],
    *,
    config: CohortExperimentConfig,
    output_dir: Path,
    timeout_seconds: float,
    shutdown_grace_seconds: float,
) -> dict[str, Any]:
    domain_min, domain_max = cohort_training_state_domain(dataset, train_indices)
    samples = sample_neural_reaction(
        model,
        domain_min=domain_min,
        domain_max=domain_max,
        n_samples=config.pysr_n_samples,
        device=config.device,
        dtype=config.dtype,
    )
    symbolic_config = SymbolicRegressionConfig(
        niterations=config.pysr_niterations,
        populations=config.pysr_populations,
        population_size=config.pysr_population_size,
        maxsize=config.pysr_maxsize,
        seed=config.pysr_seed,
        model_selection="best",
        output_directory=str(output_dir / "pysr_artifacts"),
    )
    evaluation_grid = make_midpoint_evaluation_grid(domain_min, domain_max, n_points=1000)
    neural_prediction = _neural_reaction_on_grid(model, evaluation_grid, config=config)
    try:
        payload = _run_isolated_pysr_fit(
            samples=samples,
            symbolic_config=symbolic_config,
            evaluation_grid=evaluation_grid,
            neural_prediction=neural_prediction,
            timeout_seconds=timeout_seconds,
            shutdown_grace_seconds=shutdown_grace_seconds,
        )
    except PySRWorkerError as exc:
        return {
            "completed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "fit_domain": [domain_min, domain_max],
            "worker_cleanup_required": exc.cleanup_required,
            "worker_timed_out": exc.timed_out,
        }
    return {
        "completed": True,
        "fit_domain": payload.get("fit_domain"),
        "evaluation_domain": payload.get("evaluation_domain"),
        "equation": payload.get("equation"),
        "expanded_expression": payload.get("expanded_expression"),
        "complexity": payload.get("complexity"),
        "training_loss": payload.get("training_loss"),
        "artifact_directory": payload.get("artifact_directory"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "pysr_version": payload.get("pysr_version"),
        "julia_version": payload.get("julia_version"),
        "metrics": payload.get("metrics"),
        "polynomial_interpretation": payload.get("polynomial_interpretation"),
        "worker_exited_normally": payload.get("worker_exited_normally"),
        "worker_cleanup_required": payload.get("worker_cleanup_required"),
        "worker_returncode": payload.get("worker_returncode"),
    }


def symbolic_reaction_from_expression(symbolic_summary: Mapping[str, Any]):
    expression = symbolic_summary.get("expanded_expression") or symbolic_summary.get("equation")
    if not isinstance(expression, str) or not expression.strip():
        raise SchemaError("symbolic expression is unavailable.")
    try:
        import sympy as sp

        symbol = sp.Symbol("y")
        parsed = sp.sympify(expression, locals={"y": symbol})
        function = sp.lambdify(symbol, parsed, modules="numpy")
    except Exception as exc:
        raise SchemaError("symbolic expression could not be converted to a reaction.") from exc

    def reaction_fn(y: np.ndarray) -> np.ndarray:
        values = np.asarray(function(np.asarray(y, dtype=float)), dtype=float)
        if values.shape == ():
            values = np.full_like(np.asarray(y, dtype=float), float(values))
        if values.shape != np.asarray(y).shape or not np.isfinite(values).all():
            raise SchemaError("symbolic reaction produced invalid values.")
        return values.astype(float, copy=True)

    return reaction_fn


def evaluate_frozen_test_stage(
    dataset: LongitudinalTrajectoryDataset,
    split: SubjectSplit,
    parametric_results: Mapping[str, ParametricFitResult],
    neural_model: ZeroAnchoredNeuralReactionDiffusion,
    symbolic_reaction_fn,
    *,
    config: CohortExperimentConfig,
    analysis_freeze_written: bool,
) -> dict[str, Any]:
    if not analysis_freeze_written:
        raise SchemaError("analysis freeze must be written before test evaluation.")
    test_indices = split.test_indices
    forecasts: dict[str, dict[str, np.ndarray]] = {
        "persistence": forecast_persistence(dataset, test_indices),
    }
    for name in ("diffusion", "linear", "quadratic"):
        result = parametric_results[name]
        forecasts[name] = forecast_parametric(
            dataset,
            test_indices,
            rho=result.rho,
            linear_coefficient=result.linear_coefficient,
            quadratic_coefficient=result.quadratic_coefficient,
            max_step=config.max_step,
        )
    forecasts["neural"] = forecast_neural(neural_model, dataset, test_indices, config=config)
    forecasts["symbolic"] = forecast_symbolic(
        dataset,
        test_indices,
        rho=float(neural_model.rho.detach().cpu().item()),
        reaction_fn=symbolic_reaction_fn,
        max_step=config.max_step,
    )
    aggregate_metrics: dict[str, Any] = {}
    subject_metrics: dict[str, Any] = {}
    residuals_by_model: dict[str, tuple[np.ndarray, ...]] = {}
    for name, model_forecasts in forecasts.items():
        start = time.perf_counter()
        metrics = evaluate_model_forecasts(dataset, test_indices, model_forecasts)
        aggregate_metrics[name] = {
            **_aggregate_metrics_only(metrics),
            "runtime_seconds": time.perf_counter() - start,
        }
        subject_metrics[name] = metrics["subject_metrics"]
        residuals_by_model[name] = _residuals_for_indices(dataset, test_indices, model_forecasts)
    comparisons = {
        "diffusion_vs_persistence": ("diffusion", "persistence"),
        "linear_vs_persistence": ("linear", "persistence"),
        "quadratic_vs_linear": ("quadratic", "linear"),
        "neural_vs_linear": ("neural", "linear"),
        "neural_vs_quadratic": ("neural", "quadratic"),
        "symbolic_vs_neural": ("symbolic", "neural"),
        "symbolic_vs_linear": ("symbolic", "linear"),
    }
    bootstrap = {
        name: bootstrap_paired_rmse(
            residuals_by_model[first],
            residuals_by_model[second],
            n_resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed,
        )
        for name, (first, second) in comparisons.items()
    }
    return {
        "aggregate_metrics": aggregate_metrics,
        "subject_metrics": subject_metrics,
        "bootstrap_comparisons": bootstrap,
    }


def _fit_neural_seed(
    dataset: LongitudinalTrajectoryDataset,
    split: SubjectSplit,
    config: CohortExperimentConfig,
    seed: int,
) -> tuple[NeuralTrainingResult, float]:
    training_config = NeuralTrainingConfig(
        hidden_width=config.hidden_width,
        initial_rho=config.initial_rho,
        learning_rate=config.learning_rate,
        rho_learning_rate=config.rho_learning_rate,
        weight_decay=0.0,
        max_epochs=config.max_epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        max_step=config.max_step,
        seed=seed,
        dtype=config.dtype,
        device=config.device,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ZeroAnchoredNeuralReactionDiffusion(
            hidden_width=config.hidden_width,
            initial_rho=config.initial_rho,
        )
    start = time.perf_counter()
    result = fit_recovery_model(dataset, split, training_config, model=model)
    return result, time.perf_counter() - start


def _run_isolated_pysr_fit(
    *,
    samples: ReactionSamples,
    symbolic_config: SymbolicRegressionConfig,
    evaluation_grid: np.ndarray,
    neural_prediction: np.ndarray,
    timeout_seconds: float,
    shutdown_grace_seconds: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tau_sr_cohort_pysr_worker_") as worker_dir:
        worker_root = Path(worker_dir)
        request_path = worker_root / "request.json"
        result_path = worker_root / "result.json"
        write_json_atomically(
            request_path,
            {
                "samples": {
                    "input_values": samples.input_values.tolist(),
                    "neural_reaction_values": samples.neural_reaction_values.tolist(),
                    "domain_min": samples.domain_min,
                    "domain_max": samples.domain_max,
                },
                "config": {
                    "niterations": symbolic_config.niterations,
                    "populations": symbolic_config.populations,
                    "population_size": symbolic_config.population_size,
                    "maxsize": symbolic_config.maxsize,
                    "seed": symbolic_config.seed,
                    "model_selection": symbolic_config.model_selection,
                    "output_directory": symbolic_config.output_directory,
                },
                "evaluation": {
                    "grid": np.asarray(evaluation_grid, dtype=float).tolist(),
                    "neural_prediction": np.asarray(neural_prediction, dtype=float).tolist(),
                },
            },
        )
        worker_result = run_json_worker(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_pysr_worker.py"),
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
            result_path,
            timeout_seconds=timeout_seconds,
            shutdown_grace_seconds=shutdown_grace_seconds,
        )
    return {
        **worker_result.payload,
        "worker_exited_normally": worker_result.process_exited_normally,
        "worker_cleanup_required": worker_result.cleanup_required,
        "worker_returncode": worker_result.returncode,
    }


def _integrate_numpy_polynomial(
    y0: np.ndarray,
    laplacian: np.ndarray,
    *,
    rho: float,
    linear_coefficient: float,
    quadratic_coefficient: float,
    max_step: float,
) -> np.ndarray:
    y0_vector = _vector(y0, "y0")
    graph = np.asarray(laplacian, dtype=float)

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        return (
            -rho * (graph @ state)
            + linear_coefficient * state
            + quadratic_coefficient * state**2
        )

    trajectory = integrate_rk4(
        rhs,
        y0_vector,
        np.asarray([0.0, 1.0], dtype=float),
        max_step=max_step,
    )
    return trajectory[-1].astype(float, copy=True)


def _parametric_loss(
    model: _TorchParametricReactionDiffusion,
    batch: Mapping[str, torch.Tensor],
    *,
    max_step: float,
) -> torch.Tensor:
    prediction = model.integrate(batch["y0"], batch["laplacian"], max_step=max_step)
    residual = prediction - batch["target"]
    return torch.mean(torch.mean(residual**2, dim=1))


def _torch_batch(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    checked = _validate_indices(dataset, indices)
    _validate_two_timepoints(dataset, checked)
    return {
        "y0": torch.as_tensor(
            np.stack([dataset.trajectories[index][0] for index in checked]),
            device=device,
            dtype=dtype,
        ),
        "target": torch.as_tensor(
            np.stack([dataset.trajectories[index][1] for index in checked]),
            device=device,
            dtype=dtype,
        ),
        "laplacian": torch.as_tensor(
            np.stack([dataset.laplacians[index] for index in checked]),
            device=device,
            dtype=dtype,
        ),
    }


def _validation_record(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
    forecasts: Mapping[str, np.ndarray],
    *,
    runtime_seconds: float,
) -> dict[str, Any]:
    metrics = evaluate_model_forecasts(dataset, indices, forecasts)
    return {**_aggregate_metrics_only(metrics), "runtime_seconds": runtime_seconds}


def _aggregate_metrics_only(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "pooled_rmse",
            "pooled_mae",
            "mean_prediction_error",
            "mean_subject_rmse",
            "median_subject_rmse",
            "mean_subject_mae",
            "median_subject_mae",
            "numerical_instability_count",
        )
    }


def _parametric_result_payload(result: ParametricFitResult) -> dict[str, Any]:
    return {
        "model_name": result.model_name,
        "rho": result.rho,
        "linear_coefficient": result.linear_coefficient,
        "quadratic_coefficient": result.quadratic_coefficient,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "training_loss_at_best": (
            None if not result.training_losses else result.training_losses[result.best_epoch]
        ),
        "validation_loss_at_best": (
            None if not result.validation_losses else result.validation_losses[result.best_epoch]
        ),
        "numerical_instability": result.numerical_instability,
        "runtime_seconds": result.runtime_seconds,
    }


def _analysis_freeze_payload(
    frozen: FrozenCohort,
    config: CohortExperimentConfig,
    model_configs: Mapping[str, Any],
    fitted: Mapping[str, Any],
    validation: Mapping[str, Any],
    symbolic_summary: Mapping[str, Any],
    smoke: Mapping[str, Any],
    *,
    suite_start: float,
    test_not_inspected: bool,
) -> dict[str, Any]:
    return {
        "experiment": "full_cohort_model_comparison",
        "completion_status": "analysis_frozen",
        "cohort": {
            "subject_count": len(frozen.dataset.subject_ids),
            "region_count": len(frozen.dataset.roi_labels),
            "matched_triplets": frozen.qc_summary["matched_triplets"],
            "cohort_fingerprint": frozen.cohort_fingerprint,
            "time_years": frozen.qc_summary["time_years"],
            "minimum_processed_node_degree": frozen.qc_summary[
                "minimum_processed_node_degree"
            ],
            "baseline_tau": frozen.qc_summary["baseline_tau"],
            "followup_tau": frozen.qc_summary["followup_tau"],
            "mean_tau_change": frozen.qc_summary["mean_tau_change"],
        },
        "split": {
            "seed": FROZEN_SPLIT_SEED,
            "counts": {
                "train": len(frozen.split.train_indices),
                "validation": len(frozen.split.validation_indices),
                "test": len(frozen.split.test_indices),
            },
            "test_set_role": frozen.split_manifest["test_set_role"],
        },
        "model_configurations": model_configs,
        "training_smoke_test": smoke,
        "fitted_models": fitted,
        "validation_results": validation,
        "symbolic": symbolic_summary,
        "analysis_freeze": {
            "test_outcomes_inspected": not test_not_inspected,
            "test_metrics_computed": False,
            "status": "complete_before_test",
            "runtime_seconds_to_freeze": time.perf_counter() - suite_start,
        },
    }


def _model_config_summary(config: CohortExperimentConfig) -> dict[str, Any]:
    return {
        "persistence": {"prediction": "baseline_state"},
        "diffusion": {
            "rho": "shared_nonnegative",
            "linear_coefficient": 0.0,
            "quadratic_coefficient": 0.0,
            "optimizer": "Adam",
            "learning_rate": config.parametric_learning_rate,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
        },
        "linear": {
            "rho": "shared_nonnegative",
            "reaction": "a*y",
            "optimizer": "Adam",
            "learning_rate": config.parametric_learning_rate,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
        },
        "quadratic": {
            "rho": "shared_nonnegative",
            "reaction": "a*y + b*y**2",
            "optimizer": "Adam",
            "learning_rate": config.parametric_learning_rate,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
        },
        "neural": {
            "model": "ZeroAnchoredNeuralReactionDiffusion",
            "reaction": "y*g_theta(y)",
            "hidden_width": config.hidden_width,
            "initial_rho": config.initial_rho,
            "learning_rate": config.learning_rate,
            "rho_learning_rate": config.rho_learning_rate,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "seeds": list(config.neural_seeds),
            "device": config.device,
            "dtype": str(config.dtype),
            "max_step": config.max_step,
        },
        "symbolic": {
            "binary_operators": ["+", "-", "*"],
            "unary_operators": [],
            "maxsize": config.pysr_maxsize,
            "model_selection": "best",
            "seed": config.pysr_seed,
            "n_samples": config.pysr_n_samples,
            "niterations": config.pysr_niterations,
            "populations": config.pysr_populations,
            "population_size": config.pysr_population_size,
            "parallelism": "serial",
        },
    }


def _nonlinearity_conclusion(test_results: Mapping[str, Any]) -> str:
    metrics = test_results["aggregate_metrics"]
    bootstrap = test_results["bootstrap_comparisons"]
    quadratic_diff = bootstrap["quadratic_vs_linear"]["rmse_difference_first_minus_second"]
    neural_diff = bootstrap["neural_vs_linear"]["rmse_difference_first_minus_second"]
    quadratic_ci = bootstrap["quadratic_vs_linear"]["ci95"]
    neural_ci = bootstrap["neural_vs_linear"]["ci95"]
    if quadratic_diff < 0.0 and neural_diff < 0.0 and quadratic_ci[1] < 0.0 and neural_ci[1] < 0.0:
        return (
            "Held-out prediction supports a nonlinear local component under this "
            "model and dataset."
        )
    if metrics["quadratic"]["pooled_rmse"] < metrics["linear"]["pooled_rmse"] and metrics["neural"][
        "pooled_rmse"
    ] < metrics["linear"]["pooled_rmse"]:
        return (
            "Nonlinear models improved point estimates, but bootstrap support should "
            "be interpreted cautiously."
        )
    return (
        "The present cohort does not provide evidence that a nonlinear local reaction "
        "improves one-year prediction over the linear reaction model."
    )


def _neural_reaction_on_grid(
    model: ZeroAnchoredNeuralReactionDiffusion,
    grid: np.ndarray,
    *,
    config: CohortExperimentConfig,
) -> np.ndarray:
    values = torch.as_tensor(
        np.asarray(grid, dtype=float).reshape(-1, 1),
        device=torch.device(config.device),
        dtype=config.dtype,
    )
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            prediction = model.reaction(values)
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
    output = prediction.detach().cpu().numpy().reshape(-1)
    if output.shape != np.asarray(grid).shape or not np.isfinite(output).all():
        raise SchemaError("neural reaction evaluation produced invalid values.")
    return output.copy()


def _split_from_manifest(
    dataset: LongitudinalTrajectoryDataset,
    manifest: Mapping[str, Any],
) -> SubjectSplit:
    if manifest.get("seed") != FROZEN_SPLIT_SEED:
        raise SchemaError("split artifact seed is not the frozen seed.")
    raw_labels = manifest.get("subject_labels")
    if not isinstance(raw_labels, Mapping):
        raise SchemaError("split artifact is missing subject labels.")
    label_to_index = {label: index for index, label in enumerate(dataset.subject_ids)}
    groups: dict[str, tuple[int, ...]] = {}
    for name in ("train", "validation", "test"):
        labels = raw_labels.get(name)
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise SchemaError("split artifact group is invalid.")
        checked_labels = tuple(str(label) for label in labels)
        if any(not _is_neutral_subject_label(label) for label in checked_labels):
            raise SchemaError("split artifact contains non-neutral labels.")
        try:
            groups[name] = tuple(label_to_index[label] for label in checked_labels)
        except KeyError as exc:
            raise SchemaError("split artifact contains labels outside the dataset.") from exc
    return SubjectSplit(
        train_indices=groups["train"],
        validation_indices=groups["validation"],
        test_indices=groups["test"],
    )


def _validate_frozen_prerequisites(
    qc_summary: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    dataset: LongitudinalTrajectoryDataset,
    split: SubjectSplit,
) -> None:
    counts = split_manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise SchemaError("split artifact is missing counts.")
    checks = (
        qc_summary.get("subject_count") == EXPECTED_SUBJECT_COUNT,
        qc_summary.get("matched_triplets") == EXPECTED_SUBJECT_COUNT,
        counts.get("train") == EXPECTED_TRAIN_COUNT,
        counts.get("validation") == EXPECTED_VALIDATION_COUNT,
        counts.get("test") == EXPECTED_TEST_COUNT,
        len(split.train_indices) == EXPECTED_TRAIN_COUNT,
        len(split.validation_indices) == EXPECTED_VALIDATION_COUNT,
        len(split.test_indices) == EXPECTED_TEST_COUNT,
        split_manifest.get("test_set_role") == "held_out_final_evaluation_only",
        qc_summary.get("duplicate_key_count") == 0,
        qc_summary.get("unexpected_or_malformed_csv_count") == 0,
        qc_summary.get("missing_counterpart_count") == 0,
        qc_summary.get("zero_degree_nodes") == 0,
        qc_summary.get("all_processed_matrices_symmetric") is True,
        qc_summary.get("all_processed_diagonals_zero") is True,
        qc_summary.get("time_years") == [0.0, 1.0],
        len(dataset.subject_ids) == EXPECTED_SUBJECT_COUNT,
    )
    if not all(checks):
        raise SchemaError("frozen full-cohort prerequisites differ from the verified phase.")
    _validate_dataset_and_split(dataset, split)


def _validate_dataset_and_split(
    dataset: LongitudinalTrajectoryDataset,
    split: SubjectSplit,
) -> None:
    if not isinstance(dataset, LongitudinalTrajectoryDataset):
        raise SchemaError("dataset must be a LongitudinalTrajectoryDataset.")
    if not isinstance(split, SubjectSplit):
        raise SchemaError("split must be a SubjectSplit.")
    all_indices = split.train_indices + split.validation_indices + split.test_indices
    if set(all_indices) != set(range(len(dataset.subject_ids))):
        raise SchemaError("split must cover the cohort exactly once.")
    _validate_two_timepoints(dataset, all_indices)


def _validate_two_timepoints(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
) -> None:
    for index in indices:
        times = dataset.times_by_subject[index]
        trajectory = dataset.trajectories[index]
        if not np.array_equal(times, np.asarray([0.0, 1.0], dtype=float)):
            raise SchemaError("full-cohort subjects must use one-year time coordinates.")
        if trajectory.shape != (2, len(dataset.roi_labels)):
            raise SchemaError("full-cohort trajectories must have two 160-region rows.")


def _validate_indices(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise SchemaError("subject indices must be a sequence.")
    checked: list[int] = []
    for value in indices:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError("subject indices must be integers.")
        if value < 0 or value >= len(dataset.subject_ids):
            raise SchemaError("subject index is outside the dataset range.")
        checked.append(value)
    if not checked:
        raise SchemaError("subject indices must be nonempty.")
    if len(set(checked)) != len(checked):
        raise SchemaError("subject indices must be unique.")
    return tuple(checked)


def _residuals_for_indices(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
    forecasts: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, ...]:
    return tuple(
        _vector(forecasts[dataset.subject_ids[index]], "forecast")
        - dataset.trajectories[index][1].astype(float, copy=True)
        for index in _validate_indices(dataset, indices)
    )


def _subject_groups_by_time(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    groups: dict[tuple[float, ...], list[int]] = {}
    for index in indices:
        key = tuple(float(value) for value in dataset.times_by_subject[index])
        groups.setdefault(key, []).append(index)
    return tuple(tuple(values) for values in groups.values())


def _pooled_rmse(residuals: Sequence[np.ndarray], indices: np.ndarray) -> float:
    mse = float(np.mean([np.mean(residuals[int(index)] ** 2) for index in indices]))
    return math.sqrt(mse)


def _direction(value: float) -> str:
    if value < 0.0:
        return "first_model_lower_rmse"
    if value > 0.0:
        return "second_model_lower_rmse"
    return "tie"


def _vector(value: object, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise SchemaError(f"{field_name} must be a nonempty one-dimensional vector.")
    if not np.isfinite(array).all():
        raise SchemaError(f"{field_name} must contain finite values.")
    return array.astype(float, copy=True)


def _validate_torch_finite(value: torch.Tensor, field_name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise SchemaError(f"{field_name} must be finite.")


def _validate_gradients(parameters: Sequence[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise SchemaError("model gradients must be finite.")


def _validate_parameters(parameters: Sequence[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        if not torch.isfinite(parameter).all().item():
            raise SchemaError("model parameters must be finite.")


def _read_private_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = parse_json_file(path)
    except Exception as exc:
        raise SchemaError(f"{name} artifact is unavailable or invalid.") from exc
    if not isinstance(payload, dict):
        raise SchemaError(f"{name} artifact must contain a JSON object.")
    return payload


def _write_checkpoint(output_dir: Path, payload: Mapping[str, Any]) -> None:
    checkpoint = {"checkpoint_schema": "tau_sr_full_cohort_v1", **dict(payload)}
    write_private_json(output_dir / "cohort_experiment_checkpoint.json", checkpoint)


def _is_neutral_subject_label(value: str) -> bool:
    return value.startswith("Cohort Subject ") and value.removeprefix("Cohort Subject ").isdigit()


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise SchemaError(f"{field_name} must be numeric.")
    output = float(value)
    if not math.isfinite(output):
        raise SchemaError(f"{field_name} must be finite.")
    return output


def _positive_float(value: object, field_name: str) -> float:
    output = _finite_float(value, field_name)
    if output <= 0.0:
        raise SchemaError(f"{field_name} must be positive.")
    return output


def _nonnegative_float(value: object, field_name: str) -> float:
    output = _finite_float(value, field_name)
    if output < 0.0:
        raise SchemaError(f"{field_name} must be nonnegative.")
    return output


def _positive_int(value: object, field_name: str) -> int:
    output = _nonnegative_int(value, field_name)
    if output < 1:
        raise SchemaError(f"{field_name} must be positive.")
    return output


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SchemaError(f"{field_name} must be an integer.")
    output = int(value)
    if output < 0:
        raise SchemaError(f"{field_name} must be nonnegative.")
    return output


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"
