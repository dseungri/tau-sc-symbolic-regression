"""Development-only full-cohort robustness cross-validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from tau_sr.cohort_adapter import (
    DEFAULT_COHORT_LAPLACIAN_KIND,
    _build_cohort_components,
    _cohort_fingerprint,
    _json_safe,
    _match_subject_files,
)
from tau_sr.cohort_experiment import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_PYSR_SHUTDOWN_GRACE_SECONDS,
    DEFAULT_PYSR_TIMEOUT_SECONDS,
    EXPECTED_TEST_COUNT,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VALIDATION_COUNT,
    FROZEN_SPLIT_SEED,
    CohortExperimentConfig,
    ParametricFitResult,
    _parametric_loss,
    _torch_batch,
    _TorchParametricReactionDiffusion,
    evaluate_model_forecasts,
    forecast_neural,
    forecast_parametric,
    forecast_persistence,
    forecast_symbolic,
    run_symbolic_distillation,
    symbolic_reaction_from_expression,
    write_private_json,
)
from tau_sr.identifiability import ZeroAnchoredNeuralReactionDiffusion, environment_versions
from tau_sr.pilot_adapter import LongitudinalTrajectoryDataset
from tau_sr.pysr_lifecycle import parse_json_file
from tau_sr.schemas import SchemaError
from tau_sr.training import (
    NeuralTrainingConfig,
    SubjectSplit,
    fit_fixed_epoch_recovery_model,
    fit_recovery_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_MODELS: tuple[str, ...] = (
    "persistence",
    "ridge",
    "diffusion",
    "linear",
    "quadratic",
    "neural",
    "symbolic",
)
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("linear_vs_persistence", "linear", "persistence"),
    ("quadratic_vs_linear", "quadratic", "linear"),
    ("neural_vs_linear", "neural", "linear"),
    ("symbolic_vs_neural", "symbolic", "neural"),
    ("ridge_vs_persistence", "ridge", "persistence"),
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("diffusion_vs_persistence", "diffusion", "persistence"),
    ("quadratic_vs_persistence", "quadratic", "persistence"),
    ("neural_vs_quadratic", "neural", "quadratic"),
    ("neural_vs_ridge", "neural", "ridge"),
    ("linear_vs_ridge", "linear", "ridge"),
    ("quadratic_vs_ridge", "quadratic", "ridge"),
)
RIDGE_ALPHA_GRID: tuple[float, ...] = (
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

ParametricKind = Literal["diffusion", "linear", "quadratic"]
SymbolicRunner = Callable[
    [
        ZeroAnchoredNeuralReactionDiffusion,
        LongitudinalTrajectoryDataset,
        Sequence[int],
        CohortExperimentConfig,
        Path,
        float,
        float,
    ],
    dict[str, Any],
]

__all__ = (
    "DevelopmentCohort",
    "RidgeFit",
    "RobustnessConfig",
    "ROBUSTNESS_MODELS",
    "RIDGE_ALPHA_GRID",
    "build_outer_folds",
    "development_labels_from_manifest",
    "fit_ridge_baseline",
    "inner_train_validation_split",
    "paired_subject_bootstrap",
    "run_development_cross_validation",
    "select_ridge_alpha",
    "validate_private_robustness_output_directory",
)


@dataclass(frozen=True)
class RobustnessConfig:
    outer_folds: int = 5
    master_seed: int = 42
    inner_validation_fraction: float = 0.20
    ridge_alpha_grid: tuple[float, ...] = RIDGE_ALPHA_GRID
    model_config: CohortExperimentConfig = CohortExperimentConfig()
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = 42
    pysr_worker_timeout_seconds: float = DEFAULT_PYSR_TIMEOUT_SECONDS
    pysr_worker_shutdown_grace_seconds: float = DEFAULT_PYSR_SHUTDOWN_GRACE_SECONDS

    def __post_init__(self) -> None:
        _positive_int(self.outer_folds, "outer_folds")
        _nonnegative_int(self.master_seed, "master_seed")
        fraction = _finite_float(self.inner_validation_fraction, "inner_validation_fraction")
        if fraction <= 0.0 or fraction >= 1.0:
            raise SchemaError("inner_validation_fraction must be between 0 and 1.")
        grid = tuple(_positive_float(alpha, "ridge_alpha_grid") for alpha in self.ridge_alpha_grid)
        if not grid:
            raise SchemaError("ridge_alpha_grid must be nonempty.")
        _positive_int(self.bootstrap_resamples, "bootstrap_resamples")
        _nonnegative_int(self.bootstrap_seed, "bootstrap_seed")
        _positive_float(self.pysr_worker_timeout_seconds, "pysr_worker_timeout_seconds")
        _positive_float(
            self.pysr_worker_shutdown_grace_seconds,
            "pysr_worker_shutdown_grace_seconds",
        )
        object.__setattr__(self, "ridge_alpha_grid", grid)
        object.__setattr__(
            self,
            "model_config",
            CohortExperimentConfig(**self.model_config.__dict__),
        )


@dataclass(frozen=True)
class DevelopmentCohort:
    dataset: LongitudinalTrajectoryDataset
    original_split_manifest: dict[str, Any]
    development_label_count: int
    excluded_test_label_count: int
    numerical_records_opened: int
    test_numerical_records_opened: int
    development_fingerprint: str
    source_file_match_count: int


@dataclass(frozen=True)
class RidgeFit:
    alpha: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    coefficients: np.ndarray

    @property
    def parameter_count(self) -> int:
        return int(self.coefficients.size + self.y_mean.size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        matrix = _matrix(x, "x")
        if matrix.shape[1] != self.coefficients.shape[0]:
            raise SchemaError("x has incompatible feature count.")
        standardized = (matrix - self.x_mean) / self.x_scale
        prediction = standardized @ self.coefficients + self.y_mean
        if not np.isfinite(prediction).all():
            raise SchemaError("ridge prediction must be finite.")
        return prediction.astype(float, copy=True)


def run_development_cross_validation(
    structure_dir: str | Path,
    baseline_tau_dir: str | Path,
    followup_tau_dir: str | Path,
    preparation_artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    config: RobustnessConfig | None = None,
    symbolic_runner: SymbolicRunner | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the development-only nested cross-validation robustness phase."""

    checked_config = RobustnessConfig() if config is None else RobustnessConfig(**config.__dict__)
    private_dir = validate_private_robustness_output_directory(output_dir)
    start = time.perf_counter()
    split_manifest = _read_json_object(
        Path(preparation_artifact_dir).expanduser().resolve() / "cohort_split_seed_42.json",
        "split manifest",
    )
    cohort = load_development_cohort(
        structure_dir,
        baseline_tau_dir,
        followup_tau_dir,
        split_manifest,
    )
    if cohort.development_label_count != EXPECTED_TRAIN_COUNT + EXPECTED_VALIDATION_COUNT:
        raise SchemaError("development cohort must contain exactly 130 subjects.")
    if cohort.excluded_test_label_count != EXPECTED_TEST_COUNT:
        raise SchemaError("excluded frozen test subject count must be exactly 33.")
    if cohort.test_numerical_records_opened != 0:
        raise SchemaError("frozen test numerical records were opened.")

    fingerprints = _run_fingerprints(cohort, checked_config)
    checkpoint = _load_resume_checkpoint(private_dir, fingerprints)
    if checkpoint.get("completion_status") == "complete":
        return _read_json_object(private_dir / "cohort_robustness_summary.json", "summary")

    folds = build_outer_folds(
        cohort.development_label_count,
        n_folds=checked_config.outer_folds,
        seed=checked_config.master_seed,
    )
    if any(len(fold) != 26 for fold in folds):
        raise SchemaError("each outer fold must contain exactly 26 development subjects.")
    fold_records: dict[str, Any] = dict(checkpoint.get("folds", {}))
    symbolic = symbolic_runner or _run_symbolic_with_retry

    for fold_index, outer_eval in enumerate(folds):
        key = str(fold_index)
        existing = _mapping(fold_records.get(key), "fold record") if key in fold_records else {}
        if existing.get("completion_status") == "complete":
            continue
        fold_start = time.perf_counter()
        outer_eval_set = set(outer_eval)
        outer_train = tuple(
            index
            for index in range(cohort.development_label_count)
            if index not in outer_eval_set
        )
        inner_train, inner_validation, inner_seed = inner_train_validation_split(
            outer_train,
            outer_eval,
            seed=checked_config.master_seed,
            fold_index=fold_index,
            validation_fraction=checked_config.inner_validation_fraction,
        )
        fold_record: dict[str, Any] = {
            "fold_index": fold_index,
            "outer_evaluation_count": len(outer_eval),
            "outer_training_count": len(outer_train),
            "inner_training_count": len(inner_train),
            "inner_validation_count": len(inner_validation),
            "inner_seed": inner_seed,
            "models": dict(existing.get("models", {})),
            "completion_status": "incomplete",
        }
        fold_records[key] = fold_record
        _emit_progress(progress, f"fold {fold_index + 1}/5 started")
        _write_robustness_checkpoint(private_dir, fingerprints, fold_records, "fold_started")

        _run_persistence_fold(cohort.dataset, outer_eval, private_dir, fold_record)
        _emit_progress(progress, f"fold {fold_index + 1}/5 persistence complete")
        _write_robustness_checkpoint(private_dir, fingerprints, fold_records, "persistence")

        _run_ridge_fold(
            cohort.dataset,
            inner_train,
            inner_validation,
            outer_train,
            outer_eval,
            fold_record,
            private_dir,
            checked_config,
        )
        _emit_progress(progress, f"fold {fold_index + 1}/5 ridge complete")
        _write_robustness_checkpoint(private_dir, fingerprints, fold_records, "ridge")

        parametric_results: dict[str, ParametricFitResult] = {}
        split = SubjectSplit(
            train_indices=inner_train,
            validation_indices=inner_validation,
            test_indices=outer_eval,
        )
        for model_name in ("diffusion", "linear", "quadratic"):
            parametric_results[model_name] = _run_parametric_fold(
                cohort.dataset,
                split,
                outer_train,
                outer_eval,
                model_name,  # type: ignore[arg-type]
                fold_record,
                private_dir,
                checked_config.model_config,
            )
            _emit_progress(progress, f"fold {fold_index + 1}/5 {model_name} complete")
            _write_robustness_checkpoint(private_dir, fingerprints, fold_records, model_name)

        neural_model, neural_payload = _run_neural_fold(
            cohort.dataset,
            split,
            outer_train,
            outer_eval,
            fold_record,
            private_dir,
            checked_config.model_config,
        )
        fold_record["models"]["neural"] = neural_payload
        _emit_progress(progress, f"fold {fold_index + 1}/5 neural complete")
        _write_robustness_checkpoint(private_dir, fingerprints, fold_records, "neural")

        symbolic_payload = _run_symbolic_fold(
            neural_model,
            cohort.dataset,
            outer_train,
            outer_eval,
            fold_index,
            fold_record,
            private_dir,
            checked_config,
            symbolic,
        )
        fold_record["models"]["symbolic"] = symbolic_payload
        _emit_progress(progress, f"fold {fold_index + 1}/5 symbolic complete")
        fold_record["completion_status"] = "complete"
        fold_record["runtime_seconds"] = time.perf_counter() - fold_start
        _write_robustness_checkpoint(private_dir, fingerprints, fold_records, "fold_complete")
        _emit_progress(progress, f"fold {fold_index + 1}/5 complete")
        del neural_model, parametric_results

    summary = _finalize_robustness_summary(
        cohort,
        folds,
        fold_records,
        private_dir,
        fingerprints,
        checked_config,
        runtime_seconds=time.perf_counter() - start,
    )
    write_private_json(private_dir / "cohort_robustness_summary.json", summary)
    _write_robustness_checkpoint(private_dir, fingerprints, fold_records, "complete")
    _emit_progress(progress, "robustness cross-validation complete")
    return summary


def load_development_cohort(
    structure_dir: str | Path,
    baseline_tau_dir: str | Path,
    followup_tau_dir: str | Path,
    split_manifest: Mapping[str, Any],
) -> DevelopmentCohort:
    triplets, file_counts = _match_subject_files(structure_dir, baseline_tau_dir, followup_tau_dir)
    dev_labels, test_labels = development_labels_from_manifest(split_manifest, len(triplets))
    label_to_triplet = {
        f"Cohort Subject {position:03d}": triplet
        for position, triplet in enumerate(triplets, start=1)
    }
    try:
        dev_triplets = tuple(label_to_triplet[label] for label in dev_labels)
    except KeyError as exc:
        raise SchemaError("development labels are not available in matched files.") from exc
    if any(label in test_labels for label in dev_labels):
        raise SchemaError("development labels overlap frozen test labels.")
    dev_counts = {
        "structural_connectivity_files": len(dev_triplets),
        "baseline_tau_files": len(dev_triplets),
        "followup_tau_files": len(dev_triplets),
        "duplicate_key_count": file_counts["duplicate_key_count"],
        "unexpected_or_malformed_csv_count": file_counts["unexpected_or_malformed_csv_count"],
        "missing_counterpart_count": file_counts["missing_counterpart_count"],
    }
    components = _build_cohort_components(
        dev_triplets,
        file_counts=dev_counts,
        laplacian_kind=DEFAULT_COHORT_LAPLACIAN_KIND,
    )
    return DevelopmentCohort(
        dataset=components.dataset,
        original_split_manifest=dict(split_manifest),
        development_label_count=len(dev_labels),
        excluded_test_label_count=len(test_labels),
        numerical_records_opened=len(dev_triplets) * 3,
        test_numerical_records_opened=0,
        development_fingerprint=_fingerprint(
            {"labels": dev_labels, "dataset": _cohort_fingerprint(components.dataset)}
        ),
        source_file_match_count=len(triplets),
    )


def development_labels_from_manifest(
    split_manifest: Mapping[str, Any],
    n_subjects: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if split_manifest.get("seed") != FROZEN_SPLIT_SEED:
        raise SchemaError("split manifest must use the frozen seed.")
    labels = _mapping(split_manifest.get("subject_labels"), "subject_labels")
    train = _label_tuple(labels.get("train"), "train labels")
    validation = _label_tuple(labels.get("validation"), "validation labels")
    test = _label_tuple(labels.get("test"), "test labels")
    expected = tuple(f"Cohort Subject {position:03d}" for position in range(1, n_subjects + 1))
    expected_set = set(expected)
    if set(train) | set(validation) | set(test) != expected_set:
        raise SchemaError("split manifest labels must cover matched subjects exactly.")
    if (
        (set(train) & set(validation))
        or (set(train) & set(test))
        or (set(validation) & set(test))
    ):
        raise SchemaError("split manifest groups must be disjoint.")
    development_group = set(train) | set(validation)
    development = tuple(label for label in expected if label in development_group)
    return development, tuple(label for label in expected if label in set(test))


def build_outer_folds(
    n_subjects: int,
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[tuple[int, ...], ...]:
    _positive_int(n_subjects, "n_subjects")
    _positive_int(n_folds, "n_folds")
    _nonnegative_int(seed, "seed")
    if n_subjects % n_folds != 0:
        raise SchemaError("n_subjects must divide evenly across outer folds.")
    rng = np.random.default_rng(seed)
    shuffled = tuple(int(index) for index in rng.permutation(n_subjects))
    fold_size = n_subjects // n_folds
    folds = tuple(
        tuple(sorted(shuffled[start : start + fold_size]))
        for start in range(0, n_subjects, fold_size)
    )
    combined = tuple(index for fold in folds for index in fold)
    if set(combined) != set(range(n_subjects)) or len(combined) != len(set(combined)):
        raise SchemaError("outer folds must cover each development subject exactly once.")
    return folds


def inner_train_validation_split(
    outer_train_indices: Sequence[int],
    outer_eval_indices: Sequence[int],
    *,
    seed: int,
    fold_index: int,
    validation_fraction: float = 0.20,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    outer_train = _index_tuple(outer_train_indices, "outer_train_indices")
    outer_eval = _index_tuple(outer_eval_indices, "outer_eval_indices")
    if set(outer_train) & set(outer_eval):
        raise SchemaError("inner split cannot include outer-evaluation subjects.")
    _nonnegative_int(seed, "seed")
    _nonnegative_int(fold_index, "fold_index")
    fraction = _finite_float(validation_fraction, "validation_fraction")
    if fraction <= 0.0 or fraction >= 1.0:
        raise SchemaError("validation_fraction must be between 0 and 1.")
    inner_seed = seed + 1009 * (fold_index + 1)
    rng = np.random.default_rng(inner_seed)
    shuffled = tuple(int(index) for index in rng.permutation(outer_train))
    validation_count = max(1, math.floor(len(outer_train) * fraction))
    validation = tuple(sorted(shuffled[:validation_count]))
    training = tuple(sorted(shuffled[validation_count:]))
    if not training or not validation:
        raise SchemaError("inner split must have nonempty training and validation groups.")
    return training, validation, inner_seed


def fit_ridge_baseline(x: np.ndarray, y: np.ndarray, *, alpha: float) -> RidgeFit:
    x_matrix = _matrix(x, "x")
    y_matrix = _matrix(y, "y")
    if x_matrix.shape != y_matrix.shape:
        raise SchemaError("ridge x and y must have matching shapes.")
    alpha_value = _positive_float(alpha, "alpha")
    x_mean = np.mean(x_matrix, axis=0)
    x_scale = np.std(x_matrix, axis=0)
    x_scale = np.where(x_scale > 0.0, x_scale, 1.0)
    y_mean = np.mean(y_matrix, axis=0)
    x_standardized = (x_matrix - x_mean) / x_scale
    y_centered = y_matrix - y_mean
    gram = x_standardized.T @ x_standardized
    coefficients = np.linalg.solve(
        gram + alpha_value * np.eye(gram.shape[0], dtype=float),
        x_standardized.T @ y_centered,
    )
    return RidgeFit(
        alpha=alpha_value,
        x_mean=x_mean.astype(float, copy=True),
        x_scale=x_scale.astype(float, copy=True),
        y_mean=y_mean.astype(float, copy=True),
        coefficients=coefficients.astype(float, copy=True),
    )


def select_ridge_alpha(
    dataset: LongitudinalTrajectoryDataset,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
) -> dict[str, Any]:
    train = _index_tuple(train_indices, "train_indices")
    validation = _index_tuple(validation_indices, "validation_indices")
    alphas = tuple(_positive_float(alpha, "alpha_grid") for alpha in alpha_grid)
    x_train, y_train = _xy(dataset, train)
    x_validation, y_validation = _xy(dataset, validation)
    records: list[dict[str, float]] = []
    best_alpha = alphas[0]
    best_rmse = math.inf
    for alpha in alphas:
        fit = fit_ridge_baseline(x_train, y_train, alpha=alpha)
        prediction = fit.predict(x_validation)
        rmse = float(np.sqrt(np.mean((prediction - y_validation) ** 2)))
        records.append({"alpha": alpha, "validation_rmse": rmse})
        better = rmse < best_rmse - 1e-15
        tied_with_more_regularization = (
            math.isclose(rmse, best_rmse, abs_tol=1e-15) and alpha > best_alpha
        )
        if better or tied_with_more_regularization:
            best_alpha = alpha
            best_rmse = rmse
    return {
        "selected_alpha": best_alpha,
        "validation_rmse": best_rmse,
        "alpha_records": records,
    }


def paired_subject_bootstrap(
    first_residuals: np.ndarray,
    second_residuals: np.ndarray,
    *,
    metric: Literal["rmse", "mae"],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    first = _residual_matrix(first_residuals, "first_residuals")
    second = _residual_matrix(second_residuals, "second_residuals")
    if first.shape != second.shape:
        raise SchemaError("paired residual matrices must have matching shapes.")
    _positive_int(n_resamples, "n_resamples")
    _nonnegative_int(seed, "seed")
    estimator = _pooled_rmse if metric == "rmse" else _pooled_mae
    observed = estimator(first, np.arange(first.shape[0])) - estimator(
        second,
        np.arange(second.shape[0]),
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled = rng.integers(0, first.shape[0], size=first.shape[0])
        differences[index] = estimator(first, sampled) - estimator(second, sampled)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    reference = estimator(second, np.arange(second.shape[0]))
    return {
        "metric": metric,
        "difference_first_minus_second": float(observed),
        "relative_percent_difference": float(observed / reference * 100.0),
        "ci95": [float(lower), float(upper)],
        "interval_excludes_zero": bool(upper < 0.0 or lower > 0.0),
        "direction": _comparison_direction(float(observed), metric),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def validate_private_robustness_output_directory(output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if _is_relative_to(output, REPO_ROOT):
        raise SchemaError("robustness output directory must be outside the repository.")
    if output.exists() and not output.is_dir():
        raise SchemaError("robustness output location must be a directory.")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    return output


def _run_persistence_fold(
    dataset: LongitudinalTrajectoryDataset,
    outer_eval: Sequence[int],
    private_dir: Path,
    fold_record: dict[str, Any],
) -> None:
    if "persistence" in fold_record["models"]:
        return
    start = time.perf_counter()
    forecasts = forecast_persistence(dataset, outer_eval)
    payload = _evaluate_and_store_model(
        dataset,
        outer_eval,
        forecasts,
        private_dir,
        int(fold_record["fold_index"]),
        "persistence",
        runtime_seconds=time.perf_counter() - start,
    )
    fold_record["models"]["persistence"] = payload


def _run_ridge_fold(
    dataset: LongitudinalTrajectoryDataset,
    inner_train: Sequence[int],
    inner_validation: Sequence[int],
    outer_train: Sequence[int],
    outer_eval: Sequence[int],
    fold_record: dict[str, Any],
    private_dir: Path,
    config: RobustnessConfig,
) -> None:
    if "ridge" in fold_record["models"]:
        return
    start = time.perf_counter()
    selection = select_ridge_alpha(
        dataset,
        inner_train,
        inner_validation,
        config.ridge_alpha_grid,
    )
    x_train, y_train = _xy(dataset, outer_train)
    x_eval, _y_eval = _xy(dataset, outer_eval)
    fit = fit_ridge_baseline(x_train, y_train, alpha=selection["selected_alpha"])
    predictions = fit.predict(x_eval)
    forecasts = {
        dataset.subject_ids[index]: predictions[row].astype(float, copy=True)
        for row, index in enumerate(outer_eval)
    }
    payload = _evaluate_and_store_model(
        dataset,
        outer_eval,
        forecasts,
        private_dir,
        int(fold_record["fold_index"]),
        "ridge",
        runtime_seconds=time.perf_counter() - start,
    )
    payload.update(
        {
            "selected_alpha": selection["selected_alpha"],
            "inner_validation_rmse": selection["validation_rmse"],
            "parameter_count": fit.parameter_count,
        }
    )
    fold_record["models"]["ridge"] = payload


def _run_parametric_fold(
    dataset: LongitudinalTrajectoryDataset,
    split: SubjectSplit,
    outer_train: Sequence[int],
    outer_eval: Sequence[int],
    kind: ParametricKind,
    fold_record: dict[str, Any],
    private_dir: Path,
    config: CohortExperimentConfig,
) -> ParametricFitResult:
    if kind in fold_record["models"]:
        return _parametric_from_payload(kind, fold_record["models"][kind])
    start = time.perf_counter()
    inner = _fit_parametric_inner(dataset, split, kind=kind, config=config)
    if inner.numerical_instability:
        raise SchemaError(f"{kind} inner training was numerically unstable.")
    updates = inner.best_epoch + 1
    refit = _fit_parametric_fixed_updates(
        dataset,
        outer_train,
        kind=kind,
        n_updates=updates,
        config=config,
    )
    if refit.numerical_instability:
        raise SchemaError(f"{kind} refit was numerically unstable.")
    forecasts = forecast_parametric(
        dataset,
        outer_eval,
        rho=refit.rho,
        linear_coefficient=refit.linear_coefficient,
        quadratic_coefficient=refit.quadratic_coefficient,
        max_step=config.max_step,
    )
    payload = _evaluate_and_store_model(
        dataset,
        outer_eval,
        forecasts,
        private_dir,
        int(fold_record["fold_index"]),
        kind,
        runtime_seconds=time.perf_counter() - start,
    )
    payload.update(
        {
            "rho": refit.rho,
            "linear_coefficient": refit.linear_coefficient,
            "quadratic_coefficient": refit.quadratic_coefficient,
            "selected_inner_best_epoch": inner.best_epoch,
            "refit_updates": updates,
            "inner_epochs_run": inner.epochs_run,
            "refit_training_losses": list(refit.training_losses),
        }
    )
    fold_record["models"][kind] = payload
    return refit


def _fit_parametric_inner(
    dataset: LongitudinalTrajectoryDataset,
    split: SubjectSplit,
    *,
    kind: ParametricKind,
    config: CohortExperimentConfig,
) -> ParametricFitResult:
    from tau_sr.cohort_experiment import fit_parametric_model

    return fit_parametric_model(dataset, split, kind=kind, config=config)


def _fit_parametric_fixed_updates(
    dataset: LongitudinalTrajectoryDataset,
    subject_indices: Sequence[int],
    *,
    kind: ParametricKind,
    n_updates: int,
    config: CohortExperimentConfig,
) -> ParametricFitResult:
    _positive_int(n_updates, "n_updates")
    device = torch.device(config.device)
    dtype = config.dtype
    batch = _torch_batch(dataset, subject_indices, device=device, dtype=dtype)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.parametric_seed)
        model = _TorchParametricReactionDiffusion(
            kind=kind,
            initial_rho=config.initial_rho,
            dtype=dtype,
            device=device,
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.parametric_learning_rate)
    losses: list[float] = []
    numerical_instability = False
    start = time.perf_counter()
    for _ in range(n_updates):
        model.train()
        optimizer.zero_grad()
        try:
            loss = _parametric_loss(model, batch, max_step=config.max_step)
            if not torch.isfinite(loss).all().item():
                raise SchemaError("parametric fixed-update loss must be finite.")
            loss.backward()
            for parameter in parameters:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                    raise SchemaError("parametric fixed-update gradients must be finite.")
            optimizer.step()
            for parameter in parameters:
                if not torch.isfinite(parameter).all().item():
                    raise SchemaError("parametric fixed-update parameters must be finite.")
        except SchemaError:
            numerical_instability = True
            break
        losses.append(float(loss.detach().cpu().item()))
    model.eval()
    return ParametricFitResult(
        model_name=kind,
        rho=float(model.rho.detach().cpu().item()),
        linear_coefficient=float(model.linear_coefficient.detach().cpu().item()),
        quadratic_coefficient=float(model.quadratic_coefficient.detach().cpu().item()),
        best_epoch=max(0, len(losses) - 1),
        epochs_run=len(losses),
        training_losses=tuple(losses),
        validation_losses=(),
        numerical_instability=numerical_instability,
        runtime_seconds=time.perf_counter() - start,
    )


def _run_neural_fold(
    dataset: LongitudinalTrajectoryDataset,
    split: SubjectSplit,
    outer_train: Sequence[int],
    outer_eval: Sequence[int],
    fold_record: dict[str, Any],
    private_dir: Path,
    config: CohortExperimentConfig,
) -> tuple[ZeroAnchoredNeuralReactionDiffusion, dict[str, Any]]:
    existing = fold_record["models"].get("neural")
    if existing is not None:
        raise SchemaError("resuming after neural fit requires rerunning the current fold.")
    start = time.perf_counter()
    seed_results: dict[int, dict[str, Any]] = {}
    best_seed: int | None = None
    best_rmse = math.inf
    best_epoch = 0
    for seed in config.neural_seeds:
        result = _fit_zero_anchored_inner(dataset, split, seed=seed, config=config)
        forecasts = forecast_neural(result.model, dataset, split.validation_indices, config=config)
        metrics = evaluate_model_forecasts(dataset, split.validation_indices, forecasts)
        rmse = float(metrics["pooled_rmse"])
        seed_results[int(seed)] = {
            "seed": int(seed),
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "inner_validation_rmse": rmse,
            "learned_rho": float(result.model.rho.detach().cpu().item()),
        }
        better = rmse < best_rmse - 1e-15
        tied_with_smaller_seed = (
            math.isclose(rmse, best_rmse, abs_tol=1e-15)
            and (best_seed is None or seed < best_seed)
        )
        if better or tied_with_smaller_seed:
            best_seed = int(seed)
            best_rmse = rmse
            best_epoch = result.best_epoch
    if best_seed is None:
        raise SchemaError("neural seed selection failed.")
    updates = best_epoch + 1
    refit_model = _new_zero_anchored_model(best_seed, config)
    training_config = _neural_training_config(config, best_seed, max_epochs=updates, patience=1)
    refit = fit_fixed_epoch_recovery_model(
        dataset,
        outer_train,
        training_config,
        model=refit_model,
        n_epochs=updates,
    )
    if refit.numerical_instability:
        raise SchemaError("neural fixed-update refit was numerically unstable.")
    forecasts = forecast_neural(refit.model, dataset, outer_eval, config=config)
    payload = _evaluate_and_store_model(
        dataset,
        outer_eval,
        forecasts,
        private_dir,
        int(fold_record["fold_index"]),
        "neural",
        runtime_seconds=time.perf_counter() - start,
    )
    payload.update(
        {
            "selected_seed": best_seed,
            "selected_inner_best_epoch": best_epoch,
            "refit_updates": updates,
            "seed_sensitivity": seed_results,
            "rho": float(refit.model.rho.detach().cpu().item()),
            "refit_training_losses": list(refit.training_losses),
        }
    )
    return refit.model, payload


def _fit_zero_anchored_inner(
    dataset: LongitudinalTrajectoryDataset,
    split: SubjectSplit,
    *,
    seed: int,
    config: CohortExperimentConfig,
):
    model = _new_zero_anchored_model(seed, config)
    training_config = _neural_training_config(config, seed)
    return fit_recovery_model(dataset, split, training_config, model=model)


def _new_zero_anchored_model(
    seed: int,
    config: CohortExperimentConfig,
) -> ZeroAnchoredNeuralReactionDiffusion:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ZeroAnchoredNeuralReactionDiffusion(
            hidden_width=config.hidden_width,
            initial_rho=config.initial_rho,
        )
    return model.to(device=torch.device(config.device), dtype=config.dtype)


def _neural_training_config(
    config: CohortExperimentConfig,
    seed: int,
    *,
    max_epochs: int | None = None,
    patience: int | None = None,
) -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        hidden_width=config.hidden_width,
        initial_rho=config.initial_rho,
        learning_rate=config.learning_rate,
        rho_learning_rate=config.rho_learning_rate,
        weight_decay=0.0,
        max_epochs=config.max_epochs if max_epochs is None else max_epochs,
        patience=config.patience if patience is None else patience,
        min_delta=config.min_delta,
        max_step=config.max_step,
        seed=seed,
        dtype=config.dtype,
        device=config.device,
    )


def _run_symbolic_fold(
    neural_model: ZeroAnchoredNeuralReactionDiffusion,
    dataset: LongitudinalTrajectoryDataset,
    outer_train: Sequence[int],
    outer_eval: Sequence[int],
    fold_index: int,
    fold_record: dict[str, Any],
    private_dir: Path,
    config: RobustnessConfig,
    symbolic_runner: SymbolicRunner,
) -> dict[str, Any]:
    if "symbolic" in fold_record["models"]:
        return fold_record["models"]["symbolic"]
    start = time.perf_counter()
    symbolic_dir = private_dir / "symbolic" / f"fold_{fold_index:02d}"
    summary = symbolic_runner(
        neural_model,
        dataset,
        outer_train,
        config.model_config,
        symbolic_dir,
        config.pysr_worker_timeout_seconds,
        config.pysr_worker_shutdown_grace_seconds,
    )
    payload = {
        "completed": bool(summary.get("completed")),
        "runtime_seconds": time.perf_counter() - start,
        "fit_domain": summary.get("fit_domain"),
        "evaluation_domain": summary.get("evaluation_domain"),
        "equation": summary.get("equation"),
        "expanded_expression": summary.get("expanded_expression"),
        "complexity": summary.get("complexity"),
        "training_loss": summary.get("training_loss"),
        "metrics": summary.get("metrics"),
        "error_type": summary.get("error_type"),
        "error": summary.get("error"),
        "worker_cleanup_required": summary.get("worker_cleanup_required"),
    }
    if not payload["completed"]:
        return payload
    reaction = symbolic_reaction_from_expression(summary)
    forecasts = forecast_symbolic(
        dataset,
        outer_eval,
        rho=float(neural_model.rho.detach().cpu().item()),
        reaction_fn=reaction,
        max_step=config.model_config.max_step,
    )
    metrics_payload = _evaluate_and_store_model(
        dataset,
        outer_eval,
        forecasts,
        private_dir,
        fold_index,
        "symbolic",
        runtime_seconds=float(payload["runtime_seconds"]),
    )
    return {**metrics_payload, **payload, "completed": True}


def _run_symbolic_with_retry(
    model: ZeroAnchoredNeuralReactionDiffusion,
    dataset: LongitudinalTrajectoryDataset,
    train_indices: Sequence[int],
    config: CohortExperimentConfig,
    output_dir: Path,
    timeout_seconds: float,
    shutdown_grace_seconds: float,
) -> dict[str, Any]:
    first = run_symbolic_distillation(
        model,
        dataset,
        train_indices,
        config=config,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
        shutdown_grace_seconds=shutdown_grace_seconds,
    )
    if first.get("completed") or not first.get("worker_cleanup_required"):
        return first
    return run_symbolic_distillation(
        model,
        dataset,
        train_indices,
        config=config,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
        shutdown_grace_seconds=shutdown_grace_seconds,
    )


def _evaluate_and_store_model(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
    forecasts: Mapping[str, np.ndarray],
    private_dir: Path,
    fold_index: int,
    model_name: str,
    *,
    runtime_seconds: float,
) -> dict[str, Any]:
    metrics = evaluate_model_forecasts(dataset, indices, forecasts)
    prediction_matrix = np.stack([forecasts[dataset.subject_ids[index]] for index in indices])
    target_matrix = np.stack([dataset.trajectories[index][1] for index in indices])
    residual_matrix = prediction_matrix - target_matrix
    _write_npz_private(
        private_dir / "fold_predictions" / f"fold_{fold_index:02d}_{model_name}.npz",
        prediction=prediction_matrix,
        residual=residual_matrix,
        outer_eval_indices=np.asarray(indices, dtype=int),
    )
    return {
        **_aggregate_metrics(metrics),
        "runtime_seconds": float(runtime_seconds),
        "prediction_artifact": f"fold_predictions/fold_{fold_index:02d}_{model_name}.npz",
    }


def _finalize_robustness_summary(
    cohort: DevelopmentCohort,
    folds: Sequence[Sequence[int]],
    fold_records: Mapping[str, Any],
    private_dir: Path,
    fingerprints: Mapping[str, Any],
    config: RobustnessConfig,
    *,
    runtime_seconds: float,
) -> dict[str, Any]:
    predictions, residuals = _assemble_out_of_fold_arrays(
        cohort.dataset,
        fold_records,
        private_dir,
    )
    overall = {
        model: _overall_metrics(residuals[model])
        for model in ROBUSTNESS_MODELS
        if model in residuals
    }
    fold_metrics = _fold_metric_summary(fold_records)
    comparisons = _all_bootstrap_comparisons(
        residuals,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    stability = _fold_stability(fold_records)
    private_arrays = _write_final_arrays(private_dir, predictions, residuals)
    summary = {
        "experiment": "development_only_full_cohort_robustness_cv",
        "completion_status": "complete",
        "repository_commit": _current_commit(),
        "experiment_commit": fingerprints["experiment_commit"],
        "audit_commit": fingerprints["audit_commit"],
        "development_subjects": cohort.development_label_count,
        "excluded_frozen_test_subjects": cohort.excluded_test_label_count,
        "test_firewall": {
            "test_numerical_records_opened": cohort.test_numerical_records_opened,
            "development_numerical_records_opened": cohort.numerical_records_opened,
            "passed": cohort.test_numerical_records_opened == 0,
        },
        "fold_design": {
            "outer_folds": len(folds),
            "outer_fold_size": len(folds[0]),
            "master_seed": config.master_seed,
            "inner_validation_fraction": config.inner_validation_fraction,
            "inner_seed_formula": "master_seed + 1009 * (fold_index + 1)",
            "outer_fold_fingerprint": fingerprints["outer_fold_fingerprint"],
        },
        "configuration": _config_payload(config),
        "fingerprints": fingerprints,
        "folds": fold_records,
        "overall_out_of_fold_metrics": overall,
        "fold_metric_summary": fold_metrics,
        "paired_bootstrap_comparisons": comparisons,
        "fold_stability": stability,
        "private_prediction_artifacts": private_arrays,
        "software_versions": environment_versions(),
        "runtime_seconds": float(runtime_seconds),
        "scientific_conclusion": _scientific_conclusion(comparisons),
    }
    return _json_safe(summary)  # type: ignore[return-value]


def _assemble_out_of_fold_arrays(
    dataset: LongitudinalTrajectoryDataset,
    fold_records: Mapping[str, Any],
    private_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_subjects = len(dataset.subject_ids)
    n_regions = len(dataset.roi_labels)
    predictions: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    filled: dict[str, np.ndarray] = {}
    for model in ROBUSTNESS_MODELS:
        predictions[model] = np.full((n_subjects, n_regions), np.nan, dtype=float)
        residuals[model] = np.full((n_subjects, n_regions), np.nan, dtype=float)
        filled[model] = np.zeros(n_subjects, dtype=bool)
    for record in fold_records.values():
        models = _mapping(record.get("models"), "fold models")
        for model, payload in models.items():
            if model not in predictions or "prediction_artifact" not in payload:
                continue
            artifact_path = private_dir / str(payload["prediction_artifact"])
            with np.load(artifact_path) as arrays:
                indices = np.asarray(arrays["outer_eval_indices"], dtype=int)
                predictions[model][indices] = np.asarray(arrays["prediction"], dtype=float)
                residuals[model][indices] = np.asarray(arrays["residual"], dtype=float)
                filled[model][indices] = True
    completed_predictions = {
        model: value
        for model, value in predictions.items()
        if filled[model].all() and np.isfinite(value).all()
    }
    completed_residuals = {
        model: value
        for model, value in residuals.items()
        if filled[model].all() and np.isfinite(value).all()
    }
    return completed_predictions, completed_residuals


def _all_bootstrap_comparisons(
    residuals: Mapping[str, np.ndarray],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for category, comparisons in (
        ("primary", PRIMARY_COMPARISONS),
        ("secondary", SECONDARY_COMPARISONS),
    ):
        for name, first, second in comparisons:
            if first not in residuals or second not in residuals:
                output[name] = {"category": category, "available": False}
                continue
            output[name] = {
                "category": category,
                "available": True,
                "first_model": first,
                "second_model": second,
                "rmse": paired_subject_bootstrap(
                    residuals[first],
                    residuals[second],
                    metric="rmse",
                    n_resamples=n_resamples,
                    seed=seed,
                ),
                "mae": paired_subject_bootstrap(
                    residuals[first],
                    residuals[second],
                    metric="mae",
                    n_resamples=n_resamples,
                    seed=seed,
                ),
            }
    return output


def _fold_metric_summary(fold_records: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in ROBUSTNESS_MODELS:
        rmse: list[float] = []
        mae: list[float] = []
        for record in fold_records.values():
            payload = _mapping(record.get("models"), "fold models").get(model)
            if isinstance(payload, Mapping) and "pooled_rmse" in payload:
                rmse.append(float(payload["pooled_rmse"]))
                mae.append(float(payload["pooled_mae"]))
        if rmse:
            output[model] = {
                "fold_rmse": rmse,
                "fold_mae": mae,
                "mean_fold_rmse": float(np.mean(rmse)),
                "std_fold_rmse": float(np.std(rmse)),
                "mean_fold_mae": float(np.mean(mae)),
                "std_fold_mae": float(np.std(mae)),
            }
    return output


def _fold_stability(fold_records: Mapping[str, Any]) -> dict[str, Any]:
    lower_than_persistence = {model: 0 for model in ROBUSTNESS_MODELS if model != "persistence"}
    quadratic_lower_linear = 0
    neural_lower_linear = 0
    selected_neural_seeds: dict[str, int] = {}
    selected_ridge_alphas: dict[str, float] = {}
    rho_values: dict[str, list[float]] = {
        "diffusion": [],
        "linear": [],
        "quadratic": [],
        "neural": [],
    }
    coefficients: dict[str, list[float]] = {
        "linear": [],
        "quadratic_linear": [],
        "quadratic_quadratic": [],
    }
    for fold_key, record in fold_records.items():
        models = _mapping(record.get("models"), "fold models")
        persistence = models.get("persistence")
        if isinstance(persistence, Mapping):
            p_rmse = float(persistence["pooled_rmse"])
            for model, payload in models.items():
                if (
                    model != "persistence"
                    and isinstance(payload, Mapping)
                    and "pooled_rmse" in payload
                ):
                    lower_than_persistence[model] += int(float(payload["pooled_rmse"]) < p_rmse)
        if (
            isinstance(models.get("quadratic"), Mapping)
            and isinstance(models.get("linear"), Mapping)
        ):
            quadratic_lower_linear += int(
                float(models["quadratic"]["pooled_rmse"]) < float(models["linear"]["pooled_rmse"])
            )
        if (
            isinstance(models.get("neural"), Mapping)
            and isinstance(models.get("linear"), Mapping)
        ):
            neural_lower_linear += int(
                float(models["neural"]["pooled_rmse"]) < float(models["linear"]["pooled_rmse"])
            )
        if isinstance(models.get("neural"), Mapping):
            selected_neural_seeds[fold_key] = int(models["neural"]["selected_seed"])
        if isinstance(models.get("ridge"), Mapping):
            selected_ridge_alphas[fold_key] = float(models["ridge"]["selected_alpha"])
        for model in ("diffusion", "linear", "quadratic", "neural"):
            if isinstance(models.get(model), Mapping) and "rho" in models[model]:
                rho_values[model].append(float(models[model]["rho"]))
        if isinstance(models.get("linear"), Mapping):
            coefficients["linear"].append(float(models["linear"]["linear_coefficient"]))
        if isinstance(models.get("quadratic"), Mapping):
            coefficients["quadratic_linear"].append(float(models["quadratic"]["linear_coefficient"]))
            coefficients["quadratic_quadratic"].append(
                float(models["quadratic"]["quadratic_coefficient"])
            )
    return {
        "folds_lower_than_persistence": lower_than_persistence,
        "quadratic_lower_than_linear_folds": quadratic_lower_linear,
        "neural_lower_than_linear_folds": neural_lower_linear,
        "selected_neural_seed_by_fold": selected_neural_seeds,
        "selected_ridge_alpha_by_fold": selected_ridge_alphas,
        "rho_summary": {model: _summary(values) for model, values in rho_values.items() if values},
        "coefficient_summary": {
            name: _summary(values) for name, values in coefficients.items() if values
        },
    }


def _write_final_arrays(
    private_dir: Path,
    predictions: Mapping[str, np.ndarray],
    residuals: Mapping[str, np.ndarray],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for model, prediction in predictions.items():
        if model not in residuals:
            continue
        relative = Path("out_of_fold_arrays") / f"{model}.npz"
        _write_npz_private(
            private_dir / relative,
            prediction=prediction,
            residual=residuals[model],
        )
        paths[model] = str(relative)
    return paths


def _overall_metrics(residual: np.ndarray) -> dict[str, Any]:
    values = _residual_matrix(residual, "residual")
    subject_rmse = np.sqrt(np.mean(values**2, axis=1))
    subject_mae = np.mean(np.abs(values), axis=1)
    return {
        "pooled_rmse": float(np.sqrt(np.mean(values**2))),
        "pooled_mae": float(np.mean(np.abs(values))),
        "median_subject_rmse": float(np.median(subject_rmse)),
        "subject_rmse_iqr": [
            float(np.quantile(subject_rmse, 0.25)),
            float(np.quantile(subject_rmse, 0.75)),
        ],
        "median_subject_mae": float(np.median(subject_mae)),
        "subject_count": int(values.shape[0]),
    }


def _run_fingerprints(cohort: DevelopmentCohort, config: RobustnessConfig) -> dict[str, Any]:
    outer = build_outer_folds(
        cohort.development_label_count,
        n_folds=config.outer_folds,
        seed=config.master_seed,
    )
    return {
        "repository_commit": _current_commit(),
        "experiment_commit": _previous_commit("full_cohort_comparison"),
        "audit_commit": _previous_commit("cohort_audit"),
        "implementation_fingerprint": _implementation_fingerprint(),
        "development_fingerprint": cohort.development_fingerprint,
        "dataset_fingerprint": _cohort_fingerprint(cohort.dataset),
        "outer_fold_fingerprint": _fingerprint(outer),
        "configuration_fingerprint": _fingerprint(_config_payload(config)),
    }


def _config_payload(config: RobustnessConfig) -> dict[str, Any]:
    model = config.model_config
    return {
        "outer_folds": config.outer_folds,
        "master_seed": config.master_seed,
        "inner_validation_fraction": config.inner_validation_fraction,
        "ridge_alpha_grid": list(config.ridge_alpha_grid),
        "models": list(ROBUSTNESS_MODELS),
        "model_config": {
            "hidden_width": model.hidden_width,
            "initial_rho": model.initial_rho,
            "learning_rate": model.learning_rate,
            "rho_learning_rate": model.rho_learning_rate,
            "parametric_learning_rate": model.parametric_learning_rate,
            "max_epochs": model.max_epochs,
            "patience": model.patience,
            "min_delta": model.min_delta,
            "max_step": model.max_step,
            "neural_seeds": list(model.neural_seeds),
            "parametric_seed": model.parametric_seed,
            "dtype": str(model.dtype),
            "device": model.device,
            "pysr_seed": model.pysr_seed,
            "pysr_n_samples": model.pysr_n_samples,
            "pysr_niterations": model.pysr_niterations,
            "pysr_populations": model.pysr_populations,
            "pysr_population_size": model.pysr_population_size,
            "pysr_maxsize": model.pysr_maxsize,
        },
    }


def _load_resume_checkpoint(private_dir: Path, fingerprints: Mapping[str, Any]) -> dict[str, Any]:
    path = private_dir / "cohort_robustness_checkpoint.json"
    if not path.exists():
        return {"folds": {}}
    payload = _read_json_object(path, "robustness checkpoint")
    saved = _mapping(payload.get("fingerprints"), "checkpoint fingerprints")
    for key in (
        "repository_commit",
        "development_fingerprint",
        "dataset_fingerprint",
        "outer_fold_fingerprint",
        "configuration_fingerprint",
        "implementation_fingerprint",
    ):
        if saved.get(key) != fingerprints.get(key):
            raise SchemaError("robustness checkpoint fingerprints do not match.")
    return payload


def _write_robustness_checkpoint(
    private_dir: Path,
    fingerprints: Mapping[str, Any],
    fold_records: Mapping[str, Any],
    stage: str,
) -> None:
    serializable = _json_safe(
        {
            "checkpoint_schema": "tau_sr_development_robustness_v1",
            "completion_status": "complete" if stage == "complete" else "incomplete",
            "stage": stage,
            "fingerprints": {
                key: value for key, value in fingerprints.items() if key != "output_dir"
            },
            "folds": fold_records,
        }
    )
    write_private_json(private_dir / "cohort_robustness_checkpoint.json", serializable)


def _write_npz_private(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _aggregate_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
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


def _xy(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    checked = _index_tuple(indices, "indices")
    x = np.stack([dataset.trajectories[index][0] for index in checked])
    y = np.stack([dataset.trajectories[index][1] for index in checked])
    return x.astype(float, copy=True), y.astype(float, copy=True)


def _parametric_from_payload(
    kind: ParametricKind,
    payload: Mapping[str, Any],
) -> ParametricFitResult:
    return ParametricFitResult(
        model_name=kind,
        rho=float(payload["rho"]),
        linear_coefficient=float(payload.get("linear_coefficient", 0.0)),
        quadratic_coefficient=float(payload.get("quadratic_coefficient", 0.0)),
        best_epoch=int(payload["selected_inner_best_epoch"]),
        epochs_run=int(payload["refit_updates"]),
        training_losses=(),
        validation_losses=(),
        numerical_instability=False,
        runtime_seconds=float(payload.get("runtime_seconds", 0.0)),
    )


def _label_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(f"{field_name} must be a sequence.")
    labels = tuple(str(label) for label in value)
    if not labels or any(not label.startswith("Cohort Subject ") for label in labels):
        raise SchemaError(f"{field_name} must contain neutral cohort labels.")
    return labels


def _index_tuple(value: object, field_name: str) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        raw = tuple(value.tolist())
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(f"{field_name} must be a sequence of indices.")
    else:
        raw = tuple(value)
    if not raw:
        raise SchemaError(f"{field_name} must be nonempty.")
    output: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise SchemaError(f"{field_name} must contain integers.")
        if int(item) < 0:
            raise SchemaError(f"{field_name} must contain nonnegative integers.")
        output.append(int(item))
    if len(output) != len(set(output)):
        raise SchemaError(f"{field_name} must not contain duplicates.")
    return tuple(output)


def _matrix(value: object, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise SchemaError(f"{field_name} must be a nonempty two-dimensional matrix.")
    if not np.isfinite(matrix).all():
        raise SchemaError(f"{field_name} must contain finite values.")
    return matrix.astype(float, copy=True)


def _residual_matrix(value: object, field_name: str) -> np.ndarray:
    matrix = _matrix(value, field_name)
    return matrix


def _pooled_rmse(values: np.ndarray, indices: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values[indices] ** 2)))


def _pooled_mae(values: np.ndarray, indices: np.ndarray) -> float:
    return float(np.mean(np.abs(values[indices])))


def _comparison_direction(value: float, metric: str) -> str:
    if value < 0.0:
        return f"first model lower {metric}"
    if value > 0.0:
        return f"second model lower {metric}"
    return "tie"


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _scientific_conclusion(comparisons: Mapping[str, Any]) -> str:
    linear = _mapping(comparisons.get("linear_vs_persistence"), "linear comparison")
    quadratic = _mapping(comparisons.get("quadratic_vs_linear"), "quadratic comparison")
    if linear.get("available") and _mapping(linear["rmse"], "linear rmse")[
        "interval_excludes_zero"
    ]:
        linear_statement = "Linear has paired-bootstrap support versus persistence."
    else:
        linear_statement = "Linear lacks paired-bootstrap support versus persistence."
    if quadratic.get("available") and _mapping(quadratic["rmse"], "quadratic rmse")[
        "interval_excludes_zero"
    ]:
        nonlinear_statement = "Quadratic differs from linear under the paired interval."
    else:
        nonlinear_statement = "Quadratic does not show a supported improvement over linear."
    return f"{linear_statement} {nonlinear_statement}"


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _previous_commit(_label: str) -> str:
    if _label == "cohort_audit":
        return _current_commit()
    result = subprocess.run(
        ["git", "rev-list", "--max-count=1", f"--grep={_label}", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit or _current_commit()


def _implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in (
        "src/tau_sr/cohort_robustness.py",
        "scripts/run_cohort_robustness.py",
    ):
        path = REPO_ROOT / relative
        digest.update(relative.encode("utf-8"))
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_json_object(path: Path, field_name: str) -> dict[str, Any]:
    try:
        payload = parse_json_file(path)
    except Exception as exc:
        raise SchemaError(f"{field_name} could not be read.") from exc
    if not isinstance(payload, dict):
        raise SchemaError(f"{field_name} must contain a JSON object.")
    return payload


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field_name} must be a mapping.")
    return value


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
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
