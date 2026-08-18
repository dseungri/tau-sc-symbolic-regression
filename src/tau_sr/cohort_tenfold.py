"""Participant-level nested 10-fold analysis for the full two-timepoint cohort."""

from __future__ import annotations

import hashlib
import importlib.metadata
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
    _cohort_fingerprint,
    _json_safe,
    inspect_two_timepoint_cohort,
    load_two_timepoint_cohort,
)
from tau_sr.cohort_experiment import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_PYSR_SHUTDOWN_GRACE_SECONDS,
    DEFAULT_PYSR_TIMEOUT_SECONDS,
    EXPECTED_SUBJECT_COUNT,
    CohortExperimentConfig,
    ParametricFitResult,
    _parametric_loss,
    _torch_batch,
    _TorchParametricReactionDiffusion,
    forecast_neural,
    forecast_parametric,
    forecast_persistence,
    forecast_symbolic,
    run_symbolic_distillation,
    symbolic_reaction_from_expression,
    write_private_json,
)
from tau_sr.cohort_robustness import (
    RIDGE_ALPHA_GRID,
    RidgeFit,
    fit_ridge_baseline,
    select_ridge_alpha,
)
from tau_sr.identifiability import ZeroAnchoredNeuralReactionDiffusion, environment_versions
from tau_sr.pilot_adapter import LongitudinalTrajectoryDataset
from tau_sr.pysr_lifecycle import parse_json_file, sha256_file
from tau_sr.schemas import SchemaError
from tau_sr.training import (
    NeuralTrainingConfig,
    SubjectSplit,
    fit_fixed_epoch_recovery_model,
    fit_recovery_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_JSON = REPO_ROOT / "config" / "cohort_tenfold_protocol.json"
TENFOLD_FREEZE_NAME = "cohort_tenfold_freeze.json"
TENFOLD_CHECKPOINT_NAME = "cohort_tenfold_checkpoint.json"
TENFOLD_SUMMARY_NAME = "cohort_tenfold_summary.json"
TENFOLD_FOLD_COUNT = 10
TENFOLD_SEED = 42
INNER_SEED_MULTIPLIER = 1009
TENFOLD_MODELS: tuple[str, ...] = (
    "persistence",
    "ridge",
    "diffusion",
    "linear",
    "quadratic",
    "neural",
    "symbolic",
)
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("quadratic_vs_linear", "quadratic", "linear"),
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("linear_vs_persistence", "linear", "persistence"),
    ("quadratic_vs_persistence", "quadratic", "persistence"),
    ("diffusion_vs_persistence", "diffusion", "persistence"),
    ("ridge_vs_persistence", "ridge", "persistence"),
    ("neural_vs_linear", "neural", "linear"),
    ("neural_vs_quadratic", "neural", "quadratic"),
    ("symbolic_vs_neural", "symbolic", "neural"),
    ("symbolic_vs_linear", "symbolic", "linear"),
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
    "EXPECTED_SUBJECT_COUNT",
    "PRIMARY_COMPARISONS",
    "SECONDARY_COMPARISONS",
    "TENFOLD_MODELS",
    "TenfoldCohort",
    "TenfoldConfig",
    "TenfoldFoldPlan",
    "aggregate_tenfold_results",
    "build_tenfold_outer_folds",
    "execute_tenfold_analysis",
    "freeze_tenfold_protocol",
    "inner_train_validation_split",
    "load_verified_tenfold_cohort",
    "paired_subject_bootstrap",
    "refit_updates_from_best_epoch",
    "run_tenfold_analysis",
    "validate_private_tenfold_output_directory",
)


@dataclass(frozen=True)
class TenfoldConfig:
    """Frozen 10-fold analysis settings."""

    outer_folds: int = TENFOLD_FOLD_COUNT
    master_seed: int = TENFOLD_SEED
    expected_subject_count: int = EXPECTED_SUBJECT_COUNT
    inner_validation_fraction: float = 0.20
    ridge_alpha_grid: tuple[float, ...] = RIDGE_ALPHA_GRID
    model_config: CohortExperimentConfig = CohortExperimentConfig()
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = 42
    pysr_worker_timeout_seconds: float = DEFAULT_PYSR_TIMEOUT_SECONDS
    pysr_worker_shutdown_grace_seconds: float = DEFAULT_PYSR_SHUTDOWN_GRACE_SECONDS

    def __post_init__(self) -> None:
        _positive_int(self.outer_folds, "outer_folds")
        if self.outer_folds != TENFOLD_FOLD_COUNT:
            raise SchemaError("tenfold analysis requires exactly 10 outer folds.")
        _nonnegative_int(self.master_seed, "master_seed")
        _positive_int(self.expected_subject_count, "expected_subject_count")
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
class TenfoldCohort:
    dataset: LongitudinalTrajectoryDataset
    qc_summary: dict[str, Any]
    cohort_fingerprint: str
    source_file_match_count: int


@dataclass(frozen=True)
class TenfoldFoldPlan:
    fold_index: int
    outer_evaluation_indices: tuple[int, ...]
    outer_development_indices: tuple[int, ...]
    inner_training_indices: tuple[int, ...]
    inner_validation_indices: tuple[int, ...]
    inner_seed: int


def load_verified_tenfold_cohort(
    structure_dir: str | Path,
    baseline_tau_dir: str | Path,
    followup_tau_dir: str | Path,
    preparation_artifact_dir: str | Path,
    *,
    expected_subject_count: int = EXPECTED_SUBJECT_COUNT,
) -> TenfoldCohort:
    """Load the complete validated two-timepoint cohort without using old split labels."""

    expected = _positive_int(expected_subject_count, "expected_subject_count")
    prep_dir = Path(preparation_artifact_dir).expanduser().resolve()
    saved_qc = _read_json_object(prep_dir / "cohort_qc_summary.json", "cohort QC")
    recomputed_qc = _json_safe(
        inspect_two_timepoint_cohort(structure_dir, baseline_tau_dir, followup_tau_dir)
    )
    if saved_qc != recomputed_qc:
        raise SchemaError("cohort QC artifact does not match recomputed QC.")

    dataset = load_two_timepoint_cohort(
        structure_dir,
        baseline_tau_dir,
        followup_tau_dir,
        laplacian_kind=DEFAULT_COHORT_LAPLACIAN_KIND,
    )
    if len(dataset.subject_ids) != expected:
        raise SchemaError("tenfold cohort subject count does not match the frozen protocol.")
    if len(dataset.roi_labels) != 160:
        raise SchemaError("tenfold cohort must contain 160 ROI labels.")
    if any(
        tuple(float(value) for value in times) != (0.0, 1.0)
        for times in dataset.times_by_subject
    ):
        raise SchemaError("tenfold cohort must use [0.0, 1.0] time coordinates.")
    fingerprint = _cohort_fingerprint(dataset)

    split_manifest_path = prep_dir / "cohort_split_seed_42.json"
    if split_manifest_path.exists():
        split_manifest = _read_json_object(split_manifest_path, "cohort split manifest")
        saved_fingerprint = split_manifest.get("cohort_fingerprint")
        if saved_fingerprint is not None and saved_fingerprint != fingerprint:
            raise SchemaError("cohort fingerprint changed after preparation.")

    source_file_match_count = int(saved_qc.get("matched_triplets", len(dataset.subject_ids)))
    return TenfoldCohort(
        dataset=dataset,
        qc_summary=dict(saved_qc),
        cohort_fingerprint=fingerprint,
        source_file_match_count=source_file_match_count,
    )


def build_tenfold_outer_folds(
    n_subjects: int,
    *,
    seed: int = TENFOLD_SEED,
) -> tuple[tuple[int, ...], ...]:
    """Create deterministic participant-level outer folds with complete coverage."""

    subject_count = _positive_int(n_subjects, "n_subjects")
    if subject_count < TENFOLD_FOLD_COUNT:
        raise SchemaError("n_subjects must be at least 10 for tenfold analysis.")
    seed_value = _nonnegative_int(seed, "seed")
    rng = np.random.default_rng(seed_value)
    shuffled = np.asarray(rng.permutation(subject_count), dtype=int)
    folds = tuple(
        tuple(sorted(int(index) for index in fold))
        for fold in np.array_split(shuffled, 10)
    )
    _validate_outer_folds(folds, subject_count)
    return folds


def inner_train_validation_split(
    outer_development_indices: Sequence[int],
    outer_evaluation_indices: Sequence[int],
    *,
    seed: int,
    fold_index: int,
    validation_fraction: float = 0.20,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Create one deterministic inner split for all trained paths in an outer fold."""

    outer_development = _index_tuple(outer_development_indices, "outer_development_indices")
    outer_evaluation = _index_tuple(outer_evaluation_indices, "outer_evaluation_indices")
    if set(outer_development) & set(outer_evaluation):
        raise SchemaError("inner split cannot include outer-evaluation participants.")
    seed_value = _nonnegative_int(seed, "seed")
    fold_value = _nonnegative_int(fold_index, "fold_index")
    fraction = _finite_float(validation_fraction, "validation_fraction")
    if fraction <= 0.0 or fraction >= 1.0:
        raise SchemaError("validation_fraction must be between 0 and 1.")
    inner_seed = seed_value + INNER_SEED_MULTIPLIER * (fold_value + 1)
    rng = np.random.default_rng(inner_seed)
    shuffled = tuple(int(index) for index in rng.permutation(outer_development))
    validation_count = max(1, math.floor(len(outer_development) * fraction))
    validation = tuple(sorted(shuffled[:validation_count]))
    training = tuple(sorted(shuffled[validation_count:]))
    if not training or not validation:
        raise SchemaError("inner split must have nonempty training and validation groups.")
    if set(validation) & set(outer_evaluation) or set(training) & set(outer_evaluation):
        raise SchemaError("outer-evaluation participants leaked into inner split.")
    return training, validation, inner_seed


def refit_updates_from_best_epoch(best_epoch: int) -> int:
    """Translate a validation-selected zero-based epoch into refit optimizer updates."""

    return _nonnegative_int(best_epoch, "best_epoch") + 1


def freeze_tenfold_protocol(
    structure_dir: str | Path,
    baseline_tau_dir: str | Path,
    followup_tau_dir: str | Path,
    preparation_artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    config: TenfoldConfig | None = None,
) -> dict[str, Any]:
    """Write or verify the private tenfold analysis-freeze artifact."""

    checked_config = TenfoldConfig() if config is None else TenfoldConfig(**config.__dict__)
    private_dir = validate_private_tenfold_output_directory(output_dir, refuse_completed=True)
    cohort = load_verified_tenfold_cohort(
        structure_dir,
        baseline_tau_dir,
        followup_tau_dir,
        preparation_artifact_dir,
        expected_subject_count=checked_config.expected_subject_count,
    )
    freeze = _freeze_payload(cohort, checked_config)
    freeze_path = private_dir / TENFOLD_FREEZE_NAME
    if freeze_path.exists():
        saved = _read_json_object(freeze_path, "tenfold freeze")
        if saved != freeze:
            raise SchemaError("existing tenfold freeze artifact differs from runtime inputs.")
        return saved
    write_private_json(freeze_path, freeze)
    return freeze


def execute_tenfold_analysis(
    structure_dir: str | Path,
    baseline_tau_dir: str | Path,
    followup_tau_dir: str | Path,
    preparation_artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    config: TenfoldConfig | None = None,
    symbolic_runner: SymbolicRunner | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the frozen nested tenfold analysis and aggregate only when complete."""

    checked_config = TenfoldConfig() if config is None else TenfoldConfig(**config.__dict__)
    private_dir = validate_private_tenfold_output_directory(output_dir)
    cohort = load_verified_tenfold_cohort(
        structure_dir,
        baseline_tau_dir,
        followup_tau_dir,
        preparation_artifact_dir,
        expected_subject_count=checked_config.expected_subject_count,
    )
    expected_freeze = _freeze_payload(cohort, checked_config)
    saved_freeze = _read_json_object(private_dir / TENFOLD_FREEZE_NAME, "tenfold freeze")
    if saved_freeze != expected_freeze:
        raise SchemaError("runtime inputs differ from the frozen tenfold protocol.")

    start = time.perf_counter()
    folds = build_tenfold_outer_folds(
        len(cohort.dataset.subject_ids),
        seed=checked_config.master_seed,
    )
    fold_records = dict(_load_tenfold_checkpoint(private_dir, expected_freeze).get("folds", {}))
    symbolic = symbolic_runner or _run_symbolic_with_retry
    for fold_index, _outer_eval in enumerate(folds):
        key = str(fold_index)
        existing = _mapping(fold_records.get(key), "fold record") if key in fold_records else {}
        if existing.get("completion_status") == "complete":
            continue
        plan = make_tenfold_fold_plan(folds, fold_index, config=checked_config)
        fold_start = time.perf_counter()
        fold_record: dict[str, Any] = {
            "fold_index": fold_index,
            "outer_evaluation_count": len(plan.outer_evaluation_indices),
            "outer_development_count": len(plan.outer_development_indices),
            "inner_training_count": len(plan.inner_training_indices),
            "inner_validation_count": len(plan.inner_validation_indices),
            "inner_seed": plan.inner_seed,
            "models": dict(existing.get("models", {})),
            "completion_status": "incomplete",
        }
        fold_records[key] = fold_record
        _emit_progress(progress, f"fold {fold_index + 1}/10 started")
        _write_tenfold_checkpoint(private_dir, expected_freeze, fold_records, "fold_started")

        _run_persistence_fold(cohort.dataset, plan, private_dir, fold_record)
        _write_tenfold_checkpoint(private_dir, expected_freeze, fold_records, "persistence")

        _run_ridge_fold(cohort.dataset, plan, private_dir, fold_record, checked_config)
        _write_tenfold_checkpoint(private_dir, expected_freeze, fold_records, "ridge")

        split = SubjectSplit(
            train_indices=plan.inner_training_indices,
            validation_indices=plan.inner_validation_indices,
            test_indices=plan.outer_evaluation_indices,
        )
        for model_name in ("diffusion", "linear", "quadratic"):
            _run_parametric_fold(
                cohort.dataset,
                plan,
                split,
                model_name,  # type: ignore[arg-type]
                private_dir,
                fold_record,
                checked_config.model_config,
            )
            _write_tenfold_checkpoint(private_dir, expected_freeze, fold_records, model_name)

        neural_model = _run_neural_fold(
            cohort.dataset,
            plan,
            split,
            private_dir,
            fold_record,
            checked_config.model_config,
        )
        _write_tenfold_checkpoint(private_dir, expected_freeze, fold_records, "neural")

        _run_symbolic_fold(
            neural_model,
            cohort.dataset,
            plan,
            private_dir,
            fold_record,
            checked_config,
            symbolic,
        )
        _write_tenfold_checkpoint(private_dir, expected_freeze, fold_records, "symbolic")
        fold_record["completion_status"] = "complete"
        fold_record["runtime_seconds"] = time.perf_counter() - fold_start
        _write_tenfold_checkpoint(private_dir, expected_freeze, fold_records, "fold_complete")
        _emit_progress(progress, f"fold {fold_index + 1}/10 complete")
        del neural_model

    summary = aggregate_tenfold_results(
        cohort.dataset,
        folds,
        fold_records,
        private_dir,
        expected_freeze,
        checked_config,
        runtime_seconds=time.perf_counter() - start,
    )
    write_private_json(private_dir / TENFOLD_SUMMARY_NAME, summary)
    _write_tenfold_checkpoint(private_dir, expected_freeze, fold_records, "complete")
    _write_tenfold_figures(private_dir, summary)
    return summary


def run_tenfold_analysis(
    structure_dir: str | Path,
    baseline_tau_dir: str | Path,
    followup_tau_dir: str | Path,
    preparation_artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    freeze_only: bool = False,
    execute: bool = False,
    aggregate_only: bool = False,
    audit: bool = False,
    config: TenfoldConfig | None = None,
    symbolic_runner: SymbolicRunner | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """CLI-friendly staged tenfold runner."""

    if sum(bool(value) for value in (freeze_only, execute, aggregate_only, audit)) != 1:
        raise SchemaError("choose exactly one tenfold stage.")
    checked_config = TenfoldConfig() if config is None else TenfoldConfig(**config.__dict__)
    private_dir = validate_private_tenfold_output_directory(output_dir)
    if freeze_only:
        return freeze_tenfold_protocol(
            structure_dir,
            baseline_tau_dir,
            followup_tau_dir,
            preparation_artifact_dir,
            private_dir,
            config=checked_config,
        )
    if execute:
        return execute_tenfold_analysis(
            structure_dir,
            baseline_tau_dir,
            followup_tau_dir,
            preparation_artifact_dir,
            private_dir,
            config=checked_config,
            symbolic_runner=symbolic_runner,
            progress=progress,
        )
    if aggregate_only:
        cohort = load_verified_tenfold_cohort(
            structure_dir,
            baseline_tau_dir,
            followup_tau_dir,
            preparation_artifact_dir,
            expected_subject_count=checked_config.expected_subject_count,
        )
        freeze = _read_json_object(private_dir / TENFOLD_FREEZE_NAME, "tenfold freeze")
        folds = build_tenfold_outer_folds(
            len(cohort.dataset.subject_ids),
            seed=checked_config.master_seed,
        )
        checkpoint = _load_tenfold_checkpoint(private_dir, freeze)
        summary = aggregate_tenfold_results(
            cohort.dataset,
            folds,
            _mapping(checkpoint.get("folds"), "checkpoint folds"),
            private_dir,
            freeze,
            checked_config,
            runtime_seconds=float(checkpoint.get("runtime_seconds", 0.0)),
        )
        write_private_json(private_dir / TENFOLD_SUMMARY_NAME, summary)
        _write_tenfold_figures(private_dir, summary)
        return summary

    from tau_sr.cohort_tenfold_audit import audit_tenfold_experiment

    return audit_tenfold_experiment(private_dir)


def make_tenfold_fold_plan(
    folds: Sequence[Sequence[int]],
    fold_index: int,
    *,
    config: TenfoldConfig,
) -> TenfoldFoldPlan:
    checked_folds = tuple(_index_tuple(fold, "outer fold") for fold in folds)
    _validate_outer_folds(checked_folds, config.expected_subject_count)
    fold_value = _nonnegative_int(fold_index, "fold_index")
    if fold_value >= len(checked_folds):
        raise SchemaError("fold_index is out of range.")
    outer_eval = checked_folds[fold_value]
    outer_eval_set = set(outer_eval)
    outer_development = tuple(
        index for index in range(config.expected_subject_count) if index not in outer_eval_set
    )
    inner_train, inner_validation, inner_seed = inner_train_validation_split(
        outer_development,
        outer_eval,
        seed=config.master_seed,
        fold_index=fold_value,
        validation_fraction=config.inner_validation_fraction,
    )
    return TenfoldFoldPlan(
        fold_index=fold_value,
        outer_evaluation_indices=outer_eval,
        outer_development_indices=outer_development,
        inner_training_indices=inner_train,
        inner_validation_indices=inner_validation,
        inner_seed=inner_seed,
    )


def aggregate_tenfold_results(
    dataset: LongitudinalTrajectoryDataset,
    folds: Sequence[Sequence[int]],
    fold_records: Mapping[str, Any],
    private_dir: str | Path,
    freeze: Mapping[str, Any],
    config: TenfoldConfig,
    *,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Aggregate complete out-of-fold prediction artifacts."""

    private_path = Path(private_dir).expanduser().resolve()
    checked_folds = tuple(_index_tuple(fold, "outer fold") for fold in folds)
    _validate_outer_folds(checked_folds, len(dataset.subject_ids))
    if len(checked_folds) != 10:
        raise SchemaError("tenfold aggregation requires all 10 folds.")
    records = _mapping(fold_records, "fold_records")
    predictions, residuals, targets, artifact_count = assemble_out_of_fold_arrays(
        dataset,
        checked_folds,
        records,
        private_path,
    )
    overall = {
        model: overall_prediction_metrics(residuals[model], targets[model])
        for model in TENFOLD_MODELS
    }
    fold_metric_summary = _fold_metric_summary(records)
    comparisons = all_bootstrap_comparisons(
        residuals,
        n_resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    stability = _fold_stability(records)
    arrays = _write_final_arrays(private_path, predictions, residuals, targets)
    summary = {
        "experiment": "full_cohort_nested_tenfold_cv",
        "completion_status": "complete",
        "repository_commit": _current_commit(),
        "freeze_fingerprint": _fingerprint(freeze),
        "cohort": {
            "subject_count": len(dataset.subject_ids),
            "region_count": len(dataset.roi_labels),
            "time_years": [0.0, 1.0],
            "cohort_fingerprint": freeze.get("cohort_fingerprint"),
        },
        "fold_design": {
            "outer_folds": len(checked_folds),
            "outer_fold_sizes": [len(fold) for fold in checked_folds],
            "outer_fold_size_distribution": _fold_size_distribution(checked_folds),
            "master_seed": config.master_seed,
            "inner_validation_fraction": config.inner_validation_fraction,
            "inner_seed_formula": "master_seed + 1009 * (fold_index + 1)",
            "fold_fingerprint": _fingerprint(checked_folds),
        },
        "configuration": config_payload(config),
        "folds": _fold_records_without_membership(records),
        "overall_out_of_fold_metrics": overall,
        "fold_metric_summary": fold_metric_summary,
        "paired_bootstrap_comparisons": comparisons,
        "fold_stability": stability,
        "private_prediction_artifacts": arrays,
        "prediction_artifact_count": artifact_count,
        "software_versions": environment_versions(),
        "runtime_seconds": float(runtime_seconds),
        "scientific_conclusion": _scientific_conclusion(comparisons),
    }
    return _json_safe(summary)  # type: ignore[return-value]


def assemble_out_of_fold_arrays(
    dataset: LongitudinalTrajectoryDataset,
    folds: Sequence[Sequence[int]],
    fold_records: Mapping[str, Any],
    private_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], int]:
    n_subjects = len(dataset.subject_ids)
    n_regions = len(dataset.roi_labels)
    predictions = {model: np.full((n_subjects, n_regions), np.nan) for model in TENFOLD_MODELS}
    residuals = {model: np.full((n_subjects, n_regions), np.nan) for model in TENFOLD_MODELS}
    targets = {model: np.full((n_subjects, n_regions), np.nan) for model in TENFOLD_MODELS}
    filled = {model: np.zeros(n_subjects, dtype=int) for model in TENFOLD_MODELS}
    artifact_count = 0
    expected_by_fold = {index: tuple(fold) for index, fold in enumerate(folds)}
    if set(fold_records) != {str(index) for index in range(len(folds))}:
        raise SchemaError("fold records must contain exactly the expected folds.")
    for fold_key, record_object in fold_records.items():
        record = _mapping(record_object, "fold record")
        fold_index = int(fold_key)
        if record.get("completion_status") != "complete":
            raise SchemaError("cannot aggregate incomplete tenfold records.")
        models = _mapping(record.get("models"), "fold models")
        expected_indices = np.asarray(expected_by_fold[fold_index], dtype=int)
        if set(models) != set(TENFOLD_MODELS):
            raise SchemaError("fold record is missing a required model.")
        for model_name in TENFOLD_MODELS:
            payload = _mapping(models.get(model_name), f"{model_name} payload")
            relative = _relative_artifact_path(payload.get("prediction_artifact"))
            artifact_path = private_dir / relative
            arrays = _load_prediction_artifact(artifact_path)
            indices = arrays["outer_eval_indices"].astype(int, copy=False)
            if tuple(indices.tolist()) != tuple(expected_indices.tolist()):
                raise SchemaError("prediction artifact indices do not match the frozen fold.")
            for key, store in (
                ("prediction", predictions[model_name]),
                ("residual", residuals[model_name]),
                ("target", targets[model_name]),
            ):
                values = arrays[key]
                if values.shape != (len(indices), n_regions):
                    raise SchemaError(f"{model_name} artifact {key} has an invalid shape.")
                store[indices] = values
            filled[model_name][indices] += 1
            artifact_count += 1
    for model_name in TENFOLD_MODELS:
        if not np.array_equal(filled[model_name], np.ones(n_subjects, dtype=int)):
            raise SchemaError("each participant must receive exactly one OOF prediction.")
        for matrix in (
            predictions[model_name],
            residuals[model_name],
            targets[model_name],
        ):
            if not np.isfinite(matrix).all():
                raise SchemaError("out-of-fold arrays must be complete and finite.")
    return predictions, residuals, targets, artifact_count


def overall_prediction_metrics(residuals: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    residual = _residual_matrix(residuals, "residuals")
    target = _residual_matrix(targets, "targets")
    if residual.shape != target.shape:
        raise SchemaError("residuals and targets must have matching shapes.")
    subject_rmse = np.sqrt(np.mean(residual**2, axis=1))
    subject_mae = np.mean(np.abs(residual), axis=1)
    mse = float(np.mean(residual**2))
    bias = float(np.mean(residual))
    residual_variance = float(np.mean((residual - bias) ** 2))
    sse = float(np.sum(residual**2))
    centered_target = target - float(np.mean(target))
    sst = float(np.sum(centered_target**2))
    r_squared = None if sst <= 0.0 else float(1.0 - sse / sst)
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "bias": bias,
        "residual_variance": residual_variance,
        "residual_standard_deviation": float(math.sqrt(residual_variance)),
        "r_squared_secondary_descriptive": r_squared,
        "mean_subject_rmse": float(np.mean(subject_rmse)),
        "std_subject_rmse": float(np.std(subject_rmse)),
        "median_subject_rmse": float(np.median(subject_rmse)),
        "mean_subject_mae": float(np.mean(subject_mae)),
        "std_subject_mae": float(np.std(subject_mae)),
        "median_subject_mae": float(np.median(subject_mae)),
        "subject_count": int(residual.shape[0]),
        "region_count": int(residual.shape[1]),
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
        "relative_percent_difference": float(observed / reference * 100.0)
        if reference != 0.0
        else None,
        "ci95": [float(lower), float(upper)],
        "interval_excludes_zero": bool(upper < 0.0 or lower > 0.0),
        "direction": _comparison_direction(float(observed), metric),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def all_bootstrap_comparisons(
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
            output[name] = {
                "category": category,
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


def validate_private_tenfold_output_directory(
    output_dir: str | Path,
    *,
    refuse_completed: bool = False,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if _is_relative_to(output, REPO_ROOT):
        raise SchemaError("tenfold output directory must be outside the repository.")
    if output.exists() and not output.is_dir():
        raise SchemaError("tenfold output location must be a directory.")
    if refuse_completed and (output / TENFOLD_SUMMARY_NAME).exists():
        summary = _read_json_object(output / TENFOLD_SUMMARY_NAME, "tenfold summary")
        if summary.get("completion_status") == "complete":
            raise SchemaError("tenfold output directory already contains a completed experiment.")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    return output


def config_payload(config: TenfoldConfig) -> dict[str, Any]:
    model = config.model_config
    return {
        "outer_folds": config.outer_folds,
        "master_seed": config.master_seed,
        "expected_subject_count": config.expected_subject_count,
        "inner_validation_fraction": config.inner_validation_fraction,
        "inner_seed_formula": "master_seed + 1009 * (fold_index + 1)",
        "ridge_alpha_grid": list(config.ridge_alpha_grid),
        "models": list(TENFOLD_MODELS),
        "primary_comparison": "quadratic_vs_linear",
        "secondary_comparisons": [name for name, _first, _second in SECONDARY_COMPARISONS],
        "bootstrap_resamples": config.bootstrap_resamples,
        "bootstrap_seed": config.bootstrap_seed,
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
            "pysr_worker_timeout_seconds": config.pysr_worker_timeout_seconds,
            "pysr_worker_shutdown_grace_seconds": config.pysr_worker_shutdown_grace_seconds,
        },
    }


def _run_persistence_fold(
    dataset: LongitudinalTrajectoryDataset,
    plan: TenfoldFoldPlan,
    private_dir: Path,
    fold_record: dict[str, Any],
) -> None:
    if _completed_model(fold_record, "persistence"):
        return
    start = time.perf_counter()
    forecasts = forecast_persistence(dataset, plan.outer_evaluation_indices)
    fold_record["models"]["persistence"] = _evaluate_and_store_model(
        dataset,
        plan,
        forecasts,
        private_dir,
        "persistence",
        runtime_seconds=time.perf_counter() - start,
    )


def _run_ridge_fold(
    dataset: LongitudinalTrajectoryDataset,
    plan: TenfoldFoldPlan,
    private_dir: Path,
    fold_record: dict[str, Any],
    config: TenfoldConfig,
) -> RidgeFit | None:
    if _completed_model(fold_record, "ridge"):
        return None
    start = time.perf_counter()
    selection = select_ridge_alpha(
        dataset,
        plan.inner_training_indices,
        plan.inner_validation_indices,
        config.ridge_alpha_grid,
    )
    x_train, y_train = _xy(dataset, plan.outer_development_indices)
    x_eval, _y_eval = _xy(dataset, plan.outer_evaluation_indices)
    fit = fit_ridge_baseline(x_train, y_train, alpha=selection["selected_alpha"])
    predictions = fit.predict(x_eval)
    forecasts = {
        dataset.subject_ids[index]: predictions[row].astype(float, copy=True)
        for row, index in enumerate(plan.outer_evaluation_indices)
    }
    payload = _evaluate_and_store_model(
        dataset,
        plan,
        forecasts,
        private_dir,
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
    return fit


def _run_parametric_fold(
    dataset: LongitudinalTrajectoryDataset,
    plan: TenfoldFoldPlan,
    split: SubjectSplit,
    kind: ParametricKind,
    private_dir: Path,
    fold_record: dict[str, Any],
    config: CohortExperimentConfig,
) -> ParametricFitResult | None:
    if _completed_model(fold_record, kind):
        return None
    start = time.perf_counter()
    inner = _fit_parametric_inner(dataset, split, kind=kind, config=config)
    if inner.numerical_instability:
        raise SchemaError(f"{kind} inner training was numerically unstable.")
    updates = refit_updates_from_best_epoch(inner.best_epoch)
    refit = _fit_parametric_fixed_updates(
        dataset,
        plan.outer_development_indices,
        kind=kind,
        n_updates=updates,
        config=config,
    )
    if refit.numerical_instability:
        raise SchemaError(f"{kind} refit was numerically unstable.")
    forecasts = forecast_parametric(
        dataset,
        plan.outer_evaluation_indices,
        rho=refit.rho,
        linear_coefficient=refit.linear_coefficient,
        quadratic_coefficient=refit.quadratic_coefficient,
        max_step=config.max_step,
    )
    payload = _evaluate_and_store_model(
        dataset,
        plan,
        forecasts,
        private_dir,
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
    plan: TenfoldFoldPlan,
    split: SubjectSplit,
    private_dir: Path,
    fold_record: dict[str, Any],
    config: CohortExperimentConfig,
) -> ZeroAnchoredNeuralReactionDiffusion:
    if _completed_model(fold_record, "neural"):
        return _load_neural_model_from_payload(
            fold_record["models"]["neural"],
            config,
            private_dir,
        )
    start = time.perf_counter()
    seed_results: dict[int, dict[str, Any]] = {}
    best_seed: int | None = None
    best_rmse = math.inf
    best_epoch = 0
    for seed in config.neural_seeds:
        result = _fit_zero_anchored_inner(dataset, split, seed=seed, config=config)
        forecasts = forecast_neural(result.model, dataset, split.validation_indices, config=config)
        metrics = _evaluate_forecasts(dataset, split.validation_indices, forecasts)
        rmse = float(metrics["rmse"])
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
    updates = refit_updates_from_best_epoch(best_epoch)
    refit_model = _new_zero_anchored_model(best_seed, config)
    training_config = _neural_training_config(config, best_seed, max_epochs=updates, patience=1)
    refit = fit_fixed_epoch_recovery_model(
        dataset,
        plan.outer_development_indices,
        training_config,
        model=refit_model,
        n_epochs=updates,
    )
    if refit.numerical_instability:
        raise SchemaError("neural fixed-update refit was numerically unstable.")
    forecasts = forecast_neural(refit.model, dataset, plan.outer_evaluation_indices, config=config)
    payload = _evaluate_and_store_model(
        dataset,
        plan,
        forecasts,
        private_dir,
        "neural",
        runtime_seconds=time.perf_counter() - start,
    )
    state_path = Path("model_states") / f"fold_{plan.fold_index:02d}_neural.pt"
    _write_torch_state_private(private_dir / state_path, refit.model.state_dict())
    payload.update(
        {
            "selected_seed": best_seed,
            "selected_inner_best_epoch": best_epoch,
            "refit_updates": updates,
            "seed_sensitivity": seed_results,
            "rho": float(refit.model.rho.detach().cpu().item()),
            "refit_training_losses": list(refit.training_losses),
            "model_state_artifact": str(state_path),
        }
    )
    fold_record["models"]["neural"] = payload
    return refit.model


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
    plan: TenfoldFoldPlan,
    private_dir: Path,
    fold_record: dict[str, Any],
    config: TenfoldConfig,
    symbolic_runner: SymbolicRunner,
) -> None:
    if _completed_model(fold_record, "symbolic"):
        return
    start = time.perf_counter()
    symbolic_dir = private_dir / "symbolic" / f"fold_{plan.fold_index:02d}"
    symbolic_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(symbolic_dir, 0o700)
    try:
        summary = symbolic_runner(
            neural_model,
            dataset,
            plan.outer_development_indices,
            config.model_config,
            symbolic_dir,
            config.pysr_worker_timeout_seconds,
            config.pysr_worker_shutdown_grace_seconds,
        )
    finally:
        _restrict_private_tree(symbolic_dir)
    if not summary.get("completed"):
        raise SchemaError("symbolic distillation did not complete for an outer fold.")
    reaction = symbolic_reaction_from_expression(summary)
    forecasts = forecast_symbolic(
        dataset,
        plan.outer_evaluation_indices,
        rho=float(neural_model.rho.detach().cpu().item()),
        reaction_fn=reaction,
        max_step=config.model_config.max_step,
    )
    payload = _evaluate_and_store_model(
        dataset,
        plan,
        forecasts,
        private_dir,
        "symbolic",
        runtime_seconds=time.perf_counter() - start,
    )
    payload.update(
        {
            "fit_domain": summary.get("fit_domain"),
            "evaluation_domain": summary.get("evaluation_domain"),
            "equation": summary.get("equation"),
            "expanded_expression": summary.get("expanded_expression"),
            "complexity": summary.get("complexity"),
            "training_loss": summary.get("training_loss"),
            "metrics": summary.get("metrics"),
            "polynomial_interpretation": summary.get("polynomial_interpretation"),
            "completed": True,
        }
    )
    fold_record["models"]["symbolic"] = payload


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
    plan: TenfoldFoldPlan,
    forecasts: Mapping[str, np.ndarray],
    private_dir: Path,
    model_name: str,
    *,
    runtime_seconds: float,
) -> dict[str, Any]:
    metrics = _evaluate_forecasts(dataset, plan.outer_evaluation_indices, forecasts)
    prediction_matrix = np.stack(
        [forecasts[dataset.subject_ids[index]] for index in plan.outer_evaluation_indices]
    )
    target_matrix = np.stack(
        [dataset.trajectories[index][1] for index in plan.outer_evaluation_indices]
    )
    residual_matrix = prediction_matrix - target_matrix
    relative = Path("fold_predictions") / f"fold_{plan.fold_index:02d}_{model_name}.npz"
    _write_npz_private(
        private_dir / relative,
        prediction=prediction_matrix,
        residual=residual_matrix,
        target=target_matrix,
        outer_eval_indices=np.asarray(plan.outer_evaluation_indices, dtype=int),
        fold_index=np.asarray([plan.fold_index], dtype=int),
    )
    artifact_path = private_dir / relative
    return {
        **metrics,
        "runtime_seconds": float(runtime_seconds),
        "prediction_artifact": str(relative),
        "artifact_sha256": sha256_file(artifact_path),
        "completion_status": "complete",
    }


def _evaluate_forecasts(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
    forecasts: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    checked = _index_tuple(indices, "indices")
    residuals: list[np.ndarray] = []
    subject_rmse: list[float] = []
    subject_mae: list[float] = []
    numerical_instability_count = 0
    for index in checked:
        label = dataset.subject_ids[index]
        if label not in forecasts:
            raise SchemaError("forecast mapping is missing a requested participant.")
        predicted = np.asarray(forecasts[label], dtype=float)
        observed = dataset.trajectories[index][1].astype(float, copy=True)
        if predicted.ndim != 1 or predicted.shape != observed.shape:
            raise SchemaError("forecast vector shape does not match observed follow-up.")
        if not np.isfinite(predicted).all():
            numerical_instability_count += 1
            continue
        residual = predicted - observed
        residuals.append(residual)
        subject_rmse.append(float(np.sqrt(np.mean(residual**2))))
        subject_mae.append(float(np.mean(np.abs(residual))))
    if not residuals:
        raise SchemaError("no finite forecasts were available for evaluation.")
    residual_matrix = np.stack(residuals)
    target_matrix = np.stack([dataset.trajectories[index][1] for index in checked])
    metrics = overall_prediction_metrics(residual_matrix, target_matrix)
    return {
        **metrics,
        "mean_subject_rmse": float(np.mean(subject_rmse)),
        "median_subject_rmse": float(np.median(subject_rmse)),
        "mean_subject_mae": float(np.mean(subject_mae)),
        "median_subject_mae": float(np.median(subject_mae)),
        "numerical_instability_count": int(numerical_instability_count),
    }


def _freeze_payload(cohort: TenfoldCohort, config: TenfoldConfig) -> dict[str, Any]:
    folds = build_tenfold_outer_folds(
        len(cohort.dataset.subject_ids),
        seed=config.master_seed,
    )
    if len(cohort.dataset.subject_ids) != config.expected_subject_count:
        raise SchemaError("cohort subject count does not match tenfold config.")
    return _json_safe(
        {
            "freeze_schema": "tau_sr_full_cohort_tenfold_freeze_v1",
            "completion_status": "frozen_before_execution",
            "execution_started": False,
            "repository_commit": _current_commit(),
            "required_branch": "full_cohort_tenfold_cv",
            "cohort_fingerprint": cohort.cohort_fingerprint,
            "cohort_qc_summary": cohort.qc_summary,
            "source_file_match_count": cohort.source_file_match_count,
            "protocol_file": "config/cohort_tenfold_protocol.json",
            "protocol_file_sha256": sha256_file(PROTOCOL_JSON) if PROTOCOL_JSON.exists() else None,
            "configuration_fingerprint": _fingerprint(config_payload(config)),
            "fold_fingerprint": _fingerprint(folds),
            "fold_design": {
                "outer_folds": len(folds),
                "outer_fold_sizes": [len(fold) for fold in folds],
                "outer_fold_size_distribution": _fold_size_distribution(folds),
                "master_seed": config.master_seed,
                "inner_validation_fraction": config.inner_validation_fraction,
                "inner_seed_formula": "master_seed + 1009 * (fold_index + 1)",
                "allowable_fold_sizes": [16, 17]
                if config.expected_subject_count == EXPECTED_SUBJECT_COUNT
                else sorted({len(fold) for fold in folds}),
            },
            "configuration": config_payload(config),
            "metric_definitions": {
                "residual": "predicted - observed",
                "mse": "mean(residual^2)",
                "rmse": "sqrt(mse)",
                "mae": "mean(abs(residual))",
                "bias": "mean(residual)",
                "residual_variance": "mean((residual - bias)^2)",
                "r_squared": "1 - SSE/SST; secondary descriptive metric",
            },
            "model_list": list(TENFOLD_MODELS),
            "primary_comparison": "quadratic_vs_linear",
            "secondary_comparisons": [name for name, _first, _second in SECONDARY_COMPARISONS],
            "bootstrap_settings": {
                "participant_level": True,
                "resamples": config.bootstrap_resamples,
                "seed": config.bootstrap_seed,
            },
            "software_versions": {
                **environment_versions(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "scikit_learn": _package_version("scikit-learn"),
                "pysr": _package_version("pysr"),
            },
            "privacy_contract": {
                "raw_paths_in_artifacts": False,
                "subject_identifiers_in_repository": False,
                "fold_membership_printed": False,
                "previous_outcome_artifacts_read": False,
            },
        }
    )  # type: ignore[return-value]


def _load_tenfold_checkpoint(private_dir: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    path = private_dir / TENFOLD_CHECKPOINT_NAME
    if not path.exists():
        return {"folds": {}}
    payload = _read_json_object(path, "tenfold checkpoint")
    if payload.get("freeze_fingerprint") != _fingerprint(freeze):
        raise SchemaError("tenfold checkpoint does not match the freeze artifact.")
    return payload


def _write_tenfold_checkpoint(
    private_dir: Path,
    freeze: Mapping[str, Any],
    fold_records: Mapping[str, Any],
    stage: str,
) -> None:
    payload = _json_safe(
        {
            "checkpoint_schema": "tau_sr_full_cohort_tenfold_checkpoint_v1",
            "completion_status": "complete" if stage == "complete" else "incomplete",
            "stage": stage,
            "freeze_fingerprint": _fingerprint(freeze),
            "folds": fold_records,
            "updated_at_unix_seconds": time.time(),
        }
    )
    write_private_json(private_dir / TENFOLD_CHECKPOINT_NAME, payload)


def _write_npz_private(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _write_torch_state_private(path: Path, state: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _restrict_private_tree(root: Path) -> None:
    if not root.exists():
        return
    for directory, subdirectories, filenames in os.walk(root):
        directory_path = Path(directory)
        os.chmod(directory_path, 0o700)
        for name in subdirectories:
            os.chmod(directory_path / name, 0o700)
        for name in filenames:
            os.chmod(directory_path / name, 0o600)


def _load_neural_model_from_payload(
    payload: Mapping[str, Any],
    config: CohortExperimentConfig,
    private_dir: Path,
) -> ZeroAnchoredNeuralReactionDiffusion:
    relative = _relative_artifact_path(payload.get("model_state_artifact"))
    model = _new_zero_anchored_model(int(payload["selected_seed"]), config)
    state = torch.load(private_dir / relative, map_location=torch.device(config.device))
    model.load_state_dict(state)
    model.eval()
    return model


def _completed_model(fold_record: Mapping[str, Any], model_name: str) -> bool:
    models = _mapping(fold_record.get("models"), "fold models")
    payload = models.get(model_name)
    return isinstance(payload, Mapping) and payload.get("completion_status") == "complete"


def _load_prediction_artifact(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise SchemaError("prediction artifact is missing.")
    if path.stat().st_mode & 0o077:
        raise SchemaError("prediction artifact permissions are not restrictive.")
    with np.load(path) as arrays:
        required = ("prediction", "residual", "target", "outer_eval_indices")
        if any(name not in arrays for name in required):
            raise SchemaError("prediction artifact is missing required arrays.")
        return {
            name: np.asarray(
                arrays[name],
                dtype=float if name != "outer_eval_indices" else int,
            )
            for name in required
        }


def _write_final_arrays(
    private_dir: Path,
    predictions: Mapping[str, np.ndarray],
    residuals: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for model in TENFOLD_MODELS:
        relative = Path("out_of_fold_arrays") / f"{model}.npz"
        _write_npz_private(
            private_dir / relative,
            prediction=predictions[model],
            residual=residuals[model],
            target=targets[model],
        )
        paths[model] = str(relative)
    return paths


def _fold_metric_summary(fold_records: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in TENFOLD_MODELS:
        rmse: list[float] = []
        mae: list[float] = []
        for record in fold_records.values():
            payload = _mapping(record.get("models"), "fold models").get(model)
            if isinstance(payload, Mapping) and "rmse" in payload:
                rmse.append(float(payload["rmse"]))
                mae.append(float(payload["mae"]))
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
    lower_than_persistence = {model: 0 for model in TENFOLD_MODELS if model != "persistence"}
    selected_neural_seeds: dict[str, int] = {}
    selected_neural_epochs: dict[str, int] = {}
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
    symbolic_complexities: list[float] = []
    symbolic_equations: list[str] = []
    symbolic_polynomial_orders: list[int] = []
    for fold_key, record in fold_records.items():
        models = _mapping(record.get("models"), "fold models")
        persistence = _mapping(models.get("persistence"), "persistence")
        p_rmse = float(persistence["rmse"])
        for model, payload in models.items():
            if model != "persistence" and isinstance(payload, Mapping) and "rmse" in payload:
                lower_than_persistence[model] += int(float(payload["rmse"]) < p_rmse)
        if isinstance(models.get("neural"), Mapping):
            selected_neural_seeds[fold_key] = int(models["neural"]["selected_seed"])
            selected_neural_epochs[fold_key] = int(models["neural"]["selected_inner_best_epoch"])
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
        if isinstance(models.get("symbolic"), Mapping):
            symbolic = models["symbolic"]
            if symbolic.get("equation") is not None:
                symbolic_equations.append(str(symbolic["equation"]))
            if symbolic.get("complexity") is not None:
                symbolic_complexities.append(float(symbolic["complexity"]))
            symbolic_polynomial_orders.append(_symbolic_order(symbolic))
    return {
        "folds_lower_than_persistence": lower_than_persistence,
        "selected_neural_seed_by_fold": selected_neural_seeds,
        "selected_neural_epoch_by_fold": selected_neural_epochs,
        "selected_ridge_alpha_by_fold": selected_ridge_alphas,
        "rho_summary": {model: _summary(values) for model, values in rho_values.items() if values},
        "coefficient_summary": {
            name: _summary(values) for name, values in coefficients.items() if values
        },
        "symbolic_stability": {
            "unique_equation_count": len(set(symbolic_equations)),
            "complexity_summary": _summary(symbolic_complexities)
            if symbolic_complexities
            else None,
            "linear_fold_count": int(sum(order == 1 for order in symbolic_polynomial_orders)),
            "quadratic_fold_count": int(sum(order == 2 for order in symbolic_polynomial_orders)),
            "higher_order_fold_count": int(sum(order > 2 for order in symbolic_polynomial_orders)),
        },
    }


def _symbolic_order(payload: Mapping[str, Any]) -> int:
    interpretation = payload.get("polynomial_interpretation")
    if isinstance(interpretation, Mapping):
        coeffs = interpretation.get("coefficients")
        if isinstance(coeffs, Mapping):
            orders = [
                int(key)
                for key, value in coeffs.items()
                if _finite_float(value, "symbolic coefficient") != 0.0
                for _ in [key]
                if str(key).lstrip("-").isdigit()
            ]
            return max(orders, default=0)
    expression = str(payload.get("expanded_expression") or payload.get("equation") or "")
    if "**3" in expression or "^3" in expression:
        return 3
    if "**2" in expression or "^2" in expression:
        return 2
    if "y" in expression:
        return 1
    return 0


def _scientific_conclusion(comparisons: Mapping[str, Any]) -> str:
    primary = _mapping(comparisons.get("quadratic_vs_linear"), "primary comparison")
    rmse = _mapping(primary.get("rmse"), "primary rmse comparison")
    if rmse.get("interval_excludes_zero") and float(rmse["difference_first_minus_second"]) < 0.0:
        return "Quadratic has paired-bootstrap support for lower RMSE than linear."
    if rmse.get("interval_excludes_zero"):
        return "Linear has paired-bootstrap support for lower RMSE than quadratic."
    return "The prespecified quadratic-minus-linear RMSE interval crosses zero."


def _write_tenfold_figures(private_dir: Path, summary: Mapping[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return
    figures = private_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(figures, 0o700)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )
    _figure_model_metrics(figures, summary, plt)
    _figure_paired_differences(figures, summary, plt)
    _figure_fold_stability(figures, summary, plt)
    _figure_coefficient_stability(figures, summary, plt)
    _figure_symbolic_stability(figures, summary, plt)


def _save_figure(fig: Any, stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    os.chmod(stem.with_suffix(".png"), 0o600)
    os.chmod(stem.with_suffix(".pdf"), 0o600)
    plt.close(fig)


def _figure_model_metrics(figures: Path, summary: Mapping[str, Any], plt: Any) -> None:
    metrics = _mapping(summary.get("overall_out_of_fold_metrics"), "overall metrics")
    models = [model for model in TENFOLD_MODELS if model in metrics]
    rmse = [float(metrics[model]["rmse"]) for model in models]
    mae = [float(metrics[model]["mae"]) for model in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - 0.18, rmse, width=0.36, label="RMSE", color="#3B5B92")
    ax.bar(x + 0.18, mae, width=0.36, label="MAE", color="#8F4E2F")
    ax.set_xticks(x, models, rotation=30, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Out-of-fold prediction error")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, figures / "figure_1_oof_rmse_mae")


def _figure_paired_differences(figures: Path, summary: Mapping[str, Any], plt: Any) -> None:
    comparisons = _mapping(summary.get("paired_bootstrap_comparisons"), "comparisons")
    names = [
        name
        for name in ("quadratic_vs_linear", *[c[0] for c in SECONDARY_COMPARISONS])
        if name in comparisons
    ]
    y = np.arange(len(names))
    diffs = [float(comparisons[name]["rmse"]["difference_first_minus_second"]) for name in names]
    ci = [comparisons[name]["rmse"]["ci95"] for name in names]
    lower = [diff - float(bounds[0]) for diff, bounds in zip(diffs, ci, strict=True)]
    upper = [float(bounds[1]) - diff for diff, bounds in zip(diffs, ci, strict=True)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axvline(0.0, color="0.25", linewidth=0.8)
    ax.errorbar(diffs, y, xerr=[lower, upper], fmt="o", color="#3B5B92", ecolor="0.35")
    ax.set_yticks(y, [name.replace("_", " ") for name in names])
    ax.set_xlabel("Paired RMSE difference")
    ax.set_title("Participant-level bootstrap comparisons")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, figures / "figure_2_paired_rmse_differences")


def _figure_fold_stability(figures: Path, summary: Mapping[str, Any], plt: Any) -> None:
    fold_summary = _mapping(summary.get("fold_metric_summary"), "fold metric summary")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for model in TENFOLD_MODELS:
        if model in fold_summary:
            ax.plot(
                range(1, len(fold_summary[model]["fold_rmse"]) + 1),
                fold_summary[model]["fold_rmse"],
                marker="o",
                linewidth=1.2,
                label=model,
            )
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("RMSE")
    ax.set_title("Fold-level RMSE stability")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, figures / "figure_3_fold_rmse_stability")


def _figure_coefficient_stability(figures: Path, summary: Mapping[str, Any], plt: Any) -> None:
    stability = _mapping(summary.get("fold_stability"), "fold stability")
    rho = _mapping(stability.get("rho_summary"), "rho summary")
    labels = [model for model in ("diffusion", "linear", "quadratic", "neural") if model in rho]
    means = [float(rho[model]["mean"]) for model in labels]
    errors = [float(rho[model]["std"]) for model in labels]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(labels, means, yerr=errors, color="#3B5B92", capsize=3)
    ax.set_ylabel("rho")
    ax.set_title("Learned diffusion coefficient stability")
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, figures / "figure_4_rho_stability")


def _figure_symbolic_stability(figures: Path, summary: Mapping[str, Any], plt: Any) -> None:
    stability = _mapping(summary.get("fold_stability"), "fold stability")
    symbolic = _mapping(stability.get("symbolic_stability"), "symbolic stability")
    labels = ("linear", "quadratic", "higher order")
    values = [
        int(symbolic.get("linear_fold_count", 0)),
        int(symbolic.get("quadratic_fold_count", 0)),
        int(symbolic.get("higher_order_fold_count", 0)),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.bar(labels, values, color="#3B5B92")
    ax.set_ylabel("Fold count")
    ax.set_title("Symbolic structure stability")
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, figures / "figure_5_symbolic_structure_stability")


def _load_prediction_artifact_for_audit(path: Path) -> dict[str, np.ndarray]:
    return _load_prediction_artifact(path)


def _xy(
    dataset: LongitudinalTrajectoryDataset,
    indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    checked = _index_tuple(indices, "indices")
    x = np.stack([dataset.trajectories[index][0] for index in checked])
    y = np.stack([dataset.trajectories[index][1] for index in checked])
    return x.astype(float, copy=True), y.astype(float, copy=True)


def _validate_outer_folds(folds: Sequence[Sequence[int]], n_subjects: int) -> None:
    if len(folds) != TENFOLD_FOLD_COUNT:
        raise SchemaError("exactly 10 outer folds are required.")
    sizes = [len(fold) for fold in folds]
    if any(size < 1 for size in sizes):
        raise SchemaError("outer folds must be nonempty.")
    if max(sizes) - min(sizes) > 1:
        raise SchemaError("outer fold sizes must differ by at most one participant.")
    combined = [index for fold in folds for index in fold]
    if sorted(combined) != list(range(n_subjects)):
        raise SchemaError("outer folds must cover every participant exactly once.")
    if len(combined) != len(set(combined)):
        raise SchemaError("outer folds must not overlap.")


def _fold_size_distribution(folds: Sequence[Sequence[int]]) -> dict[str, int]:
    output: dict[str, int] = {}
    for fold in folds:
        key = str(len(fold))
        output[key] = output.get(key, 0) + 1
    return output


def _fold_records_without_membership(records: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in records.items():
        record = dict(_mapping(value, "fold record"))
        record.pop("outer_evaluation_indices", None)
        record.pop("outer_development_indices", None)
        record.pop("inner_training_indices", None)
        record.pop("inner_validation_indices", None)
        output[str(key)] = record
    return output


def _relative_artifact_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("artifact path must be a relative string.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise SchemaError("artifact path must remain inside the private output directory.")
    return path


def _residual_matrix(value: object, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise SchemaError(f"{field_name} must be a nonempty two-dimensional matrix.")
    if not np.isfinite(matrix).all():
        raise SchemaError(f"{field_name} must contain finite values.")
    return matrix.astype(float, copy=True)


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


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
