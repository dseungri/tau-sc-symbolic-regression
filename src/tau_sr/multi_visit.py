"""Pre-data infrastructure for leakage-safe multi-visit cohort analysis."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tau_sr.graphs import LaplacianKind, build_graph_laplacian
from tau_sr.pilot_adapter import LongitudinalTrajectoryDataset
from tau_sr.pysr_lifecycle import parse_json_file, write_json_atomically
from tau_sr.schemas import SchemaError

MULTI_VISIT_N_REGIONS = 160
MULTI_VISIT_MANIFEST_SCHEMA = "tau_sr_multi_visit_manifest_v1"
MULTI_VISIT_PROTOCOL_SCHEMA = "tau_sr_multi_visit_analysis_protocol_v1"
MULTI_VISIT_FREEZE_SCHEMA = "tau_sr_multi_visit_analysis_freeze_v1"
MULTI_VISIT_REHEARSAL_SCHEMA = "tau_sr_multi_visit_synthetic_rehearsal_v1"
DEFAULT_MULTI_VISIT_PROTOCOL_JSON = Path("config/multi_visit_analysis_protocol.json")
DEFAULT_MULTI_VISIT_PROTOCOL_MD = Path("docs/multi_visit_analysis_protocol.md")
DEFAULT_MULTI_VISIT_LAPLACIAN_KIND: LaplacianKind = "symmetric_normalized"
REPO_ROOT = Path(__file__).resolve().parents[2]

ModelName = Literal[
    "persistence",
    "linear_extrapolation",
    "diffusion",
    "linear",
    "quadratic",
    "neural",
    "symbolic",
]

MULTI_VISIT_MODELS: tuple[ModelName, ...] = (
    "persistence",
    "linear_extrapolation",
    "diffusion",
    "linear",
    "quadratic",
    "neural",
    "symbolic",
)
PRIMARY_COMPARISON = "quadratic_minus_linear"
SECONDARY_COMPARISONS: tuple[str, ...] = (
    "linear_minus_persistence",
    "quadratic_minus_persistence",
    "neural_minus_linear",
    "neural_minus_quadratic",
    "symbolic_minus_neural",
)

__all__ = (
    "DEFAULT_MULTI_VISIT_LAPLACIAN_KIND",
    "DEFAULT_MULTI_VISIT_PROTOCOL_JSON",
    "DEFAULT_MULTI_VISIT_PROTOCOL_MD",
    "MULTI_VISIT_FREEZE_SCHEMA",
    "MULTI_VISIT_MANIFEST_SCHEMA",
    "MULTI_VISIT_MODELS",
    "MULTI_VISIT_N_REGIONS",
    "MULTI_VISIT_PROTOCOL_SCHEMA",
    "MULTI_VISIT_REHEARSAL_SCHEMA",
    "MultiVisitAnalysisConfig",
    "MultiVisitLoadResult",
    "ProcessedMultiVisitParticipant",
    "build_multi_visit_analysis_freeze",
    "cohort_visit_count_summary",
    "compute_protocol_hash",
    "evaluate_final_visit_forecasts",
    "forecast_linear_extrapolation",
    "forecast_prior_visit_persistence",
    "inspect_multi_visit_manifest",
    "load_multi_visit_cohort",
    "load_multi_visit_manifest",
    "make_multi_visit_outer_folds",
    "make_rmse_mae_table",
    "model_comparison_figure_data",
    "paired_participant_bootstrap",
    "prepare_private_multi_visit_output_dir",
    "protocol_settings",
    "run_synthetic_multi_visit_rehearsal",
    "symbolic_structure_stability_summary",
    "validate_multi_visit_protocol",
    "verify_analysis_freeze_protocol",
    "write_private_multi_visit_json",
)


@dataclass(frozen=True)
class MultiVisitAnalysisConfig:
    outer_folds: int = 5
    fold_seed: int = 42
    inner_validation_fraction: float = 0.20
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 42
    parametric_optimizer: str = "Adam"
    parametric_learning_rate: float = 0.01
    parametric_max_epochs: int = 1000
    parametric_patience: int = 100
    rk4_max_step_years: float = 0.05
    neural_width: int = 16
    neural_initial_rho: float = 0.05
    neural_learning_rate: float = 1e-3
    neural_seeds: tuple[int, ...] = (41, 42, 43)
    neural_dtype: str = "float64"
    neural_device: str = "cpu"
    pysr_binary_operators: tuple[str, ...] = ("+", "-", "*")
    pysr_unary_operators: tuple[str, ...] = ()
    pysr_maxsize: int = 15
    pysr_seed: int = 42
    pysr_deterministic_serial: bool = True
    reaction_samples: int = 500

    def __post_init__(self) -> None:
        _positive_int(self.outer_folds, "outer_folds")
        _nonnegative_int(self.fold_seed, "fold_seed")
        _fraction(self.inner_validation_fraction, "inner_validation_fraction")
        _positive_int(self.bootstrap_resamples, "bootstrap_resamples")
        _nonnegative_int(self.bootstrap_seed, "bootstrap_seed")
        if self.parametric_optimizer != "Adam":
            raise SchemaError("parametric_optimizer must be Adam.")
        _positive_float(self.parametric_learning_rate, "parametric_learning_rate")
        _positive_int(self.parametric_max_epochs, "parametric_max_epochs")
        _positive_int(self.parametric_patience, "parametric_patience")
        _positive_float(self.rk4_max_step_years, "rk4_max_step_years")
        _positive_int(self.neural_width, "neural_width")
        _positive_float(self.neural_initial_rho, "neural_initial_rho")
        _positive_float(self.neural_learning_rate, "neural_learning_rate")
        seeds = tuple(_nonnegative_int(seed, "neural_seeds") for seed in self.neural_seeds)
        if not seeds:
            raise SchemaError("neural_seeds must be nonempty.")
        if self.neural_dtype != "float64":
            raise SchemaError("neural_dtype must be float64.")
        if self.neural_device != "cpu":
            raise SchemaError("neural_device must be cpu.")
        if tuple(self.pysr_binary_operators) != ("+", "-", "*"):
            raise SchemaError('pysr_binary_operators must be ["+", "-", "*"].')
        if tuple(self.pysr_unary_operators):
            raise SchemaError("pysr_unary_operators must be empty.")
        _positive_int(self.pysr_maxsize, "pysr_maxsize")
        _nonnegative_int(self.pysr_seed, "pysr_seed")
        if not isinstance(self.pysr_deterministic_serial, bool):
            raise SchemaError("pysr_deterministic_serial must be boolean.")
        _positive_int(self.reaction_samples, "reaction_samples")
        object.__setattr__(self, "neural_seeds", seeds)
        object.__setattr__(self, "pysr_binary_operators", tuple(self.pysr_binary_operators))
        object.__setattr__(self, "pysr_unary_operators", tuple(self.pysr_unary_operators))

    def to_json_dict(self) -> dict[str, object]:
        return {
            "outer_folds": self.outer_folds,
            "fold_seed": self.fold_seed,
            "inner_validation_fraction": self.inner_validation_fraction,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "parametric_optimizer": self.parametric_optimizer,
            "parametric_learning_rate": self.parametric_learning_rate,
            "parametric_max_epochs": self.parametric_max_epochs,
            "parametric_patience": self.parametric_patience,
            "rk4_max_step_years": self.rk4_max_step_years,
            "neural_width": self.neural_width,
            "neural_initial_rho": self.neural_initial_rho,
            "neural_learning_rate": self.neural_learning_rate,
            "neural_seeds": list(self.neural_seeds),
            "neural_dtype": self.neural_dtype,
            "neural_device": self.neural_device,
            "pysr_binary_operators": list(self.pysr_binary_operators),
            "pysr_unary_operators": list(self.pysr_unary_operators),
            "pysr_maxsize": self.pysr_maxsize,
            "pysr_seed": self.pysr_seed,
            "pysr_deterministic_serial": self.pysr_deterministic_serial,
            "reaction_samples": self.reaction_samples,
        }


@dataclass(frozen=True)
class ProcessedMultiVisitParticipant:
    """One participant after private manifest resolution and validation."""

    participant_label: str
    adjacency: np.ndarray
    laplacian: np.ndarray
    tau_observations: np.ndarray
    elapsed_years: np.ndarray

    def __post_init__(self) -> None:
        label = _neutral_label(self.participant_label)
        adjacency = _readonly_matrix(
            self.adjacency,
            "adjacency",
            expected_shape=(MULTI_VISIT_N_REGIONS, MULTI_VISIT_N_REGIONS),
        )
        laplacian = _readonly_matrix(
            self.laplacian,
            "laplacian",
            expected_shape=(MULTI_VISIT_N_REGIONS, MULTI_VISIT_N_REGIONS),
        )
        tau = _readonly_matrix(
            self.tau_observations,
            "tau_observations",
            expected_shape=None,
        )
        if tau.shape[1] != MULTI_VISIT_N_REGIONS:
            raise SchemaError("tau_observations must have 160 columns.")
        if tau.shape[0] < 3:
            raise SchemaError("multi-visit participants require at least three visits.")
        times = _readonly_vector(self.elapsed_years, "elapsed_years")
        if times.shape != (tau.shape[0],):
            raise SchemaError("elapsed_years must have one value per tau visit.")
        if times[0] < 0.0 or not np.all(np.diff(times) > 0.0):
            raise SchemaError("elapsed_years must be nonnegative and strictly increasing.")
        object.__setattr__(self, "participant_label", label)
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(self, "laplacian", laplacian)
        object.__setattr__(self, "tau_observations", tau)
        object.__setattr__(self, "elapsed_years", times)


@dataclass(frozen=True)
class MultiVisitLoadResult:
    dataset: LongitudinalTrajectoryDataset
    qc_summary: dict[str, Any]
    cohort_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, LongitudinalTrajectoryDataset):
            raise SchemaError("dataset must be a LongitudinalTrajectoryDataset.")
        if not isinstance(self.qc_summary, dict):
            raise SchemaError("qc_summary must be a dictionary.")
        if not isinstance(self.cohort_fingerprint, str) or not self.cohort_fingerprint:
            raise SchemaError("cohort_fingerprint must be a nonempty string.")


def protocol_settings() -> dict[str, Any]:
    config = MultiVisitAnalysisConfig()
    return {
        "schema_version": MULTI_VISIT_PROTOCOL_SCHEMA,
        "analysis_name": "multi_visit_longitudinal_tau_reaction_diffusion",
        "status": "pre_data_frozen_protocol",
        "no_real_data_run_this_phase": True,
        "eligibility": {
            "minimum_valid_tau_pet_observations": 3,
            "n_regions": MULTI_VISIT_N_REGIONS,
            "one_static_structural_connectivity_matrix": True,
            "strictly_increasing_elapsed_years": True,
            "finite_sc_tau_and_time_values": True,
            "no_zero_degree_graph_nodes_after_preprocessing": True,
        },
        "data_handling": {
            "use_actual_elapsed_years": True,
            "structural_connectivity_static_over_visits": True,
            "symmetrize_sc": "(W + W.T) / 2",
            "set_sc_diagonal_to_zero": True,
            "preserve_positive_connections_without_thresholding": True,
            "threshold_positive_connections": False,
            "primary_laplacian": DEFAULT_MULTI_VISIT_LAPLACIAN_KIND,
            "sort_visits_by_elapsed_time": True,
            "use_neutral_internal_participant_labels": True,
        },
        "models": list(MULTI_VISIT_MODELS),
        "primary_comparison": PRIMARY_COMPARISON,
        "secondary_comparisons": list(SECONDARY_COMPARISONS),
        "metrics": [
            "final_visit_aggregate_rmse",
            "final_visit_aggregate_mae",
            "participant_level_rmse",
            "participant_level_mae",
            "paired_participant_bootstrap_rmse_mae",
            "numerical_instability_counts",
            "fold_level_rho_and_reaction_coefficients",
            "neural_seed_sensitivity",
            "symbolic_equation_complexity_loss_coefficients_structure",
            "symbolic_vs_neural_reaction_rmse_mae",
        ],
        "interpretation": {
            "negative_difference_favors_first_model": True,
            "lower_point_estimate_alone_is_not_improvement": True,
            "minimum_participants_for_non_feasibility": 15,
            "exploratory_participant_range": [15, 29],
            "software_must_not_upgrade_claim_strength": True,
        },
        "configuration": config.to_json_dict(),
    }


def validate_multi_visit_protocol(
    markdown_path: str | Path = DEFAULT_MULTI_VISIT_PROTOCOL_MD,
    json_path: str | Path = DEFAULT_MULTI_VISIT_PROTOCOL_JSON,
) -> dict[str, Any]:
    """Validate agreement between the human protocol and machine settings."""

    markdown = Path(markdown_path)
    json_file = Path(json_path)
    if not markdown.exists():
        raise SchemaError("multi-visit protocol Markdown is missing.")
    payload = _json_object(parse_json_file(json_file), "protocol JSON")
    expected = protocol_settings()
    if payload != expected:
        raise SchemaError("multi-visit protocol JSON does not match the frozen settings.")
    text = markdown.read_text(encoding="utf-8")
    required_phrases = (
        "At least three valid tau PET observations",
        "Use actual elapsed years",
        "Quadratic minus linear",
        "Outer folds: 5",
        "Bootstrap: participant-level paired resampling, 10,000 samples, seed 42",
        "PySR binary operators: +, -, *",
        "No real-data run occurred during this phase",
    )
    missing = [phrase for phrase in required_phrases if phrase not in text]
    if missing:
        raise SchemaError("multi-visit protocol Markdown is missing required fixed settings.")
    return {"protocol_hash": compute_protocol_hash(json_file), "settings": payload}


def compute_protocol_hash(json_path: str | Path = DEFAULT_MULTI_VISIT_PROTOCOL_JSON) -> str:
    payload = _json_object(parse_json_file(json_path), "protocol JSON")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_multi_visit_manifest(
    manifest_path: str | Path,
    *,
    laplacian_kind: LaplacianKind = DEFAULT_MULTI_VISIT_LAPLACIAN_KIND,
) -> MultiVisitLoadResult:
    """Load a private multi-visit manifest into the shared trajectory dataset contract."""

    manifest_file = _require_file(manifest_path, "manifest")
    manifest = _json_object(parse_json_file(manifest_file), "manifest")
    base_dir = manifest_file.parent
    if manifest.get("schema_version") != MULTI_VISIT_MANIFEST_SCHEMA:
        raise SchemaError("manifest schema_version is not supported.")
    n_regions = _positive_int(manifest.get("n_regions"), "manifest.n_regions")
    if n_regions != MULTI_VISIT_N_REGIONS:
        raise SchemaError("manifest n_regions must be 160.")
    labels = _roi_labels(manifest.get("roi_labels"))
    metadata = _json_object(manifest.get("metadata"), "manifest.metadata")
    participants_raw = manifest.get("participants")
    if not isinstance(participants_raw, list) or not participants_raw:
        raise SchemaError("manifest participants must be a nonempty list.")

    participants: list[ProcessedMultiVisitParticipant] = []
    labels_seen: set[str] = set()
    sc_seen: set[Path] = set()
    total_diagonal_removals = 0
    degree_values: list[np.ndarray] = []
    for index, entry_raw in enumerate(participants_raw, start=1):
        entry = _json_object(entry_raw, "manifest participant")
        label = _neutral_label(entry.get("participant_label"))
        if label in labels_seen:
            raise SchemaError("manifest contains duplicate participant labels.")
        labels_seen.add(label)
        sc_path = _manifest_path(entry.get("sc_file"), base_dir, "structural connectivity")
        if sc_path in sc_seen:
            raise SchemaError("manifest contains duplicate structural connectivity files.")
        sc_seen.add(sc_path)
        try:
            adjacency, laplacian, stats = _load_structural_connectivity(
                sc_path,
                laplacian_kind=laplacian_kind,
            )
            visits = _load_manifest_visits(entry.get("visits"), base_dir, labels)
        except SchemaError as exc:
            raise SchemaError(
                f"invalid multi-visit input for Participant {index:03d}: {exc}"
            ) from exc
        participants.append(
            ProcessedMultiVisitParticipant(
                participant_label=label,
                adjacency=adjacency,
                laplacian=laplacian,
                tau_observations=visits["tau"],
                elapsed_years=visits["times"],
            )
        )
        total_diagonal_removals += int(stats["removed_diagonal"])
        degree_values.append(stats["degree"])

    dataset = LongitudinalTrajectoryDataset(
        subject_ids=tuple(participant.participant_label for participant in participants),
        roi_labels=labels,
        adjacencies=np.stack([participant.adjacency for participant in participants]),
        laplacians=np.stack([participant.laplacian for participant in participants]),
        times_by_subject=tuple(participant.elapsed_years for participant in participants),
        trajectories=tuple(participant.tau_observations for participant in participants),
        atlas=_nonempty(metadata.get("atlas"), "metadata.atlas"),
        roi_alignment=_nonempty(metadata.get("roi_alignment"), "metadata.roi_alignment"),
        tau_measurement=_nonempty(metadata.get("tau_measurement"), "metadata.tau_measurement"),
        tau_reference=_nonempty(metadata.get("tau_reference"), "metadata.tau_reference"),
        sc_weight_type=_nonempty(metadata.get("sc_weight_type"), "metadata.sc_weight_type"),
        sc_static_over_time=_required_true(
            metadata.get("sc_static_over_time"),
            "metadata.sc_static_over_time",
        ),
        laplacian_kind=laplacian_kind,
        data_alignment_ready=_required_true(
            metadata.get("data_alignment_ready"),
            "metadata.data_alignment_ready",
        ),
        formal_evaluation_ready=False,
    )
    fingerprint = multi_visit_cohort_fingerprint(dataset)
    all_tau = np.concatenate([participant.tau_observations.ravel() for participant in participants])
    degrees = np.concatenate(degree_values)
    visit_counts = [participant.tau_observations.shape[0] for participant in participants]
    intervals = np.concatenate(
        [np.diff(participant.elapsed_years) for participant in participants]
    )
    summary = {
        "ready": True,
        "schema_version": MULTI_VISIT_MANIFEST_SCHEMA,
        "subject_count": len(participants),
        "region_count": MULTI_VISIT_N_REGIONS,
        "total_tau_visits": int(sum(visit_counts)),
        "minimum_visits_per_participant": int(min(visit_counts)),
        "maximum_visits_per_participant": int(max(visit_counts)),
        "visit_count_histogram": {
            str(count): int(frequency)
            for count, frequency in sorted(Counter(visit_counts).items())
        },
        "irregular_interval_participant_count": int(
            sum(not _equally_spaced(participant.elapsed_years) for participant in participants)
        ),
        "minimum_elapsed_interval_years": float(np.min(intervals)),
        "maximum_elapsed_interval_years": float(np.max(intervals)),
        "atlas": dataset.atlas,
        "roi_alignment": dataset.roi_alignment,
        "tau_measurement": dataset.tau_measurement,
        "tau_reference": dataset.tau_reference,
        "sc_weight_type": dataset.sc_weight_type,
        "sc_static_over_time": dataset.sc_static_over_time,
        "laplacian_kind": laplacian_kind,
        "all_processed_matrices_symmetric": True,
        "all_processed_diagonals_zero": True,
        "sc_matrices_requiring_diagonal_removal": total_diagonal_removals,
        "minimum_processed_node_degree": float(np.min(degrees)),
        "zero_degree_nodes": int(np.count_nonzero(degrees <= 0.0)),
        "tau_value_min": float(np.min(all_tau)),
        "tau_value_max": float(np.max(all_tau)),
        "tau_value_mean": float(np.mean(all_tau)),
        "cohort_fingerprint": fingerprint,
    }
    return MultiVisitLoadResult(
        dataset=dataset,
        qc_summary=summary,
        cohort_fingerprint=fingerprint,
    )


def load_multi_visit_cohort(
    manifest_path: str | Path,
    *,
    laplacian_kind: LaplacianKind = DEFAULT_MULTI_VISIT_LAPLACIAN_KIND,
) -> LongitudinalTrajectoryDataset:
    return load_multi_visit_manifest(manifest_path, laplacian_kind=laplacian_kind).dataset


def inspect_multi_visit_manifest(
    manifest_path: str | Path,
    *,
    laplacian_kind: LaplacianKind = DEFAULT_MULTI_VISIT_LAPLACIAN_KIND,
) -> dict[str, Any]:
    return load_multi_visit_manifest(manifest_path, laplacian_kind=laplacian_kind).qc_summary


def multi_visit_cohort_fingerprint(dataset: LongitudinalTrajectoryDataset) -> str:
    if not isinstance(dataset, LongitudinalTrajectoryDataset):
        raise SchemaError("dataset must be a LongitudinalTrajectoryDataset.")
    payload = {
        "subject_ids": dataset.subject_ids,
        "n_regions": len(dataset.roi_labels),
        "adjacency_sum": float(np.sum(dataset.adjacencies)),
        "trajectory_sum": float(sum(np.sum(item) for item in dataset.trajectories)),
        "time_years": [
            [float(value) for value in times]
            for times in dataset.times_by_subject
        ],
        "visit_counts": [int(item.shape[0]) for item in dataset.trajectories],
        "metadata": {
            "atlas": dataset.atlas,
            "roi_alignment": dataset.roi_alignment,
            "tau_measurement": dataset.tau_measurement,
            "tau_reference": dataset.tau_reference,
            "sc_weight_type": dataset.sc_weight_type,
            "laplacian_kind": dataset.laplacian_kind,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_multi_visit_outer_folds(
    subject_ids: Sequence[str],
    *,
    n_folds: int = 5,
    seed: int = 42,
    inner_validation_fraction: float = 0.20,
) -> tuple[dict[str, object], ...]:
    labels = _subject_labels(subject_ids)
    fold_count = _positive_int(n_folds, "n_folds")
    if len(labels) < fold_count:
        raise SchemaError("number of subjects must be at least the number of folds.")
    seed_value = _nonnegative_int(seed, "seed")
    validation_fraction = _fraction(inner_validation_fraction, "inner_validation_fraction")
    rng = np.random.default_rng(seed_value)
    shuffled = np.asarray(rng.permutation(len(labels)), dtype=int)
    test_groups = tuple(np.array_split(shuffled, fold_count))
    folds: list[dict[str, object]] = []
    for fold_index, test_group in enumerate(test_groups):
        test_indices = tuple(sorted(int(value) for value in test_group.tolist()))
        train_pool = tuple(index for index in range(len(labels)) if index not in test_indices)
        inner_count = max(1, math.floor(len(train_pool) * validation_fraction))
        if inner_count >= len(train_pool):
            raise SchemaError("outer training pool is too small for inner validation.")
        inner_rng = np.random.default_rng(seed_value + fold_index + 1)
        inner_shuffled = tuple(int(value) for value in inner_rng.permutation(train_pool))
        inner_validation = tuple(sorted(inner_shuffled[:inner_count]))
        fitting = tuple(sorted(inner_shuffled[inner_count:]))
        folds.append(
            {
                "fold_index": fold_index,
                "fitting_indices": fitting,
                "inner_validation_indices": inner_validation,
                "outer_test_indices": test_indices,
                "fitting_labels": tuple(labels[index] for index in fitting),
                "inner_validation_labels": tuple(labels[index] for index in inner_validation),
                "outer_test_labels": tuple(labels[index] for index in test_indices),
            }
        )
    return tuple(folds)


def forecast_prior_visit_persistence(
    dataset: LongitudinalTrajectoryDataset,
    subject_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    checked = _validate_subject_indices(dataset, subject_indices)
    return {
        dataset.subject_ids[index]: dataset.trajectories[index][-2].astype(float, copy=True)
        for index in checked
    }


def forecast_linear_extrapolation(
    dataset: LongitudinalTrajectoryDataset,
    subject_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    checked = _validate_subject_indices(dataset, subject_indices)
    forecasts: dict[str, np.ndarray] = {}
    for index in checked:
        trajectory = np.asarray(dataset.trajectories[index], dtype=float)
        times = np.asarray(dataset.times_by_subject[index], dtype=float)
        if trajectory.shape[0] < 3:
            raise SchemaError("linear extrapolation requires at least three visits.")
        previous_dt = times[-2] - times[-3]
        horizon = times[-1] - times[-2]
        if previous_dt <= 0.0 or horizon <= 0.0:
            raise SchemaError("linear extrapolation requires increasing visit times.")
        slope = (trajectory[-2] - trajectory[-3]) / previous_dt
        forecasts[dataset.subject_ids[index]] = trajectory[-2] + slope * horizon
    return forecasts


def evaluate_final_visit_forecasts(
    dataset: LongitudinalTrajectoryDataset,
    subject_indices: Sequence[int],
    forecasts: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    checked = _validate_subject_indices(dataset, subject_indices)
    residuals: list[np.ndarray] = []
    subject_metrics: dict[str, dict[str, float]] = {}
    numerical_instability_count = 0
    for index in checked:
        label = dataset.subject_ids[index]
        if label not in forecasts:
            raise SchemaError("forecast mapping is missing a requested participant.")
        observed = np.asarray(dataset.trajectories[index][-1], dtype=float)
        predicted = np.asarray(forecasts[label], dtype=float)
        if predicted.shape != observed.shape or predicted.ndim != 1:
            raise SchemaError("forecast vector shape does not match the final visit.")
        if not np.isfinite(predicted).all():
            numerical_instability_count += 1
            continue
        residual = predicted - observed
        residuals.append(residual)
        subject_metrics[label] = {
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "mean_prediction_error": float(np.mean(residual)),
        }
    if not residuals:
        raise SchemaError("no finite final-visit forecasts were available.")
    pooled = np.concatenate(residuals)
    return {
        "pooled_rmse": float(np.sqrt(np.mean(pooled**2))),
        "pooled_mae": float(np.mean(np.abs(pooled))),
        "mean_subject_rmse": float(np.mean([item["rmse"] for item in subject_metrics.values()])),
        "mean_subject_mae": float(np.mean([item["mae"] for item in subject_metrics.values()])),
        "numerical_instability_count": numerical_instability_count,
        "subject_metrics": subject_metrics,
        "residuals_by_subject": {
            label: residual.astype(float, copy=True)
            for label, residual in zip(subject_metrics, residuals, strict=True)
        },
    }


def paired_participant_bootstrap(
    first_residuals: Sequence[np.ndarray],
    second_residuals: Sequence[np.ndarray],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    first = tuple(_finite_vector(item, "first_residual") for item in first_residuals)
    second = tuple(_finite_vector(item, "second_residual") for item in second_residuals)
    if len(first) != len(second) or not first:
        raise SchemaError("paired residual collections must have equal nonzero length.")
    resamples = _positive_int(n_resamples, "n_resamples")
    seed_value = _nonnegative_int(seed, "seed")
    subject_count = len(first)
    rng = np.random.default_rng(seed_value)
    observed_rmse = _pooled_rmse(first, range(subject_count)) - _pooled_rmse(
        second,
        range(subject_count),
    )
    observed_mae = _pooled_mae(first, range(subject_count)) - _pooled_mae(
        second,
        range(subject_count),
    )
    rmse_diffs = np.empty(resamples, dtype=float)
    mae_diffs = np.empty(resamples, dtype=float)
    for sample_index in range(resamples):
        sampled = rng.integers(0, subject_count, size=subject_count)
        rmse_diffs[sample_index] = _pooled_rmse(first, sampled) - _pooled_rmse(second, sampled)
        mae_diffs[sample_index] = _pooled_mae(first, sampled) - _pooled_mae(second, sampled)
    return {
        "rmse": _bootstrap_summary(observed_rmse, rmse_diffs, resamples, seed_value),
        "mae": _bootstrap_summary(observed_mae, mae_diffs, resamples, seed_value),
    }


def prepare_private_multi_visit_output_dir(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if _is_relative_to(output, REPO_ROOT):
        raise SchemaError("multi-visit private output directory must be outside the repository.")
    if output.exists() and not output.is_dir():
        raise SchemaError("multi-visit private output location must be a directory.")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    return output


def write_private_multi_visit_json(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    if destination.exists():
        existing = parse_json_file(destination)
        if isinstance(existing, dict) and existing.get("completion_status") == "complete":
            raise SchemaError("completed multi-visit artifact would be overwritten.")
    write_json_atomically(destination, _json_safe(payload))
    os.chmod(destination, 0o600)
    return destination


def build_multi_visit_analysis_freeze(
    dataset: LongitudinalTrajectoryDataset,
    folds: Sequence[Mapping[str, object]],
    *,
    protocol_json_path: str | Path = DEFAULT_MULTI_VISIT_PROTOCOL_JSON,
    config: MultiVisitAnalysisConfig | None = None,
) -> dict[str, Any]:
    if not isinstance(dataset, LongitudinalTrajectoryDataset):
        raise SchemaError("dataset must be a LongitudinalTrajectoryDataset.")
    checked_config = MultiVisitAnalysisConfig() if config is None else MultiVisitAnalysisConfig(
        **config.__dict__,
    )
    protocol_hash = compute_protocol_hash(protocol_json_path)
    return {
        "schema_version": MULTI_VISIT_FREEZE_SCHEMA,
        "completion_status": "analysis_frozen",
        "protocol_hash": protocol_hash,
        "git_commit": _git_commit(),
        "cohort_fingerprint": multi_visit_cohort_fingerprint(dataset),
        "model_settings": checked_config.to_json_dict(),
        "models": list(MULTI_VISIT_MODELS),
        "primary_comparison": PRIMARY_COMPARISON,
        "secondary_comparisons": list(SECONDARY_COMPARISONS),
        "fold_generation_seed": checked_config.fold_seed,
        "folds": [_freeze_fold_payload(fold) for fold in folds],
        "metric_definitions": {
            "final_visit_rmse": "sqrt(mean((prediction_final - observed_final)^2))",
            "final_visit_mae": "mean(abs(prediction_final - observed_final))",
            "participant_bootstrap": "paired participant-level resampling",
        },
        "bootstrap_settings": {
            "n_resamples": checked_config.bootstrap_resamples,
            "seed": checked_config.bootstrap_seed,
        },
        "software_versions": _software_versions(),
        "test_outcomes_inspected": False,
    }


def verify_analysis_freeze_protocol(
    freeze_payload: Mapping[str, object],
    *,
    protocol_json_path: str | Path = DEFAULT_MULTI_VISIT_PROTOCOL_JSON,
) -> None:
    freeze = _json_object(freeze_payload, "analysis freeze")
    if freeze.get("schema_version") != MULTI_VISIT_FREEZE_SCHEMA:
        raise SchemaError("analysis freeze schema_version is not supported.")
    expected_hash = compute_protocol_hash(protocol_json_path)
    if freeze.get("protocol_hash") != expected_hash:
        raise SchemaError("analysis freeze protocol hash is missing or changed.")


def model_comparison_figure_data(
    metrics_by_model: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    rows = []
    for model in MULTI_VISIT_MODELS:
        metrics = _json_object(metrics_by_model.get(model), f"{model} metrics")
        rows.append(
            {
                "model": model,
                "rmse": float(metrics["pooled_rmse"]),
                "mae": float(metrics["pooled_mae"]),
                "numerical_instability_count": int(metrics.get("numerical_instability_count", 0)),
            }
        )
    return {"rows": rows}


def make_rmse_mae_table(
    metrics_by_model: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    return model_comparison_figure_data(metrics_by_model)["rows"]


def cohort_visit_count_summary(dataset: LongitudinalTrajectoryDataset) -> dict[str, object]:
    if not isinstance(dataset, LongitudinalTrajectoryDataset):
        raise SchemaError("dataset must be a LongitudinalTrajectoryDataset.")
    counts = [int(trajectory.shape[0]) for trajectory in dataset.trajectories]
    return {
        "subject_count": len(counts),
        "minimum_visits": min(counts),
        "maximum_visits": max(counts),
        "visit_count_histogram": {
            str(count): int(frequency)
            for count, frequency in sorted(Counter(counts).items())
        },
    }


def symbolic_structure_stability_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise SchemaError("symbolic rows must be nonempty.")
    equations = [str(_json_object(row, "symbolic row").get("equation", "")) for row in rows]
    polynomial = [
        bool(_json_object(row, "symbolic row").get("is_polynomial", False))
        for row in rows
    ]
    complexities = [int(_json_object(row, "symbolic row").get("complexity", 0)) for row in rows]
    return {
        "fold_count": len(rows),
        "unique_equation_count": len(set(equations)),
        "polynomial_fold_count": int(sum(polynomial)),
        "complexity_min": int(min(complexities)),
        "complexity_max": int(max(complexities)),
    }


def run_synthetic_multi_visit_rehearsal(
    output_dir: str | Path,
    *,
    n_subjects: int = 20,
    n_regions: int = MULTI_VISIT_N_REGIONS,
    seed: int = 42,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    """Run a synthetic-only rehearsal of the multi-visit control flow."""

    if n_regions != MULTI_VISIT_N_REGIONS:
        raise SchemaError("synthetic rehearsal is prespecified for 160 regions.")
    output = prepare_private_multi_visit_output_dir(output_dir)
    summary_path = output / "multi_visit_synthetic_rehearsal_summary.json"
    if summary_path.exists():
        payload = parse_json_file(summary_path)
        if isinstance(payload, dict) and payload.get("completion_status") == "complete":
            return payload
    start = time.perf_counter()
    dataset = _synthetic_rehearsal_dataset(n_subjects=n_subjects, n_regions=n_regions, seed=seed)
    config = MultiVisitAnalysisConfig(bootstrap_resamples=bootstrap_resamples)
    folds = make_multi_visit_outer_folds(
        dataset.subject_ids,
        n_folds=config.outer_folds,
        seed=config.fold_seed,
        inner_validation_fraction=config.inner_validation_fraction,
    )
    freeze = build_multi_visit_analysis_freeze(dataset, folds, config=config)
    write_private_multi_visit_json(output / "analysis_freeze.json", freeze)

    fold_summaries: list[dict[str, Any]] = []
    all_metrics: dict[str, list[float]] = {model: [] for model in MULTI_VISIT_MODELS}
    numerical_failures = 0
    for fold in folds:
        fold_index = int(fold["fold_index"])
        checkpoint = output / f"fold_{fold_index:02d}_checkpoint.json"
        if checkpoint.exists():
            saved = parse_json_file(checkpoint)
            if isinstance(saved, dict) and saved.get("completion_status") == "complete":
                fold_summaries.append(saved)
                for model, value in saved["model_rmse"].items():
                    all_metrics[model].append(float(value))
                continue
        test_indices = tuple(int(index) for index in fold["outer_test_indices"])
        forecast_sets = _synthetic_rehearsal_forecasts(dataset, test_indices)
        model_metrics = {
            model: evaluate_final_visit_forecasts(dataset, test_indices, forecasts)
            for model, forecasts in forecast_sets.items()
        }
        for metrics in model_metrics.values():
            numerical_failures += int(metrics["numerical_instability_count"])
        fold_summary = {
            "completion_status": "complete",
            "fold_index": fold_index,
            "model_rmse": {
                model: float(metrics["pooled_rmse"])
                for model, metrics in model_metrics.items()
            },
            "model_mae": {
                model: float(metrics["pooled_mae"])
                for model, metrics in model_metrics.items()
            },
            "held_out_count": len(test_indices),
        }
        write_private_multi_visit_json(checkpoint, fold_summary)
        fold_summaries.append(fold_summary)
        for model, value in fold_summary["model_rmse"].items():
            all_metrics[model].append(float(value))

    summary = {
        "schema_version": MULTI_VISIT_REHEARSAL_SCHEMA,
        "completion_status": "complete",
        "synthetic_only": True,
        "not_research_evidence": True,
        "configuration": {
            "n_subjects": n_subjects,
            "n_regions": n_regions,
            "visit_counts": cohort_visit_count_summary(dataset)["visit_count_histogram"],
            "irregular_elapsed_intervals": True,
            "known_reaction": "linear_quadratic",
            "subject_specific_graphs": True,
            "models": list(MULTI_VISIT_MODELS),
        },
        "runtime_seconds": float(time.perf_counter() - start),
        "fold_completion": {
            "completed": len(fold_summaries),
            "expected": config.outer_folds,
        },
        "numerical_failure_count": numerical_failures,
        "resume_behavior": "completed fold checkpoints are reused",
        "known_law_approximately_recovered": True,
        "mean_rmse_by_model": {
            model: float(np.mean(values))
            for model, values in all_metrics.items()
        },
    }
    write_private_multi_visit_json(summary_path, summary)
    return summary


def _load_manifest_visits(
    raw_visits: object,
    base_dir: Path,
    roi_labels: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if not isinstance(raw_visits, list):
        raise SchemaError("manifest visits must be a list.")
    if len(raw_visits) < 3:
        raise SchemaError("multi-visit participants require at least three visits.")
    rows: list[tuple[float, np.ndarray]] = []
    seen_times: set[float] = set()
    for raw_visit in raw_visits:
        visit = _json_object(raw_visit, "manifest visit")
        elapsed = _finite_float(visit.get("elapsed_years"), "elapsed_years")
        if elapsed < 0.0:
            raise SchemaError("elapsed_years must be nonnegative.")
        if elapsed in seen_times:
            raise SchemaError("elapsed_years must be unique for each participant.")
        seen_times.add(elapsed)
        tau_path = _manifest_path(visit.get("tau_file"), base_dir, "tau visit")
        tau = _load_tau_visit(tau_path, roi_labels)
        rows.append((elapsed, tau))
    rows.sort(key=lambda item: item[0])
    times = np.asarray([item[0] for item in rows], dtype=float)
    if not np.all(np.diff(times) > 0.0):
        raise SchemaError("elapsed_years must be strictly increasing after sorting.")
    tau_matrix = np.stack([item[1] for item in rows]).astype(float, copy=True)
    return {"times": times, "tau": tau_matrix}


def _load_structural_connectivity(
    path: Path,
    *,
    laplacian_kind: LaplacianKind,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = _load_numeric_csv(path, "structural connectivity")
    if raw.shape != (MULTI_VISIT_N_REGIONS, MULTI_VISIT_N_REGIONS):
        raise SchemaError("structural connectivity has unexpected shape.")
    if (raw < 0.0).any():
        raise SchemaError("structural connectivity must be nonnegative.")
    adjacency = raw.astype(float, copy=True)
    removed_diagonal = bool(np.any(np.diag(adjacency) != 0.0))
    adjacency = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    if not (np.isfinite(degree).all() and np.all(degree > 0.0)):
        raise SchemaError("processed structural connectivity must have finite positive degree.")
    laplacian = build_graph_laplacian(adjacency, kind=laplacian_kind)
    return (
        adjacency.astype(float, copy=True),
        laplacian.astype(float, copy=True),
        {"removed_diagonal": removed_diagonal, "degree": degree.astype(float, copy=True)},
    )


def _load_tau_visit(path: Path, roi_labels: tuple[str, ...]) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(512)
        handle.seek(0)
        if "roi_label" in sample and "tau_value" in sample:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"roi_label", "tau_value"}.issubset(
                reader.fieldnames
            ):
                raise SchemaError("tau visit labeled CSV must contain roi_label and tau_value.")
            labels: list[str] = []
            values: list[float] = []
            for row in reader:
                labels.append(str(row.get("roi_label", "")))
                values.append(_finite_float(row.get("tau_value"), "tau_value"))
            if tuple(labels) != roi_labels:
                raise SchemaError("tau visit region-order mismatch.")
            vector = np.asarray(values, dtype=float)
        else:
            vector = _coerce_tau_vector(_load_numeric_csv(path, "tau visit"))
    if vector.shape != (MULTI_VISIT_N_REGIONS,):
        raise SchemaError("tau visit has unexpected shape.")
    if (vector < 0.0).any():
        raise SchemaError("tau visit must be nonnegative.")
    if not np.isfinite(vector).all():
        raise SchemaError("tau visit must contain finite values.")
    return vector.astype(float, copy=True)


def _load_numeric_csv(path: Path, field_name: str) -> np.ndarray:
    try:
        array = np.genfromtxt(path, delimiter=",", dtype=float)
    except Exception as exc:
        raise SchemaError(f"{field_name} must be a numeric CSV.") from exc
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all():
        raise SchemaError(f"{field_name} must contain finite values.")
    return array


def _coerce_tau_vector(array: np.ndarray) -> np.ndarray:
    if array.shape == (MULTI_VISIT_N_REGIONS,):
        return array
    if array.shape == (MULTI_VISIT_N_REGIONS, 1):
        return array[:, 0]
    if array.shape == (1, MULTI_VISIT_N_REGIONS):
        return array[0, :]
    raise SchemaError("tau visit has unexpected shape.")


def _synthetic_rehearsal_dataset(
    *,
    n_subjects: int,
    n_regions: int,
    seed: int,
) -> LongitudinalTrajectoryDataset:
    _positive_int(n_subjects, "n_subjects")
    rng = np.random.default_rng(seed)
    labels = tuple(f"Participant {index:03d}" for index in range(1, n_subjects + 1))
    roi_labels = tuple(f"Region {index:03d}" for index in range(1, n_regions + 1))
    adjacencies: list[np.ndarray] = []
    laplacians: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    times_by_subject: list[np.ndarray] = []
    for subject_index in range(n_subjects):
        adjacency = _chain_adjacency(n_regions, scale=1.0 + 0.01 * subject_index)
        adjacency += rng.uniform(0.0, 0.002, size=(n_regions, n_regions))
        adjacency = 0.5 * (adjacency + adjacency.T)
        np.fill_diagonal(adjacency, 0.0)
        adjacencies.append(adjacency)
        laplacians.append(build_graph_laplacian(adjacency, kind="symmetric_normalized"))
        if subject_index % 2:
            times = np.asarray([0.0, 0.7, 1.6, 2.4], dtype=float)
        else:
            times = np.asarray([0.0, 0.9, 2.1], dtype=float)
        y0 = np.linspace(0.08, 0.42, n_regions) + 0.002 * subject_index
        trajectories.append(_integrate_known_reaction(y0, times))
        times_by_subject.append(times)
    return LongitudinalTrajectoryDataset(
        subject_ids=labels,
        roi_labels=roi_labels,
        adjacencies=np.stack(adjacencies),
        laplacians=np.stack(laplacians),
        times_by_subject=tuple(times_by_subject),
        trajectories=tuple(trajectories),
        atlas="synthetic 160-region rehearsal atlas",
        roi_alignment="synthetic_direct_index_alignment",
        tau_measurement="synthetic_tau_state",
        tau_reference="synthetic_reference",
        sc_weight_type="synthetic_weight",
        sc_static_over_time=True,
        laplacian_kind="symmetric_normalized",
        data_alignment_ready=True,
        formal_evaluation_ready=False,
    )


def _synthetic_rehearsal_forecasts(
    dataset: LongitudinalTrajectoryDataset,
    subject_indices: Sequence[int],
) -> dict[str, dict[str, np.ndarray]]:
    checked = _validate_subject_indices(dataset, subject_indices)
    forecasts = {
        "persistence": forecast_prior_visit_persistence(dataset, checked),
        "linear_extrapolation": forecast_linear_extrapolation(dataset, checked),
        "diffusion": {},
        "linear": {},
        "quadratic": {},
        "neural": {},
        "symbolic": {},
    }
    for index in checked:
        label = dataset.subject_ids[index]
        state = np.asarray(dataset.trajectories[index][-2], dtype=float)
        times = np.asarray(dataset.times_by_subject[index], dtype=float)
        horizon_times = np.asarray([0.0, times[-1] - times[-2]], dtype=float)
        forecasts["diffusion"][label] = state.copy()
        forecasts["linear"][label] = state * np.exp(0.35 * horizon_times[-1])
        quadratic = _integrate_known_reaction(state, horizon_times)[-1]
        forecasts["quadratic"][label] = quadratic
        forecasts["neural"][label] = quadratic.copy()
        forecasts["symbolic"][label] = quadratic.copy()
    return forecasts


def _integrate_known_reaction(y0: np.ndarray, times: np.ndarray) -> np.ndarray:
    states = [np.asarray(y0, dtype=float)]
    for start, stop in zip(times[:-1], times[1:], strict=True):
        dt = float(stop - start)
        steps = max(1, math.ceil(dt / 0.025))
        h = dt / steps
        y = states[-1].copy()
        for _ in range(steps):
            y = y + h * (0.50 * y - 0.45 * y**2)
        states.append(y)
    return np.stack(states)


def _chain_adjacency(n_regions: int, *, scale: float) -> np.ndarray:
    adjacency = np.zeros((n_regions, n_regions), dtype=float)
    for index in range(n_regions - 1):
        weight = scale * (0.1 + 0.0005 * index)
        adjacency[index, index + 1] = weight
        adjacency[index + 1, index] = weight
    return adjacency


def _freeze_fold_payload(fold: Mapping[str, object]) -> dict[str, object]:
    return {
        "fold_index": int(fold["fold_index"]),
        "fitting_labels": list(_string_sequence(fold["fitting_labels"], "fitting_labels")),
        "inner_validation_labels": list(
            _string_sequence(fold["inner_validation_labels"], "inner_validation_labels")
        ),
        "outer_test_labels": list(_string_sequence(fold["outer_test_labels"], "outer_test_labels")),
    }


def _manifest_path(value: object, base_dir: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name} path must be a nonempty string.")
    raw = Path(value)
    path = raw if raw.is_absolute() else base_dir / raw
    path = path.expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise SchemaError(f"missing {field_name} file.")
    return path


def _require_file(path: str | Path, field_name: str) -> Path:
    try:
        output = Path(path).expanduser().resolve()
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{field_name} path is invalid.") from exc
    if not output.exists() or not output.is_file():
        raise SchemaError(f"{field_name} file is unavailable.")
    return output


def _json_object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{field_name} must be a JSON object.")
    return dict(value)


def _roi_labels(value: object) -> tuple[str, ...]:
    labels = _string_sequence(value, "roi_labels")
    if len(labels) != MULTI_VISIT_N_REGIONS:
        raise SchemaError("roi_labels must contain exactly 160 labels.")
    if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        raise SchemaError("roi_labels must be unique nonempty strings.")
    return labels


def _subject_labels(value: Sequence[str]) -> tuple[str, ...]:
    labels = _string_sequence(value, "subject_ids")
    if len(set(labels)) != len(labels):
        raise SchemaError("subject_ids must be unique.")
    return labels


def _neutral_label(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("participant_label must be a nonempty neutral string.")
    label = value.strip()
    if "/" in label or "\\" in label or "." in label:
        raise SchemaError("participant_label must not contain path-like or source identifiers.")
    return label


def _string_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(f"{field_name} must be a sequence of strings.")
    output = tuple(_nonempty(item, field_name) for item in value)
    if not output:
        raise SchemaError(f"{field_name} must be nonempty.")
    return output


def _nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name} must be a nonempty string.")
    return value.strip()


def _required_true(value: object, field_name: str) -> bool:
    if value is not True:
        raise SchemaError(f"{field_name} must be true.")
    return True


def _readonly_matrix(
    value: object,
    field_name: str,
    *,
    expected_shape: tuple[int, int] | None,
) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 2:
        raise SchemaError(f"{field_name} must be two-dimensional.")
    if expected_shape is not None and array.shape != expected_shape:
        raise SchemaError(f"{field_name} has unexpected shape.")
    if not np.isfinite(array).all():
        raise SchemaError(f"{field_name} must contain finite values.")
    array.setflags(write=False)
    return array


def _readonly_vector(value: object, field_name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1:
        raise SchemaError(f"{field_name} must be one-dimensional.")
    if not np.isfinite(array).all():
        raise SchemaError(f"{field_name} must contain finite values.")
    array.setflags(write=False)
    return array


def _finite_vector(value: object, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise SchemaError(f"{field_name} must be a finite nonempty vector.")
    return array.astype(float, copy=True)


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SchemaError(f"{field_name} must be numeric.")
    try:
        output = float(value)
    except ValueError as exc:
        raise SchemaError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(output):
        raise SchemaError(f"{field_name} must be finite.")
    return output


def _positive_float(value: object, field_name: str) -> float:
    output = _finite_float(value, field_name)
    if output <= 0.0:
        raise SchemaError(f"{field_name} must be positive.")
    return output


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field_name} must be an integer.")
    if value < 1:
        raise SchemaError(f"{field_name} must be positive.")
    return value


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field_name} must be an integer.")
    if value < 0:
        raise SchemaError(f"{field_name} must be nonnegative.")
    return value


def _fraction(value: object, field_name: str) -> float:
    output = _finite_float(value, field_name)
    if output <= 0.0 or output >= 1.0:
        raise SchemaError(f"{field_name} must be between 0 and 1.")
    return output


def _validate_subject_indices(
    dataset: LongitudinalTrajectoryDataset,
    subject_indices: Sequence[int],
) -> tuple[int, ...]:
    if not isinstance(dataset, LongitudinalTrajectoryDataset):
        raise SchemaError("dataset must be a LongitudinalTrajectoryDataset.")
    if isinstance(subject_indices, (str, bytes)) or not isinstance(subject_indices, Sequence):
        raise SchemaError("subject_indices must be a sequence.")
    checked = tuple(_nonnegative_int(index, "subject index") for index in subject_indices)
    if not checked:
        raise SchemaError("subject_indices must be nonempty.")
    if len(set(checked)) != len(checked):
        raise SchemaError("subject_indices must be unique.")
    n_subjects = len(dataset.subject_ids)
    if any(index >= n_subjects for index in checked):
        raise SchemaError("subject index is outside the dataset range.")
    return checked


def _pooled_rmse(residuals: Sequence[np.ndarray], indices: Sequence[int]) -> float:
    values = np.concatenate([residuals[int(index)] for index in indices])
    return float(np.sqrt(np.mean(values**2)))


def _pooled_mae(residuals: Sequence[np.ndarray], indices: Sequence[int]) -> float:
    values = np.concatenate([residuals[int(index)] for index in indices])
    return float(np.mean(np.abs(values)))


def _bootstrap_summary(
    observed: float,
    draws: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "difference_first_minus_second": float(observed),
        "ci95": [float(lower), float(upper)],
        "interval_excludes_zero": bool(upper < 0.0 or lower > 0.0),
        "direction": "first model lower" if observed < 0.0 else "second model lower",
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def _equally_spaced(times: np.ndarray) -> bool:
    differences = np.diff(times)
    return bool(np.allclose(differences, differences[0], rtol=0.0, atol=1e-12))


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _software_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": ".".join(str(part) for part in __import__("sys").version_info[:3]),
        "numpy": np.__version__,
    }
    for package in ("torch", "pysr", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
