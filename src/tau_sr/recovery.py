"""Noise-free synthetic recovery-dataset assembly."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from tau_sr.graphs import build_graph_laplacian
from tau_sr.reactions import make_linear_quadratic_reaction
from tau_sr.schemas import SchemaError
from tau_sr.simulation import simulate_reaction_diffusion_cohort
from tau_sr.synthetic import (
    SyntheticCohortConfig,
    build_connectome_representations,
    generate_baseline_tau,
    generate_count_connectome,
    generate_edge_attributes,
    generate_synthetic_roi_labels,
    validate_synthetic_config,
)

SUPPORTED_CONNECTOME_REPRESENTATIONS: tuple[str, ...] = (
    "count",
    "fa_weighted",
    "length_adjusted",
)
SUPPORTED_LAPLACIAN_KINDS: tuple[str, ...] = (
    "unnormalized",
    "symmetric_normalized",
)


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


def build_noise_free_recovery_dataset(
    config: SyntheticCohortConfig,
    times_by_subject: Sequence[np.ndarray],
    *,
    representation: str,
    laplacian_kind: str,
    rho: float,
    growth_rate: float,
    saturation_rate: float,
    max_step: float = 0.05,
) -> SyntheticRecoveryDataset:
    """Build a deterministic noise-free synthetic recovery dataset."""

    checked_config = _validate_recovery_config(config)
    checked_times = _validate_times_by_subject(
        times_by_subject,
        n_subjects=checked_config.n_subjects,
    )
    _validate_representation(representation)
    _validate_laplacian_kind(laplacian_kind)
    rho_value = _require_real_scalar(rho, "rho", minimum=0.0)
    growth_rate_value = _require_real_scalar(growth_rate, "growth_rate", positive=True)
    saturation_rate_value = _require_real_scalar(
        saturation_rate,
        "saturation_rate",
        minimum=0.0,
    )
    max_step_value = _require_real_scalar(max_step, "max_step", positive=True)

    rng = np.random.default_rng(checked_config.seed)
    roi_labels = generate_synthetic_roi_labels(checked_config)
    subject_ids = tuple(
        f"synthetic_subject_{subject_index:04d}"
        for subject_index in range(checked_config.n_subjects)
    )

    adjacency_list: list[np.ndarray] = []
    laplacian_list: list[np.ndarray] = []
    baseline_list: list[np.ndarray] = []

    for _subject_index in range(checked_config.n_subjects):
        count_matrix = generate_count_connectome(
            checked_config.n_regions,
            rng,
            edge_probability=checked_config.edge_probability,
        )
        fa_matrix, length_matrix = generate_edge_attributes(count_matrix, rng)
        representations = build_connectome_representations(
            count_matrix,
            fa_matrix,
            length_matrix,
        )
        adjacency = representations[representation]
        laplacian = build_graph_laplacian(adjacency, kind=laplacian_kind)
        baseline = generate_baseline_tau(
            checked_config.n_regions,
            rng,
            tau_min=checked_config.baseline_tau_min,
            tau_max=checked_config.baseline_tau_max,
        )

        adjacency_list.append(adjacency.astype(float, copy=True))
        laplacian_list.append(laplacian.astype(float, copy=True))
        baseline_list.append(baseline.astype(float, copy=True))

    baseline_states = np.stack(baseline_list).astype(float, copy=True)
    adjacencies = np.stack(adjacency_list).astype(float, copy=True)
    laplacians = np.stack(laplacian_list).astype(float, copy=True)

    reaction_fn = make_linear_quadratic_reaction(
        growth_rate_value,
        saturation_rate_value,
    )
    trajectories = simulate_reaction_diffusion_cohort(
        baseline_states,
        checked_times,
        laplacians,
        rho_value,
        reaction_fn,
        max_step=max_step_value,
    )

    return SyntheticRecoveryDataset(
        subject_ids=subject_ids,
        roi_labels=roi_labels,
        baseline_states=baseline_states.copy(),
        adjacencies=adjacencies.copy(),
        laplacians=laplacians.copy(),
        times_by_subject=tuple(times.copy() for times in checked_times),
        trajectories=tuple(trajectory.copy() for trajectory in trajectories),
        representation=representation,
        laplacian_kind=laplacian_kind,
        rho=rho_value,
        growth_rate=growth_rate_value,
        saturation_rate=saturation_rate_value,
        seed=checked_config.seed,
    )


def _validate_recovery_config(config: object) -> SyntheticCohortConfig:
    if not isinstance(config, SyntheticCohortConfig):
        raise SchemaError("config must be a SyntheticCohortConfig.")
    checked_config = validate_synthetic_config(config)
    if checked_config.n_regions < 2:
        raise SchemaError("n_regions must be at least 2 for recovery datasets.")
    if checked_config.add_observation_noise:
        raise SchemaError("Observation noise must be disabled for noise-free recovery datasets.")
    return checked_config


def _validate_times_by_subject(
    times_by_subject: object,
    *,
    n_subjects: int,
) -> tuple[np.ndarray, ...]:
    if isinstance(times_by_subject, (str, bytes)):
        raise SchemaError("times_by_subject must be a sequence of NumPy time arrays.")
    if isinstance(times_by_subject, np.ndarray):
        if times_by_subject.ndim != 2:
            raise SchemaError("times_by_subject as a NumPy array must be two-dimensional.")
        raw_times = tuple(times_by_subject[index] for index in range(times_by_subject.shape[0]))
    elif isinstance(times_by_subject, Sequence):
        raw_times = tuple(times_by_subject)
    else:
        raise SchemaError("times_by_subject must be a sequence of NumPy time arrays.")

    if len(raw_times) != n_subjects:
        raise SchemaError(
            f"times_by_subject has {len(raw_times)} entries; expected {n_subjects}."
        )

    return tuple(
        _validate_subject_times(times, subject_index=subject_index)
        for subject_index, times in enumerate(raw_times)
    )


def _validate_subject_times(times: object, *, subject_index: int) -> np.ndarray:
    field_name = f"times_by_subject[{subject_index}]"
    if not isinstance(times, np.ndarray):
        raise SchemaError(f"{field_name} must be a NumPy array.")
    if times.ndim != 1:
        raise SchemaError(f"{field_name} must be one-dimensional.")
    if times.size == 0:
        raise SchemaError(f"{field_name} must be nonempty.")
    if not _is_real_numeric_dtype(times):
        raise SchemaError(f"{field_name} must have a real numeric dtype.")
    if not np.isfinite(times).all():
        raise SchemaError(f"{field_name} must contain only finite values.")
    times_float = times.astype(float, copy=False)
    if times_float.shape[0] > 1 and not np.all(np.diff(times_float) > 0.0):
        raise SchemaError(f"{field_name} must be strictly increasing.")
    return times.astype(float, copy=True)


def _validate_representation(representation: object) -> None:
    if representation not in SUPPORTED_CONNECTOME_REPRESENTATIONS:
        raise SchemaError(
            "representation must be one of: "
            f"{', '.join(SUPPORTED_CONNECTOME_REPRESENTATIONS)}."
        )


def _validate_laplacian_kind(laplacian_kind: object) -> None:
    if laplacian_kind not in SUPPORTED_LAPLACIAN_KINDS:
        raise SchemaError(
            "laplacian_kind must be one of: "
            f"{', '.join(SUPPORTED_LAPLACIAN_KINDS)}."
        )


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


def _is_real_numeric_dtype(array: np.ndarray) -> bool:
    if array.dtype.kind in {"b", "c", "O", "S", "U"}:
        return False
    return bool(np.issubdtype(array.dtype, np.number))
