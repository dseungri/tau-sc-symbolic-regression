"""Synthetic-cohort config helpers"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from tau_sr.schemas import SchemaError


@dataclass(frozen=True)
class SyntheticCohortConfig:
    n_subjects: int = 12
    n_regions: int = 148
    visits_per_subject: int = 3
    seed: int = 42
    edge_probability: float = 0.08
    baseline_tau_min: float = 0.05
    baseline_tau_max: float = 0.40
    add_observation_noise: bool = False
    observation_noise_std: float = 0.0

    def __post_init__(self) -> None:
        validate_synthetic_config(self)


def validate_synthetic_config(config: SyntheticCohortConfig) -> SyntheticCohortConfig:
    """Validate synthetic-cohort configuration values."""

    if not isinstance(config, SyntheticCohortConfig):
        raise SchemaError("config must be a SyntheticCohortConfig.")

    _require_int(config.n_subjects, "n_subjects", minimum=1)
    _require_int(config.n_regions, "n_regions", minimum=1)
    _require_int(config.visits_per_subject, "visits_per_subject", minimum=2)
    _require_int(config.seed, "seed", minimum=0)
    _require_probability(config.edge_probability, "edge_probability")
    _require_float(config.baseline_tau_min, "baseline_tau_min", minimum=0.0)
    _require_float(config.baseline_tau_max, "baseline_tau_max", minimum=0.0)

    if config.baseline_tau_max <= config.baseline_tau_min:
        raise SchemaError("baseline_tau_max must be greater than baseline_tau_min.")
    if not isinstance(config.add_observation_noise, bool):
        raise SchemaError("add_observation_noise must be a bool.")

    _require_float(config.observation_noise_std, "observation_noise_std", minimum=0.0)
    if config.add_observation_noise and config.observation_noise_std == 0.0:
        raise SchemaError("observation_noise_std must be positive when observation noise is enabled.")

    return config


def generate_synthetic_roi_labels(
    config: SyntheticCohortConfig | None = None,
) -> tuple[str, ...]:
    """Generate fake ROI labels for a cohort config"""

    checked = SyntheticCohortConfig() if config is None else validate_synthetic_config(config)
    return tuple(f"syn_roi_{index + 1:03d}" for index in range(checked.n_regions))


def generate_count_connectome(
    n_regions: int,
    rng: np.random.Generator,
    *,
    edge_probability: float = 0.08,
    weight_min: float = 0.1,
    weight_max: float = 1.0,
) -> np.ndarray:
    """Generate one artificial symmetric W_count adjacency matrix for tests."""

    _require_int(n_regions, "n_regions", minimum=1)
    if not isinstance(rng, np.random.Generator):
        raise SchemaError("rng must be a numpy.random.Generator.")
    _require_probability(edge_probability, "edge_probability")
    _require_float(weight_min, "weight_min", minimum=0.0)
    _require_float(weight_max, "weight_max", minimum=0.0)
    if weight_min <= 0.0:
        raise SchemaError("weight_min must be greater than 0.")
    if weight_max <= weight_min:
        raise SchemaError("weight_max must be greater than weight_min.")

    edge_mask = rng.random((n_regions, n_regions)) < edge_probability
    edge_mask = np.triu(edge_mask, k=1)
    weights = rng.uniform(weight_min, weight_max, size=(n_regions, n_regions))
    if n_regions > 1:
        chain_indices = np.arange(n_regions - 1)
        edge_mask[chain_indices, chain_indices + 1] = True
    upper = np.triu(weights * edge_mask, k=1)
    return upper + upper.T


def generate_edge_attributes(
    count_matrix: np.ndarray,
    rng: np.random.Generator,
    *,
    fa_min: float = 0.2,
    fa_max: float = 0.8,
    length_min: float = 20.0,
    length_max: float = 150.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic FA and tract-length matrices on count support."""

    _validate_count_matrix(count_matrix)
    if not isinstance(rng, np.random.Generator):
        raise SchemaError("rng must be a numpy.random.Generator.")
    _validate_fa_bounds(fa_min, fa_max)
    _validate_length_bounds(length_min, length_max)

    fa_matrix = np.zeros(count_matrix.shape, dtype=float)
    length_matrix = np.zeros(count_matrix.shape, dtype=float)

    support = count_matrix > 0.0
    upper_support = np.triu(support, k=1)
    rows, cols = np.nonzero(upper_support)

    fa_values = rng.uniform(fa_min, fa_max, size=len(rows))
    length_values = rng.uniform(length_min, length_max, size=len(rows))

    fa_matrix[rows, cols] = fa_values
    length_matrix[rows, cols] = length_values

    fa_matrix[cols, rows] = fa_values
    length_matrix[cols, rows] = length_values

    np.fill_diagonal(fa_matrix, 0.0)
    np.fill_diagonal(length_matrix, 0.0)

    return fa_matrix, length_matrix


def _require_int(value: object, field_name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field_name} must be an integer.")
    if value < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")


def _require_float(value: object, field_name: str, *, minimum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SchemaError(f"{field_name} must be numeric.")
    if not math.isfinite(float(value)):
        raise SchemaError(f"{field_name} must be finite.")
    if float(value) < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")


def _require_probability(value: object, field_name: str) -> None:
    _require_float(value, field_name, minimum=0.0)
    if float(value) > 1.0:
        raise SchemaError(f"{field_name} must be between 0 and 1.")


def _validate_count_matrix(count_matrix: np.ndarray) -> None:
    if not isinstance(count_matrix, np.ndarray):
        raise SchemaError("count_matrix must be a NumPy array.")
    if count_matrix.ndim != 2:
        raise SchemaError("count_matrix must be two-dimensional.")
    if count_matrix.shape[0] != count_matrix.shape[1]:
        raise SchemaError("count_matrix must be square.")
    if count_matrix.shape[0] < 2:
        raise SchemaError("count_matrix must have at least 2 rows.")
    if (
        count_matrix.dtype.kind in {"b", "O", "S", "U", "c"}
        or not np.issubdtype(count_matrix.dtype, np.number)
    ):
        raise SchemaError("count_matrix must have a real numeric dtype.")
    if not np.isfinite(count_matrix).all():
        raise SchemaError("count_matrix must contain only finite values.")
    if (count_matrix < 0.0).any():
        raise SchemaError("count_matrix must be nonnegative.")
    if not np.allclose(count_matrix, count_matrix.T, rtol=1e-7, atol=1e-12):
        raise SchemaError("count_matrix must be symmetric.")
    if not np.array_equal(np.diag(count_matrix), np.zeros(count_matrix.shape[0])):
        raise SchemaError("count_matrix diagonal must be exactly zero.")
    if not (count_matrix > 0.0).any():
        raise SchemaError("count_matrix must contain at least one positive edge.")


def _validate_fa_bounds(fa_min: float, fa_max: float) -> None:
    _require_float(fa_min, "fa_min", minimum=0.0)
    _require_float(fa_max, "fa_max", minimum=0.0)
    if fa_min <= 0.0:
        raise SchemaError("fa_min must be greater than 0.")
    if fa_max <= fa_min:
        raise SchemaError("fa_max must be greater than fa_min.")
    if fa_max > 1.0:
        raise SchemaError("fa_max must be at most 1.")


def _validate_length_bounds(length_min: float, length_max: float) -> None:
    _require_float(length_min, "length_min", minimum=0.0)
    _require_float(length_max, "length_max", minimum=0.0)
    if length_min <= 0.0:
        raise SchemaError("length_min must be greater than 0.")
    if length_max <= length_min:
        raise SchemaError("length_max must be greater than length_min.")
