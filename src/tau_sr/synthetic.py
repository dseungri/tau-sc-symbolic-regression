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
    upper = np.triu(weights * edge_mask, k=1)
    return upper + upper.T


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
