from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tau_sr.schemas import SchemaError
from tau_sr.synthetic import (
    SyntheticCohortConfig,
    generate_count_connectome,
    generate_synthetic_roi_labels,
    validate_synthetic_config,
)


def test_synthetic_config_defaults_are_valid() -> None:
    config = SyntheticCohortConfig()

    assert validate_synthetic_config(config) is config
    assert config.n_subjects == 12
    assert config.n_regions == 148
    assert config.visits_per_subject == 3
    assert config.seed == 42
    assert config.edge_probability == 0.08
    assert config.baseline_tau_min == 0.05
    assert config.baseline_tau_max == 0.40
    assert config.add_observation_noise is False
    assert config.observation_noise_std == 0.0


def test_synthetic_config_is_frozen() -> None:
    config = SyntheticCohortConfig()

    with pytest.raises(FrozenInstanceError):
        config.seed = 7  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_subjects": 0}, "n_subjects"),
        ({"n_regions": 0}, "n_regions"),
        ({"visits_per_subject": 1}, "visits_per_subject"),
        ({"seed": -1}, "seed"),
        ({"edge_probability": -0.1}, "edge_probability"),
        ({"edge_probability": 1.1}, "edge_probability"),
        ({"baseline_tau_min": -0.01}, "baseline_tau_min"),
        ({"baseline_tau_min": 0.4, "baseline_tau_max": 0.4}, "baseline_tau_max"),
        ({"baseline_tau_max": float("inf")}, "baseline_tau_max"),
        ({"add_observation_noise": "yes"}, "add_observation_noise"),
        ({"observation_noise_std": -0.1}, "observation_noise_std"),
        ({"add_observation_noise": True, "observation_noise_std": 0.0}, "observation_noise_std"),
    ],
)
def test_synthetic_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SchemaError, match=message):
        SyntheticCohortConfig(**kwargs)  # type: ignore[arg-type]


def test_validate_synthetic_config_rejects_non_config_input() -> None:
    with pytest.raises(SchemaError, match="SyntheticCohortConfig"):
        validate_synthetic_config({"n_subjects": 12})  # type: ignore[arg-type]


def test_generate_synthetic_roi_labels_is_deterministic() -> None:
    config = SyntheticCohortConfig(n_regions=4, seed=123)

    labels = generate_synthetic_roi_labels(config)

    assert labels == ("syn_roi_001", "syn_roi_002", "syn_roi_003", "syn_roi_004")
    assert generate_synthetic_roi_labels(config) == labels


def test_generate_synthetic_roi_labels_uses_region_count_only() -> None:
    labels_a = generate_synthetic_roi_labels(SyntheticCohortConfig(n_regions=3, seed=1))
    labels_b = generate_synthetic_roi_labels(SyntheticCohortConfig(n_regions=3, seed=999))

    assert labels_a == labels_b
    assert labels_a[-1] == "syn_roi_003"


def test_generate_count_connectome_has_expected_adjacency_structure() -> None:
    matrix = generate_count_connectome(
        8,
        np.random.default_rng(123),
        edge_probability=1.0,
        weight_min=0.25,
        weight_max=0.75,
    )

    assert matrix.shape == (8, 8)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.isfinite(matrix).all()
    assert np.all(matrix >= 0.0)
    off_diagonal = matrix[~np.eye(8, dtype=bool)]
    assert np.all(off_diagonal >= 0.25)
    assert np.all(off_diagonal < 0.75)


def test_generate_count_connectome_can_generate_empty_graph() -> None:
    matrix = generate_count_connectome(5, np.random.default_rng(123), edge_probability=0.0)

    np.testing.assert_array_equal(matrix, np.zeros((5, 5)))


def test_generate_count_connectome_is_deterministic_for_seeded_rng() -> None:
    matrix_a = generate_count_connectome(6, np.random.default_rng(42), edge_probability=0.5)
    matrix_b = generate_count_connectome(6, np.random.default_rng(42), edge_probability=0.5)

    np.testing.assert_array_equal(matrix_a, matrix_b)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_regions": 0}, "n_regions"),
        ({"edge_probability": -0.1}, "edge_probability"),
        ({"edge_probability": 1.1}, "edge_probability"),
        ({"weight_min": 0.0}, "weight_min"),
        ({"weight_min": float("nan")}, "weight_min"),
        ({"weight_max": 0.1, "weight_min": 0.2}, "weight_max"),
        ({"weight_max": float("inf")}, "weight_max"),
    ],
)
def test_generate_count_connectome_rejects_invalid_parameters(
    kwargs: dict[str, object],
    message: str,
) -> None:
    n_regions = kwargs.pop("n_regions", 4)

    with pytest.raises(SchemaError, match=message):
        generate_count_connectome(
            n_regions,  # type: ignore[arg-type]
            np.random.default_rng(123),
            **kwargs,  # type: ignore[arg-type]
        )


def test_generate_count_connectome_rejects_non_generator_rng() -> None:
    with pytest.raises(SchemaError, match="rng"):
        generate_count_connectome(4, 123)  # type: ignore[arg-type]
