from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tau_sr.schemas import SchemaError
from tau_sr.synthetic import (
    SyntheticCohortConfig,
    generate_count_connectome,
    generate_edge_attributes,
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


def test_generate_count_connectome_has_chain_backbone() -> None:
    matrix = generate_count_connectome(
        5,
        np.random.default_rng(123),
        edge_probability=0.0,
        weight_min=0.25,
        weight_max=0.75,
    )

    expected_edges = {(0, 1), (1, 2), (2, 3), (3, 4)}
    nonzero_edges = {
        (row, col)
        for row, col in zip(*np.nonzero(np.triu(matrix, k=1)), strict=True)
    }

    assert nonzero_edges == expected_edges
    assert np.all(matrix[matrix > 0.0] >= 0.25)
    assert np.all(matrix[matrix > 0.0] < 0.75)


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


def test_generate_edge_attributes_standard_case() -> None:
    count = generate_count_connectome(148, np.random.default_rng(42))

    fa, length = generate_edge_attributes(count, np.random.default_rng(123))

    assert fa.shape == (148, 148)
    assert length.shape == (148, 148)
    assert np.issubdtype(fa.dtype, np.floating)
    assert np.issubdtype(length.dtype, np.floating)
    assert np.isfinite(fa).all()
    assert np.isfinite(length).all()
    assert np.all(fa >= 0.0)
    assert np.all(length >= 0.0)
    assert np.allclose(fa, fa.T)
    assert np.allclose(length, length.T)
    np.testing.assert_array_equal(np.diag(fa), np.zeros(148))
    np.testing.assert_array_equal(np.diag(length), np.zeros(148))
    np.testing.assert_array_equal(fa > 0.0, count > 0.0)
    np.testing.assert_array_equal(length > 0.0, count > 0.0)


def test_generate_edge_attributes_respects_custom_bounds() -> None:
    count = generate_count_connectome(12, np.random.default_rng(42))

    fa, length = generate_edge_attributes(
        count,
        np.random.default_rng(123),
        fa_min=0.35,
        fa_max=0.55,
        length_min=40.0,
        length_max=80.0,
    )

    positive_fa = fa[fa > 0.0]
    positive_length = length[length > 0.0]
    assert np.all(positive_fa >= 0.35)
    assert np.all(positive_fa < 0.55)
    assert np.all(positive_length >= 40.0)
    assert np.all(positive_length < 80.0)


def test_generate_edge_attributes_preserves_chain_only_support() -> None:
    count = generate_count_connectome(6, np.random.default_rng(42), edge_probability=0.0)

    fa, length = generate_edge_attributes(count, np.random.default_rng(123))

    np.testing.assert_array_equal(fa > 0.0, count > 0.0)
    np.testing.assert_array_equal(length > 0.0, count > 0.0)
    assert np.count_nonzero(np.triu(fa > 0.0, k=1)) == 5
    assert np.count_nonzero(np.triu(length > 0.0, k=1)) == 5


def test_generate_edge_attributes_is_reproducible_for_same_seed() -> None:
    count = generate_count_connectome(24, np.random.default_rng(42))

    fa_a, length_a = generate_edge_attributes(count, np.random.default_rng(123))
    fa_b, length_b = generate_edge_attributes(count, np.random.default_rng(123))

    np.testing.assert_array_equal(fa_a, fa_b)
    np.testing.assert_array_equal(length_a, length_b)


def test_generate_edge_attributes_differs_for_different_seeds() -> None:
    count = generate_count_connectome(24, np.random.default_rng(42))

    fa_a, length_a = generate_edge_attributes(count, np.random.default_rng(123))
    fa_b, length_b = generate_edge_attributes(count, np.random.default_rng(456))

    assert not (np.array_equal(fa_a, fa_b) and np.array_equal(length_a, length_b))


def test_generate_edge_attributes_preserves_input_and_allocates_outputs() -> None:
    count = generate_count_connectome(10, np.random.default_rng(42))
    count_before = count.copy()

    fa, length = generate_edge_attributes(count, np.random.default_rng(123))

    np.testing.assert_array_equal(count, count_before)
    assert not np.shares_memory(fa, count)
    assert not np.shares_memory(length, count)
    assert not np.shares_memory(fa, length)


def test_generate_edge_attributes_small_graph() -> None:
    count = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 4.0],
            [0.0, 4.0, 0.0],
        ]
    )

    fa, length = generate_edge_attributes(count, np.random.default_rng(123))

    assert fa.shape == (3, 3)
    assert length.shape == (3, 3)
    assert np.allclose(fa, fa.T)
    assert np.allclose(length, length.T)
    np.testing.assert_array_equal(np.diag(fa), np.zeros(3))
    np.testing.assert_array_equal(np.diag(length), np.zeros(3))
    np.testing.assert_array_equal(fa > 0.0, count > 0.0)
    np.testing.assert_array_equal(length > 0.0, count > 0.0)


@pytest.mark.parametrize(
    ("count_factory", "message"),
    [
        (lambda: [[0.0, 1.0], [1.0, 0.0]], "NumPy"),
        (lambda: np.array([0.0, 1.0]), "two-dimensional"),
        (lambda: np.zeros((2, 3)), "square"),
        (lambda: np.zeros((1, 1)), "at least 2"),
        (lambda: np.array([[False, True], [True, False]]), "real numeric"),
        (lambda: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=object), "real numeric"),
        (lambda: np.array([["0", "1"], ["1", "0"]]), "real numeric"),
        (lambda: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), "real numeric"),
        (lambda: np.array([[0.0, np.nan], [np.nan, 0.0]]), "finite"),
        (lambda: np.array([[0.0, np.inf], [np.inf, 0.0]]), "finite"),
        (lambda: np.array([[0.0, -1.0], [-1.0, 0.0]]), "nonnegative"),
        (lambda: np.array([[0.0, 1.0], [2.0, 0.0]]), "symmetric"),
        (lambda: np.array([[1.0, 2.0], [2.0, 0.0]]), "diagonal"),
        (lambda: np.zeros((2, 2)), "positive edge"),
    ],
)
def test_generate_edge_attributes_rejects_invalid_count_matrices(
    count_factory: object,
    message: str,
) -> None:
    with pytest.raises(SchemaError, match=message):
        generate_edge_attributes(count_factory(), np.random.default_rng(123))  # type: ignore[operator]


@pytest.mark.parametrize("rng", [123, None, object()])
def test_generate_edge_attributes_rejects_invalid_rng(rng: object) -> None:
    count = generate_count_connectome(6, np.random.default_rng(42))

    with pytest.raises(SchemaError, match="rng"):
        generate_edge_attributes(count, rng)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fa_min": 0.0}, "fa_min"),
        ({"fa_min": -0.1}, "fa_min"),
        ({"fa_min": 0.5, "fa_max": 0.5}, "fa_max"),
        ({"fa_min": 0.6, "fa_max": 0.5}, "fa_max"),
        ({"fa_max": 1.1}, "fa_max"),
        ({"fa_min": float("nan")}, "fa_min"),
        ({"fa_max": float("inf")}, "fa_max"),
        ({"fa_min": True}, "fa_min"),
        ({"fa_max": False}, "fa_max"),
        ({"fa_min": "bad"}, "fa_min"),
        ({"fa_max": "bad"}, "fa_max"),
    ],
)
def test_generate_edge_attributes_rejects_invalid_fa_bounds(
    kwargs: dict[str, object],
    message: str,
) -> None:
    count = generate_count_connectome(6, np.random.default_rng(42))

    with pytest.raises(SchemaError, match=message):
        generate_edge_attributes(count, np.random.default_rng(123), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"length_min": 0.0}, "length_min"),
        ({"length_min": -1.0}, "length_min"),
        ({"length_min": 40.0, "length_max": 40.0}, "length_max"),
        ({"length_min": 50.0, "length_max": 40.0}, "length_max"),
        ({"length_min": float("nan")}, "length_min"),
        ({"length_max": float("inf")}, "length_max"),
        ({"length_min": True}, "length_min"),
        ({"length_max": False}, "length_max"),
        ({"length_min": "bad"}, "length_min"),
        ({"length_max": "bad"}, "length_max"),
    ],
)
def test_generate_edge_attributes_rejects_invalid_length_bounds(
    kwargs: dict[str, object],
    message: str,
) -> None:
    count = generate_count_connectome(6, np.random.default_rng(42))

    with pytest.raises(SchemaError, match=message):
        generate_edge_attributes(count, np.random.default_rng(123), **kwargs)  # type: ignore[arg-type]
