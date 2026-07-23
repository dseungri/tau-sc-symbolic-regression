from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tau_sr.schemas import SchemaError
from tau_sr.synthetic import (
    SyntheticCohortConfig,
    build_connectome_representations,
    generate_baseline_tau,
    generate_count_connectome,
    generate_edge_attributes,
    generate_synthetic_roi_labels,
    validate_synthetic_config,
)


def _valid_representation_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = generate_count_connectome(148, np.random.default_rng(42))
    fa, length = generate_edge_attributes(count, np.random.default_rng(123))
    return count, fa, length


def _small_valid_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 4.0],
            [0.0, 4.0, 0.0],
        ]
    )
    fa = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.25],
            [0.0, 0.25, 0.0],
        ]
    )
    length = np.array(
        [
            [0.0, 20.0, 0.0],
            [20.0, 0.0, 40.0],
            [0.0, 40.0, 0.0],
        ]
    )
    return count, fa, length


def _expected_length_adjusted(
    count: np.ndarray,
    length: np.ndarray,
    *,
    length_epsilon: float = 1e-8,
) -> np.ndarray:
    support = count > 0.0
    upper_support = np.triu(support, k=1)
    median_length = np.median(length[upper_support])
    scaled_length = np.zeros_like(length, dtype=float)
    scaled_length[support] = length[support] / median_length
    expected = np.zeros_like(count, dtype=float)
    expected[support] = count[support] / (scaled_length[support] + length_epsilon)
    return expected


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


def test_generate_baseline_tau_standard_case() -> None:
    baseline = generate_baseline_tau(148, np.random.default_rng(42))

    assert isinstance(baseline, np.ndarray)
    assert baseline.shape == (148,)
    assert len(baseline) == 148
    assert np.issubdtype(baseline.dtype, np.floating)
    assert np.isfinite(baseline).all()
    assert np.all(baseline >= 0.05)
    assert np.all(baseline < 0.40)


def test_generate_baseline_tau_matches_config_defaults() -> None:
    config = SyntheticCohortConfig()

    baseline = generate_baseline_tau(
        config.n_regions,
        np.random.default_rng(config.seed),
        tau_min=config.baseline_tau_min,
        tau_max=config.baseline_tau_max,
    )

    assert baseline.shape == (config.n_regions,)
    assert np.all(baseline >= config.baseline_tau_min)
    assert np.all(baseline < config.baseline_tau_max)


def test_generate_baseline_tau_is_reproducible_for_same_seed() -> None:
    baseline_a = generate_baseline_tau(148, np.random.default_rng(42))
    baseline_b = generate_baseline_tau(148, np.random.default_rng(42))

    np.testing.assert_array_equal(baseline_a, baseline_b)


def test_generate_baseline_tau_differs_for_different_seeds() -> None:
    baseline_a = generate_baseline_tau(148, np.random.default_rng(42))
    baseline_b = generate_baseline_tau(148, np.random.default_rng(43))

    assert not np.array_equal(baseline_a, baseline_b)


def test_generate_baseline_tau_advances_supplied_rng_state() -> None:
    rng = np.random.default_rng(42)

    first = generate_baseline_tau(10, rng)
    second = generate_baseline_tau(10, rng)

    replay_rng = np.random.default_rng(42)
    replay_first = generate_baseline_tau(10, replay_rng)
    replay_second = generate_baseline_tau(10, replay_rng)

    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, replay_first)
    np.testing.assert_array_equal(second, replay_second)


def test_generate_baseline_tau_respects_custom_bounds() -> None:
    baseline = generate_baseline_tau(
        12,
        np.random.default_rng(42),
        tau_min=1.2,
        tau_max=2.4,
    )

    assert np.all(baseline >= 1.2)
    assert np.all(baseline < 2.4)


def test_generate_baseline_tau_accepts_zero_lower_bound() -> None:
    baseline = generate_baseline_tau(
        12,
        np.random.default_rng(42),
        tau_min=0.0,
        tau_max=0.5,
    )

    assert np.all(baseline >= 0.0)
    assert np.all(baseline < 0.5)


def test_generate_baseline_tau_accepts_two_regions() -> None:
    baseline = generate_baseline_tau(2, np.random.default_rng(42))

    assert baseline.shape == (2,)
    assert np.issubdtype(baseline.dtype, np.floating)
    assert np.all(baseline >= 0.05)
    assert np.all(baseline < 0.40)


@pytest.mark.parametrize("n_regions", [0, 1, -1, True, False, 3.5, "148", None])
def test_generate_baseline_tau_rejects_invalid_region_counts(n_regions: object) -> None:
    with pytest.raises(SchemaError, match="n_regions"):
        generate_baseline_tau(n_regions, np.random.default_rng(42))  # type: ignore[arg-type]


@pytest.mark.parametrize("rng", [42, None, "rng", np.random.RandomState(42)])
def test_generate_baseline_tau_rejects_invalid_rng(rng: object) -> None:
    with pytest.raises(SchemaError, match="rng"):
        generate_baseline_tau(12, rng)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tau_min": -0.1}, "tau_min"),
        ({"tau_min": np.nan}, "tau_min"),
        ({"tau_min": np.inf}, "tau_min"),
        ({"tau_min": -np.inf}, "tau_min"),
        ({"tau_min": True}, "tau_min"),
        ({"tau_min": "0.05"}, "tau_min"),
        ({"tau_min": None}, "tau_min"),
        ({"tau_max": np.nan}, "tau_max"),
        ({"tau_max": np.inf}, "tau_max"),
        ({"tau_max": -np.inf}, "tau_max"),
        ({"tau_max": False}, "tau_max"),
        ({"tau_max": "0.40"}, "tau_max"),
        ({"tau_max": None}, "tau_max"),
        ({"tau_min": 0.2, "tau_max": 0.2}, "tau_max"),
        ({"tau_min": 0.4, "tau_max": 0.2}, "tau_max"),
    ],
)
def test_generate_baseline_tau_rejects_invalid_bounds(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SchemaError, match=message):
        generate_baseline_tau(
            12,
            np.random.default_rng(42),
            **kwargs,  # type: ignore[arg-type]
        )


def test_generate_baseline_tau_outputs_independent_arrays() -> None:
    baseline_a = generate_baseline_tau(12, np.random.default_rng(42))
    baseline_b = generate_baseline_tau(12, np.random.default_rng(42))

    np.testing.assert_array_equal(baseline_a, baseline_b)
    assert baseline_a is not baseline_b
    assert not np.shares_memory(baseline_a, baseline_b)


def test_generate_baseline_tau_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(12345)
    expected = np.random.random(5)

    np.random.seed(12345)
    generate_baseline_tau(10, np.random.default_rng(42))
    observed = np.random.random(5)

    np.testing.assert_array_equal(observed, expected)


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


def test_build_connectome_representations_standard_case() -> None:
    count, fa, length = _valid_representation_inputs()

    representations = build_connectome_representations(count, fa, length)

    assert isinstance(representations, dict)
    assert list(representations) == ["count", "fa_weighted", "length_adjusted"]
    for network in representations.values():
        assert network.shape == (148, 148)
        assert np.issubdtype(network.dtype, np.floating)
        assert np.isfinite(network).all()
        assert np.all(network >= 0.0)
        assert np.allclose(network, network.T)
        np.testing.assert_array_equal(np.diag(network), np.zeros(148))
        np.testing.assert_array_equal(network > 0.0, count > 0.0)


def test_build_connectome_representations_exact_count_formula() -> None:
    count, fa, length = _valid_representation_inputs()

    representations = build_connectome_representations(count, fa, length)

    np.testing.assert_array_equal(representations["count"], count)
    assert not np.shares_memory(representations["count"], count)


def test_build_connectome_representations_exact_fa_formula() -> None:
    count, fa, length = _valid_representation_inputs()

    representations = build_connectome_representations(count, fa, length)

    np.testing.assert_allclose(representations["fa_weighted"], count * fa)


def test_build_connectome_representations_exact_length_formula() -> None:
    count, fa, length = _valid_representation_inputs()

    representations = build_connectome_representations(count, fa, length)

    np.testing.assert_allclose(
        representations["length_adjusted"],
        _expected_length_adjusted(count, length),
    )


def test_build_connectome_representations_median_scaling_interpretation() -> None:
    count = np.array(
        [
            [0.0, 6.0, 0.0, 0.0],
            [6.0, 0.0, 6.0, 0.0],
            [0.0, 6.0, 0.0, 6.0],
            [0.0, 0.0, 6.0, 0.0],
        ]
    )
    fa = (count > 0.0).astype(float) * 0.5
    length = np.array(
        [
            [0.0, 10.0, 0.0, 0.0],
            [10.0, 0.0, 20.0, 0.0],
            [0.0, 20.0, 0.0, 30.0],
            [0.0, 0.0, 30.0, 0.0],
        ]
    )
    length_epsilon = 1e-4

    representations = build_connectome_representations(
        count,
        fa,
        length,
        length_epsilon=length_epsilon,
    )
    adjusted = representations["length_adjusted"]

    assert adjusted[0, 1] > adjusted[1, 2]
    assert adjusted[2, 3] < adjusted[1, 2]
    assert adjusted[1, 2] == pytest.approx(6.0 / (1.0 + length_epsilon))


def test_build_connectome_representations_preserves_chain_only_support() -> None:
    n_regions = 6
    count = generate_count_connectome(
        n_regions,
        np.random.default_rng(42),
        edge_probability=0.0,
    )
    fa, length = generate_edge_attributes(count, np.random.default_rng(123))

    representations = build_connectome_representations(count, fa, length)

    chain_support = count > 0.0
    assert np.count_nonzero(np.triu(chain_support, k=1)) == n_regions - 1
    for network in representations.values():
        np.testing.assert_array_equal(network > 0.0, chain_support)
        assert np.count_nonzero(np.triu(network > 0.0, k=1)) == n_regions - 1


def test_build_connectome_representations_preserves_inputs() -> None:
    count, fa, length = _valid_representation_inputs()
    count_before = count.copy()
    fa_before = fa.copy()
    length_before = length.copy()

    build_connectome_representations(count, fa, length)

    np.testing.assert_array_equal(count, count_before)
    np.testing.assert_array_equal(fa, fa_before)
    np.testing.assert_array_equal(length, length_before)


def test_build_connectome_representations_allocates_independent_outputs() -> None:
    count, fa, length = _valid_representation_inputs()

    representations = build_connectome_representations(count, fa, length)

    for network in representations.values():
        assert not np.shares_memory(network, count)
        assert not np.shares_memory(network, fa)
        assert not np.shares_memory(network, length)
    assert not np.shares_memory(representations["count"], representations["fa_weighted"])
    assert not np.shares_memory(representations["count"], representations["length_adjusted"])
    assert not np.shares_memory(
        representations["fa_weighted"],
        representations["length_adjusted"],
    )


def test_build_connectome_representations_small_graph() -> None:
    count, fa, length = _small_valid_matrices()

    representations = build_connectome_representations(count, fa, length)

    assert representations["count"].shape == (3, 3)
    np.testing.assert_array_equal(representations["count"], count)
    np.testing.assert_allclose(representations["fa_weighted"], count * fa)
    np.testing.assert_allclose(
        representations["length_adjusted"],
        _expected_length_adjusted(count, length),
    )
    for network in representations.values():
        assert np.allclose(network, network.T)
        np.testing.assert_array_equal(np.diag(network), np.zeros(3))
        np.testing.assert_array_equal(network > 0.0, count > 0.0)


def test_build_connectome_representations_respects_custom_epsilon() -> None:
    count, fa, length = _small_valid_matrices()

    small_epsilon_reps = build_connectome_representations(
        count,
        fa,
        length,
        length_epsilon=1e-8,
    )
    large_epsilon_reps = build_connectome_representations(
        count,
        fa,
        length,
        length_epsilon=0.5,
    )

    np.testing.assert_allclose(
        large_epsilon_reps["length_adjusted"],
        _expected_length_adjusted(count, length, length_epsilon=0.5),
    )
    support = count > 0.0
    np.testing.assert_array_equal(
        large_epsilon_reps["length_adjusted"] > 0.0,
        support,
    )
    assert np.all(
        large_epsilon_reps["length_adjusted"][support]
        < small_epsilon_reps["length_adjusted"][support]
    )


@pytest.mark.parametrize(
    ("matrix_name", "replacement"),
    [
        ("fa_matrix", np.zeros((4, 4))),
        ("length_matrix", np.zeros((4, 4))),
    ],
)
def test_build_connectome_representations_rejects_shape_mismatch(
    matrix_name: str,
    replacement: np.ndarray,
) -> None:
    count, fa, length = _small_valid_matrices()
    matrices = {
        "count_matrix": count,
        "fa_matrix": fa,
        "length_matrix": length,
    }
    matrices[matrix_name] = replacement

    with pytest.raises(SchemaError, match=matrix_name):
        build_connectome_representations(
            matrices["count_matrix"],
            matrices["fa_matrix"],
            matrices["length_matrix"],
        )


@pytest.mark.parametrize(
    ("matrix_name", "row", "col", "value"),
    [
        ("fa_matrix", 0, 1, 0.0),
        ("length_matrix", 0, 1, 0.0),
        ("fa_matrix", 0, 2, 0.5),
        ("length_matrix", 0, 2, 10.0),
    ],
)
def test_build_connectome_representations_rejects_support_mismatch(
    matrix_name: str,
    row: int,
    col: int,
    value: float,
) -> None:
    count, fa, length = _small_valid_matrices()
    matrices = {
        "count_matrix": count,
        "fa_matrix": fa.copy(),
        "length_matrix": length.copy(),
    }
    matrices[matrix_name][row, col] = value
    matrices[matrix_name][col, row] = value

    with pytest.raises(SchemaError, match=rf"{matrix_name} support"):
        build_connectome_representations(
            matrices["count_matrix"],
            matrices["fa_matrix"],
            matrices["length_matrix"],
        )


@pytest.mark.parametrize("matrix_name", ["count_matrix", "fa_matrix", "length_matrix"])
@pytest.mark.parametrize(
    ("matrix_factory", "message"),
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
    ],
)
def test_build_connectome_representations_rejects_invalid_matrices(
    matrix_name: str,
    matrix_factory: object,
    message: str,
) -> None:
    count, fa, length = _small_valid_matrices()
    matrices = {
        "count_matrix": count,
        "fa_matrix": fa,
        "length_matrix": length,
    }
    matrices[matrix_name] = matrix_factory()  # type: ignore[operator]

    with pytest.raises(SchemaError, match=rf"{matrix_name}.*{message}"):
        build_connectome_representations(
            matrices["count_matrix"],  # type: ignore[arg-type]
            matrices["fa_matrix"],  # type: ignore[arg-type]
            matrices["length_matrix"],  # type: ignore[arg-type]
        )


def test_build_connectome_representations_rejects_count_without_positive_edge() -> None:
    count, fa, length = _small_valid_matrices()
    count = np.zeros_like(count)
    fa = np.zeros_like(fa)
    length = np.zeros_like(length)

    with pytest.raises(SchemaError, match="positive edge"):
        build_connectome_representations(count, fa, length)


@pytest.mark.parametrize("length_epsilon", [0.0, -1.0, np.nan, np.inf, -np.inf, True, "0.1", None])
def test_build_connectome_representations_rejects_invalid_epsilon(
    length_epsilon: object,
) -> None:
    count, fa, length = _small_valid_matrices()

    with pytest.raises(SchemaError, match="length_epsilon"):
        build_connectome_representations(
            count,
            fa,
            length,
            length_epsilon=length_epsilon,  # type: ignore[arg-type]
        )
