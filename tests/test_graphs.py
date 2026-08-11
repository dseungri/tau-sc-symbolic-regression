from __future__ import annotations

import numpy as np
import pytest

from tau_sr.graphs import build_graph_laplacian
from tau_sr.schemas import SchemaError
from tau_sr.synthetic import (
    build_connectome_representations,
    generate_count_connectome,
    generate_edge_attributes,
)


def _small_adjacency() -> np.ndarray:
    return np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )


def _synthetic_representations() -> dict[str, np.ndarray]:
    count = generate_count_connectome(148, np.random.default_rng(42))
    fa, length = generate_edge_attributes(count, np.random.default_rng(123))
    return build_connectome_representations(count, fa, length)


def _off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def test_build_graph_laplacian_unnormalized_small_example() -> None:
    adjacency = _small_adjacency()
    expected = np.array(
        [
            [2.0, -2.0, 0.0],
            [-2.0, 3.0, -1.0],
            [0.0, -1.0, 1.0],
        ]
    )

    laplacian = build_graph_laplacian(adjacency, kind="unnormalized")

    np.testing.assert_allclose(laplacian, expected)
    np.testing.assert_allclose(laplacian @ np.ones(3), np.zeros(3), atol=1e-12)


def test_build_graph_laplacian_symmetric_normalized_small_example() -> None:
    adjacency = _small_adjacency()
    degree = adjacency.sum(axis=1)
    inverse_sqrt_degree = 1.0 / np.sqrt(degree)
    expected = (
        np.eye(3)
        - inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )

    laplacian = build_graph_laplacian(adjacency, kind="symmetric_normalized")

    np.testing.assert_allclose(laplacian, expected)
    np.testing.assert_allclose(laplacian @ np.sqrt(degree), np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(np.diag(laplacian), np.ones(3))


def test_build_graph_laplacian_random_walk_small_example() -> None:
    adjacency = _small_adjacency()
    degree = adjacency.sum(axis=1)
    expected = np.eye(3) - adjacency / degree[:, None]

    laplacian = build_graph_laplacian(adjacency, kind="random_walk")

    np.testing.assert_allclose(laplacian, expected)
    np.testing.assert_allclose(laplacian @ np.ones(3), np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(np.diag(laplacian), np.ones(3))


def test_build_graph_laplacian_combinatorial_aliases_unnormalized() -> None:
    adjacency = _small_adjacency()

    np.testing.assert_allclose(
        build_graph_laplacian(adjacency, kind="combinatorial"),
        build_graph_laplacian(adjacency, kind="unnormalized"),
    )


def test_build_graph_laplacian_defaults_to_unnormalized() -> None:
    adjacency = _small_adjacency()

    default_laplacian = build_graph_laplacian(adjacency)
    explicit_laplacian = build_graph_laplacian(adjacency, kind="unnormalized")

    np.testing.assert_allclose(default_laplacian, explicit_laplacian)


def test_build_graph_laplacian_standard_synthetic_representations() -> None:
    representations = _synthetic_representations()

    for adjacency in representations.values():
        for kind in ("unnormalized", "combinatorial", "symmetric_normalized", "random_walk"):
            laplacian = build_graph_laplacian(adjacency, kind=kind)

            assert laplacian.shape == (148, 148)
            assert np.issubdtype(laplacian.dtype, np.floating)
            assert np.isfinite(laplacian).all()
            assert not np.shares_memory(laplacian, adjacency)
            assert np.all(np.diag(laplacian) >= 0.0)
            assert np.all(_off_diagonal_values(laplacian) <= 1e-12)

            if kind in ("unnormalized", "combinatorial"):
                assert np.allclose(laplacian, laplacian.T)
                np.testing.assert_allclose(
                    laplacian.sum(axis=1),
                    0.0,
                    rtol=1e-7,
                    atol=1e-10,
                )
            elif kind == "symmetric_normalized":
                assert np.allclose(laplacian, laplacian.T)
                degree = adjacency.sum(axis=1)
                np.testing.assert_allclose(
                    laplacian @ np.sqrt(degree),
                    0.0,
                    rtol=1e-7,
                    atol=1e-10,
                )
                np.testing.assert_allclose(np.diag(laplacian), np.ones(148))
            else:
                np.testing.assert_allclose(
                    laplacian.sum(axis=1),
                    0.0,
                    rtol=1e-7,
                    atol=1e-10,
                )
                np.testing.assert_allclose(np.diag(laplacian), np.ones(148))


def test_build_graph_laplacian_positive_semidefinite_behavior() -> None:
    adjacency = generate_count_connectome(12, np.random.default_rng(42))

    for kind in ("unnormalized", "combinatorial", "symmetric_normalized"):
        laplacian = build_graph_laplacian(adjacency, kind=kind)
        eigenvalues = np.linalg.eigvalsh(laplacian)

        assert eigenvalues.min() >= -1e-10
        if kind == "symmetric_normalized":
            assert eigenvalues.max() <= 2.0 + 1e-10


def test_build_graph_laplacian_connected_synthetic_graph_has_zero_eigenvalue() -> None:
    adjacency = generate_count_connectome(12, np.random.default_rng(42))

    for kind in ("unnormalized", "combinatorial", "symmetric_normalized"):
        laplacian = build_graph_laplacian(adjacency, kind=kind)
        eigenvalues = np.linalg.eigvalsh(laplacian)

        assert abs(eigenvalues[0]) <= 1e-10

    random_walk = build_graph_laplacian(adjacency, kind="random_walk")
    eigenvalues = np.linalg.eigvals(random_walk)
    assert np.min(np.abs(eigenvalues)) <= 1e-10


def test_build_graph_laplacian_scaling_behavior() -> None:
    adjacency = _small_adjacency()
    scale = 7.5

    np.testing.assert_allclose(
        build_graph_laplacian(scale * adjacency, kind="unnormalized"),
        scale * build_graph_laplacian(adjacency, kind="unnormalized"),
    )
    np.testing.assert_allclose(
        build_graph_laplacian(scale * adjacency, kind="combinatorial"),
        scale * build_graph_laplacian(adjacency, kind="combinatorial"),
    )
    np.testing.assert_allclose(
        build_graph_laplacian(scale * adjacency, kind="symmetric_normalized"),
        build_graph_laplacian(adjacency, kind="symmetric_normalized"),
    )
    np.testing.assert_allclose(
        build_graph_laplacian(scale * adjacency, kind="random_walk"),
        build_graph_laplacian(adjacency, kind="random_walk"),
    )


def test_build_graph_laplacian_preserves_input_and_allocates_output() -> None:
    adjacency = _small_adjacency()

    for kind in ("unnormalized", "combinatorial", "symmetric_normalized", "random_walk"):
        adjacency_before = adjacency.copy()
        laplacian = build_graph_laplacian(adjacency, kind=kind)

        np.testing.assert_array_equal(adjacency, adjacency_before)
        assert not np.shares_memory(laplacian, adjacency)


def test_build_graph_laplacian_accepts_integer_input() -> None:
    adjacency = np.array(
        [
            [0, 2, 1],
            [2, 0, 3],
            [1, 3, 0],
        ],
        dtype=int,
    )
    expected = np.array(
        [
            [3.0, -2.0, -1.0],
            [-2.0, 5.0, -3.0],
            [-1.0, -3.0, 4.0],
        ]
    )

    laplacian = build_graph_laplacian(adjacency)

    assert np.issubdtype(laplacian.dtype, np.floating)
    np.testing.assert_allclose(laplacian, expected)


@pytest.mark.parametrize(
    ("adjacency_factory", "message"),
    [
        (lambda: [[0.0, 1.0], [1.0, 0.0]], "NumPy"),
        (lambda: np.array([0.0, 1.0]), "two-dimensional"),
        (lambda: np.zeros((2, 3)), "square"),
        (lambda: np.zeros((1, 1)), "at least 2"),
        (lambda: np.array([[False, True], [True, False]]), "real numeric"),
        (lambda: np.array([["0", "1"], ["1", "0"]]), "real numeric"),
        (lambda: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=object), "real numeric"),
        (lambda: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), "real numeric"),
        (lambda: np.array([[0.0, np.nan], [np.nan, 0.0]]), "finite"),
        (lambda: np.array([[0.0, np.inf], [np.inf, 0.0]]), "finite"),
        (lambda: np.array([[0.0, -np.inf], [-np.inf, 0.0]]), "finite"),
        (lambda: np.array([[0.0, -1.0], [-1.0, 0.0]]), "nonnegative"),
        (lambda: np.array([[0.0, 1.0], [2.0, 0.0]]), "symmetric"),
        (lambda: np.array([[0.0, 1.0], [1.000001, 0.0]]), "symmetric"),
        (lambda: np.array([[1.0, 2.0], [2.0, 0.0]]), "diagonal"),
        (lambda: np.zeros((2, 2)), "positive edge"),
        (
            lambda: np.array(
                [
                    [0.0, np.finfo(float).max, np.finfo(float).max],
                    [np.finfo(float).max, 0.0, np.finfo(float).max],
                    [np.finfo(float).max, np.finfo(float).max, 0.0],
                ]
            ),
            "finite, positive weighted degree",
        ),
        (
            lambda: np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "weighted degree",
        ),
    ],
)
def test_build_graph_laplacian_rejects_invalid_adjacency(
    adjacency_factory: object,
    message: str,
) -> None:
    with pytest.raises(SchemaError, match=message):
        build_graph_laplacian(adjacency_factory())  # type: ignore[operator]


@pytest.mark.parametrize(
    "kind",
    ["normalized", "symmetric", "UNNORMALIZED", "", None, 1, True],
)
def test_build_graph_laplacian_rejects_invalid_kind(kind: object) -> None:
    adjacency = _small_adjacency()

    with pytest.raises(SchemaError, match="kind"):
        build_graph_laplacian(adjacency, kind=kind)  # type: ignore[arg-type]
