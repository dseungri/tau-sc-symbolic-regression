from __future__ import annotations

import numpy as np

from tau_sr.graphs.laplacian import (
    build_symmetric_normalized_laplacian,
    build_unnormalized_laplacian,
    transport_direction_test,
)


def test_unnormalized_laplacian_known_graph() -> None:
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(build_unnormalized_laplacian(adjacency), expected)


def test_symmetric_normalized_laplacian_known_graph() -> None:
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(build_symmetric_normalized_laplacian(adjacency), expected)


def test_negative_laplacian_sign_direction() -> None:
    assert transport_direction_test()
