from __future__ import annotations

import numpy as np

from tau_sr.graphs.representations import build_fa_weighted_network, build_length_adjusted_network


def test_fa_weighting_preserves_edge_support() -> None:
    count = np.array([[0.0, 2.0], [2.0, 0.0]])
    fa = np.array([[0.0, 0.5], [0.5, 0.0]])
    result = build_fa_weighted_network(count, fa)
    assert np.array_equal(result.matrix > 0, count > 0)
    assert result.matrix[0, 1] == 1.0


def test_length_adjustment_handles_zeros_safely() -> None:
    count = np.array([[0.0, 2.0], [2.0, 0.0]])
    length = np.array([[0.0, 10.0], [10.0, 0.0]])
    result = build_length_adjusted_network(count, length, 1.0e-8)
    assert result.matrix[0, 0] == 0.0
    assert result.matrix[0, 1] > 0.0
