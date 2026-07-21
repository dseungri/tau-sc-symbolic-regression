"""Optional adjacency scaling functions."""

from __future__ import annotations

import numpy as np

from tau_sr.graphs.validation import validate_nonnegative_adjacency


def scale_adjacency(matrix: np.ndarray, method: str, *, epsilon: float = 1.0e-8) -> np.ndarray:
    """Scale an adjacency matrix using an explicit configured method."""

    array = validate_nonnegative_adjacency(matrix)
    if method == "none":
        return array.copy()
    if method == "total_weight":
        total = float(array.sum())
        return array / (total + epsilon)
    if method == "frobenius":
        return array / (float(np.linalg.norm(array, ord="fro")) + epsilon)
    raise ValueError(f"unsupported adjacency scaling method: {method}")
