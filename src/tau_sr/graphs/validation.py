"""Adjacency validation helpers."""

from __future__ import annotations

import numpy as np


def validate_square_numeric_matrix(matrix: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    """Return a float array after square numeric validation."""

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_nonnegative_adjacency(matrix: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    """Validate a finite nonnegative square adjacency matrix."""

    array = validate_square_numeric_matrix(matrix, name=name)
    if (array < 0).any():
        raise ValueError(f"{name} must be nonnegative")
    return array
