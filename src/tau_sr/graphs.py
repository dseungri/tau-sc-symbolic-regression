"""Generic graph construction helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np

from tau_sr.schemas import SchemaError

LaplacianKind = Literal["unnormalized", "symmetric_normalized"]

_VALID_LAPLACIAN_KINDS = ("unnormalized", "symmetric_normalized")


def build_graph_laplacian(
    adjacency: np.ndarray,
    *,
    kind: LaplacianKind = "unnormalized",
) -> np.ndarray:
    """Build an unnormalized or symmetric normalized graph Laplacian."""

    _validate_kind(kind)
    degree = _validate_adjacency(adjacency)
    adjacency_float = adjacency.astype(float, copy=False)

    if kind == "unnormalized":
        degree_matrix = np.diag(degree)
        return degree_matrix - adjacency_float

    inverse_sqrt_degree = 1.0 / np.sqrt(degree)
    normalized_adjacency = (
        inverse_sqrt_degree[:, None]
        * adjacency_float
        * inverse_sqrt_degree[None, :]
    )
    return np.eye(adjacency.shape[0], dtype=float) - normalized_adjacency


def _validate_kind(kind: object) -> None:
    if kind not in _VALID_LAPLACIAN_KINDS:
        raise SchemaError(
            "kind must be one of: unnormalized, symmetric_normalized."
        )


def _validate_adjacency(adjacency: np.ndarray) -> np.ndarray:
    if not isinstance(adjacency, np.ndarray):
        raise SchemaError("adjacency must be a NumPy array.")
    if adjacency.ndim != 2:
        raise SchemaError("adjacency must be two-dimensional.")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise SchemaError("adjacency must be square.")
    if adjacency.shape[0] < 2:
        raise SchemaError("adjacency must have at least 2 rows.")
    if (
        adjacency.dtype.kind in {"b", "O", "S", "U", "c"}
        or not np.issubdtype(adjacency.dtype, np.number)
    ):
        raise SchemaError("adjacency must have a real numeric dtype.")
    if not np.isfinite(adjacency).all():
        raise SchemaError("adjacency must contain only finite values.")
    if (adjacency < 0.0).any():
        raise SchemaError("adjacency must be nonnegative.")
    if not np.allclose(adjacency, adjacency.T, rtol=1e-7, atol=1e-12):
        raise SchemaError("adjacency must be symmetric.")
    if not np.array_equal(np.diag(adjacency), np.zeros(adjacency.shape[0])):
        raise SchemaError("adjacency diagonal must be exactly zero.")
    if not (adjacency > 0.0).any():
        raise SchemaError("adjacency must contain at least one positive edge.")

    degree = adjacency.astype(float, copy=False).sum(axis=1)
    if not np.all(degree > 0.0):
        raise SchemaError("adjacency must have positive weighted degree at every node.")
    return degree
