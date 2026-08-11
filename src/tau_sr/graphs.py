"""Generic graph construction helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np

from tau_sr.schemas import SchemaError

LaplacianKind = Literal[
    "unnormalized",
    "combinatorial",
    "symmetric_normalized",
    "random_walk",
]

_VALID_LAPLACIAN_KINDS = (
    "unnormalized",
    "combinatorial",
    "symmetric_normalized",
    "random_walk",
)


def build_graph_laplacian(
    adjacency: np.ndarray,
    *,
    kind: LaplacianKind = "unnormalized",
) -> np.ndarray:
    """Build a graph Laplacian from an adjacency matrix.

    Every node must have a finite, positive weighted degree before any
    Laplacian variant is constructed.
    """

    _validate_kind(kind)
    degree = _validate_adjacency(adjacency)
    adjacency_float = adjacency.astype(float, copy=False)

    if kind in ("unnormalized", "combinatorial"):
        degree_matrix = np.diag(degree)
        return degree_matrix - adjacency_float

    if kind == "random_walk":
        inverse_degree = 1.0 / degree
        return np.eye(adjacency.shape[0], dtype=float) - inverse_degree[:, None] * adjacency_float

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
            "kind must be one of: unnormalized, combinatorial, "
            "symmetric_normalized, random_walk."
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
    if not np.array_equal(adjacency, adjacency.T):
        raise SchemaError("adjacency must be symmetric.")
    if not np.array_equal(np.diag(adjacency), np.zeros(adjacency.shape[0])):
        raise SchemaError("adjacency diagonal must be exactly zero.")
    if not (adjacency > 0.0).any():
        raise SchemaError("adjacency must contain at least one positive edge.")

    with np.errstate(over="ignore", invalid="ignore"):
        degree = adjacency.astype(float, copy=False).sum(axis=1)
    if not (np.isfinite(degree).all() and np.all(degree > 0.0)):
        raise SchemaError("adjacency must have finite, positive weighted degree at every node.")
    return degree
