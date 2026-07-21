"""Structural-connectivity matrix QC."""

from __future__ import annotations

from collections import deque

import numpy as np

from tau_sr.constants import EXPECTED_ROI_COUNT
from tau_sr.qc.common import QCReport


def connected_component_count(matrix: np.ndarray) -> int:
    """Count connected components in a nonnegative adjacency matrix."""

    adjacency = np.asarray(matrix) > 0
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    components = 0
    for start in range(n):
        if seen[start]:
            continue
        components += 1
        queue: deque[int] = deque([start])
        seen[start] = True
        while queue:
            node = queue.popleft()
            for neighbor in np.flatnonzero(adjacency[node]):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(int(neighbor))
    return components


def normalized_frobenius_symmetry_error(matrix: np.ndarray, *, epsilon: float = 1.0e-8) -> float:
    """Compute ||W - W.T||_F / (||W||_F + epsilon)."""

    return float(
        np.linalg.norm(matrix - matrix.T, ord="fro") / (np.linalg.norm(matrix, ord="fro") + epsilon)
    )


def validate_connectome_matrix(
    matrix: np.ndarray,
    *,
    expected_shape: tuple[int, int] = (EXPECTED_ROI_COUNT, EXPECTED_ROI_COUNT),
    symmetry_tolerance: float = 1.0e-6,
    epsilon: float = 1.0e-8,
) -> QCReport:
    """Validate a structural-connectivity matrix."""

    report = QCReport("connectome_matrix")
    array = np.asarray(matrix)
    report.metrics["shape"] = list(array.shape)
    if array.shape != expected_shape:
        report.add_error("wrong_shape", f"expected {expected_shape}, found {array.shape}")
        return report
    if not np.issubdtype(array.dtype, np.number):
        report.add_error("nonnumeric_dtype", "matrix dtype must be numeric")
        return report
    if not np.isfinite(array).all():
        report.add_error("nonfinite_values", "matrix contains NaN or infinity")
    if (array < 0).any():
        report.add_error("negative_weights", "matrix contains negative weights")
    diagonal_max = float(np.max(np.abs(np.diag(array))))
    if diagonal_max > symmetry_tolerance:
        report.add_warning("nonzero_diagonal", "matrix diagonal contains nonzero values")
    symmetry_error = normalized_frobenius_symmetry_error(array, epsilon=epsilon)
    if symmetry_error > symmetry_tolerance:
        report.add_error("asymmetry", "matrix exceeds normalized Frobenius symmetry tolerance")
    nonzero = int(np.count_nonzero(array))
    possible = array.size - array.shape[0]
    strengths = array.sum(axis=1)
    isolated = int(np.count_nonzero(np.isclose(strengths, 0.0)))
    report.metrics.update(
        {
            "symmetry_error": symmetry_error,
            "density": float(nonzero / possible) if possible else 0.0,
            "nonzero_edge_count": nonzero,
            "isolated_node_count": isolated,
            "connected_component_count": connected_component_count(array),
            "min_strength": float(np.min(strengths)),
            "max_strength": float(np.max(strengths)),
            "mean_strength": float(np.mean(strengths)),
        }
    )
    if isolated:
        report.add_warning("isolated_nodes", "one or more isolated nodes detected")
    return report


def symmetrize_matrix(matrix: np.ndarray, *, approved: bool = False) -> np.ndarray:
    """Explicit, opt-in symmetrization transform."""

    if not approved:
        raise ValueError("symmetrization requires explicit configuration approval")
    array = np.asarray(matrix, dtype=float)
    return (array + array.T) / 2.0


def validate_cross_matrix_consistency(
    count_matrix: np.ndarray,
    fa_matrix: np.ndarray,
    length_matrix: np.ndarray,
    *,
    roi_order_verified: bool,
    allow_missing_edge_values: bool = False,
) -> QCReport:
    """Validate count, FA, and length matrix consistency."""

    report = QCReport("connectome_cross_matrix_consistency")
    count = np.asarray(count_matrix)
    fa = np.asarray(fa_matrix)
    length = np.asarray(length_matrix)
    if count.shape != fa.shape or count.shape != length.shape:
        report.add_error(
            "shape_mismatch", "count, FA, and length matrices must have identical shape"
        )
        return report
    if not roi_order_verified:
        report.add_error("roi_order_unverified", "ROI order must be externally verified")
    support = count > 0
    if np.any((fa > 0) & ~support):
        report.add_error(
            "fa_edges_without_count", "FA matrix creates edges absent from count matrix"
        )
    if np.any((length > 0) & ~support):
        report.add_error(
            "length_edges_without_count", "length matrix creates edges absent from count matrix"
        )
    if not allow_missing_edge_values:
        if np.any(support & ~(fa > 0)):
            report.add_error("missing_fa_on_count_edges", "count-supported edges require FA values")
        if np.any(support & ~(length > 0)):
            report.add_error(
                "missing_length_on_count_edges", "count-supported edges require length values"
            )
    return report
