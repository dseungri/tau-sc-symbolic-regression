"""Graph Laplacian construction and validation."""

from __future__ import annotations

import numpy as np

from tau_sr.graphs.validation import validate_nonnegative_adjacency, validate_square_numeric_matrix
from tau_sr.qc.common import QCReport
from tau_sr.qc.connectome import connected_component_count


def build_unnormalized_laplacian(matrix: np.ndarray) -> np.ndarray:
    """Build L = D - W."""

    adjacency = validate_nonnegative_adjacency(matrix)
    degree = adjacency.sum(axis=1)
    return np.diag(degree) - adjacency


def build_symmetric_normalized_laplacian(matrix: np.ndarray) -> np.ndarray:
    """Build L = I - D^(-1/2) W D^(-1/2), safely handling zero-degree nodes."""

    adjacency = validate_nonnegative_adjacency(matrix)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree, dtype=float)
    positive = degree > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized_adjacency = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    laplacian = np.eye(adjacency.shape[0]) - normalized_adjacency
    laplacian[~positive, ~positive] = 0.0
    return laplacian


def validate_laplacian(
    laplacian: np.ndarray,
    *,
    adjacency: np.ndarray | None = None,
    eigenvalue_tolerance: float = 1.0e-8,
    symmetry_tolerance: float = 1.0e-6,
) -> QCReport:
    """Validate common Laplacian properties."""

    matrix = validate_square_numeric_matrix(laplacian, name="laplacian")
    report = QCReport("laplacian")
    if not np.allclose(matrix, matrix.T, atol=symmetry_tolerance):
        report.add_error("asymmetric_laplacian", "laplacian is not symmetric")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    report.metrics["min_eigenvalue"] = float(np.min(eigenvalues))
    report.metrics["near_zero_eigenvalue_count"] = int(
        np.count_nonzero(np.abs(eigenvalues) <= eigenvalue_tolerance)
    )
    if np.min(eigenvalues) < -eigenvalue_tolerance:
        report.add_error(
            "negative_eigenvalues", "laplacian has negative eigenvalues beyond tolerance"
        )
    if adjacency is not None:
        adjacency_array = validate_nonnegative_adjacency(adjacency, name="adjacency")
        expected_components = connected_component_count(adjacency_array)
        report.metrics["adjacency_component_count"] = expected_components
        if report.metrics["near_zero_eigenvalue_count"] != expected_components:
            report.add_warning(
                "component_nullity_mismatch",
                "near-zero eigenvalue count differs from connected components",
            )
        if np.allclose(matrix.sum(axis=1), 0.0, atol=symmetry_tolerance):
            report.metrics["row_sums_zero"] = True
        else:
            report.metrics["row_sums_zero"] = False
    return report


def transport_direction_test() -> bool:
    """Check that -rho * L @ y diffuses high tau to connected lower neighbors."""

    adjacency = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    y = np.array([2.0, 0.0, 0.0])
    rho = 0.5
    laplacian = build_unnormalized_laplacian(adjacency)
    # With the negative-Laplacian convention, a high node decreases and lower neighbors increase.
    derivative = -rho * laplacian @ y
    return bool(derivative[0] < 0 and derivative[1] > 0 and derivative[2] > 0)
