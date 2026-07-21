"""Subject-specific structural-network representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tau_sr.graphs.validation import validate_nonnegative_adjacency


@dataclass(frozen=True)
class NetworkBuildResult:
    """Network matrix with transformation metadata."""

    matrix: np.ndarray
    metadata: dict[str, object]


def build_count_network(count_matrix: np.ndarray) -> NetworkBuildResult:
    """Return the streamline-density count network."""

    count = validate_nonnegative_adjacency(count_matrix, name="count_matrix")
    return NetworkBuildResult(count.copy(), {"representation": "count"})


def build_fa_weighted_network(
    count_matrix: np.ndarray, fa_matrix: np.ndarray
) -> NetworkBuildResult:
    """Return W_count * FA while preserving zero-edge support."""

    count = validate_nonnegative_adjacency(count_matrix, name="count_matrix")
    fa = validate_nonnegative_adjacency(fa_matrix, name="fa_matrix")
    if count.shape != fa.shape:
        raise ValueError("count and FA matrices must have identical shape")
    support = count > 0
    if np.any((fa > 0) & ~support):
        raise ValueError("FA matrix contains edges absent from count matrix")
    output = np.zeros_like(count, dtype=float)
    output[support] = count[support] * fa[support]
    return NetworkBuildResult(output, {"representation": "fa", "formula": "count * fa"})


def build_length_adjusted_network(
    count_matrix: np.ndarray,
    length_matrix: np.ndarray,
    epsilon: float,
    *,
    scale_length: bool = True,
) -> NetworkBuildResult:
    """Return count / (scaled_length + epsilon) on count-supported edges."""

    count = validate_nonnegative_adjacency(count_matrix, name="count_matrix")
    length = validate_nonnegative_adjacency(length_matrix, name="length_matrix")
    if count.shape != length.shape:
        raise ValueError("count and length matrices must have identical shape")
    support = count > 0
    if np.any((length > 0) & ~support):
        raise ValueError("length matrix contains edges absent from count matrix")
    if np.any(support & ~(length > 0)):
        raise ValueError("count-supported edges require positive tract lengths")
    positive_lengths = length[support]
    scale = float(np.median(positive_lengths)) if scale_length and positive_lengths.size else 1.0
    scaled = length / scale
    output = np.zeros_like(count, dtype=float)
    output[support] = count[support] / (scaled[support] + epsilon)
    return NetworkBuildResult(
        output,
        {
            "representation": "length",
            "formula": "count / (scaled_length + epsilon)",
            "epsilon": epsilon,
            "scale_length": scale_length,
            "length_scale": scale,
        },
    )
