"""Synthetic data generators for tests and smoke checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tau_sr.constants import EXPECTED_ROI_COUNT


def synthetic_roi_labels(n: int = EXPECTED_ROI_COUNT) -> list[str]:
    """Return unique, obviously synthetic ROI labels."""

    hemispheres = ["left", "right"]
    return [f"syn_roi_{index + 1:03d}_{hemispheres[index % 2]}" for index in range(n)]


def synthetic_subject_ids(n: int) -> list[str]:
    """Return generated synthetic subject identifiers."""

    return [f"sub-syn{index + 1:03d}" for index in range(n)]


def synthetic_tau_table(
    subject_count: int = 6, visits: int = 2, roi_count: int = EXPECTED_ROI_COUNT
) -> pd.DataFrame:
    """Generate deterministic longitudinal tau data."""

    rows = []
    labels = synthetic_roi_labels(roi_count)
    for subject_index, subject_id in enumerate(synthetic_subject_ids(subject_count)):
        for visit_index in range(visits):
            time = float(visit_index)
            for roi_index, label in enumerate(labels):
                value = 0.1 + 0.01 * subject_index + 0.001 * roi_index + 0.02 * visit_index
                rows.append(
                    {
                        "subject_id": subject_id,
                        "visit_id": f"visit-{visit_index + 1:02d}",
                        "scan_date": f"2026-01-{visit_index + 1:02d}",
                        "time_from_baseline_years": time,
                        "roi_name": label,
                        "tau_value": value,
                    }
                )
    return pd.DataFrame(rows)


def synthetic_connectome_matrices(
    roi_count: int = EXPECTED_ROI_COUNT, *, seed: int = 13
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate symmetric count, FA, and tract-length matrices."""

    rng = np.random.default_rng(seed)
    upper = rng.integers(0, 8, size=(roi_count, roi_count)).astype(float)
    upper = np.triu(upper, k=1)
    count = upper + upper.T
    chain = np.eye(roi_count, k=1) + np.eye(roi_count, k=-1)
    count = np.maximum(count, chain)
    support = count > 0
    fa = np.zeros_like(count)
    length = np.zeros_like(count)
    fa[support] = rng.uniform(0.2, 0.8, size=int(np.count_nonzero(support)))
    length[support] = rng.uniform(20.0, 120.0, size=int(np.count_nonzero(support)))
    fa = (fa + fa.T) / 2.0
    length = (length + length.T) / 2.0
    np.fill_diagonal(count, 0.0)
    np.fill_diagonal(fa, 0.0)
    np.fill_diagonal(length, 0.0)
    return count, fa, length


def malformed_negative_matrix(n: int = EXPECTED_ROI_COUNT) -> np.ndarray:
    matrix, _, _ = synthetic_connectome_matrices(n)
    matrix[0, 1] = -1.0
    matrix[1, 0] = -1.0
    return matrix


def toy_transport_network() -> np.ndarray:
    """Small graph for transport-direction tests."""

    return np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
