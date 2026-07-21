"""ROI label loading, comparison, and reorder-map validation."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tau_sr.constants import EXPECTED_ROI_COUNT
from tau_sr.exceptions import ValidationError
from tau_sr.qc.common import QCReport


def normalize_roi_label(label: str) -> str:
    """Normalize harmless formatting differences while preserving anatomy."""

    lowered = str(label).strip().lower()
    collapsed = re.sub(r"[-_\s]+", "_", lowered)
    return collapsed.strip("_")


def load_roi_labels(path: str | Path, *, label_column: str = "roi_name") -> list[str]:
    """Load ROI labels from CSV, TSV, or text files."""

    file_path = Path(path)
    if file_path.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(file_path, sep="	" if file_path.suffix.lower() == ".tsv" else ",")
        if label_column not in frame.columns:
            raise ValidationError(f"missing ROI label column: {label_column}")
        return frame[label_column].astype(str).tolist()
    return [
        line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def validate_roi_labels(
    labels: list[str], *, expected_count: int = EXPECTED_ROI_COUNT, strict: bool = True
) -> QCReport:
    """Validate uniqueness and optional expected count."""

    report = QCReport("roi_labels")
    normalized = [normalize_roi_label(label) for label in labels]
    duplicates = sorted({label for label in normalized if normalized.count(label) > 1})
    if duplicates:
        report.add_error("duplicate_normalized_labels", ", ".join(duplicates[:10]))
    if strict and len(set(normalized)) != expected_count:
        report.add_error(
            "unexpected_roi_count", f"expected {expected_count}, found {len(set(normalized))}"
        )
    report.metrics["roi_count"] = len(labels)
    report.metrics["unique_normalized_roi_count"] = len(set(normalized))
    return report


def build_reorder_map(source_labels: list[str], target_labels: list[str]) -> list[int]:
    """Return source indices needed to reorder source into target order."""

    source_normalized = [normalize_roi_label(label) for label in source_labels]
    target_normalized = [normalize_roi_label(label) for label in target_labels]
    if len(set(source_normalized)) != len(source_normalized):
        raise ValidationError("source labels contain duplicate normalized labels")
    if len(set(target_normalized)) != len(target_normalized):
        raise ValidationError("target labels contain duplicate normalized labels")
    source_lookup = {label: index for index, label in enumerate(source_normalized)}
    missing = [label for label in target_normalized if label not in source_lookup]
    if missing:
        raise ValidationError(f"target labels missing from source: {missing[:5]}")
    return [source_lookup[label] for label in target_normalized]


def compare_roi_sets(tau_labels: list[str], connectome_labels: list[str]) -> QCReport:
    """Compare two ROI label sets and report order differences."""

    report = QCReport("roi_alignment")
    tau = [normalize_roi_label(label) for label in tau_labels]
    connectome = [normalize_roi_label(label) for label in connectome_labels]
    if set(tau) != set(connectome):
        report.add_error("roi_set_mismatch", "tau and connectome ROI sets differ")
    elif tau != connectome:
        report.add_warning(
            "roi_order_difference", "same ROI set but order differs; explicit reorder required"
        )
    report.metrics["tau_roi_count"] = len(tau)
    report.metrics["connectome_roi_count"] = len(connectome)
    return report


def round_trip_reorder(labels: list[str], reorder_map: list[int]) -> bool:
    """Check that a reorder map is a permutation with a valid inverse."""

    if sorted(reorder_map) != list(range(len(labels))):
        return False
    reordered = [labels[index] for index in reorder_map]
    inverse = [0] * len(reorder_map)
    for new_index, old_index in enumerate(reorder_map):
        inverse[old_index] = new_index
    return [reordered[index] for index in inverse] == labels
