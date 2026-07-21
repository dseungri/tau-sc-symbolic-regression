"""Schema validation for manifests and subject/visit fields."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tau_sr.qc.common import QCReport

SUBJECT_PATTERN = re.compile(r"^sub-[A-Za-z0-9][A-Za-z0-9_-]*$")
VISIT_PATTERN = re.compile(r"^(visit|ses)-?[A-Za-z0-9][A-Za-z0-9_-]*$", re.I)

REQUIRED_MANIFEST_COLUMNS = [
    "subject_id",
    "visit_id",
    "modality",
    "scan_date",
    "time_from_baseline_years",
    "tau_file",
    "count_matrix_file",
    "fa_matrix_file",
    "length_matrix_file",
    "roi_label_file",
]


def validate_subject_id(value: str) -> bool:
    return bool(SUBJECT_PATTERN.match(str(value)))


def validate_visit_id(value: str) -> bool:
    return bool(VISIT_PATTERN.match(str(value)))


def validate_manifest(
    manifest: pd.DataFrame,
    *,
    file_root: str | Path | None = None,
    require_modalities: list[str] | None = None,
) -> QCReport:
    """Validate a subject/visit manifest without declaring final eligibility."""

    report = QCReport("subject_visit_manifest")
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        report.add_error("missing_columns", ", ".join(missing))
        return report

    if manifest.duplicated(["subject_id", "visit_id"]).any():
        report.add_error("duplicate_subject_visit", "duplicate subject/visit rows detected")

    for column, validator in {
        "subject_id": validate_subject_id,
        "visit_id": validate_visit_id,
    }.items():
        bad = manifest[~manifest[column].map(validator)]
        if not bad.empty:
            report.add_error("invalid_identifier", f"{column} has invalid values")

    dates = pd.to_datetime(manifest["scan_date"], errors="coerce")
    if dates.isna().any():
        report.add_error("invalid_dates", "scan_date contains invalid dates")

    follow_up = pd.to_numeric(manifest["time_from_baseline_years"], errors="coerce")
    if follow_up.isna().any() or (follow_up < 0).any():
        report.add_error("invalid_follow_up", "follow-up times must be nonnegative numbers")

    if require_modalities:
        modalities = set(manifest["modality"].dropna().astype(str))
        missing_modalities = sorted(set(require_modalities) - modalities)
        if missing_modalities:
            report.add_warning("missing_required_modalities", ", ".join(missing_modalities))

    if file_root is not None:
        root = Path(file_root)
        for column in [
            "tau_file",
            "count_matrix_file",
            "fa_matrix_file",
            "length_matrix_file",
            "roi_label_file",
        ]:
            for value in manifest[column].dropna().astype(str):
                if value and value != "UNSET" and not (root / value).exists():
                    report.add_warning("missing_file", f"{column} path does not exist", column)

    visits_per_subject = manifest.groupby("subject_id")["visit_id"].nunique()
    if (visits_per_subject < 2).any():
        report.add_warning(
            "missing_longitudinal_tau", "one or more subjects have fewer than two visits"
        )
    return report
