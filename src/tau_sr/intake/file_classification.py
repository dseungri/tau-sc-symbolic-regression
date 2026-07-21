"""Conservative file classification for intake inventory."""

from __future__ import annotations

import re
from pathlib import Path

ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar"}
MATRIX_EXTENSIONS = {".csv", ".tsv", ".npy", ".npz", ".parquet"}
IMAGE_EXTENSIONS = {".nii", ".nii.gz", ".dcm", ".dicom"}
TEXT_EXTENSIONS = {".json", ".yaml", ".yml", ".txt"}
SUPPORTED_EXTENSIONS = ARCHIVE_EXTENSIONS | MATRIX_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS


def normalized_extension(path: str | Path) -> str:
    """Return an extension, preserving compound suffixes."""

    name = Path(path).name.lower()
    for compound in (".nii.gz", ".tar.gz"):
        if name.endswith(compound):
            return compound
    return Path(name).suffix


def classify_file(path: str | Path) -> str:
    """Classify supported intake files by extension and filename hints."""

    name = Path(path).name.lower()
    extension = normalized_extension(path)
    if extension in IMAGE_EXTENSIONS:
        return "dwi_or_pet_image" if "pet" in name or "dwi" in name else "imaging"
    if extension in {".npy", ".npz"} or "matrix" in name or "connectome" in name:
        return "connectome_matrix"
    if "tau" in name and extension in {".csv", ".tsv", ".parquet"}:
        return "tau_table"
    if "roi" in name and extension in {".csv", ".tsv", ".txt"}:
        return "roi_labels"
    if extension in ARCHIVE_EXTENSIONS:
        return "archive"
    if extension in TEXT_EXTENSIONS:
        return "metadata_or_text"
    return "unknown"


def suspect_subject_from_name(path: str | Path) -> str | None:
    """Return synthetic subject IDs, redacting other subject-like tokens."""

    match = re.search(r"sub-syn\d+|sub-[A-Za-z0-9][A-Za-z0-9-]*", Path(path).name)
    if not match:
        return None
    value = match.group(0)
    if value.startswith("sub-syn"):
        return value
    return "REDACTED_POTENTIAL_IDENTIFIER"


def suspect_visit_from_name(path: str | Path) -> str | None:
    """Return a visit-like token when present."""

    match = re.search(r"(visit|ses)[-_]?[A-Za-z0-9]+", Path(path).name, flags=re.IGNORECASE)
    return match.group(0) if match else None
