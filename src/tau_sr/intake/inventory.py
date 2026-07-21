"""Recursive file inventory with conservative metadata extraction."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tau_sr.intake.checksums import sha256_file
from tau_sr.intake.file_classification import (
    classify_file,
    normalized_extension,
    suspect_subject_from_name,
    suspect_visit_from_name,
)


def _matrix_shape(path: Path) -> str:
    extension = normalized_extension(path)
    try:
        if extension == ".npy":
            array = np.load(path, mmap_mode="r")
            return "x".join(str(item) for item in array.shape)
        if extension == ".npz":
            with np.load(path) as archive:
                first = archive.files[0] if archive.files else ""
                if first:
                    return "x".join(str(item) for item in archive[first].shape)
        if extension in {".csv", ".tsv"}:
            frame = pd.read_csv(path, sep="	" if extension == ".tsv" else ",", nrows=5)
            return f"unknown_rowsx{len(frame.columns)}"
        if extension == ".parquet":
            frame = pd.read_parquet(path, columns=[])
            return f"{len(frame)}xunknown_columns"
        if extension == ".nii.gz" or extension == ".nii":
            try:
                import nibabel as nib
            except ImportError:
                return "header_unchecked_nibabel_unavailable"
            image = nib.load(str(path))
            return "x".join(str(item) for item in image.header.get_data_shape())
    except Exception as exc:  # noqa: BLE001 - inventory must continue.
        return f"unavailable:{exc.__class__.__name__}"
    return ""


def inventory_file(
    root: str | Path, path: str | Path, *, hash_files: bool = True
) -> dict[str, Any]:
    """Inventory one file without modifying it."""

    base = Path(root).resolve()
    file_path = Path(path).resolve()
    stat = file_path.stat()
    readable = True
    notes = ""
    digest = ""
    try:
        if hash_files:
            digest = sha256_file(file_path)
    except OSError as exc:
        readable = False
        notes = f"hash_unavailable:{exc.__class__.__name__}"
    extension = normalized_extension(file_path)
    return {
        "relative_path": file_path.relative_to(base).as_posix(),
        "filename": file_path.name,
        "extension": extension,
        "size_bytes": stat.st_size,
        "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "suspected_modality": classify_file(file_path),
        "suspected_subject": suspect_subject_from_name(file_path) or "",
        "suspected_visit": suspect_visit_from_name(file_path) or "",
        "matrix_shape": _matrix_shape(file_path),
        "readable": readable,
        "sha256": digest,
        "notes": notes,
    }


def inventory_directory(
    root: str | Path, *, recursive: bool = True, hash_files: bool = True
) -> list[dict[str, Any]]:
    """Inventory files under a root."""

    base = Path(root).resolve()
    files = base.rglob("*") if recursive else base.glob("*")
    return [
        inventory_file(base, path, hash_files=hash_files)
        for path in sorted(files)
        if path.is_file()
    ]


def write_inventory(records: list[dict[str, Any]], output_csv: str | Path) -> None:
    """Write inventory records to CSV."""

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "relative_path",
        "filename",
        "extension",
        "size_bytes",
        "modified_time_utc",
        "suspected_modality",
        "suspected_subject",
        "suspected_visit",
        "matrix_shape",
        "readable",
        "sha256",
        "notes",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
