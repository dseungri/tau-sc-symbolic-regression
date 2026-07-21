"""Potential identifier flagging without storing discovered values."""

from __future__ import annotations

import csv
import re
from pathlib import Path

SENSITIVE_FIELD_PATTERNS = {
    "direct_name": re.compile(r"(^|_)(name|first_name|last_name|full_name)($|_)", re.I),
    "date_of_birth": re.compile(r"(dob|date_of_birth|birth_date)", re.I),
    "medical_record_number": re.compile(r"(mrn|medical_record|record_number)", re.I),
    "email": re.compile(r"email", re.I),
    "address": re.compile(r"address|street|zipcode|zip_code", re.I),
    "phone": re.compile(r"phone|telephone|mobile", re.I),
    "government_id": re.compile(r"ssn|social_security|passport|driver", re.I),
}
SUSPICIOUS_FILENAME = re.compile(r"(dob|mrn|medical_record|email|phone|address|ssn)", re.I)


def flag_path(path: str | Path) -> list[dict[str, str]]:
    """Flag suspicious filename patterns without exposing values."""

    candidate = Path(path)
    if SUSPICIOUS_FILENAME.search(candidate.name):
        return [{"path": candidate.as_posix(), "field_name": "", "reason": "suspicious_filename"}]
    return []


def flag_columns(path: str | Path, columns: list[str]) -> list[dict[str, str]]:
    """Flag sensitive column names only."""

    flags: list[dict[str, str]] = []
    for column in columns:
        for reason, pattern in SENSITIVE_FIELD_PATTERNS.items():
            if pattern.search(column):
                flags.append(
                    {"path": Path(path).as_posix(), "field_name": column, "reason": reason}
                )
                break
    return flags


def header_columns(path: str | Path) -> list[str]:
    """Read only a delimited text header."""

    delimiter = "	" if Path(path).suffix.lower() == ".tsv" else ","
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        return next(reader, [])


def scan_file(path: str | Path) -> list[dict[str, str]]:
    """Scan a file path and safe headers for potential identifiers."""

    flags = flag_path(path)
    if Path(path).suffix.lower() in {".csv", ".tsv"}:
        try:
            flags.extend(flag_columns(path, header_columns(path)))
        except UnicodeDecodeError:
            flags.append(
                {
                    "path": Path(path).as_posix(),
                    "field_name": "",
                    "reason": "unreadable_text_header",
                }
            )
    return flags
