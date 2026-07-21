"""SHA-256 checksum utilities."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file in chunks and return a lowercase SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def checksum_records(root: str | Path, *, recursive: bool = True) -> list[dict[str, object]]:
    """Create checksum records for files under a root without modifying them."""

    base = Path(root).resolve()
    files: Iterable[Path] = base.rglob("*") if recursive else base.glob("*")
    records = []
    for path in sorted(p for p in files if p.is_file()):
        stat = path.stat()
        records.append(
            {
                "relative_path": path.relative_to(base).as_posix(),
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def write_checksum_manifest(records: list[dict[str, object]], output_csv: str | Path) -> None:
    """Write checksum records to a CSV manifest."""

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(records)


def verify_checksum_manifest(root: str | Path, manifest_csv: str | Path) -> list[dict[str, str]]:
    """Verify a checksum manifest and return mismatch records."""

    base = Path(root).resolve()
    mismatches: list[dict[str, str]] = []
    with Path(manifest_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = base / row["relative_path"]
            if not path.exists():
                mismatches.append({"relative_path": row["relative_path"], "reason": "missing"})
                continue
            current = sha256_file(path)
            if current != row["sha256"]:
                mismatches.append(
                    {"relative_path": row["relative_path"], "reason": "sha256_mismatch"}
                )
    return mismatches
