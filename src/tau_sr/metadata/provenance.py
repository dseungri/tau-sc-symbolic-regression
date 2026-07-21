"""Provenance record helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tau_sr.intake.checksums import sha256_file


def provenance_record(
    artifact_path: str | Path,
    *,
    artifact_type: str,
    source_paths: list[str],
    script: str,
    config: str,
    git_commit: str = "UNSET",
) -> dict[str, object]:
    """Create a provenance record for a generated artifact."""

    path = Path(artifact_path)
    return {
        "artifact_path": path.as_posix(),
        "artifact_type": artifact_type,
        "source_paths": source_paths,
        "script": script,
        "config": config,
        "git_commit": git_commit,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
        "status": "created" if path.exists() else "missing",
    }
