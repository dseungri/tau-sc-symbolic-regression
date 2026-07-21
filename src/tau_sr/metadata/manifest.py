"""Draft manifest generation from approved mappings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from tau_sr.metadata.schemas import validate_manifest


def build_draft_manifest(mapping_yaml: str | Path, output_csv: str | Path) -> dict[str, object]:
    """Build a draft manifest from an explicit mapping file."""

    with Path(mapping_yaml).open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle) or {}
    rows = mapping.get("rows", [])
    frame = pd.DataFrame(rows)
    report = validate_manifest(frame)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return {
        "output_csv": destination.as_posix(),
        "row_count": int(len(frame)),
        "validation": report.to_dict(),
        "unmatched_files": mapping.get("unmatched_files", []),
        "duplicate_mappings": mapping.get("duplicate_mappings", []),
    }
