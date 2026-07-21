"""QC report serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tau_sr.qc.common import QCReport


def write_reports_json(reports: list[QCReport], output_path: str | Path) -> None:
    """Write multiple QC reports as machine-readable JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [report.to_dict() for report in reports]
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
