"""Prepare privacy-safe QC artifacts for a future multi-visit cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.multi_visit import (  # noqa: E402
    inspect_multi_visit_manifest,
    prepare_private_multi_visit_output_dir,
    validate_multi_visit_protocol,
    write_private_multi_visit_json,
)
from tau_sr.schemas import SchemaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a private multi-visit manifest and write aggregate QC.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-md", default="docs/multi_visit_analysis_protocol.md")
    parser.add_argument("--protocol-json", default="config/multi_visit_analysis_protocol.json")
    args = parser.parse_args()

    try:
        protocol = validate_multi_visit_protocol(args.protocol_md, args.protocol_json)
        qc_summary = inspect_multi_visit_manifest(args.manifest)
        output = prepare_private_multi_visit_output_dir(args.output_dir)
        write_private_multi_visit_json(
            output / "multi_visit_qc_summary.json",
            {
                "completion_status": "complete",
                "protocol_hash": protocol["protocol_hash"],
                "qc_summary": qc_summary,
            },
        )
    except SchemaError as exc:
        print(f"Multi-visit preparation failed: {exc}", file=sys.stderr)
        return 1

    print("Multi-visit QC passed.")
    print(f"Participants: {qc_summary['subject_count']}")
    print(f"Total visits: {qc_summary['total_tau_visits']}")
    print("Private aggregate QC artifact was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
