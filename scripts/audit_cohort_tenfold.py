"""Run the read-only audit for a completed tenfold experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.cohort_tenfold_audit import audit_tenfold_experiment  # noqa: E402
from tau_sr.schemas import SchemaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a completed full-cohort nested tenfold analysis.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    try:
        audit = audit_tenfold_experiment(args.output_dir)
    except SchemaError as exc:
        print(f"Tenfold audit failed: {exc}", file=sys.stderr)
        return 1

    print(f"Audit pass: {audit['audit_pass']}")
    print(
        "Maximum metric discrepancy: "
        f"{audit['metric_reproduction']['max_aggregate_metric_discrepancy']}"
    )
    print("Read-only tenfold audit complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
