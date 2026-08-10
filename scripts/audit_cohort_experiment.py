"""Run a privacy-safe read-only audit of frozen full-cohort artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.cohort_audit import AuditInputs, audit_frozen_experiment  # noqa: E402
from tau_sr.schemas import SchemaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen full-cohort experiment artifacts without rerunning models.",
    )
    parser.add_argument("--frozen-artifact-dir", required=True)
    parser.add_argument("--audit-output-dir", required=True)
    parser.add_argument("--structure-dir")
    parser.add_argument("--baseline-tau-dir")
    parser.add_argument("--followup-tau-dir")
    parser.add_argument("--preparation-artifact-dir")
    parser.add_argument("--experiment-commit")
    args = parser.parse_args()

    try:
        audit = audit_frozen_experiment(
            AuditInputs(
                frozen_artifact_dir=args.frozen_artifact_dir,
                audit_output_dir=args.audit_output_dir,
                structure_dir=args.structure_dir,
                baseline_tau_dir=args.baseline_tau_dir,
                followup_tau_dir=args.followup_tau_dir,
                preparation_artifact_dir=args.preparation_artifact_dir,
                experiment_commit=args.experiment_commit,
            )
        )
    except SchemaError as exc:
        print(f"Full-cohort audit failed: {exc}", file=sys.stderr)
        return 1

    metric = audit["metric_reproduction"]
    integrity_checks = audit["artifact_integrity"]["checks"]
    print(f"Audit pass: {audit['audit_pass']}")
    print(f"Subject-count check: {integrity_checks['subject_count']}")
    print(f"Split-count check: {integrity_checks['split_counts']}")
    print(f"Max aggregate discrepancy: {metric['max_aggregate_discrepancy']:.3g}")
    print("Audit artifact written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
