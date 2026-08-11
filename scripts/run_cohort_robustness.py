"""Run development-only full-cohort robustness cross-validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.cohort_robustness import (  # noqa: E402
    RobustnessConfig,
    run_development_cross_validation,
)
from tau_sr.schemas import SchemaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run development-only nested CV without using the frozen test set.",
    )
    parser.add_argument("--structure-dir", required=True)
    parser.add_argument("--baseline-tau-dir", required=True)
    parser.add_argument("--followup-tau-dir", required=True)
    parser.add_argument("--preparation-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pysr-timeout-seconds", type=float, default=1200.0)
    args = parser.parse_args()

    try:
        summary = run_development_cross_validation(
            args.structure_dir,
            args.baseline_tau_dir,
            args.followup_tau_dir,
            args.preparation_artifact_dir,
            args.output_dir,
            config=RobustnessConfig(
                pysr_worker_timeout_seconds=args.pysr_timeout_seconds,
            ),
            progress=print,
        )
    except SchemaError as exc:
        print(f"Robustness CV failed: {exc}", file=sys.stderr)
        return 1

    print(f"Completion status: {summary['completion_status']}")
    print(f"Development subjects: {summary['development_subjects']}")
    print(f"Excluded frozen test subjects: {summary['excluded_frozen_test_subjects']}")
    print(
        "Frozen test numerical records opened: "
        f"{summary['test_firewall']['test_numerical_records_opened']}"
    )
    print(f"Outer folds: {summary['fold_design']['outer_folds']}")
    print("Private robustness artifacts written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
