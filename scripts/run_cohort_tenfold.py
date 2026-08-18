"""Run the full-cohort nested tenfold analysis in explicit stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.cohort_tenfold import (  # noqa: E402
    TenfoldConfig,
    require_clean_repository,
    run_tenfold_analysis,
)
from tau_sr.schemas import SchemaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the privacy-safe full-cohort nested tenfold CV analysis.",
    )
    parser.add_argument("--structure-dir", required=True)
    parser.add_argument("--baseline-tau-dir", required=True)
    parser.add_argument("--followup-tau-dir", required=True)
    parser.add_argument("--preparation-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pysr-timeout-seconds", type=float, default=1200.0)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--freeze-only", action="store_true")
    stage.add_argument("--execute", action="store_true")
    stage.add_argument("--aggregate", action="store_true")
    stage.add_argument("--audit", action="store_true")
    args = parser.parse_args()

    config = TenfoldConfig(pysr_worker_timeout_seconds=args.pysr_timeout_seconds)
    try:
        if args.freeze_only or args.execute or args.aggregate:
            require_clean_repository()
        result = run_tenfold_analysis(
            args.structure_dir,
            args.baseline_tau_dir,
            args.followup_tau_dir,
            args.preparation_artifact_dir,
            args.output_dir,
            freeze_only=args.freeze_only,
            execute=args.execute,
            aggregate_only=args.aggregate,
            audit=args.audit,
            config=config,
            progress=print,
        )
    except SchemaError as exc:
        print(f"Tenfold analysis failed: {exc}", file=sys.stderr)
        return 1

    print(f"Completion status: {result.get('completion_status', result.get('audit_pass'))}")
    if "fold_design" in result:
        print(f"Outer folds: {result['fold_design']['outer_folds']}")
        print(f"Fold-size distribution: {result['fold_design']['outer_fold_size_distribution']}")
    if "audit_pass" in result:
        print(f"Audit pass: {result['audit_pass']}")
    print("Private tenfold artifacts are stored outside the repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
