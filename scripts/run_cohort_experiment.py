"""Run the privacy-safe full-cohort model-comparison experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.cohort_experiment import (  # noqa: E402
    CohortExperimentConfig,
    run_full_cohort_experiment,
)
from tau_sr.schemas import SchemaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen full-cohort model-comparison experiment.",
    )
    parser.add_argument("--structure-dir", required=True)
    parser.add_argument("--baseline-tau-dir", required=True)
    parser.add_argument("--followup-tau-dir", required=True)
    parser.add_argument("--preparation-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--pysr-timeout-seconds", type=float, default=1200.0)
    args = parser.parse_args()

    try:
        summary = run_full_cohort_experiment(
            args.structure_dir,
            args.baseline_tau_dir,
            args.followup_tau_dir,
            args.preparation_artifact_dir,
            args.output_dir,
            config=CohortExperimentConfig(),
            resume_checkpoint=args.resume_checkpoint,
            pysr_worker_timeout_seconds=args.pysr_timeout_seconds,
        )
    except SchemaError as exc:
        print(f"Full-cohort experiment failed: {exc}", file=sys.stderr)
        return 1

    print(f"Completion status: {summary['completion_status']}")
    cohort = summary["cohort"]
    split = summary["split"]
    print(f"Subjects: {cohort['subject_count']}")
    print(f"Matched triplets: {cohort['matched_triplets']}")
    counts = split["counts"]
    print(
        "Split counts: "
        f"train={counts['train']}, "
        f"validation={counts['validation']}, "
        f"test={counts['test']}"
    )
    if summary["completion_status"] == "complete":
        selected = summary["fitted_models"]["neural"]["selected_seed"]
        print(f"Validation-selected neural seed: {selected}")
        print("Final test metrics are available in the private artifact.")
    else:
        print("Test metrics were not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
