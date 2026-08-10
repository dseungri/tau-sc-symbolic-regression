"""Prepare privacy-safe QC and split artifacts for the full real cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.cohort_adapter import (  # noqa: E402
    DEFAULT_SPLIT_SEED,
    inspect_two_timepoint_cohort,
    load_two_timepoint_cohort,
    make_frozen_cohort_split,
    write_private_cohort_artifacts,
)
from tau_sr.schemas import SchemaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare full-cohort QC and frozen split artifacts.",
    )
    parser.add_argument("--structure-dir", required=True)
    parser.add_argument("--baseline-tau-dir", required=True)
    parser.add_argument("--followup-tau-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()

    try:
        qc_summary = inspect_two_timepoint_cohort(
            args.structure_dir,
            args.baseline_tau_dir,
            args.followup_tau_dir,
        )
        dataset = load_two_timepoint_cohort(
            args.structure_dir,
            args.baseline_tau_dir,
            args.followup_tau_dir,
        )
        split = make_frozen_cohort_split(dataset.subject_ids, seed=args.split_seed)
        write_private_cohort_artifacts(
            dataset,
            qc_summary,
            split,
            args.output_dir,
            seed=args.split_seed,
        )
    except SchemaError as exc:
        print(f"Cohort preparation failed: {exc}", file=sys.stderr)
        return 1

    print("Cohort QC passed.")
    print(f"Subjects: {qc_summary['subject_count']}")
    print(f"Matched triplets: {qc_summary['matched_triplets']}")
    print(
        "Split counts: "
        f"train={len(split.train_indices)}, "
        f"validation={len(split.validation_indices)}, "
        f"test={len(split.test_indices)}"
    )
    print("Private QC and split artifacts were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
