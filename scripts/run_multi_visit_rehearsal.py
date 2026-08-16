"""Run a synthetic-only rehearsal of the multi-visit analysis control flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.multi_visit import run_synthetic_multi_visit_rehearsal  # noqa: E402
from tau_sr.schemas import SchemaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the synthetic-only multi-visit rehearsal.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()

    try:
        summary = run_synthetic_multi_visit_rehearsal(
            args.output_dir,
            n_subjects=args.subjects,
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    except SchemaError as exc:
        print(f"Multi-visit synthetic rehearsal failed: {exc}", file=sys.stderr)
        return 1

    print("Synthetic-only rehearsal complete.")
    print(f"Participants: {summary['configuration']['n_subjects']}")
    print(f"Fold completion: {summary['fold_completion']['completed']}/"
          f"{summary['fold_completion']['expected']}")
    print(f"Numerical failures: {summary['numerical_failure_count']}")
    print("Results are software-validation artifacts, not research evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
