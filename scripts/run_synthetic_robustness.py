"""Run the controlled synthetic robustness sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.schemas import SchemaError  # noqa: E402
from tau_sr.synthetic_robustness import (  # noqa: E402
    SyntheticRobustnessConfig,
    run_synthetic_robustness_condition,
    run_synthetic_robustness_sweep,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the prespecified synthetic robustness sweep.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/synthetic_robustness",
        help="Ignored directory for resumable detailed sweep artifacts.",
    )
    parser.add_argument(
        "--publication-dir",
        default="docs/spie/synthetic_robustness",
        help="Repository documentation directory for aggregate synthetic artifacts.",
    )
    parser.add_argument("--pysr-timeout-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one tiny synthetic smoke condition without using results as evidence.",
    )
    args = parser.parse_args()

    try:
        if args.smoke:
            config = SyntheticRobustnessConfig(
                n_subjects=6,
                n_regions=6,
                times=(0.0, 0.2, 0.4),
                validation_fraction=1 / 6,
                test_fraction=1 / 6,
                hidden_width=2,
                max_epochs=2,
                patience=2,
                pysr_niterations=1,
                pysr_populations=1,
                pysr_population_size=33,
                pysr_maxsize=7,
                pysr_n_samples=20,
                reference_grid_points=40,
                pysr_worker_timeout_seconds=args.pysr_timeout_seconds,
                require_canonical_reproduction=False,
            )
            from tau_sr.synthetic_robustness import build_sweep_manifest

            condition = build_sweep_manifest()[2]
            result = run_synthetic_robustness_condition(
                condition,
                Path(args.results_dir) / "smoke",
                config=config,
            )
            print(f"Smoke status: {result['status']}")
            print("Smoke result is not scientific evidence.")
            return 0 if result["status"] == "complete" else 1

        summary = run_synthetic_robustness_sweep(
            args.results_dir,
            args.publication_dir,
            config=SyntheticRobustnessConfig(
                pysr_worker_timeout_seconds=args.pysr_timeout_seconds,
            ),
            progress=print,
        )
    except SchemaError as exc:
        print(f"Synthetic robustness sweep failed: {exc}", file=sys.stderr)
        return 1

    print(f"Completion: {summary['completed_run_count']}/23 complete")
    print(f"Failures: {summary['failed_run_count']}")
    print(f"Canonical reproduction passed: {summary['canonical_reproduction']['passed']}")
    print("Synthetic robustness artifacts written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
