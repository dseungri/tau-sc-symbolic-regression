"""Generate audit-gated aggregate tenfold figures and tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.cohort_tenfold_reporting import generate_tenfold_report  # noqa: E402
from tau_sr.schemas import SchemaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate privacy-safe aggregate tenfold report artifacts.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--publication-dir")
    args = parser.parse_args()

    try:
        manifest = generate_tenfold_report(
            args.output_dir,
            publication_dir=args.publication_dir,
        )
    except SchemaError as exc:
        print(f"Tenfold reporting failed: {exc}", file=sys.stderr)
        return 1

    print(f"Completion status: {manifest['completion_status']}")
    print(f"Figure count: {len(manifest['figures'])}")
    print(f"Table count: {len(manifest['tables'])}")
    print("Aggregate tenfold reporting complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
