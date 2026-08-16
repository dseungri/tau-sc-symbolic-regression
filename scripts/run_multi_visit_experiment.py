"""Create the leakage-safe multi-visit analysis-freeze scaffold.

The full private model execution is intentionally not run in this pre-data
phase. This command validates the protocol, manifest, output directory, and
fold assignments, then writes the private analysis-freeze artifact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.multi_visit import (  # noqa: E402
    MultiVisitAnalysisConfig,
    build_multi_visit_analysis_freeze,
    load_multi_visit_manifest,
    make_multi_visit_outer_folds,
    prepare_private_multi_visit_output_dir,
    validate_multi_visit_protocol,
    write_private_multi_visit_json,
)
from tau_sr.schemas import SchemaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and freeze a future multi-visit analysis plan.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-md", default="docs/multi_visit_analysis_protocol.md")
    parser.add_argument("--protocol-json", default="config/multi_visit_analysis_protocol.json")
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Required in this pre-data phase; writes analysis_freeze.json only.",
    )
    args = parser.parse_args()

    if not args.freeze_only:
        print(
            "Multi-visit model execution is not enabled in this pre-data phase; "
            "rerun with --freeze-only to validate and write the freeze scaffold.",
            file=sys.stderr,
        )
        return 1

    try:
        validate_multi_visit_protocol(args.protocol_md, args.protocol_json)
        loaded = load_multi_visit_manifest(args.manifest)
        config = MultiVisitAnalysisConfig()
        folds = make_multi_visit_outer_folds(
            loaded.dataset.subject_ids,
            n_folds=config.outer_folds,
            seed=config.fold_seed,
            inner_validation_fraction=config.inner_validation_fraction,
        )
        output = prepare_private_multi_visit_output_dir(args.output_dir)
        freeze = build_multi_visit_analysis_freeze(
            loaded.dataset,
            folds,
            protocol_json_path=args.protocol_json,
            config=config,
        )
        write_private_multi_visit_json(output / "analysis_freeze.json", freeze)
    except SchemaError as exc:
        print(f"Multi-visit analysis freeze failed: {exc}", file=sys.stderr)
        return 1

    print("Multi-visit analysis freeze scaffold written.")
    print(f"Participants: {loaded.qc_summary['subject_count']}")
    print(f"Outer folds: {len(folds)}")
    print("No model fitting or final-visit evaluation was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
