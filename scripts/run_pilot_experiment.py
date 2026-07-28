"""Run the privacy-preserving two-subject pilot experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.pilot_experiment import run_pilot_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the two-subject temporal pilot.")
    parser.add_argument("--tau-workbook", required=True, help="Local tau workbook path.")
    parser.add_argument(
        "--connectome",
        action="append",
        required=True,
        help="Local subject-specific connectome path. Provide exactly two.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Private output directory. Defaults outside the repository.",
    )
    parser.add_argument(
        "--pysr-worker-timeout-seconds",
        type=float,
        default=600.0,
        help="Maximum seconds for the isolated PySR worker.",
    )
    parser.add_argument(
        "--pysr-worker-shutdown-grace-seconds",
        type=float,
        default=10.0,
        help="Grace period for a completed PySR worker to exit normally.",
    )
    args = parser.parse_args()

    if len(args.connectome) != 2:
        parser.error("provide exactly two --connectome values")

    print("Loading pilot inputs through the validated adapter...")
    print("Running QC and fixed temporal-holdout protocol...")
    result = run_pilot_experiment(
        args.tau_workbook,
        tuple(args.connectome),
        output_directory=args.output_dir,
        pysr_worker_timeout_seconds=args.pysr_worker_timeout_seconds,
        pysr_worker_shutdown_grace_seconds=args.pysr_worker_shutdown_grace_seconds,
    )
    dataset = result["dataset_summary"]
    training = result["training_diagnostics"]
    symbolic = result["symbolic"]
    print(f"Neutral subjects: {', '.join(dataset['subject_ids'])}")
    print(f"Visit counts: {dataset['visit_counts']}")
    print(f"Region count: {dataset['region_count']}")
    print(f"Alignment: {dataset['roi_alignment']}")
    print(f"Data alignment ready: {dataset['data_alignment_ready']}")
    print(f"Formal evaluation ready: {dataset['formal_evaluation_ready']}")
    print(f"Fixed epochs: {training['epochs_run']}")
    print(f"Initial training loss: {training['initial_training_loss']:.6g}")
    print(f"Final training loss: {training['final_training_loss']:.6g}")
    print(f"Learned rho: {training['learned_rho']:.6g}")
    print(f"Numerical instability: {training['numerical_instability']}")
    print(f"PySR exited normally: {symbolic.get('worker_exited_normally')}")
    print(f"Private artifact: {result['private_artifact_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
