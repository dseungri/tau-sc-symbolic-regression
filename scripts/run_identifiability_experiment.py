"""Run controlled neural identifiability diagnostics for the synthetic dataset."""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.identifiability import (  # noqa: E402
    BASELINE_TEST_REACTION_RMSE,
    BASELINE_TEST_RHO,
    BASELINE_TEST_TRAJECTORY_RMSE,
    CONTROLLED_TRUE_GROWTH_RATE,
    CONTROLLED_TRUE_RHO,
    CONTROLLED_TRUE_SATURATION_RATE,
    FixedRhoZeroAnchoredReactionDiffusion,
    IdentifiabilityExperimentResult,
    OracleReactionDiffusion,
    ZeroAnchoredNeuralReactionDiffusion,
    build_controlled_recovery_dataset,
    controlled_reaction_grid,
    controlled_subject_split,
    environment_versions,
    fit_identifiability_model,
    make_baseline_training_config,
    make_oracle_training_config,
    make_zero_anchored_training_config,
    summarize_identifiability_result,
)
from tau_sr.neural import NeuralReactionDiffusion  # noqa: E402
from tau_sr.pysr_lifecycle import (  # noqa: E402
    parse_json_file,
    run_json_worker,
    write_json_atomically,
)
from tau_sr.reactions import make_linear_quadratic_reaction  # noqa: E402
from tau_sr.symbolic import (  # noqa: E402
    ReactionSamples,
    SymbolicRegressionConfig,
    make_midpoint_evaluation_grid,
    sample_neural_reaction,
    training_state_domain,
)
from tau_sr.training import NeuralTrainingConfig, NeuralTrainingResult  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/identifiability_experiment.json",
        help="Path for the machine-readable JSON summary.",
    )
    parser.add_argument(
        "--run-extended-baseline",
        action="store_true",
        help="Also run the optional 1000-epoch unconstrained baseline.",
    )
    parser.add_argument(
        "--force-pysr",
        action="store_true",
        help="Run PySR even if the zero-anchored joint model is not materially improved.",
    )
    parser.add_argument(
        "--pysr-worker-timeout-seconds",
        type=float,
        default=600.0,
        help="Maximum time allowed for the isolated PySR worker.",
    )
    parser.add_argument(
        "--pysr-worker-shutdown-grace-seconds",
        type=float,
        default=10.0,
        help="Grace period for a completed PySR worker to exit normally.",
    )
    parser.add_argument(
        "--pysr-smoke",
        action="store_true",
        help="Run only a tiny real-PySR lifecycle smoke test.",
    )
    parser.add_argument(
        "--pysr-smoke-output",
        default=None,
        help="Output path for --pysr-smoke. Defaults to the system temp directory.",
    )
    args = parser.parse_args()

    if args.pysr_smoke:
        smoke_output = (
            Path(args.pysr_smoke_output)
            if args.pysr_smoke_output is not None
            else Path(tempfile.gettempdir()) / "tau_sr_pysr_lifecycle_smoke.json"
        )
        _run_pysr_smoke(
            smoke_output,
            timeout_seconds=args.pysr_worker_timeout_seconds,
            shutdown_grace_seconds=args.pysr_worker_shutdown_grace_seconds,
        )
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suite_start = time.perf_counter()
    dataset = build_controlled_recovery_dataset()
    split = controlled_subject_split()
    reaction_grid = controlled_reaction_grid(dataset, split, n_points=200)

    records: list[IdentifiabilityExperimentResult] = []
    trained: dict[str, tuple[NeuralTrainingResult, NeuralTrainingConfig, float]] = {}

    baseline_model = _seeded_model(lambda: NeuralReactionDiffusion(hidden_width=16))
    baseline_config = make_baseline_training_config()
    baseline_result = _fit_and_record(
        records,
        trained,
        experiment_name="A_original_baseline",
        candidate_name="baseline",
        model=baseline_model,
        config=baseline_config,
        dataset=dataset,
        split=split,
        reaction_grid=reaction_grid,
        validation_selected=True,
        evaluate_test=True,
    )

    fixed_rho_model = _seeded_model(
        lambda: FixedRhoZeroAnchoredReactionDiffusion(
            fixed_rho=CONTROLLED_TRUE_RHO,
            hidden_width=16,
        )
    )
    fixed_rho_config = make_oracle_training_config(max_epochs=1000, patience=100)
    fixed_rho_result = _fit_and_record(
        records,
        trained,
        experiment_name="B_fixed_rho_reaction_oracle",
        candidate_name="fixed_rho_0.1",
        model=fixed_rho_model,
        config=fixed_rho_config,
        dataset=dataset,
        split=split,
        reaction_grid=reaction_grid,
        validation_selected=True,
        evaluate_test=True,
    )

    oracle_candidates: list[IdentifiabilityExperimentResult] = []
    for rho_lr in (1e-3, 1e-2):
        candidate = _fit_and_record(
            records,
            trained,
            experiment_name="C_oracle_reaction_rho_recovery",
            candidate_name=f"rho_lr_{rho_lr:g}",
            model=OracleReactionDiffusion(
                growth_rate=CONTROLLED_TRUE_GROWTH_RATE,
                saturation_rate=CONTROLLED_TRUE_SATURATION_RATE,
                initial_rho=0.05,
            ),
            config=make_oracle_training_config(
                max_epochs=1000,
                patience=100,
                rho_learning_rate=rho_lr,
            ),
            dataset=dataset,
            split=split,
            reaction_grid=reaction_grid,
            validation_selected=False,
            evaluate_test=False,
        )
        oracle_candidates.append(candidate)
    selected_oracle = _select_validation_candidate(oracle_candidates)
    selected_oracle_result = _summarize_existing(
        records,
        trained,
        selected_oracle,
        dataset=dataset,
        split=split,
        reaction_grid=reaction_grid,
        validation_selected=True,
        evaluate_test=True,
    )

    zero_candidates: list[IdentifiabilityExperimentResult] = []
    for rho_lr in (1e-3, 5e-3, 1e-2):
        candidate = _fit_and_record(
            records,
            trained,
            experiment_name="D_zero_anchored_joint_recovery",
            candidate_name=f"rho_lr_{rho_lr:g}",
            model=_seeded_model(lambda: ZeroAnchoredNeuralReactionDiffusion(hidden_width=16)),
            config=make_zero_anchored_training_config(
                rho_learning_rate=rho_lr,
                max_epochs=1000,
                patience=100,
            ),
            dataset=dataset,
            split=split,
            reaction_grid=reaction_grid,
            validation_selected=False,
            evaluate_test=False,
        )
        zero_candidates.append(candidate)
    selected_zero = _select_validation_candidate(zero_candidates)
    selected_zero_result = _summarize_existing(
        records,
        trained,
        selected_zero,
        dataset=dataset,
        split=split,
        reaction_grid=reaction_grid,
        validation_selected=True,
        evaluate_test=True,
    )

    extended_baseline_result = None
    if args.run_extended_baseline:
        extended_config = make_baseline_training_config()
        extended_config = NeuralTrainingConfig(
            hidden_width=extended_config.hidden_width,
            initial_rho=extended_config.initial_rho,
            learning_rate=extended_config.learning_rate,
            rho_learning_rate=extended_config.rho_learning_rate,
            weight_decay=extended_config.weight_decay,
            max_epochs=1000,
            patience=100,
            min_delta=extended_config.min_delta,
            max_step=extended_config.max_step,
            seed=extended_config.seed,
            dtype=extended_config.dtype,
            device=extended_config.device,
        )
        extended_baseline_result = _fit_and_record(
            records,
            trained,
            experiment_name="E_extended_unconstrained_baseline",
            candidate_name="max_epochs_1000",
            model=_seeded_model(lambda: NeuralReactionDiffusion(hidden_width=16)),
            config=extended_config,
            dataset=dataset,
            split=split,
            reaction_grid=reaction_grid,
            validation_selected=True,
            evaluate_test=True,
        )

    pysr_summary = _maybe_run_pysr(
        force=args.force_pysr,
        selected_zero=selected_zero_result,
        baseline=baseline_result,
        dataset=dataset,
        split=split,
        trained=trained,
        timeout_seconds=args.pysr_worker_timeout_seconds,
        shutdown_grace_seconds=args.pysr_worker_shutdown_grace_seconds,
    )

    summary = {
        "environment": environment_versions(),
        "dataset": {
            "n_subjects": len(dataset.subject_ids),
            "n_regions": len(dataset.roi_labels),
            "times": list(dataset.times_by_subject[0]),
            "representation": dataset.representation,
            "laplacian_kind": dataset.laplacian_kind,
            "true_rho": dataset.rho,
            "growth_rate": dataset.growth_rate,
            "saturation_rate": dataset.saturation_rate,
            "seed": dataset.seed,
        },
        "fixed_split": {
            "train_indices": list(split.train_indices),
            "validation_indices": list(split.validation_indices),
            "test_indices": list(split.test_indices),
        },
        "baseline_reference": {
            "test_trajectory_rmse": BASELINE_TEST_TRAJECTORY_RMSE,
            "learned_rho": BASELINE_TEST_RHO,
            "reaction_curve_rmse": BASELINE_TEST_REACTION_RMSE,
        },
        "records": [record.to_dict() for record in records],
        "selected": {
            "oracle_reaction": selected_oracle_result.to_dict(),
            "zero_anchored_joint": selected_zero_result.to_dict(),
        },
        "fixed_rho_oracle": fixed_rho_result.to_dict(),
        "original_baseline": baseline_result.to_dict(),
        "extended_baseline": (
            None if extended_baseline_result is None else extended_baseline_result.to_dict()
        ),
        "pysr": pysr_summary,
        "runtime_seconds": time.perf_counter() - suite_start,
    }
    artifact_validated = _write_validated_summary(output_path, summary)

    _print_table(records)
    _print_completion(
        artifact_path=output_path,
        artifact_validated=artifact_validated,
        pysr_summary=pysr_summary,
        exit_status=0,
    )
    return 0


def _seeded_model(factory: Any) -> NeuralReactionDiffusion:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return factory()


def _fit_and_record(
    records: list[IdentifiabilityExperimentResult],
    trained: dict[str, tuple[NeuralTrainingResult, NeuralTrainingConfig, float]],
    *,
    experiment_name: str,
    candidate_name: str,
    model: NeuralReactionDiffusion,
    config: NeuralTrainingConfig,
    dataset: Any,
    split: Any,
    reaction_grid: np.ndarray,
    validation_selected: bool,
    evaluate_test: bool,
) -> IdentifiabilityExperimentResult:
    training_result, runtime_seconds = fit_identifiability_model(dataset, split, config, model)
    key = _record_key(experiment_name, candidate_name)
    trained[key] = (training_result, config, runtime_seconds)
    record = summarize_identifiability_result(
        experiment_name=experiment_name,
        candidate_name=candidate_name,
        training_result=training_result,
        runtime_seconds=runtime_seconds,
        dataset=dataset,
        split=split,
        config=config,
        reaction_grid=reaction_grid,
        validation_selected=validation_selected,
        evaluate_test=evaluate_test,
    )
    records.append(record)
    return record


def _summarize_existing(
    records: list[IdentifiabilityExperimentResult],
    trained: dict[str, tuple[NeuralTrainingResult, NeuralTrainingConfig, float]],
    selected: IdentifiabilityExperimentResult,
    *,
    dataset: Any,
    split: Any,
    reaction_grid: np.ndarray,
    validation_selected: bool,
    evaluate_test: bool,
) -> IdentifiabilityExperimentResult:
    training_result, config, runtime_seconds = trained[
        _record_key(selected.experiment_name, selected.candidate_name)
    ]
    record = summarize_identifiability_result(
        experiment_name=selected.experiment_name,
        candidate_name=selected.candidate_name,
        training_result=training_result,
        runtime_seconds=runtime_seconds,
        dataset=dataset,
        split=split,
        config=config,
        reaction_grid=reaction_grid,
        validation_selected=validation_selected,
        evaluate_test=evaluate_test,
    )
    records.append(record)
    return record


def _select_validation_candidate(
    candidates: list[IdentifiabilityExperimentResult],
) -> IdentifiabilityExperimentResult:
    return min(
        candidates,
        key=lambda candidate: math.inf
        if candidate.best_validation_loss is None
        else candidate.best_validation_loss,
    )


def _maybe_run_pysr(
    *,
    force: bool,
    selected_zero: IdentifiabilityExperimentResult,
    baseline: IdentifiabilityExperimentResult,
    dataset: Any,
    split: Any,
    trained: dict[str, tuple[NeuralTrainingResult, NeuralTrainingConfig, float]],
    timeout_seconds: float,
    shutdown_grace_seconds: float,
) -> dict[str, Any]:
    zero_rho_error = selected_zero.rho_absolute_error
    zero_reaction_rmse = selected_zero.reaction_curve_rmse
    baseline_rho_error = baseline.rho_absolute_error
    baseline_reaction_rmse = baseline.reaction_curve_rmse
    materially_improved = (
        zero_rho_error is not None
        and zero_reaction_rmse is not None
        and baseline_rho_error is not None
        and baseline_reaction_rmse is not None
        and zero_rho_error < 0.9 * baseline_rho_error
        and zero_reaction_rmse < 0.9 * baseline_reaction_rmse
    )
    if not (force or materially_improved):
        return {
            "ran": False,
            "reason": (
                "zero-anchored joint model did not materially improve rho and "
                "reaction recovery"
            ),
            "materially_improved": materially_improved,
            "worker_exited_normally": None,
            "worker_cleanup_required": False,
        }

    training_result, config, _runtime = trained[
        _record_key(selected_zero.experiment_name, selected_zero.candidate_name)
    ]
    domain_min, domain_max = training_state_domain(dataset, split)
    samples = sample_neural_reaction(
        training_result.model,
        domain_min=domain_min,
        domain_max=domain_max,
        n_samples=500,
        device=config.device,
        dtype=config.dtype,
    )
    symbolic_config = SymbolicRegressionConfig(
        niterations=100,
        populations=15,
        population_size=33,
        maxsize=15,
        seed=42,
        model_selection="best",
        output_directory=None,
    )
    evaluation_grid = make_midpoint_evaluation_grid(domain_min, domain_max, n_points=1000)
    neural_prediction = _neural_reaction_on_grid(
        training_result.model,
        evaluation_grid,
        config=config,
    )
    true_reaction = make_linear_quadratic_reaction(
        dataset.growth_rate,
        dataset.saturation_rate,
    )
    true_prediction = true_reaction(evaluation_grid.copy())

    payload = _run_isolated_pysr_fit(
        samples=samples,
        symbolic_config=symbolic_config,
        evaluation_grid=evaluation_grid,
        neural_prediction=neural_prediction,
        true_prediction=true_prediction,
        true_growth_rate=dataset.growth_rate,
        true_saturation_rate=dataset.saturation_rate,
        timeout_seconds=timeout_seconds,
        shutdown_grace_seconds=shutdown_grace_seconds,
    )
    return {
        "ran": True,
        "materially_improved": materially_improved,
        **payload,
    }


def _run_isolated_pysr_fit(
    *,
    samples: ReactionSamples,
    symbolic_config: SymbolicRegressionConfig,
    evaluation_grid: np.ndarray,
    neural_prediction: np.ndarray,
    true_prediction: np.ndarray,
    true_growth_rate: float,
    true_saturation_rate: float,
    timeout_seconds: float,
    shutdown_grace_seconds: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tau_sr_pysr_worker_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        request_path = temporary_root / "request.json"
        result_path = temporary_root / "result.json"
        write_json_atomically(
            request_path,
            _pysr_worker_request(
                samples=samples,
                symbolic_config=symbolic_config,
                evaluation_grid=evaluation_grid,
                neural_prediction=neural_prediction,
                true_prediction=true_prediction,
                true_growth_rate=true_growth_rate,
                true_saturation_rate=true_saturation_rate,
            ),
        )
        worker_result = run_json_worker(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_pysr_worker.py"),
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
            result_path,
            timeout_seconds=timeout_seconds,
            shutdown_grace_seconds=shutdown_grace_seconds,
        )

    return {
        **worker_result.payload,
        "worker_exited_normally": worker_result.process_exited_normally,
        "worker_cleanup_required": worker_result.cleanup_required,
        "worker_timed_out_after_result": worker_result.timed_out,
        "worker_returncode": worker_result.returncode,
    }


def _pysr_worker_request(
    *,
    samples: ReactionSamples,
    symbolic_config: SymbolicRegressionConfig,
    evaluation_grid: np.ndarray,
    neural_prediction: np.ndarray,
    true_prediction: np.ndarray,
    true_growth_rate: float,
    true_saturation_rate: float,
) -> dict[str, Any]:
    return {
        "samples": {
            "input_values": samples.input_values.tolist(),
            "neural_reaction_values": samples.neural_reaction_values.tolist(),
            "domain_min": samples.domain_min,
            "domain_max": samples.domain_max,
        },
        "config": _symbolic_config_dict(symbolic_config),
        "evaluation": {
            "grid": np.asarray(evaluation_grid, dtype=float).tolist(),
            "neural_prediction": np.asarray(neural_prediction, dtype=float).tolist(),
            "true_prediction": np.asarray(true_prediction, dtype=float).tolist(),
            "true_growth_rate": float(true_growth_rate),
            "true_saturation_rate": float(true_saturation_rate),
        },
    }


def _symbolic_config_dict(config: SymbolicRegressionConfig) -> dict[str, Any]:
    return {
        "niterations": config.niterations,
        "populations": config.populations,
        "population_size": config.population_size,
        "maxsize": config.maxsize,
        "seed": config.seed,
        "model_selection": config.model_selection,
        "output_directory": config.output_directory,
    }


def _neural_reaction_on_grid(
    model: NeuralReactionDiffusion,
    grid: np.ndarray,
    *,
    config: NeuralTrainingConfig,
) -> np.ndarray:
    y_values = torch.as_tensor(
        np.asarray(grid, dtype=float).reshape(-1, 1),
        device=torch.device(config.device),
        dtype=config.dtype,
    )
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            prediction = model.reaction(y_values)
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    values = prediction.detach().cpu().numpy().reshape(-1)
    if values.shape != np.asarray(grid).shape or not np.isfinite(values).all():
        raise RuntimeError("neural reaction evaluation produced invalid values.")
    return values.copy()


def _run_pysr_smoke(
    output_path: Path,
    *,
    timeout_seconds: float,
    shutdown_grace_seconds: float,
) -> None:
    y_values = np.linspace(0.0, 1.0, 8, dtype=float)
    reaction_values = 0.5 * y_values - 0.5 * y_values**2
    samples = ReactionSamples(
        input_values=y_values.reshape(-1, 1),
        neural_reaction_values=reaction_values,
        domain_min=float(y_values[0]),
        domain_max=float(y_values[-1]),
    )
    symbolic_config = SymbolicRegressionConfig(
        niterations=1,
        populations=1,
        population_size=20,
        maxsize=7,
        seed=42,
        model_selection="best",
        output_directory=None,
    )
    evaluation_grid = make_midpoint_evaluation_grid(0.0, 1.0, n_points=20)
    true_prediction = 0.5 * evaluation_grid - 0.5 * evaluation_grid**2
    pysr_summary = _run_isolated_pysr_fit(
        samples=samples,
        symbolic_config=symbolic_config,
        evaluation_grid=evaluation_grid,
        neural_prediction=true_prediction.copy(),
        true_prediction=true_prediction,
        true_growth_rate=0.5,
        true_saturation_rate=0.5,
        timeout_seconds=timeout_seconds,
        shutdown_grace_seconds=shutdown_grace_seconds,
    )
    summary = {
        "smoke": True,
        "pysr": {"ran": True, **pysr_summary},
        "runtime_note": "Tiny real-PySR lifecycle smoke; not the controlled experiment.",
    }
    artifact_validated = _write_validated_summary(output_path, summary)
    _print_completion(
        artifact_path=output_path,
        artifact_validated=artifact_validated,
        pysr_summary=summary["pysr"],
        exit_status=0,
    )


def _write_validated_summary(path: Path, summary: dict[str, Any]) -> bool:
    _validate_summary_structure(summary)
    write_json_atomically(path, summary)
    parsed = parse_json_file(path)
    if not isinstance(parsed, dict):
        raise RuntimeError("final artifact did not parse as a JSON object.")
    return True


def _validate_summary_structure(summary: dict[str, Any]) -> None:
    if not isinstance(summary, dict):
        raise RuntimeError("summary must be a JSON object.")
    for key in ("pysr",):
        if key not in summary or not isinstance(summary[key], dict):
            raise RuntimeError(f"summary must contain object field {key!r}.")


def _print_completion(
    *,
    artifact_path: Path,
    artifact_validated: bool,
    pysr_summary: dict[str, Any],
    exit_status: int,
) -> None:
    print(f"\nFinal artifact path: {artifact_path}")
    print(f"Artifact validated successfully: {artifact_validated}")
    print(f"PySR exited normally: {pysr_summary.get('worker_exited_normally')}")
    print(f"Worker cleanup required: {pysr_summary.get('worker_cleanup_required')}")
    print(f"Final script exit status: {exit_status}")


def _record_key(experiment_name: str, candidate_name: str) -> str:
    return f"{experiment_name}:{candidate_name}"


def _print_table(records: list[IdentifiabilityExperimentResult]) -> None:
    headers = (
        "experiment",
        "candidate",
        "selected",
        "test",
        "val_rmse",
        "test_rmse",
        "rho",
        "rho_abs",
        "rxn_rmse",
    )
    print("\t".join(headers))
    for record in records:
        values = (
            record.experiment_name,
            record.candidate_name,
            str(record.validation_selected),
            str(record.test_evaluated),
            _format_optional(record.validation_trajectory_rmse),
            _format_optional(record.test_trajectory_rmse),
            _format_optional(record.learned_rho),
            _format_optional(record.rho_absolute_error),
            _format_optional(record.reaction_curve_rmse),
        )
        print("\t".join(values))


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
