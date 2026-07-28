"""Privacy-preserving two-subject pilot experiment orchestration."""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tau_sr.identifiability import ZeroAnchoredNeuralReactionDiffusion, environment_versions
from tau_sr.pilot_adapter import (
    DATA_ALIGNMENT_READY,
    FORMAL_EVALUATION_READY,
    PILOT_N_LEFT_CORTICAL,
    PILOT_N_REGIONS,
    PILOT_N_RIGHT_CORTICAL,
    PILOT_N_SUBCORTICAL,
    PILOT_SUBCORTICAL_START,
    ROI_ALIGNMENT_DIRECT,
    LongitudinalTrajectoryDataset,
    inspect_pilot_inputs,
    load_pilot_dataset,
)
from tau_sr.pysr_lifecycle import (
    PySRWorkerError,
    parse_json_file,
    run_json_worker,
    write_json_atomically,
)
from tau_sr.schemas import SchemaError
from tau_sr.symbolic import (
    ReactionSamples,
    SymbolicRegressionConfig,
    make_midpoint_evaluation_grid,
    sample_neural_reaction,
)
from tau_sr.training import (
    NeuralTrainingConfig,
    fit_fixed_epoch_recovery_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_ARTIFACT = REPO_ROOT / "experiments/results/identifiability_experiment.json"
DEFAULT_PRIVATE_OUTPUT_DIR = Path.home() / ".tau_sr_private_pilot"
DEFAULT_PYSR_N_SAMPLES = 500
DEFAULT_PYSR_NITERATIONS = 100
DEFAULT_PYSR_POPULATIONS = 15
DEFAULT_PYSR_POPULATION_SIZE = 33
DEFAULT_PYSR_MAXSIZE = 15
DEFAULT_PYSR_SEED = 42

__all__ = (
    "DEFAULT_CONFIG_ARTIFACT",
    "PilotTemporalHoldout",
    "build_temporal_holdout",
    "forecast_metrics",
    "load_selected_pilot_training_config",
    "pilot_qc_summary",
    "run_pilot_experiment",
    "training_state_domain",
    "validate_private_output_directory",
)


@dataclass(frozen=True)
class PilotTemporalHoldout:
    full_dataset: LongitudinalTrajectoryDataset
    training_dataset: LongitudinalTrajectoryDataset
    two_visit_subject_index: int
    three_visit_subject_index: int
    heldout_subject_index: int
    heldout_visit_index: int
    training_visit_indices_by_subject: tuple[tuple[int, ...], ...]
    heldout_time: float
    heldout_state: np.ndarray
    previous_time: float
    previous_state: np.ndarray

    @property
    def train_subject_indices(self) -> tuple[int, int]:
        return (0, 1)


def pilot_qc_summary(
    dataset: LongitudinalTrajectoryDataset,
    *,
    missing_tau_values: int = 0,
) -> dict[str, object]:
    """Return aggregate QC fields for the confirmed two-subject pilot contract."""

    if not isinstance(dataset, LongitudinalTrajectoryDataset):
        raise SchemaError("dataset must be a LongitudinalTrajectoryDataset.")
    missing_count = _nonnegative_int(missing_tau_values, "missing_tau_values")
    nonfinite_tau_count = sum(
        int(trajectory.size - np.count_nonzero(np.isfinite(trajectory)))
        for trajectory in dataset.trajectories
    )
    degrees = dataset.adjacencies.sum(axis=2)
    visit_counts = tuple(int(trajectory.shape[0]) for trajectory in dataset.trajectories)
    return {
        "subject_count": len(dataset.subject_ids),
        "subject_ids": tuple(dataset.subject_ids),
        "visit_counts": {
            subject_id: int(dataset.trajectories[index].shape[0])
            for index, subject_id in enumerate(dataset.subject_ids)
        },
        "relative_times_years": {
            subject_id: tuple(float(value) for value in dataset.times_by_subject[index])
            for index, subject_id in enumerate(dataset.subject_ids)
        },
        "region_count": len(dataset.roi_labels),
        "tau_values_finite": all(
            np.isfinite(trajectory).all() for trajectory in dataset.trajectories
        ),
        "missing_tau_values": missing_count,
        "tau_values_missing": missing_count > 0 or nonfinite_tau_count > 0,
        "adjacency_shapes": tuple(tuple(adjacency.shape) for adjacency in dataset.adjacencies),
        "laplacian_shapes": tuple(tuple(laplacian.shape) for laplacian in dataset.laplacians),
        "adjacencies_finite": bool(np.isfinite(dataset.adjacencies).all()),
        "adjacencies_nonnegative": bool((dataset.adjacencies >= 0.0).all()),
        "adjacencies_symmetric": bool(
            all(np.allclose(adjacency, adjacency.T) for adjacency in dataset.adjacencies)
        ),
        "adjacencies_zero_diagonal": bool(
            all(np.allclose(np.diag(adjacency), 0.0) for adjacency in dataset.adjacencies)
        ),
        "minimum_degree": float(np.min(degrees)),
        "positive_finite_degree": bool(np.isfinite(degrees).all() and (degrees > 0.0).all()),
        "subject_specific_laplacians": bool(
            dataset.laplacians.shape[0] == 2
            and not np.array_equal(dataset.laplacians[0], dataset.laplacians[1])
        ),
        "roi_alignment": dataset.roi_alignment,
        "data_alignment_ready": dataset.data_alignment_ready,
        "formal_evaluation_ready": dataset.formal_evaluation_ready,
        "structural_ordering": {
            "left_cortical": PILOT_N_LEFT_CORTICAL,
            "right_cortical": PILOT_N_RIGHT_CORTICAL,
            "subcortical": PILOT_N_SUBCORTICAL,
            "subcortical_start_zero_based": PILOT_SUBCORTICAL_START,
        },
        "final_12_nodes_subcortical": bool(PILOT_N_SUBCORTICAL == 12),
        "visit_count_pattern": tuple(sorted(visit_counts)),
    }


def validate_pilot_qc(summary: dict[str, object]) -> None:
    expected_shapes = ((PILOT_N_REGIONS, PILOT_N_REGIONS),) * 2
    checks = (
        summary.get("subject_count") == 2,
        summary.get("region_count") == PILOT_N_REGIONS,
        summary.get("visit_count_pattern") == (2, 3),
        summary.get("tau_values_finite") is True,
        summary.get("tau_values_missing") is False,
        summary.get("adjacency_shapes") == expected_shapes,
        summary.get("laplacian_shapes") == expected_shapes,
        summary.get("adjacencies_finite") is True,
        summary.get("adjacencies_nonnegative") is True,
        summary.get("adjacencies_symmetric") is True,
        summary.get("adjacencies_zero_diagonal") is True,
        summary.get("positive_finite_degree") is True,
        summary.get("subject_specific_laplacians") is True,
        summary.get("roi_alignment") == ROI_ALIGNMENT_DIRECT,
        summary.get("data_alignment_ready") is DATA_ALIGNMENT_READY,
        summary.get("formal_evaluation_ready") is FORMAL_EVALUATION_READY,
        summary.get("final_12_nodes_subcortical") is True,
    )
    if not all(checks):
        raise SchemaError("pilot QC failed; training was not started.")


def build_temporal_holdout(dataset: LongitudinalTrajectoryDataset) -> PilotTemporalHoldout:
    """Build the predeclared two-subject temporal holdout split."""

    if len(dataset.subject_ids) != 2:
        raise SchemaError("pilot temporal holdout requires exactly two subjects.")
    visit_counts = tuple(int(trajectory.shape[0]) for trajectory in dataset.trajectories)
    two_visit = tuple(index for index, count in enumerate(visit_counts) if count == 2)
    three_visit = tuple(index for index, count in enumerate(visit_counts) if count == 3)
    if len(two_visit) != 1 or len(three_visit) != 1:
        raise SchemaError(
            "pilot temporal holdout requires one two-visit and one three-visit subject."
        )

    two_index = two_visit[0]
    three_index = three_visit[0]
    training_times = list(dataset.times_by_subject)
    training_trajectories = list(dataset.trajectories)
    training_times[three_index] = dataset.times_by_subject[three_index][:2].copy()
    training_trajectories[three_index] = dataset.trajectories[three_index][:2].copy()
    training_dataset = LongitudinalTrajectoryDataset(
        subject_ids=dataset.subject_ids,
        roi_labels=dataset.roi_labels,
        adjacencies=dataset.adjacencies,
        laplacians=dataset.laplacians,
        times_by_subject=tuple(training_times),
        trajectories=tuple(training_trajectories),
        atlas=dataset.atlas,
        roi_alignment=dataset.roi_alignment,
        tau_measurement=dataset.tau_measurement,
        tau_reference=dataset.tau_reference,
        sc_weight_type=dataset.sc_weight_type,
        sc_static_over_time=dataset.sc_static_over_time,
        laplacian_kind=dataset.laplacian_kind,
        data_alignment_ready=dataset.data_alignment_ready,
        formal_evaluation_ready=dataset.formal_evaluation_ready,
    )
    holdout = PilotTemporalHoldout(
        full_dataset=dataset,
        training_dataset=training_dataset,
        two_visit_subject_index=two_index,
        three_visit_subject_index=three_index,
        heldout_subject_index=three_index,
        heldout_visit_index=2,
        training_visit_indices_by_subject=tuple(
            tuple(range(times.shape[0])) for times in training_dataset.times_by_subject
        ),
        heldout_time=float(dataset.times_by_subject[three_index][2]),
        heldout_state=dataset.trajectories[three_index][2].copy(),
        previous_time=float(dataset.times_by_subject[three_index][1]),
        previous_state=dataset.trajectories[three_index][1].copy(),
    )
    validate_temporal_holdout(holdout)
    return holdout


def validate_temporal_holdout(holdout: PilotTemporalHoldout) -> None:
    three_index = holdout.three_visit_subject_index
    if holdout.heldout_visit_index in holdout.training_visit_indices_by_subject[three_index]:
        raise SchemaError("held-out visit appears in the training visit indices.")
    if holdout.training_dataset.trajectories[three_index].shape[0] != 2:
        raise SchemaError("three-visit subject training trajectory must contain only two visits.")
    if holdout.training_dataset.trajectories[holdout.two_visit_subject_index].shape[0] != 2:
        raise SchemaError("two-visit subject must contribute both visits to training.")
    if np.shares_memory(
        holdout.heldout_state,
        holdout.training_dataset.trajectories[three_index],
    ):
        raise SchemaError("held-out state must not share memory with training trajectories.")
    if not np.array_equal(
        holdout.training_dataset.laplacians[three_index],
        holdout.full_dataset.laplacians[three_index],
    ):
        raise SchemaError("subject-specific Laplacian association changed.")


def load_selected_pilot_training_config(
    artifact_path: str | Path = DEFAULT_CONFIG_ARTIFACT,
) -> tuple[NeuralTrainingConfig, int, dict[str, object]]:
    """Load the selected zero-anchored synthetic configuration and fixed epoch count."""

    artifact = Path(artifact_path)
    payload = parse_json_file(artifact)
    if not isinstance(payload, dict):
        raise SchemaError("identifiability artifact must be a JSON object.")
    selected = payload.get("selected")
    if not isinstance(selected, dict) or not isinstance(selected.get("zero_anchored_joint"), dict):
        raise SchemaError("identifiability artifact is missing the selected zero-anchored record.")
    record = selected["zero_anchored_joint"]
    if record.get("model_type") != "ZeroAnchoredNeuralReactionDiffusion":
        raise SchemaError("selected synthetic record must be zero-anchored.")
    best_epoch = _int_field(record, "best_epoch")
    n_epochs = best_epoch + 1
    config_payload = record.get("training_config")
    if not isinstance(config_payload, dict):
        raise SchemaError("selected synthetic record is missing training_config.")
    config = NeuralTrainingConfig(
        hidden_width=_int_field(config_payload, "hidden_width"),
        initial_rho=_float_field(config_payload, "initial_rho"),
        learning_rate=_float_field(config_payload, "learning_rate"),
        rho_learning_rate=_optional_float_field(config_payload, "rho_learning_rate"),
        weight_decay=_float_field(config_payload, "weight_decay"),
        max_epochs=_int_field(config_payload, "max_epochs"),
        patience=_int_field(config_payload, "patience"),
        min_delta=_float_field(config_payload, "min_delta"),
        max_step=_float_field(config_payload, "max_step"),
        seed=_int_field(config_payload, "seed"),
        dtype=_dtype_from_string(str(config_payload.get("dtype", "torch.float64"))),
        device=str(config_payload.get("device", "cpu")),
    )
    try:
        artifact_label = str(artifact.resolve().relative_to(REPO_ROOT))
    except ValueError:
        artifact_label = artifact.name
    source = {
        "artifact": artifact_label,
        "selected_candidate": str(record.get("candidate_name")),
        "model_type": str(record.get("model_type")),
        "best_epoch_zero_indexed": best_epoch,
        "fixed_epoch_count": n_epochs,
        "epoch_conversion": "n_epochs = best_epoch + 1",
    }
    return config, n_epochs, source


def training_state_domain(
    dataset: LongitudinalTrajectoryDataset,
    subject_indices: Sequence[int],
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for subject_index in subject_indices:
        trajectory = dataset.trajectories[subject_index]
        values.append(trajectory.ravel())
    combined = np.concatenate(values)
    if combined.size == 0 or not np.isfinite(combined).all():
        raise SchemaError("training trajectories must contain finite values.")
    return float(combined.min()), float(combined.max())


def run_pilot_experiment(
    tau_workbook_path: str | Path,
    connectome_paths: Sequence[str | Path],
    *,
    output_directory: str | Path | None = None,
    config_artifact_path: str | Path = DEFAULT_CONFIG_ARTIFACT,
    pysr_worker_timeout_seconds: float = 600.0,
    pysr_worker_shutdown_grace_seconds: float = 10.0,
) -> dict[str, object]:
    """Run the predeclared privacy-safe two-subject pilot experiment."""

    suite_start = time.perf_counter()
    output_dir = validate_private_output_directory(
        DEFAULT_PRIVATE_OUTPUT_DIR if output_directory is None else output_directory
    )
    adapter_summary = inspect_pilot_inputs(tau_workbook_path, connectome_paths)
    dataset = load_pilot_dataset(tau_workbook_path, connectome_paths)
    qc = pilot_qc_summary(
        dataset,
        missing_tau_values=_nonnegative_int(
            adapter_summary.get("missing_tau_values", 0),
            "missing_tau_values",
        ),
    )
    validate_pilot_qc(qc)
    holdout = build_temporal_holdout(dataset)
    config, n_epochs, config_source = load_selected_pilot_training_config(config_artifact_path)
    model = _seeded_zero_anchored_model(config)

    train_start = time.perf_counter()
    training_result = fit_fixed_epoch_recovery_model(
        holdout.training_dataset,
        holdout.train_subject_indices,
        config,
        model=model,
        n_epochs=n_epochs,
    )
    training_runtime = time.perf_counter() - train_start

    forecasts = evaluate_forecasts(
        training_result.model,
        holdout,
        config=config,
    )
    reaction_diagnostics = neural_reaction_diagnostics(
        training_result.model,
        holdout.training_dataset,
        holdout.train_subject_indices,
        config=config,
    )
    symbolic_summary = run_symbolic_distillation(
        training_result.model,
        holdout.training_dataset,
        holdout.train_subject_indices,
        config=config,
        output_dir=output_dir,
        timeout_seconds=pysr_worker_timeout_seconds,
        shutdown_grace_seconds=pysr_worker_shutdown_grace_seconds,
    )

    summary = {
        "experiment": "two_subject_temporal_holdout_pilot",
        "limitations": (
            "Exploratory two-subject feasibility pilot; within-subject temporal holdout; "
            "not subject-level generalization, clinical prediction, or validated biology."
        ),
        "dataset_summary": _safe_dataset_summary(qc, adapter_summary),
        "temporal_holdout": {
            "protocol": (
                "two-visit subject contributes both visits; three-visit subject "
                "contributes visits 1 and 2; visit 3 is the sole holdout"
            ),
            "heldout_subject": dataset.subject_ids[holdout.heldout_subject_index],
            "heldout_visit_index_zero_based": holdout.heldout_visit_index,
            "training_visit_indices_by_subject": [
                list(indices) for indices in holdout.training_visit_indices_by_subject
            ],
            "leakage_checks": {
                "heldout_visit_excluded_from_training": True,
                "training_domain_excludes_holdout": True,
                "fixed_config_loaded_before_performance_review": True,
                "subject_specific_laplacian_used": True,
            },
        },
        "fixed_training_configuration": {
            **config_source,
            "hidden_width": config.hidden_width,
            "initial_rho": config.initial_rho,
            "learning_rate": config.learning_rate,
            "rho_learning_rate": config.rho_learning_rate,
            "weight_decay": config.weight_decay,
            "max_step": config.max_step,
            "dtype": str(config.dtype),
            "device": config.device,
            "seed": config.seed,
        },
        "training_diagnostics": {
            "epochs_run": training_result.epochs_run,
            "initial_training_loss": training_result.training_losses[0],
            "final_training_loss": training_result.training_losses[-1],
            "learned_rho": float(training_result.model.rho.detach().cpu().item()),
            "rho_units": "inverse years for the symmetric-normalized Laplacian convention",
            "numerical_instability": training_result.numerical_instability,
            "runtime_seconds": training_runtime,
            **reaction_diagnostics,
        },
        "forecast_metrics": forecasts,
        "symbolic": symbolic_summary,
        "software_versions": environment_versions(),
        "runtime_seconds": time.perf_counter() - suite_start,
    }
    artifact_path = output_dir / "pilot_experiment_summary.json"
    write_json_atomically(artifact_path, summary)
    parsed = parse_json_file(artifact_path)
    if not isinstance(parsed, dict):
        raise SchemaError("private pilot artifact did not parse as a JSON object.")
    summary["private_artifact_path"] = str(artifact_path)
    return summary


def evaluate_forecasts(
    model: ZeroAnchoredNeuralReactionDiffusion,
    holdout: PilotTemporalHoldout,
    *,
    config: NeuralTrainingConfig,
) -> dict[str, object]:
    dataset = holdout.full_dataset
    subject_index = holdout.three_visit_subject_index
    times = dataset.times_by_subject[subject_index]
    trajectory = dataset.trajectories[subject_index]
    laplacian = dataset.laplacians[subject_index]
    primary_prediction = _integrate_prediction(
        model,
        y0=trajectory[0],
        laplacian=laplacian,
        times=times,
        config=config,
    )[-1]
    secondary_prediction = _integrate_prediction(
        model,
        y0=trajectory[1],
        laplacian=laplacian,
        times=np.array([0.0, times[2] - times[1]], dtype=float),
        config=config,
    )[-1]
    persistence_primary = trajectory[0].copy()
    persistence_secondary = trajectory[1].copy()
    slope = (trajectory[1] - trajectory[0]) / (times[1] - times[0])
    linear_prediction = trajectory[1] + slope * (times[2] - times[1])
    observed = trajectory[2]
    previous = trajectory[1]

    return {
        "primary_model": forecast_metrics(primary_prediction, observed, previous),
        "secondary_model": forecast_metrics(secondary_prediction, observed, previous),
        "persistence_long_horizon": forecast_metrics(persistence_primary, observed, previous),
        "persistence_one_step": forecast_metrics(persistence_secondary, observed, previous),
        "linear_extrapolation": forecast_metrics(linear_prediction, observed, previous),
    }


def forecast_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    previous: np.ndarray,
) -> dict[str, object]:
    pred = _vector(predicted, "predicted")
    obs = _vector(observed, "observed")
    prev = _vector(previous, "previous")
    if pred.shape != obs.shape or pred.shape != prev.shape:
        raise SchemaError("forecast vectors must have matching shapes.")
    groups = {
        "all": slice(0, PILOT_N_REGIONS),
        "left_cortical": slice(0, PILOT_N_LEFT_CORTICAL),
        "right_cortical": slice(PILOT_N_LEFT_CORTICAL, PILOT_SUBCORTICAL_START),
        "subcortical": slice(PILOT_SUBCORTICAL_START, PILOT_N_REGIONS),
    }
    return {
        name: {
            **_point_metrics(pred[group], obs[group]),
            "change": _change_metrics(pred[group] - prev[group], obs[group] - prev[group]),
        }
        for name, group in groups.items()
    }


def neural_reaction_diagnostics(
    model: ZeroAnchoredNeuralReactionDiffusion,
    dataset: LongitudinalTrajectoryDataset,
    subject_indices: Sequence[int],
    *,
    config: NeuralTrainingConfig,
) -> dict[str, float]:
    domain_min, domain_max = training_state_domain(dataset, subject_indices)
    samples = sample_neural_reaction(
        model,
        domain_min=domain_min,
        domain_max=domain_max,
        n_samples=DEFAULT_PYSR_N_SAMPLES,
        device=config.device,
        dtype=config.dtype,
    )
    zero_sample = sample_neural_reaction(
        model,
        domain_min=0.0,
        domain_max=1e-12,
        n_samples=2,
        device=config.device,
        dtype=config.dtype,
    )
    return {
        "reaction_at_zero": float(zero_sample.neural_reaction_values[0]),
        "training_state_domain_min": samples.domain_min,
        "training_state_domain_max": samples.domain_max,
        "neural_reaction_min": float(np.min(samples.neural_reaction_values)),
        "neural_reaction_max": float(np.max(samples.neural_reaction_values)),
    }


def run_symbolic_distillation(
    model: ZeroAnchoredNeuralReactionDiffusion,
    dataset: LongitudinalTrajectoryDataset,
    subject_indices: Sequence[int],
    *,
    config: NeuralTrainingConfig,
    output_dir: Path,
    timeout_seconds: float,
    shutdown_grace_seconds: float,
) -> dict[str, object]:
    domain_min, domain_max = training_state_domain(dataset, subject_indices)
    samples = sample_neural_reaction(
        model,
        domain_min=domain_min,
        domain_max=domain_max,
        n_samples=DEFAULT_PYSR_N_SAMPLES,
        device=config.device,
        dtype=config.dtype,
    )
    symbolic_config = SymbolicRegressionConfig(
        niterations=DEFAULT_PYSR_NITERATIONS,
        populations=DEFAULT_PYSR_POPULATIONS,
        population_size=DEFAULT_PYSR_POPULATION_SIZE,
        maxsize=DEFAULT_PYSR_MAXSIZE,
        seed=DEFAULT_PYSR_SEED,
        model_selection="best",
        output_directory=str(output_dir / "pysr_artifacts"),
    )
    evaluation_grid = make_midpoint_evaluation_grid(domain_min, domain_max, n_points=1000)
    neural_prediction = _neural_reaction_on_grid(model, evaluation_grid, config=config)
    try:
        payload = _run_isolated_pysr_fit(
            samples=samples,
            symbolic_config=symbolic_config,
            evaluation_grid=evaluation_grid,
            neural_prediction=neural_prediction,
            timeout_seconds=timeout_seconds,
            shutdown_grace_seconds=shutdown_grace_seconds,
        )
    except PySRWorkerError as exc:
        return {
            "ran": True,
            "worker_exited_normally": False,
            "worker_cleanup_required": exc.cleanup_required,
            "worker_timed_out": exc.timed_out,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "fit_domain": [domain_min, domain_max],
        }
    interpretation = payload.get("polynomial_interpretation", {})
    if isinstance(interpretation, dict):
        interpretation = {
            key: interpretation.get(key)
            for key in (
                "is_polynomial",
                "expression_string",
                "expanded_expression_string",
                "constant",
                "linear",
                "quadratic",
                "highest_degree",
            )
        }
    return {
        "ran": True,
        "fit_domain": payload.get("fit_domain"),
        "evaluation_domain": payload.get("evaluation_domain"),
        "equation": payload.get("equation"),
        "expanded_expression": payload.get("expanded_expression"),
        "complexity": payload.get("complexity"),
        "training_loss": payload.get("training_loss"),
        "artifact_directory": payload.get("artifact_directory"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "pysr_version": payload.get("pysr_version"),
        "julia_version": payload.get("julia_version"),
        "metrics": payload.get("metrics"),
        "polynomial_interpretation": interpretation,
        "worker_exited_normally": payload.get("worker_exited_normally"),
        "worker_cleanup_required": payload.get("worker_cleanup_required"),
        "worker_timed_out_after_result": payload.get("worker_timed_out_after_result"),
        "worker_returncode": payload.get("worker_returncode"),
    }


def validate_private_output_directory(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    if _is_relative_to(output_dir, REPO_ROOT):
        artifact = output_dir / "pilot_experiment_summary.json"
        try:
            ignored = subprocess.run(  # noqa: S603, S607 - fixed argv, no shell
                ["git", "check-ignore", "-q", str(artifact)],
                cwd=REPO_ROOT,
                check=False,
            )
        except OSError as exc:
            raise SchemaError(
                "could not run 'git check-ignore' to confirm the output directory "
                f"is ignored: {exc}"
            ) from exc
        if ignored.returncode != 0:
            raise SchemaError("output directory inside the repository must be git-ignored.")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    return output_dir


def _run_isolated_pysr_fit(
    *,
    samples: ReactionSamples,
    symbolic_config: SymbolicRegressionConfig,
    evaluation_grid: np.ndarray,
    neural_prediction: np.ndarray,
    timeout_seconds: float,
    shutdown_grace_seconds: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tau_sr_pilot_pysr_worker_") as worker_dir:
        worker_root = Path(worker_dir)
        request_path = worker_root / "request.json"
        result_path = worker_root / "result.json"
        write_json_atomically(
            request_path,
            {
                "samples": {
                    "input_values": samples.input_values.tolist(),
                    "neural_reaction_values": samples.neural_reaction_values.tolist(),
                    "domain_min": samples.domain_min,
                    "domain_max": samples.domain_max,
                },
                "config": {
                    "niterations": symbolic_config.niterations,
                    "populations": symbolic_config.populations,
                    "population_size": symbolic_config.population_size,
                    "maxsize": symbolic_config.maxsize,
                    "seed": symbolic_config.seed,
                    "model_selection": symbolic_config.model_selection,
                    "output_directory": symbolic_config.output_directory,
                },
                "evaluation": {
                    "grid": np.asarray(evaluation_grid, dtype=float).tolist(),
                    "neural_prediction": np.asarray(neural_prediction, dtype=float).tolist(),
                },
            },
        )
        worker_result = run_json_worker(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_pysr_worker.py"),
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


def _safe_dataset_summary(
    qc: dict[str, object],
    adapter_summary: dict[str, object],
) -> dict[str, object]:
    return {
        "subject_count": qc["subject_count"],
        "subject_ids": qc["subject_ids"],
        "visit_counts": qc["visit_counts"],
        "relative_times_years": adapter_summary["relative_times_years"],
        "region_count": qc["region_count"],
        "adjacency_shapes": qc["adjacency_shapes"],
        "laplacian_shapes": qc["laplacian_shapes"],
        "symmetrization_valid": qc["adjacencies_symmetric"],
        "missing_tau_values": qc["missing_tau_values"],
        "tau_values_missing": qc["tau_values_missing"],
        "minimum_degree": qc["minimum_degree"],
        "roi_alignment": qc["roi_alignment"],
        "data_alignment_ready": qc["data_alignment_ready"],
        "formal_evaluation_ready": qc["formal_evaluation_ready"],
        "structural_ordering": qc["structural_ordering"],
    }


def _seeded_zero_anchored_model(
    config: NeuralTrainingConfig,
) -> ZeroAnchoredNeuralReactionDiffusion:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = ZeroAnchoredNeuralReactionDiffusion(
            hidden_width=config.hidden_width,
            initial_rho=config.initial_rho,
        )
    return model.to(device=torch.device(config.device), dtype=config.dtype)


def _integrate_prediction(
    model: ZeroAnchoredNeuralReactionDiffusion,
    *,
    y0: np.ndarray,
    laplacian: np.ndarray,
    times: np.ndarray,
    config: NeuralTrainingConfig,
) -> np.ndarray:
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            prediction = model.integrate(
                torch.as_tensor(y0.copy(), device=torch.device(config.device), dtype=config.dtype),
                torch.as_tensor(
                    laplacian.copy(),
                    device=torch.device(config.device),
                    dtype=config.dtype,
                ),
                torch.as_tensor(
                    times.copy(),
                    device=torch.device(config.device),
                    dtype=config.dtype,
                ),
                max_step=config.max_step,
            )
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
    values = prediction.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise SchemaError("forecast prediction must contain finite values.")
    return values.copy()


def _neural_reaction_on_grid(
    model: ZeroAnchoredNeuralReactionDiffusion,
    grid: np.ndarray,
    *,
    config: NeuralTrainingConfig,
) -> np.ndarray:
    values = torch.as_tensor(
        np.asarray(grid, dtype=float).reshape(-1, 1),
        device=torch.device(config.device),
        dtype=config.dtype,
    )
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            prediction = model.reaction(values)
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
    output = prediction.detach().cpu().numpy().reshape(-1)
    if output.shape != np.asarray(grid).shape or not np.isfinite(output).all():
        raise SchemaError("neural reaction evaluation produced invalid values.")
    return output.copy()


def _point_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float | None]:
    residual = predicted - observed
    mse = float(np.mean(residual**2))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "pearson_correlation": _pearson(predicted, observed),
    }


def _change_metrics(
    predicted_change: np.ndarray,
    observed_change: np.ndarray,
) -> dict[str, float | None]:
    residual = predicted_change - observed_change
    mse = float(np.mean(residual**2))
    return {
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(residual))),
        "pearson_correlation": _pearson(predicted_change, observed_change),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _vector(value: np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.shape[0] != PILOT_N_REGIONS:
        raise SchemaError(f"{field_name} must have shape ({PILOT_N_REGIONS},).")
    if not np.isfinite(array).all():
        raise SchemaError(f"{field_name} must contain finite values.")
    return array.copy()


def _dtype_from_string(value: str) -> torch.dtype:
    if value == "torch.float64":
        return torch.float64
    if value == "torch.float32":
        return torch.float32
    raise SchemaError("unsupported torch dtype in synthetic artifact.")


def _int_field(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{key} must be an integer.")
    return value


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SchemaError(f"{field_name} must be an integer.")
    output = int(value)
    if output < 0:
        raise SchemaError(f"{field_name} must be nonnegative.")
    return output


def _float_field(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{key} must be numeric.")
    output = float(value)
    if not math.isfinite(output):
        raise SchemaError(f"{key} must be finite.")
    return output


def _optional_float_field(mapping: dict[str, object], key: str) -> float | None:
    if mapping.get(key) is None:
        return None
    return _float_field(mapping, key)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
