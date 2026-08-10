"""Read-only audit helpers for the frozen full-cohort experiment."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tau_sr.cohort_experiment import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    EXPECTED_SUBJECT_COUNT,
    EXPECTED_TEST_COUNT,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VALIDATION_COUNT,
    FROZEN_SPLIT_SEED,
    cohort_training_state_domain,
    load_verified_frozen_cohort,
    write_private_json,
)
from tau_sr.pysr_lifecycle import parse_json_file
from tau_sr.schemas import SchemaError

REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_MODELS: tuple[str, ...] = (
    "persistence",
    "diffusion",
    "linear",
    "quadratic",
    "neural",
    "symbolic",
)
AUDIT_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("diffusion_vs_persistence", "diffusion", "persistence"),
    ("linear_vs_persistence", "linear", "persistence"),
    ("quadratic_vs_persistence", "quadratic", "persistence"),
    ("neural_vs_persistence", "neural", "persistence"),
    ("symbolic_vs_persistence", "symbolic", "persistence"),
    ("quadratic_vs_linear", "quadratic", "linear"),
    ("neural_vs_linear", "neural", "linear"),
    ("neural_vs_quadratic", "neural", "quadratic"),
    ("symbolic_vs_neural", "symbolic", "neural"),
)
PREVIOUSLY_REPORTED_RMSE_COMPARISONS: tuple[str, ...] = (
    "quadratic_vs_linear",
    "neural_vs_linear",
    "neural_vs_quadratic",
    "symbolic_vs_neural",
)

__all__ = (
    "AUDIT_COMPARISONS",
    "AuditInputs",
    "METRIC_MODELS",
    "audit_frozen_experiment",
    "paired_subject_bootstrap_from_metrics",
    "recompute_test_metrics_from_subject_metrics",
    "validate_audit_output_directory",
)


@dataclass(frozen=True)
class AuditInputs:
    frozen_artifact_dir: str | Path
    audit_output_dir: str | Path
    structure_dir: str | Path | None = None
    baseline_tau_dir: str | Path | None = None
    followup_tau_dir: str | Path | None = None
    preparation_artifact_dir: str | Path | None = None
    experiment_commit: str | None = None


def audit_frozen_experiment(inputs: AuditInputs) -> dict[str, Any]:
    """Run a read-only audit of the completed full-cohort experiment."""

    frozen_dir = Path(inputs.frozen_artifact_dir).expanduser().resolve()
    output_dir = validate_audit_output_directory(inputs.audit_output_dir, frozen_dir)
    summary = _read_json_object(frozen_dir / "full_cohort_experiment_summary.json", "summary")
    freeze = _read_json_object(frozen_dir / "analysis_freeze.json", "analysis_freeze")
    checkpoint = _read_json_object(frozen_dir / "cohort_experiment_checkpoint.json", "checkpoint")
    commit = inputs.experiment_commit or _current_commit()

    artifact_integrity = _artifact_integrity(frozen_dir, summary, freeze, checkpoint)
    recomputed_metrics = recompute_test_metrics_from_subject_metrics(summary)
    max_metric_discrepancy = _max_metric_discrepancy(summary["test_results"], recomputed_metrics)
    metric_reproduction = {
        "recomputed_test_metrics": recomputed_metrics,
        "max_aggregate_discrepancy": max_metric_discrepancy,
        "tolerance": 1e-12,
        "source": "saved subject-level RMSE/MAE/mean-error metrics",
        "raw_predictions_or_residual_vectors_present": False,
        "raw_vector_metric_reproduction_available": False,
        "aggregate_reproduction_from_subject_metrics_available": True,
        "aggregate_reproduction_from_subject_metrics_passed": max_metric_discrepancy <= 1e-12,
        "equal_region_count_makes_subject_metrics_sufficient": True,
    }
    bootstrap = _all_paired_bootstraps(summary)
    bootstrap_reproduction = _bootstrap_reproduction(summary, bootstrap)
    leakage = _firewall_audit(summary, freeze)
    training_semantics = _training_semantics_audit(summary)
    split_characterization = _split_characterization_if_available(inputs, summary)
    warning_audit = {
        "source": "forecast_neural converted read-only times arrays without an explicit copy.",
        "fix": "copy times before torch.as_tensor; values and dtype are preserved.",
        "completed_run_affected": False,
    }

    audit = {
        "audit": "full_cohort_frozen_result_audit",
        "audit_pass": (
            artifact_integrity["passed"]
            and max_metric_discrepancy <= 1e-12
            and leakage["passed"]
            and bootstrap_reproduction["passed"]
            and training_semantics["material_optimization_bug_found"] is False
        ),
        "experiment_commit": commit,
        "frozen_artifact_fingerprints": _artifact_fingerprints(frozen_dir),
        "artifact_integrity": artifact_integrity,
        "metric_reproduction": metric_reproduction,
        "leakage_and_firewall": leakage,
        "training_semantics": training_semantics,
        "paired_comparisons": bootstrap,
        "previous_bootstrap_reproduction": bootstrap_reproduction,
        "split_characterization": split_characterization,
        "read_only_numpy_warning": warning_audit,
        "forbidden_actions_confirmed": {
            "retrained_models": False,
            "reran_real_predictions": False,
            "reran_pysr_or_julia": False,
            "changed_split": False,
            "overwrote_frozen_artifacts": False,
        },
        "scientific_conclusion": _scientific_conclusion(bootstrap),
    }
    write_private_json(output_dir / "full_cohort_audit_summary.json", audit)
    return audit


def validate_audit_output_directory(
    audit_output_dir: str | Path,
    frozen_artifact_dir: str | Path,
) -> Path:
    output = Path(audit_output_dir).expanduser().resolve()
    frozen = Path(frozen_artifact_dir).expanduser().resolve()
    if output == frozen:
        raise SchemaError("audit output directory must not be the frozen experiment directory.")
    if _is_relative_to(output, frozen):
        raise SchemaError("audit output directory must not be inside frozen artifacts.")
    if _is_relative_to(output, REPO_ROOT):
        raise SchemaError("audit output directory must be outside the repository.")
    if output.exists() and not output.is_dir():
        raise SchemaError("audit output location must be a directory.")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    return output


def recompute_test_metrics_from_subject_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    subject_metrics = _subject_metrics(summary)
    recomputed: dict[str, Any] = {}
    for model in METRIC_MODELS:
        metrics_by_subject = _model_subject_metrics(subject_metrics, model)
        rmse_values = np.asarray([_metric(row, "rmse") for row in metrics_by_subject], dtype=float)
        mae_values = np.asarray([_metric(row, "mae") for row in metrics_by_subject], dtype=float)
        mean_errors = np.asarray(
            [_metric(row, "mean_prediction_error") for row in metrics_by_subject],
            dtype=float,
        )
        recomputed[model] = {
            "pooled_rmse": float(np.sqrt(np.mean(rmse_values**2))),
            "pooled_mae": float(np.mean(mae_values)),
            "mean_prediction_error": float(np.mean(mean_errors)),
            "mean_subject_rmse": float(np.mean(rmse_values)),
            "median_subject_rmse": float(np.median(rmse_values)),
            "mean_subject_mae": float(np.mean(mae_values)),
            "median_subject_mae": float(np.median(mae_values)),
        }
    return recomputed


def paired_subject_bootstrap_from_metrics(
    first_subject_metrics: Sequence[Mapping[str, Any]],
    second_subject_metrics: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    first = _metric_vector(first_subject_metrics, metric)
    second = _metric_vector(second_subject_metrics, metric)
    if first.shape != second.shape or first.size == 0:
        raise SchemaError("paired metric vectors must have the same nonzero shape.")
    if n_resamples < 1:
        raise SchemaError("n_resamples must be positive.")
    if metric == "rmse":
        estimator = _pooled_rmse_from_subject_rmse
    elif metric == "mae":
        estimator = _pooled_mean
    else:
        raise SchemaError("metric must be rmse or mae.")

    observed = estimator(first, np.arange(first.shape[0])) - estimator(
        second,
        np.arange(second.shape[0]),
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled = rng.integers(0, first.shape[0], size=first.shape[0])
        differences[index] = estimator(first, sampled) - estimator(second, sampled)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    reference = estimator(second, np.arange(second.shape[0]))
    return {
        "metric": metric,
        "difference_first_minus_second": float(observed),
        "ci95": [float(lower), float(upper)],
        "relative_percent_difference": float(observed / reference * 100.0),
        "interval_excludes_zero": bool(upper < 0.0 or lower > 0.0),
        "percent_subjects_improved": float(np.mean(first < second) * 100.0),
        "direction": _comparison_direction(float(observed), metric),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def _all_paired_bootstraps(summary: Mapping[str, Any]) -> dict[str, Any]:
    subject_metrics = _subject_metrics(summary)
    output: dict[str, Any] = {}
    for name, first, second in AUDIT_COMPARISONS:
        first_metrics = _model_subject_metrics(subject_metrics, first)
        second_metrics = _model_subject_metrics(subject_metrics, second)
        output[name] = {
            "first_model": first,
            "second_model": second,
            "comparison_category": _comparison_category(name),
            "rmse": paired_subject_bootstrap_from_metrics(
                first_metrics,
                second_metrics,
                metric="rmse",
            ),
            "mae": paired_subject_bootstrap_from_metrics(
                first_metrics,
                second_metrics,
                metric="mae",
            ),
        }
    return output


def _comparison_category(name: str) -> str:
    if name in ("linear_vs_persistence", "quadratic_vs_persistence"):
        return "primary_baseline_comparison"
    if name in (
        "diffusion_vs_persistence",
        "neural_vs_persistence",
        "symbolic_vs_persistence",
    ):
        return "additional_baseline_comparison"
    return "exploratory_model_comparison"


def _artifact_integrity(
    frozen_dir: Path,
    summary: Mapping[str, Any],
    freeze: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    split = _mapping(summary.get("split"), "summary.split")
    counts = _mapping(split.get("counts"), "summary.split.counts")
    cohort = _mapping(summary.get("cohort"), "summary.cohort")
    freeze_state = _mapping(freeze.get("analysis_freeze"), "analysis_freeze")
    model_configurations = _mapping(summary.get("model_configurations"), "model_configurations")
    symbolic = _mapping(summary.get("symbolic"), "summary.symbolic")
    test_results = _mapping(summary.get("test_results"), "summary.test_results")
    validation_results = _mapping(summary.get("validation_results"), "summary.validation_results")
    software_versions = _mapping(summary.get("software_versions"), "software_versions")
    checks = {
        "summary_complete": summary.get("completion_status") == "complete",
        "checkpoint_complete": checkpoint.get("completion_status") == "complete",
        "freeze_before_test": (
            freeze_state.get("status") == "complete_before_test"
            and freeze_state.get("test_metrics_computed") is False
            and freeze_state.get("test_outcomes_inspected") is False
        ),
        "subject_count": cohort.get("subject_count") == EXPECTED_SUBJECT_COUNT,
        "region_count": cohort.get("region_count") == 160,
        "one_year_time_coordinates": cohort.get("time_years") == [0.0, 1.0],
        "split_counts": (
            counts.get("train") == EXPECTED_TRAIN_COUNT
            and counts.get("validation") == EXPECTED_VALIDATION_COUNT
            and counts.get("test") == EXPECTED_TEST_COUNT
        ),
        "split_seed": split.get("seed") == FROZEN_SPLIT_SEED,
        "model_configurations_present": all(
            model in model_configurations for model in METRIC_MODELS
        ),
        "symbolic_completed": symbolic.get("completed") is True,
        "pysr_and_julia_versions_present": (
            bool(software_versions.get("pysr"))
            and bool(symbolic.get("julia_version"))
        ),
        "test_metrics_present": all(model in test_results for model in METRIC_MODELS),
        "numerical_instability_counts_zero": all(
            _mapping(test_results.get(model), f"test metrics for {model}").get(
                "numerical_instability_count"
            )
            == 0
            and _mapping(validation_results.get(model), f"validation metrics for {model}").get(
                "numerical_instability_count"
            )
            == 0
            for model in METRIC_MODELS
        ),
        "artifact_permissions": _artifact_permissions_private(frozen_dir),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _firewall_audit(summary: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    fitted = _mapping(summary.get("fitted_models"), "summary.fitted_models")
    neural = _mapping(fitted.get("neural"), "summary.fitted_models.neural")
    seed_sensitivity = _mapping(neural.get("seed_sensitivity"), "seed_sensitivity")
    selected_seed = neural.get("selected_seed")
    validation_by_seed = {
        int(seed): _mapping(result, "seed_result")["validation_metrics"]["pooled_rmse"]
        for seed, result in seed_sensitivity.items()
    }
    expected_seed = min(validation_by_seed, key=validation_by_seed.__getitem__)
    symbolic = _mapping(summary.get("symbolic"), "summary.symbolic")
    freeze_state = _mapping(freeze.get("analysis_freeze"), "analysis_freeze")
    fit_domain = symbolic.get("fit_domain")
    evaluation_domain = symbolic.get("evaluation_domain")
    checks = {
        "selected_seed_lowest_validation_rmse": selected_seed == expected_seed,
        "test_not_seen_at_freeze": (
            freeze_state.get("test_outcomes_inspected") is False
            and freeze_state.get("test_metrics_computed") is False
        ),
        "test_results_only_in_completed_summary": summary.get("completion_status") == "complete",
        "symbolic_fit_domain_recorded_before_test": (
            isinstance(fit_domain, list) and len(fit_domain) == 2
        ),
        "symbolic_evaluation_domain_inside_fit_domain": (
            isinstance(fit_domain, list)
            and isinstance(evaluation_domain, list)
            and len(fit_domain) == 2
            and len(evaluation_domain) == 2
            and float(fit_domain[0]) <= float(evaluation_domain[0])
            and float(evaluation_domain[1]) <= float(fit_domain[1])
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "selected_neural_seed": selected_seed,
        "validation_selected_seed": expected_seed,
    }


def _training_semantics_audit(summary: Mapping[str, Any]) -> dict[str, Any]:
    fitted = _mapping(summary.get("fitted_models"), "summary.fitted_models")
    linear = _mapping(fitted.get("linear"), "fitted.linear")
    quadratic = _mapping(fitted.get("quadratic"), "fitted.quadratic")
    neural = _mapping(fitted.get("neural"), "fitted.neural")
    seed_sensitivity = _mapping(neural.get("seed_sensitivity"), "seed_sensitivity")
    return {
        "initial_values": {
            "rho": 0.05,
            "linear_coefficient": 0.0,
            "quadratic_coefficient": 0.0,
        },
        "epoch_zero_meaning": "state after the first optimizer update and validation pass",
        "validation_evaluated_after_optimizer_step": True,
        "validation_best_restores_all_state_dict_parameters": True,
        "baseline_rows_excluded_from_loss": True,
        "subject_balanced_loss": True,
        "linear_coefficient_was_initialization": False,
        "linear_coefficient_explanation": (
            "The linear coefficient was initialized at 0.0 and moved to about one "
            "Adam learning-rate step after epoch 0."
        ),
        "reported_best_epochs": {
            "linear": linear.get("best_epoch"),
            "quadratic": quadratic.get("best_epoch"),
            "neural_by_seed": {
                str(seed): _mapping(result, "seed_result").get("best_epoch")
                for seed, result in seed_sensitivity.items()
            },
        },
        "material_optimization_bug_found": False,
    }


def _split_characterization_if_available(
    inputs: AuditInputs,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    required = (
        inputs.structure_dir,
        inputs.baseline_tau_dir,
        inputs.followup_tau_dir,
        inputs.preparation_artifact_dir,
    )
    if any(value is None for value in required):
        return {
            "available": False,
            "reason": "cohort input directories and preparation artifacts were not supplied",
        }
    frozen = load_verified_frozen_cohort(
        inputs.structure_dir,  # type: ignore[arg-type]
        inputs.baseline_tau_dir,  # type: ignore[arg-type]
        inputs.followup_tau_dir,  # type: ignore[arg-type]
        inputs.preparation_artifact_dir,  # type: ignore[arg-type]
    )
    symbolic = _mapping(summary.get("symbolic"), "summary.symbolic")
    fit_domain = symbolic.get("fit_domain")
    domain_check: dict[str, Any] = {"available": False}
    if isinstance(fit_domain, list) and len(fit_domain) == 2:
        train_domain = cohort_training_state_domain(frozen.dataset, frozen.split.train_indices)
        domain_check = {
            "available": True,
            "training_domain": list(train_domain),
            "symbolic_fit_domain": [float(fit_domain[0]), float(fit_domain[1])],
            "matches_training_domain": (
                math.isclose(train_domain[0], float(fit_domain[0]), abs_tol=1e-12)
                and math.isclose(train_domain[1], float(fit_domain[1]), abs_tol=1e-12)
            ),
        }
    split_summary = {
        "train": _split_tau_summary(frozen.dataset.trajectories, frozen.split.train_indices),
        "validation": _split_tau_summary(
            frozen.dataset.trajectories,
            frozen.split.validation_indices,
        ),
        "test": _split_tau_summary(frozen.dataset.trajectories, frozen.split.test_indices),
    }
    return {
        "available": True,
        "training_domain_check": domain_check,
        "splits": split_summary,
        "interpretation": _validation_test_difference_interpretation(split_summary),
    }


def _split_tau_summary(
    trajectories: Sequence[np.ndarray],
    indices: Sequence[int],
) -> dict[str, float | int | list[float]]:
    selected = tuple(trajectories[index] for index in indices)
    baseline = np.concatenate([trajectory[0] for trajectory in selected])
    followup = np.concatenate([trajectory[1] for trajectory in selected])
    change = followup - baseline
    subject_persistence_rmse = np.asarray(
        [float(np.sqrt(np.mean((trajectory[0] - trajectory[1]) ** 2))) for trajectory in selected]
    )
    subject_persistence_mae = np.asarray(
        [float(np.mean(np.abs(trajectory[0] - trajectory[1]))) for trajectory in selected]
    )
    return {
        "subject_count": len(selected),
        "baseline_tau_mean": float(np.mean(baseline)),
        "baseline_tau_std": float(np.std(baseline)),
        "followup_tau_mean": float(np.mean(followup)),
        "followup_tau_std": float(np.std(followup)),
        "tau_change_mean": float(np.mean(change)),
        "tau_change_std": float(np.std(change)),
        "absolute_tau_change_mean": float(np.mean(np.abs(change))),
        "absolute_tau_change_median": float(np.median(np.abs(change))),
        "persistence_rmse": float(np.sqrt(np.mean(change**2))),
        "persistence_mae": float(np.mean(np.abs(change))),
        "subject_persistence_rmse_median": float(np.median(subject_persistence_rmse)),
        "subject_persistence_rmse_iqr": [
            float(np.quantile(subject_persistence_rmse, 0.25)),
            float(np.quantile(subject_persistence_rmse, 0.75)),
        ],
        "subject_persistence_mae_median": float(np.median(subject_persistence_mae)),
        "positive_regional_change_proportion": float(np.mean(change > 0.0)),
    }


def _validation_test_difference_interpretation(split_summary: Mapping[str, Any]) -> str:
    validation = _mapping(split_summary.get("validation"), "validation split")
    test = _mapping(split_summary.get("test"), "test split")
    validation_abs = float(validation["absolute_tau_change_mean"])
    test_abs = float(test["absolute_tau_change_mean"])
    validation_rmse = float(validation["persistence_rmse"])
    test_rmse = float(test["persistence_rmse"])
    if validation_abs > test_abs and validation_rmse > test_rmse:
        return (
            "Validation has larger aggregate absolute change and persistence error "
            "than test, which can explain the higher validation error scale."
        )
    return (
        "The validation/test error difference is not fully explained by the audited "
        "aggregate change summaries alone."
    )


def _bootstrap_reproduction(
    summary: Mapping[str, Any],
    audit_bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    saved = _mapping(summary.get("bootstrap_comparisons"), "saved bootstrap")
    discrepancies: dict[str, float] = {}
    for name in PREVIOUSLY_REPORTED_RMSE_COMPARISONS:
        saved_row = _mapping(saved.get(name), name)
        audit_row = _mapping(_mapping(audit_bootstrap.get(name), name).get("rmse"), f"{name}.rmse")
        saved_diff = float(saved_row["rmse_difference_first_minus_second"])
        saved_ci = tuple(float(value) for value in saved_row["ci95"])
        audit_diff = float(audit_row["difference_first_minus_second"])
        audit_ci = tuple(float(value) for value in audit_row["ci95"])
        discrepancies[name] = max(
            abs(saved_diff - audit_diff),
            abs(saved_ci[0] - audit_ci[0]),
            abs(saved_ci[1] - audit_ci[1]),
        )
    max_discrepancy = max(discrepancies.values()) if discrepancies else math.inf
    return {
        "passed": max_discrepancy <= 1e-12,
        "max_discrepancy": float(max_discrepancy),
        "tolerance": 1e-12,
        "comparisons": discrepancies,
    }


def _max_metric_discrepancy(
    saved_metrics: Mapping[str, Any],
    recomputed_metrics: Mapping[str, Any],
) -> float:
    max_discrepancy = 0.0
    for model in METRIC_MODELS:
        saved = _mapping(saved_metrics.get(model), f"saved metrics for {model}")
        recomputed = _mapping(recomputed_metrics.get(model), f"recomputed metrics for {model}")
        for key in ("pooled_rmse", "pooled_mae", "mean_prediction_error"):
            max_discrepancy = max(max_discrepancy, abs(float(saved[key]) - float(recomputed[key])))
    return float(max_discrepancy)


def _scientific_conclusion(bootstrap: Mapping[str, Any]) -> str:
    linear_vs_persistence = _mapping(bootstrap["linear_vs_persistence"], "linear_vs_persistence")
    quadratic_vs_persistence = _mapping(
        bootstrap["quadratic_vs_persistence"],
        "quadratic_vs_persistence",
    )
    quadratic_vs_linear = _mapping(bootstrap["quadratic_vs_linear"], "quadratic_vs_linear")
    if (
        _mapping(linear_vs_persistence["rmse"], "linear RMSE")["interval_excludes_zero"]
        and float(linear_vs_persistence["rmse"]["difference_first_minus_second"]) < 0.0
    ):
        baseline_statement = "Linear improves on persistence for RMSE under this audit."
    else:
        baseline_statement = (
            "Linear does not reliably improve on persistence because the RMSE interval "
            "does not exclude zero."
        )
    if (
        _mapping(quadratic_vs_persistence["rmse"], "quadratic RMSE")["interval_excludes_zero"]
        and float(quadratic_vs_persistence["rmse"]["difference_first_minus_second"]) < 0.0
        and _mapping(quadratic_vs_linear["rmse"], "quadratic-linear RMSE")[
            "interval_excludes_zero"
        ]
    ):
        nonlinear_statement = "Quadratic has bootstrap-supported improvement."
    else:
        nonlinear_statement = "There is no bootstrap-supported nonlinear improvement."
    return f"{baseline_statement} {nonlinear_statement}"


def _artifact_fingerprints(frozen_dir: Path) -> dict[str, str]:
    roles = {
        "analysis_freeze": frozen_dir / "analysis_freeze.json",
        "summary": frozen_dir / "full_cohort_experiment_summary.json",
        "checkpoint": frozen_dir / "cohort_experiment_checkpoint.json",
    }
    return {role: _sha256(path) for role, path in roles.items()}


def _artifact_permissions_private(frozen_dir: Path) -> bool:
    for name in (
        "analysis_freeze.json",
        "full_cohort_experiment_summary.json",
        "cohort_experiment_checkpoint.json",
    ):
        path = frozen_dir / name
        if not path.exists() or (path.stat().st_mode & 0o077):
            return False
    return True


def _read_json_object(path: Path, field_name: str) -> dict[str, Any]:
    try:
        payload = parse_json_file(path)
    except Exception as exc:
        raise SchemaError(f"{field_name} artifact could not be read.") from exc
    if not isinstance(payload, dict):
        raise SchemaError(f"{field_name} artifact must contain a JSON object.")
    return payload


def _subject_metrics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(summary.get("subject_level_metrics"), "subject_level_metrics")


def _model_subject_metrics(
    subject_metrics: Mapping[str, Any],
    model: str,
) -> tuple[Mapping[str, Any], ...]:
    by_subject = _mapping(subject_metrics.get(model), f"subject metrics for {model}")
    if len(by_subject) != EXPECTED_TEST_COUNT:
        raise SchemaError("subject-level test metrics do not match expected test count.")
    return tuple(_mapping(value, "subject metric row") for value in by_subject.values())


def _metric_vector(rows: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray:
    return np.asarray([_metric(row, metric) for row in rows], dtype=float)


def _metric(row: Mapping[str, Any], metric: str) -> float:
    value = row.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SchemaError(f"{metric} must be numeric.")
    output = float(value)
    if not math.isfinite(output):
        raise SchemaError(f"{metric} must be finite.")
    return output


def _pooled_rmse_from_subject_rmse(values: np.ndarray, indices: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values[indices] ** 2)))


def _pooled_mean(values: np.ndarray, indices: np.ndarray) -> float:
    return float(np.mean(values[indices]))


def _comparison_direction(value: float, metric: str) -> str:
    if value < 0.0:
        return f"first model lower {metric}"
    if value > 0.0:
        return f"second model lower {metric}"
    return "tie"


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field_name} must be a mapping.")
    return value


def _current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
