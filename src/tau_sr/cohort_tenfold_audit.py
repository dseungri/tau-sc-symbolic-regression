"""Read-only audit for the full-cohort nested tenfold analysis."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tau_sr.cohort_tenfold import (
    TENFOLD_CHECKPOINT_NAME,
    TENFOLD_FREEZE_NAME,
    TENFOLD_MODELS,
    TENFOLD_SUMMARY_NAME,
    TenfoldConfig,
    all_bootstrap_comparisons,
    all_model_bootstrap_intervals,
    assemble_out_of_fold_arrays,
    build_tenfold_outer_folds,
    overall_prediction_metrics,
    validate_private_tenfold_output_directory,
)
from tau_sr.pilot_adapter import LongitudinalTrajectoryDataset
from tau_sr.pysr_lifecycle import parse_json_file
from tau_sr.schemas import SchemaError

REPO_ROOT = Path(__file__).resolve().parents[2]
TENFOLD_AUDIT_NAME = "cohort_tenfold_audit_summary.json"

__all__ = (
    "TENFOLD_AUDIT_NAME",
    "audit_tenfold_experiment",
    "recompute_tenfold_summary_from_artifacts",
)


def audit_tenfold_experiment(output_dir: str | Path) -> dict[str, Any]:
    """Audit a completed tenfold experiment without fitting or model selection."""

    private_dir = validate_private_tenfold_output_directory(output_dir)
    freeze = _read_json_object(private_dir / TENFOLD_FREEZE_NAME, "tenfold freeze")
    summary = _read_json_object(private_dir / TENFOLD_SUMMARY_NAME, "tenfold summary")
    checkpoint = _read_json_object(private_dir / TENFOLD_CHECKPOINT_NAME, "tenfold checkpoint")
    dataset = _dataset_contract_from_artifacts(summary, private_dir)
    config = _config_from_summary(summary)
    recomputed = recompute_tenfold_summary_from_artifacts(
        dataset,
        summary,
        private_dir,
        config,
    )
    metric_discrepancy = _max_metric_discrepancy(
        _mapping(summary.get("overall_out_of_fold_metrics"), "summary metrics"),
        recomputed["overall_out_of_fold_metrics"],
    )
    bootstrap_discrepancy = _max_bootstrap_discrepancy(
        _mapping(summary.get("paired_bootstrap_comparisons"), "summary bootstrap"),
        recomputed["paired_bootstrap_comparisons"],
    )
    interval_discrepancy = _max_bootstrap_discrepancy(
        _mapping(summary.get("model_bootstrap_intervals"), "summary model intervals"),
        recomputed["model_bootstrap_intervals"],
        paired=False,
    )
    identity_checks = _metric_identity_checks(
        recomputed["overall_out_of_fold_metrics"],
    )
    artifact_checks = _artifact_checks(summary, private_dir)
    privacy = _privacy_scan(summary, freeze, checkpoint)
    fold_checks = _fold_checks(summary, freeze, checkpoint)
    audit_pass = (
        summary.get("completion_status") == "complete"
        and checkpoint.get("completion_status") == "complete"
        and freeze.get("completion_status") == "frozen_before_execution"
        and metric_discrepancy <= 1e-12
        and bootstrap_discrepancy <= 1e-12
        and interval_discrepancy <= 1e-12
        and identity_checks["passed"]
        and artifact_checks["passed"]
        and privacy["passed"]
        and fold_checks["passed"]
    )
    audit = {
        "audit": "full_cohort_nested_tenfold_audit",
        "audit_pass": audit_pass,
        "repository_commit": _current_commit(),
        "metric_reproduction": {
            "max_aggregate_metric_discrepancy": metric_discrepancy,
            "max_bootstrap_discrepancy": bootstrap_discrepancy,
            "max_model_interval_discrepancy": interval_discrepancy,
            "tolerance": 1e-12,
        },
        "metric_identity_checks": identity_checks,
        "fold_checks": fold_checks,
        "artifact_checks": artifact_checks,
        "privacy_scan": privacy,
        "recomputed_overall_out_of_fold_metrics": recomputed["overall_out_of_fold_metrics"],
        "recomputed_model_bootstrap_intervals": recomputed["model_bootstrap_intervals"],
        "recomputed_paired_bootstrap_comparisons": recomputed["paired_bootstrap_comparisons"],
        "forbidden_actions_confirmed": {
            "trained_models": False,
            "ran_pysr_or_julia": False,
            "opened_raw_cohort_files": False,
            "performed_model_selection": False,
        },
    }
    from tau_sr.cohort_experiment import write_private_json

    write_private_json(private_dir / TENFOLD_AUDIT_NAME, audit)
    return audit


def recompute_tenfold_summary_from_artifacts(
    dataset: LongitudinalTrajectoryDataset,
    summary: Mapping[str, Any],
    private_dir: Path,
    config: TenfoldConfig,
) -> dict[str, Any]:
    fold_design = _mapping(summary.get("fold_design"), "summary.fold_design")
    folds = build_tenfold_outer_folds(
        int(_mapping(summary.get("cohort"), "summary.cohort")["subject_count"]),
        seed=int(fold_design["master_seed"]),
    )
    fold_records = _mapping(summary.get("folds"), "summary.folds")
    predictions, residuals, targets, artifact_count = assemble_out_of_fold_arrays(
        dataset,
        folds,
        fold_records,
        private_dir,
    )
    del predictions
    return {
        "overall_out_of_fold_metrics": {
            model: overall_prediction_metrics(residuals[model], targets[model])
            for model in TENFOLD_MODELS
        },
        "paired_bootstrap_comparisons": all_bootstrap_comparisons(
            residuals,
            n_resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed,
        ),
        "model_bootstrap_intervals": all_model_bootstrap_intervals(
            residuals,
            n_resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed,
        ),
        "prediction_artifact_count": artifact_count,
    }


def _dataset_contract_from_artifacts(
    summary: Mapping[str, Any],
    private_dir: Path,
) -> LongitudinalTrajectoryDataset:
    cohort = _mapping(summary.get("cohort"), "summary.cohort")
    n_subjects = int(cohort["subject_count"])
    n_regions = int(cohort["region_count"])
    subject_ids = tuple(f"Audit Subject {index:03d}" for index in range(n_subjects))
    roi_labels = tuple(f"Region {index:03d}" for index in range(n_regions))
    trajectories = [np.zeros((2, n_regions), dtype=float) for _ in range(n_subjects)]
    first_model = TENFOLD_MODELS[0]
    fold_records = _mapping(summary.get("folds"), "summary.folds")
    for record in fold_records.values():
        models = _mapping(_mapping(record, "fold record").get("models"), "fold models")
        payload = _mapping(models[first_model], "first model payload")
        relative = _relative_artifact_path(payload["prediction_artifact"])
        with np.load(private_dir / relative) as arrays:
            indices = np.asarray(arrays["outer_eval_indices"], dtype=int)
            targets = np.asarray(arrays["target"], dtype=float)
            for row, index in enumerate(indices):
                trajectories[int(index)][1] = targets[row]
    return LongitudinalTrajectoryDataset(
        subject_ids=subject_ids,
        roi_labels=roi_labels,
        adjacencies=np.zeros((n_subjects, n_regions, n_regions), dtype=float),
        laplacians=np.zeros((n_subjects, n_regions, n_regions), dtype=float),
        times_by_subject=tuple(np.asarray([0.0, 1.0], dtype=float) for _ in subject_ids),
        trajectories=tuple(trajectories),
    )


def _config_from_summary(summary: Mapping[str, Any]) -> TenfoldConfig:
    payload = _mapping(summary.get("configuration"), "summary.configuration")
    model = _mapping(payload.get("model_config"), "summary.configuration.model_config")
    from tau_sr.cohort_experiment import CohortExperimentConfig

    model_config = CohortExperimentConfig(
        hidden_width=int(model["hidden_width"]),
        initial_rho=float(model["initial_rho"]),
        learning_rate=float(model["learning_rate"]),
        rho_learning_rate=None
        if model.get("rho_learning_rate") is None
        else float(model["rho_learning_rate"]),
        parametric_learning_rate=float(model["parametric_learning_rate"]),
        max_epochs=int(model["max_epochs"]),
        patience=int(model["patience"]),
        min_delta=float(model["min_delta"]),
        max_step=float(model["max_step"]),
        neural_seeds=tuple(int(seed) for seed in model["neural_seeds"]),
        parametric_seed=int(model["parametric_seed"]),
        bootstrap_resamples=int(payload["bootstrap_resamples"]),
        bootstrap_seed=int(payload["bootstrap_seed"]),
        pysr_seed=int(model["pysr_seed"]),
        pysr_n_samples=int(model["pysr_n_samples"]),
        pysr_niterations=int(model["pysr_niterations"]),
        pysr_populations=int(model["pysr_populations"]),
        pysr_population_size=int(model["pysr_population_size"]),
        pysr_maxsize=int(model["pysr_maxsize"]),
    )
    return TenfoldConfig(
        expected_subject_count=int(payload["expected_subject_count"]),
        master_seed=int(payload["master_seed"]),
        inner_validation_fraction=float(payload["inner_validation_fraction"]),
        ridge_alpha_grid=tuple(float(alpha) for alpha in payload["ridge_alpha_grid"]),
        model_config=model_config,
        bootstrap_resamples=int(payload["bootstrap_resamples"]),
        bootstrap_seed=int(payload["bootstrap_seed"]),
    )


def _max_metric_discrepancy(
    saved: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> float:
    maximum = 0.0
    for model in TENFOLD_MODELS:
        saved_metrics = _mapping(saved.get(model), f"saved {model}")
        recomputed_metrics = _mapping(recomputed.get(model), f"recomputed {model}")
        for key, value in recomputed_metrics.items():
            if isinstance(value, (int, float)):
                maximum = max(maximum, abs(float(saved_metrics[key]) - float(value)))
            elif value is None and saved_metrics.get(key) is not None:
                return math.inf
    return maximum


def _max_bootstrap_discrepancy(
    saved: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    paired: bool = True,
) -> float:
    maximum = 0.0
    for comparison, recomputed_payload in recomputed.items():
        saved_payload = _mapping(saved.get(comparison), f"saved {comparison}")
        for metric in ("rmse", "mae"):
            saved_metric = _mapping(saved_payload.get(metric), f"saved {comparison}.{metric}")
            recomputed_metric = _mapping(
                _mapping(recomputed_payload, f"recomputed {comparison}").get(metric),
                f"recomputed {comparison}.{metric}",
            )
            if paired:
                maximum = max(
                    maximum,
                    abs(
                        float(saved_metric["difference_first_minus_second"])
                        - float(recomputed_metric["difference_first_minus_second"])
                    ),
                )
            else:
                maximum = max(
                    maximum,
                    abs(float(saved_metric["estimate"]) - float(recomputed_metric["estimate"])),
                )
            for left, right in zip(saved_metric["ci95"], recomputed_metric["ci95"], strict=True):
                maximum = max(maximum, abs(float(left) - float(right)))
    return maximum


def _metric_identity_checks(metrics: Mapping[str, Any]) -> dict[str, Any]:
    discrepancies: dict[str, float] = {}
    for model, payload in metrics.items():
        row = _mapping(payload, f"{model} metrics")
        mse = float(row["mse"])
        variance = float(row["residual_variance"])
        bias = float(row["bias"])
        discrepancies[str(model)] = abs(mse - (variance + bias**2))
    maximum = max(discrepancies.values(), default=0.0)
    return {
        "passed": maximum <= 1e-12,
        "max_mse_variance_bias_discrepancy": maximum,
        "per_model_discrepancy": discrepancies,
    }


def _fold_checks(
    summary: Mapping[str, Any],
    freeze: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    fold_design = _mapping(summary.get("fold_design"), "summary.fold_design")
    sizes = tuple(int(size) for size in fold_design.get("outer_fold_sizes", ()))
    records = _mapping(summary.get("folds"), "summary.folds")
    ten_complete_folds = len(records) == 10 and all(
        _mapping(record, "fold").get("completion_status") == "complete"
        for record in records.values()
    )
    freeze_fold_design = _mapping(freeze.get("fold_design"), "freeze.fold_design")
    freeze_size_distribution = _mapping(
        freeze_fold_design.get("outer_fold_size_distribution"),
        "freeze fold sizes",
    )
    checkpoint_folds = _mapping(checkpoint.get("folds"), "checkpoint.folds")
    checks = {
        "ten_complete_folds": ten_complete_folds,
        "fold_sizes_are_16_or_17_for_real_protocol": (
            set(sizes) <= {16, 17}
            if int(_mapping(summary.get("cohort"), "summary.cohort")["subject_count"]) == 163
            else True
        ),
        "fold_size_distribution_matches_freeze": fold_design.get("outer_fold_size_distribution")
        == freeze_size_distribution,
        "checkpoint_matches_summary_folds": set(checkpoint_folds) == set(records),
        "no_outer_counts_missing": all(
            int(_mapping(record, "fold").get("outer_evaluation_count", 0)) > 0
            for record in records.values()
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _artifact_checks(summary: Mapping[str, Any], private_dir: Path) -> dict[str, Any]:
    paths: list[Path] = []
    records = _mapping(summary.get("folds"), "summary.folds")
    for record in records.values():
        models = _mapping(_mapping(record, "fold").get("models"), "fold models")
        for payload in models.values():
            model_payload = _mapping(payload, "model payload")
            relative = _relative_artifact_path(model_payload["prediction_artifact"])
            paths.append(private_dir / relative)
    missing = [str(path.name) for path in paths if not path.exists()]
    loose = [str(path.name) for path in paths if path.exists() and (path.stat().st_mode & 0o077)]
    return {
        "passed": not missing and not loose and len(paths) == 70,
        "artifact_count": len(paths),
        "missing_count": len(missing),
        "loose_permission_count": len(loose),
    }


def _privacy_scan(*payloads: Mapping[str, Any]) -> dict[str, Any]:
    text = json.dumps(payloads, sort_keys=True)
    forbidden = (
        "/Users/",
        "Private" + "Research",
        "Down" + "loads",
        "002" + "_S_",
        ".c" + "sv",
        ".xl" + "sx",
    )
    hits = [value for value in forbidden if value in text]
    return {"passed": not hits, "forbidden_token_hits": hits}


def _relative_artifact_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("artifact path must be a relative string.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise SchemaError("artifact path must be relative.")
    return path


def _read_json_object(path: Path, field_name: str) -> dict[str, Any]:
    try:
        payload = parse_json_file(path)
    except Exception as exc:
        raise SchemaError(f"{field_name} could not be read.") from exc
    if not isinstance(payload, dict):
        raise SchemaError(f"{field_name} must contain a JSON object.")
    return payload


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
