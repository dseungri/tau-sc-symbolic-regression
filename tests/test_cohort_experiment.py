from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tau_sr.cohort_experiment import (
    CohortExperimentConfig,
    ParametricFitResult,
    bootstrap_paired_rmse,
    cohort_training_state_domain,
    evaluate_frozen_test_stage,
    evaluate_model_forecasts,
    fit_parametric_model,
    forecast_parametric,
    forecast_persistence,
    forecast_symbolic,
    load_completed_checkpoint,
    select_neural_result,
    symbolic_reaction_from_expression,
    write_private_json,
)
from tau_sr.identifiability import ZeroAnchoredNeuralReactionDiffusion
from tau_sr.pilot_adapter import LongitudinalTrajectoryDataset
from tau_sr.schemas import SchemaError
from tau_sr.training import SubjectSplit


def _dataset(n_subjects: int = 6, n_regions: int = 4) -> LongitudinalTrajectoryDataset:
    subject_ids = tuple(f"Cohort Subject {index:03d}" for index in range(1, n_subjects + 1))
    roi_labels = tuple(f"Region {index:03d}" for index in range(1, n_regions + 1))
    adjacency = np.zeros((n_subjects, n_regions, n_regions), dtype=float)
    laplacian = np.zeros((n_subjects, n_regions, n_regions), dtype=float)
    trajectories = []
    for subject_index in range(n_subjects):
        baseline = np.linspace(0.1, 0.4, n_regions) + 0.01 * subject_index
        followup = baseline * np.exp(0.1)
        trajectories.append(np.stack([baseline, followup]))
    return LongitudinalTrajectoryDataset(
        subject_ids=subject_ids,
        roi_labels=roi_labels,
        adjacencies=adjacency,
        laplacians=laplacian,
        times_by_subject=tuple(np.asarray([0.0, 1.0], dtype=float) for _ in subject_ids),
        trajectories=tuple(trajectories),
    )


def _split() -> SubjectSplit:
    return SubjectSplit(
        train_indices=(0, 1, 2),
        validation_indices=(3,),
        test_indices=(4, 5),
    )


def test_persistence_returns_baseline_copy() -> None:
    dataset = _dataset()

    forecasts = forecast_persistence(dataset, (0, 1))

    np.testing.assert_allclose(forecasts["Cohort Subject 001"], dataset.trajectories[0][0])
    forecasts["Cohort Subject 001"][0] = 99.0
    assert dataset.trajectories[0][0, 0] != 99.0


def test_parametric_forecast_uses_rk4_for_linear_reaction() -> None:
    dataset = _dataset(n_subjects=1, n_regions=3)

    forecasts = forecast_parametric(
        dataset,
        (0,),
        rho=0.0,
        linear_coefficient=0.2,
        quadratic_coefficient=0.0,
        max_step=0.01,
    )

    expected = dataset.trajectories[0][0] * np.exp(0.2)
    np.testing.assert_allclose(forecasts["Cohort Subject 001"], expected, rtol=1e-9, atol=1e-9)


def test_fit_diffusion_model_keeps_rho_nonnegative_and_excludes_baseline() -> None:
    dataset = _dataset()
    config = CohortExperimentConfig(max_epochs=3, patience=2, neural_seeds=(41,))

    result = fit_parametric_model(dataset, _split(), kind="diffusion", config=config)

    assert result.rho >= 0.0
    assert result.epochs_run >= 1
    assert result.numerical_instability is False


def test_evaluation_metrics_exclude_baseline_rows() -> None:
    dataset = _dataset(n_subjects=1, n_regions=3)
    forecast = {"Cohort Subject 001": dataset.trajectories[0][1].copy()}

    metrics = evaluate_model_forecasts(dataset, (0,), forecast)

    assert metrics["pooled_rmse"] == pytest.approx(0.0)
    assert metrics["pooled_mae"] == pytest.approx(0.0)


def test_select_neural_seed_uses_validation_metric_only() -> None:
    selected = select_neural_result(
        {
            41: {"validation_metrics": {"pooled_rmse": 0.4}},
            42: {"validation_metrics": {"pooled_rmse": 0.2}},
            43: {"validation_metrics": {"pooled_rmse": 0.3}},
        }
    )

    assert selected == 42


def test_test_stage_requires_analysis_freeze() -> None:
    dataset = _dataset()
    split = _split()
    model = ZeroAnchoredNeuralReactionDiffusion(hidden_width=2, initial_rho=0.05).double()
    fitted = ParametricFitResult(
        model_name="diffusion",
        rho=0.01,
        linear_coefficient=0.0,
        quadratic_coefficient=0.0,
        best_epoch=0,
        epochs_run=1,
        training_losses=(1.0,),
        validation_losses=(1.0,),
        numerical_instability=False,
        runtime_seconds=0.0,
    )

    with pytest.raises(SchemaError, match="analysis freeze"):
        evaluate_frozen_test_stage(
            dataset,
            split,
            {"diffusion": fitted, "linear": fitted, "quadratic": fitted},
            model,
            lambda y: np.zeros_like(y),
            config=CohortExperimentConfig(max_epochs=1, patience=1, neural_seeds=(41,)),
            analysis_freeze_written=False,
        )


def test_bootstrap_is_deterministic_subject_paired_and_does_not_mutate_global_rng() -> None:
    first = (np.asarray([0.0, 1.0]), np.asarray([0.0, 2.0]))
    second = (np.asarray([0.0, 2.0]), np.asarray([0.0, 3.0]))
    np.random.seed(123)
    before = np.random.get_state()

    first_result = bootstrap_paired_rmse(first, second, n_resamples=200, seed=42)
    second_result = bootstrap_paired_rmse(first, second, n_resamples=200, seed=42)
    after = np.random.get_state()

    assert first_result == second_result
    assert first_result["rmse_difference_first_minus_second"] < 0.0
    assert all(np.array_equal(left, right) for left, right in zip(before, after, strict=True))


def test_training_state_domain_uses_training_subjects_only() -> None:
    dataset = _dataset(n_subjects=4, n_regions=2)
    modified = list(dataset.trajectories)
    modified[3] = np.full((2, 2), 99.0)
    dataset = LongitudinalTrajectoryDataset(
        subject_ids=dataset.subject_ids,
        roi_labels=dataset.roi_labels,
        adjacencies=dataset.adjacencies,
        laplacians=dataset.laplacians,
        times_by_subject=dataset.times_by_subject,
        trajectories=tuple(modified),
    )

    domain = cohort_training_state_domain(dataset, (0, 1))

    assert domain[1] < 99.0


def test_symbolic_expression_is_frozen_before_symbolic_forecast() -> None:
    dataset = _dataset(n_subjects=1, n_regions=3)
    reaction = symbolic_reaction_from_expression({"equation": "0.1 * y"})

    forecasts = forecast_symbolic(
        dataset,
        (0,),
        rho=0.0,
        reaction_fn=reaction,
        max_step=0.01,
    )

    expected = dataset.trajectories[0][0] * np.exp(0.1)
    np.testing.assert_allclose(forecasts["Cohort Subject 001"], expected, rtol=1e-9, atol=1e-9)


def test_private_atomic_artifacts_and_checkpoint_resume(tmp_path: Path) -> None:
    path = tmp_path / "private.json"
    payload = {"completion_status": "complete", "value": 1}

    write_private_json(path, payload)

    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert load_completed_checkpoint(path) == payload

    incomplete = tmp_path / "incomplete.json"
    write_private_json(incomplete, {"completion_status": "incomplete"})
    with pytest.raises(SchemaError, match="not a completed"):
        load_completed_checkpoint(incomplete)
