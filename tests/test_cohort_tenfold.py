from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tau_sr.cohort_experiment import CohortExperimentConfig, ParametricFitResult
from tau_sr.cohort_tenfold import (
    TENFOLD_MODELS,
    TenfoldCohort,
    TenfoldConfig,
    _run_parametric_fold,
    _run_persistence_fold,
    aggregate_tenfold_results,
    build_tenfold_outer_folds,
    execute_tenfold_analysis,
    freeze_tenfold_protocol,
    inner_train_validation_split,
    make_tenfold_fold_plan,
    overall_prediction_metrics,
    paired_subject_bootstrap,
    refit_updates_from_best_epoch,
)
from tau_sr.pilot_adapter import LongitudinalTrajectoryDataset
from tau_sr.schemas import SchemaError
from tau_sr.training import SubjectSplit


def _dataset(n_subjects: int = 20, n_regions: int = 3) -> LongitudinalTrajectoryDataset:
    subject_ids = tuple(f"Synthetic Subject {index:03d}" for index in range(n_subjects))
    roi_labels = tuple(f"Region {index:03d}" for index in range(n_regions))
    adjacencies = np.zeros((n_subjects, n_regions, n_regions), dtype=float)
    laplacians = np.zeros_like(adjacencies)
    trajectories = []
    for subject_index in range(n_subjects):
        baseline = np.linspace(0.1, 0.3, n_regions) + 0.001 * subject_index
        followup = baseline + 0.02 * baseline + 0.0005 * subject_index
        trajectories.append(np.stack([baseline, followup]))
    return LongitudinalTrajectoryDataset(
        subject_ids=subject_ids,
        roi_labels=roi_labels,
        adjacencies=adjacencies,
        laplacians=laplacians,
        times_by_subject=tuple(np.asarray([0.0, 1.0], dtype=float) for _ in subject_ids),
        trajectories=tuple(trajectories),
    )


def _cohort(n_subjects: int = 20, n_regions: int = 3) -> TenfoldCohort:
    return TenfoldCohort(
        dataset=_dataset(n_subjects=n_subjects, n_regions=n_regions),
        qc_summary={
            "ready": True,
            "subject_count": n_subjects,
            "matched_triplets": n_subjects,
            "region_count": n_regions,
            "time_years": [0.0, 1.0],
        },
        cohort_fingerprint=f"synthetic-{n_subjects}-{n_regions}",
        source_file_match_count=n_subjects,
    )


def _fast_config(n_subjects: int = 20) -> TenfoldConfig:
    return TenfoldConfig(
        expected_subject_count=n_subjects,
        model_config=CohortExperimentConfig(
            hidden_width=2,
            max_epochs=1,
            patience=1,
            neural_seeds=(41,),
            pysr_niterations=1,
            pysr_populations=1,
            pysr_population_size=5,
        ),
        bootstrap_resamples=20,
    )


def _fake_symbolic(model, dataset, train_indices, config, output_dir, timeout, shutdown):
    del model, config, output_dir, timeout, shutdown
    values = np.concatenate([dataset.trajectories[index].ravel() for index in train_indices])
    return {
        "completed": True,
        "fit_domain": [float(values.min()), float(values.max())],
        "evaluation_domain": [float(values.min()), float(values.max())],
        "equation": "0.0 * y",
        "expanded_expression": "0",
        "complexity": 1,
        "training_loss": 0.0,
        "metrics": {"symbolic_vs_neural_rmse": 0.0, "symbolic_vs_neural_mae": 0.0},
        "polynomial_interpretation": {"coefficients": {"0": 0.0}},
    }


def test_tenfold_outer_folds_are_deterministic_for_163_subjects() -> None:
    first = build_tenfold_outer_folds(163, seed=42)
    second = build_tenfold_outer_folds(163, seed=42)

    assert first == second
    assert sorted(len(fold) for fold in first) == [16] * 7 + [17] * 3
    combined = [index for fold in first for index in fold]
    assert sorted(combined) == list(range(163))
    assert len(combined) == len(set(combined))


def test_inner_split_excludes_outer_fold_and_is_shared_by_model_plan() -> None:
    config = _fast_config()
    folds = build_tenfold_outer_folds(20, seed=42)
    plan = make_tenfold_fold_plan(folds, 0, config=config)
    train, validation, inner_seed = inner_train_validation_split(
        plan.outer_development_indices,
        plan.outer_evaluation_indices,
        seed=42,
        fold_index=0,
    )

    assert inner_seed == plan.inner_seed == 1051
    assert train == plan.inner_training_indices
    assert validation == plan.inner_validation_indices
    assert not (set(train) & set(plan.outer_evaluation_indices))
    assert not (set(validation) & set(plan.outer_evaluation_indices))


def test_refit_updates_are_best_epoch_plus_one() -> None:
    assert refit_updates_from_best_epoch(0) == 1
    assert refit_updates_from_best_epoch(9) == 10
    with pytest.raises(SchemaError, match="nonnegative"):
        refit_updates_from_best_epoch(-1)


def test_metrics_formulas_include_variance_and_secondary_r_squared() -> None:
    residuals = np.asarray([[1.0, -1.0], [2.0, 0.0]])
    targets = np.asarray([[2.0, 2.0], [4.0, 4.0]])

    metrics = overall_prediction_metrics(residuals, targets)

    assert metrics["mse"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1.5))
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(0.5)
    assert metrics["residual_variance"] == pytest.approx(np.mean((residuals - 0.5) ** 2))
    assert metrics["r_squared_secondary_descriptive"] == pytest.approx(1.0 - 6.0 / 4.0)


def test_participant_bootstrap_is_deterministic() -> None:
    first = np.asarray([[0.0, 1.0], [0.0, 2.0]])
    second = np.asarray([[0.0, 2.0], [0.0, 3.0]])

    left = paired_subject_bootstrap(first, second, metric="rmse", n_resamples=200, seed=42)
    right = paired_subject_bootstrap(first, second, metric="rmse", n_resamples=200, seed=42)

    assert left == right
    assert left["difference_first_minus_second"] < 0.0


def test_parametric_refit_uses_selected_best_epoch_plus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(n_subjects=20, n_regions=2)
    config = _fast_config()
    folds = build_tenfold_outer_folds(20, seed=42)
    plan = make_tenfold_fold_plan(folds, 0, config=config)
    split = SubjectSplit(
        train_indices=plan.inner_training_indices,
        validation_indices=plan.inner_validation_indices,
        test_indices=plan.outer_evaluation_indices,
    )
    calls: list[int] = []

    def fake_inner(*_args, **_kwargs):
        return ParametricFitResult(
            model_name="linear",
            rho=0.0,
            linear_coefficient=0.0,
            quadratic_coefficient=0.0,
            best_epoch=2,
            epochs_run=3,
            training_losses=(1.0, 0.5, 0.25),
            validation_losses=(1.0, 0.5, 0.25),
            numerical_instability=False,
            runtime_seconds=0.0,
        )

    def fake_refit(*_args, n_updates, **_kwargs):
        calls.append(n_updates)
        return ParametricFitResult(
            model_name="linear",
            rho=0.0,
            linear_coefficient=0.0,
            quadratic_coefficient=0.0,
            best_epoch=n_updates - 1,
            epochs_run=n_updates,
            training_losses=(0.1,) * n_updates,
            validation_losses=(),
            numerical_instability=False,
            runtime_seconds=0.0,
        )

    monkeypatch.setattr("tau_sr.cohort_tenfold._fit_parametric_inner", fake_inner)
    monkeypatch.setattr("tau_sr.cohort_tenfold._fit_parametric_fixed_updates", fake_refit)

    _run_parametric_fold(
        dataset,
        plan,
        split,
        "linear",
        tmp_path,
        {"models": {}},
        config.model_config,
    )

    assert calls == [3]


def test_completed_fold_model_is_not_overwritten(tmp_path: Path) -> None:
    dataset = _dataset(n_subjects=20, n_regions=2)
    plan = make_tenfold_fold_plan(
        build_tenfold_outer_folds(20, seed=42),
        0,
        config=_fast_config(),
    )
    payload = {"completion_status": "complete", "prediction_artifact": "x.npz"}
    fold_record = {"models": {"persistence": payload}}

    _run_persistence_fold(dataset, plan, tmp_path, fold_record)

    assert fold_record["models"]["persistence"] is payload


def test_aggregation_refuses_incomplete_folds(tmp_path: Path) -> None:
    dataset = _dataset(n_subjects=20, n_regions=2)
    folds = build_tenfold_outer_folds(20, seed=42)
    records = {str(index): {"completion_status": "incomplete", "models": {}} for index in range(10)}

    with pytest.raises(SchemaError, match="incomplete"):
        aggregate_tenfold_results(
            dataset,
            folds,
            records,
            tmp_path,
            {"cohort_fingerprint": "synthetic"},
            _fast_config(),
            runtime_seconds=0.0,
        )


def test_tiny_end_to_end_tenfold_synthetic_rehearsal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = _cohort(n_subjects=20, n_regions=2)
    config = _fast_config(n_subjects=20)
    monkeypatch.setattr(
        "tau_sr.cohort_tenfold.load_verified_tenfold_cohort",
        lambda *_a, **_k: cohort,
    )

    freeze = freeze_tenfold_protocol(
        "structure",
        "baseline",
        "followup",
        "prep",
        tmp_path / "tenfold",
        config=config,
    )
    summary = execute_tenfold_analysis(
        "structure",
        "baseline",
        "followup",
        "prep",
        tmp_path / "tenfold",
        config=config,
        symbolic_runner=_fake_symbolic,
    )

    assert freeze["fold_design"]["outer_folds"] == 10
    assert summary["completion_status"] == "complete"
    assert set(summary["overall_out_of_fold_metrics"]) == set(TENFOLD_MODELS)
    assert summary["prediction_artifact_count"] == 70
    assert summary["fold_design"]["outer_fold_size_distribution"] == {"2": 10}
