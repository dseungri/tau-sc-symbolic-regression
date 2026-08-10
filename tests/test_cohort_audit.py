from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from tau_sr.cohort_audit import (
    METRIC_MODELS,
    AuditInputs,
    audit_frozen_experiment,
    paired_subject_bootstrap_from_metrics,
    recompute_test_metrics_from_subject_metrics,
    validate_audit_output_directory,
)
from tau_sr.cohort_experiment import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    EXPECTED_SUBJECT_COUNT,
    EXPECTED_TEST_COUNT,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_VALIDATION_COUNT,
    FROZEN_SPLIT_SEED,
    CohortExperimentConfig,
    _parametric_loss,
    _torch_batch,
    _TorchParametricReactionDiffusion,
    fit_parametric_model,
    forecast_neural,
    write_private_json,
)
from tau_sr.neural import NeuralReactionDiffusion
from tau_sr.pilot_adapter import LongitudinalTrajectoryDataset
from tau_sr.schemas import SchemaError
from tau_sr.training import SubjectSplit


def _dataset(n_subjects: int = 6, n_regions: int = 4) -> LongitudinalTrajectoryDataset:
    subject_ids = tuple(f"Audit Subject {index:03d}" for index in range(1, n_subjects + 1))
    roi_labels = tuple(f"Region {index:03d}" for index in range(1, n_regions + 1))
    adjacency = np.zeros((n_subjects, n_regions, n_regions), dtype=float)
    laplacian = np.zeros((n_subjects, n_regions, n_regions), dtype=float)
    base_laplacian = np.eye(n_regions, dtype=float)
    if n_regions > 1:
        for index in range(n_regions - 1):
            base_laplacian[index, index + 1] = -1.0
            base_laplacian[index + 1, index] = -1.0
        base_laplacian[0, 0] = 1.0
        base_laplacian[-1, -1] = 1.0
        if n_regions > 2:
            base_laplacian[1:-1, 1:-1] = np.diag(np.full(n_regions - 2, 2.0))
    trajectories = []
    for subject_index in range(n_subjects):
        baseline = np.linspace(0.1, 0.4, n_regions) + 0.02 * subject_index
        followup = baseline + 0.04 * baseline - 0.03 * (base_laplacian @ baseline)
        trajectories.append(np.stack([baseline, followup]))
        laplacian[subject_index] = base_laplacian
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


def _subject_metric_rows(scale: float) -> dict[str, dict[str, float]]:
    return {
        f"heldout-{index:02d}": {
            "rmse": float((0.09 + 0.001 * index) * scale),
            "mae": float((0.06 + 0.0007 * index) * scale),
            "mean_prediction_error": float(((-1) ** index) * 0.001 * scale),
        }
        for index in range(EXPECTED_TEST_COUNT)
    }


def _fake_frozen_summary() -> dict[str, object]:
    subject_level_metrics = {
        "persistence": _subject_metric_rows(1.0),
        "diffusion": _subject_metric_rows(1.01),
        "linear": _subject_metric_rows(0.98),
        "quadratic": _subject_metric_rows(0.975),
        "neural": _subject_metric_rows(0.99),
        "symbolic": _subject_metric_rows(0.991),
    }
    summary: dict[str, object] = {
        "completion_status": "complete",
        "cohort": {
            "subject_count": EXPECTED_SUBJECT_COUNT,
            "matched_triplets": EXPECTED_SUBJECT_COUNT,
            "region_count": 160,
            "time_years": [0.0, 1.0],
        },
        "split": {
            "counts": {
                "train": EXPECTED_TRAIN_COUNT,
                "validation": EXPECTED_VALIDATION_COUNT,
                "test": EXPECTED_TEST_COUNT,
            },
            "seed": FROZEN_SPLIT_SEED,
            "test_set_role": "held_out_final_evaluation_only",
        },
        "model_configurations": {model: {} for model in METRIC_MODELS},
        "fitted_models": {
            "diffusion": {"best_epoch": 12, "numerical_instability": False},
            "linear": {"best_epoch": 0, "numerical_instability": False},
            "quadratic": {"best_epoch": 6, "numerical_instability": False},
            "neural": {
                "selected_seed": 42,
                "seed_sensitivity": {
                    "41": {"best_epoch": 3, "validation_metrics": {"pooled_rmse": 0.3}},
                    "42": {"best_epoch": 2, "validation_metrics": {"pooled_rmse": 0.2}},
                    "43": {"best_epoch": 4, "validation_metrics": {"pooled_rmse": 0.25}},
                },
            },
        },
        "symbolic": {
            "completed": True,
            "fit_domain": [0.0, 1.0],
            "evaluation_domain": [0.1, 0.9],
            "julia_version": "1.0.0",
        },
        "software_versions": {"pysr": "1.5.10"},
        "subject_level_metrics": subject_level_metrics,
    }
    test_results = recompute_test_metrics_from_subject_metrics(summary)
    for metrics in test_results.values():
        metrics["numerical_instability_count"] = 0
        metrics["runtime_seconds"] = 0.0
    summary["test_results"] = test_results
    summary["validation_results"] = copy.deepcopy(test_results)
    bootstrap = {}
    for name, first, second in (
        ("quadratic_vs_linear", "quadratic", "linear"),
        ("neural_vs_linear", "neural", "linear"),
        ("neural_vs_quadratic", "neural", "quadratic"),
        ("symbolic_vs_neural", "symbolic", "neural"),
    ):
        result = paired_subject_bootstrap_from_metrics(
            tuple(subject_level_metrics[first].values()),
            tuple(subject_level_metrics[second].values()),
            metric="rmse",
            n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            seed=42,
        )
        bootstrap[name] = {
            "rmse_difference_first_minus_second": result["difference_first_minus_second"],
            "ci95": result["ci95"],
        }
    summary["bootstrap_comparisons"] = bootstrap
    return summary


def _write_fake_frozen_artifacts(directory: Path, summary: dict[str, object]) -> None:
    directory.mkdir(mode=0o700)
    freeze = {
        "analysis_freeze": {
            "status": "complete_before_test",
            "test_metrics_computed": False,
            "test_outcomes_inspected": False,
        },
        "completion_status": "analysis_freeze_complete",
    }
    checkpoint = {"completion_status": "complete"}
    write_private_json(directory / "full_cohort_experiment_summary.json", summary)
    write_private_json(directory / "analysis_freeze.json", freeze)
    write_private_json(directory / "cohort_experiment_checkpoint.json", checkpoint)


def test_recompute_metrics_from_subject_metrics_matches_saved_summary() -> None:
    summary = _fake_frozen_summary()

    recomputed = recompute_test_metrics_from_subject_metrics(summary)

    for model, metrics in recomputed.items():
        saved = summary["test_results"][model]
        assert metrics["pooled_rmse"] == pytest.approx(saved["pooled_rmse"])
        assert metrics["pooled_mae"] == pytest.approx(saved["pooled_mae"])


def test_paired_subject_bootstrap_is_deterministic_and_keeps_global_rng_state() -> None:
    first = tuple(_subject_metric_rows(0.9).values())
    second = tuple(_subject_metric_rows(1.0).values())
    np.random.seed(123)
    before = np.random.get_state()

    first_result = paired_subject_bootstrap_from_metrics(
        first,
        second,
        metric="rmse",
        n_resamples=200,
        seed=42,
    )
    second_result = paired_subject_bootstrap_from_metrics(
        first,
        second,
        metric="rmse",
        n_resamples=200,
        seed=42,
    )
    after = np.random.get_state()

    assert first_result == second_result
    assert first_result["difference_first_minus_second"] < 0.0
    assert all(np.array_equal(left, right) for left, right in zip(before, after, strict=True))


def test_audit_reads_frozen_artifacts_and_writes_private_summary(tmp_path: Path) -> None:
    frozen_dir = tmp_path / "frozen"
    output_dir = tmp_path / "audit"
    _write_fake_frozen_artifacts(frozen_dir, _fake_frozen_summary())

    audit = audit_frozen_experiment(
        AuditInputs(
            frozen_artifact_dir=frozen_dir,
            audit_output_dir=output_dir,
            experiment_commit="synthetic-commit",
        )
    )

    assert audit["audit_pass"] is True
    assert audit["metric_reproduction"]["raw_vector_metric_reproduction_available"] is False
    assert audit["metric_reproduction"]["max_aggregate_discrepancy"] == pytest.approx(0.0)
    private_summary = output_dir / "full_cohort_audit_summary.json"
    assert private_summary.exists()
    assert oct(output_dir.stat().st_mode & 0o777) == "0o700"
    assert oct(private_summary.stat().st_mode & 0o777) == "0o600"
    assert json.loads(private_summary.read_text(encoding="utf-8"))["audit_pass"] is True


def test_validate_audit_output_directory_rejects_frozen_or_repo_locations(tmp_path: Path) -> None:
    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()

    with pytest.raises(SchemaError, match="frozen experiment directory"):
        validate_audit_output_directory(frozen_dir, frozen_dir)
    with pytest.raises(SchemaError, match="outside the repository"):
        validate_audit_output_directory(Path.cwd() / "audit-output", frozen_dir)


def test_parametric_parameters_receive_gradients_and_move() -> None:
    dataset = _dataset()
    model = _TorchParametricReactionDiffusion(
        kind="quadratic",
        initial_rho=0.05,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    batch = _torch_batch(dataset, (0, 1, 2), device=torch.device("cpu"), dtype=torch.float64)
    parameters = (
        model.raw_rho,
        model.linear_coefficient,
        model.quadratic_coefficient,
    )
    initial = tuple(parameter.detach().clone() for parameter in parameters)

    loss = _parametric_loss(model, batch, max_step=1.0)
    loss.backward()
    optimizer = torch.optim.Adam(parameters, lr=1e-2)
    optimizer.step()

    for parameter, initial_value in zip(parameters, initial, strict=True):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.max(torch.abs(parameter.grad)).item() > 0.0
        assert not torch.equal(parameter.detach(), initial_value)


def test_epoch_zero_is_after_first_optimizer_step() -> None:
    dataset = _dataset()
    config = CohortExperimentConfig(
        max_epochs=1,
        patience=1,
        max_step=1.0,
        parametric_learning_rate=1e-2,
    )

    result = fit_parametric_model(dataset, _split(), kind="linear", config=config)

    assert result.best_epoch == 0
    assert result.epochs_run == 1
    assert result.linear_coefficient != pytest.approx(0.0)


def test_parametric_fit_restores_validation_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    validation_losses = iter((0.3, 0.1, 0.2))

    def fake_loss(
        model: _TorchParametricReactionDiffusion,
        batch: dict[str, torch.Tensor],
        *,
        max_step: float,
    ) -> torch.Tensor:
        del max_step
        if batch["y0"].shape[0] == len(_split().train_indices):
            return -model.linear_coefficient + model.raw_rho * 0.0
        return model.linear_coefficient * 0.0 + torch.tensor(
            next(validation_losses),
            dtype=model.linear_coefficient.dtype,
            device=model.linear_coefficient.device,
        )

    monkeypatch.setattr("tau_sr.cohort_experiment._parametric_loss", fake_loss)
    config = CohortExperimentConfig(
        max_epochs=3,
        patience=10,
        max_step=1.0,
        parametric_learning_rate=1e-2,
    )

    result = fit_parametric_model(dataset, _split(), kind="linear", config=config)

    assert result.best_epoch == 1
    assert result.epochs_run == 3
    assert result.linear_coefficient == pytest.approx(0.02, abs=1e-8)


def test_test_data_cannot_influence_parametric_early_stopping() -> None:
    dataset = _dataset()
    modified_trajectories = list(dataset.trajectories)
    for index in _split().test_indices:
        modified_trajectories[index] = modified_trajectories[index] + 100.0
    modified = LongitudinalTrajectoryDataset(
        subject_ids=dataset.subject_ids,
        roi_labels=dataset.roi_labels,
        adjacencies=dataset.adjacencies,
        laplacians=dataset.laplacians,
        times_by_subject=dataset.times_by_subject,
        trajectories=tuple(modified_trajectories),
    )
    config = CohortExperimentConfig(max_epochs=3, patience=3, max_step=1.0)

    original = fit_parametric_model(dataset, _split(), kind="linear", config=config)
    changed_test = fit_parametric_model(modified, _split(), kind="linear", config=config)

    assert changed_test.best_epoch == original.best_epoch
    assert changed_test.training_losses == pytest.approx(original.training_losses)
    assert changed_test.validation_losses == pytest.approx(original.validation_losses)
    assert changed_test.rho == pytest.approx(original.rho)
    assert changed_test.linear_coefficient == pytest.approx(original.linear_coefficient)


def test_forecast_neural_copies_readonly_times_without_warning() -> None:
    dataset = _dataset(n_subjects=2, n_regions=3)
    assert dataset.times_by_subject[0].flags.writeable is False
    original_times = dataset.times_by_subject[0].copy()
    model = NeuralReactionDiffusion(hidden_width=2, initial_rho=0.05).double()
    config = CohortExperimentConfig(max_epochs=1, patience=1, neural_seeds=(42,))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecast_neural(model, dataset, (0, 1), config=config)

    assert not any("not writable" in str(warning.message) for warning in caught)
    np.testing.assert_allclose(dataset.times_by_subject[0], original_times)
    assert dataset.times_by_subject[0].flags.writeable is False
