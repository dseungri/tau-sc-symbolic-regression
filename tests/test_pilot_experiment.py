from __future__ import annotations

import importlib.util
import json
import math
import stat
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tau_sr.pilot_experiment as pilot
from tau_sr.identifiability import ZeroAnchoredNeuralReactionDiffusion
from tau_sr.pilot_adapter import (
    DATA_ALIGNMENT_READY,
    FORMAL_EVALUATION_READY,
    PILOT_ATLAS,
    PILOT_N_REGIONS,
    ROI_ALIGNMENT_DIRECT,
    SC_STATIC_OVER_TIME,
    SC_WEIGHT_TYPE,
    TAU_MEASUREMENT,
    TAU_REFERENCE_PENDING,
    LongitudinalTrajectoryDataset,
    make_pilot_roi_labels,
)
from tau_sr.schemas import SchemaError
from tau_sr.training import NeuralTrainingConfig


def _chain_adjacency() -> np.ndarray:
    adjacency = np.zeros((PILOT_N_REGIONS, PILOT_N_REGIONS), dtype=float)
    indices = np.arange(PILOT_N_REGIONS - 1)
    adjacency[indices, indices + 1] = 1.0
    adjacency[indices + 1, indices] = 1.0
    return adjacency


def _symmetric_normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    degrees = adjacency.sum(axis=1)
    normalizer = np.sqrt(np.outer(degrees, degrees))
    return np.eye(adjacency.shape[0]) - adjacency / normalizer


def _fake_pilot_dataset(*, heldout_offset: float = 3.0) -> LongitudinalTrajectoryDataset:
    first_adjacency = _chain_adjacency()
    second_adjacency = _chain_adjacency()
    second_adjacency[0, 2] = 0.5
    second_adjacency[2, 0] = 0.5
    adjacencies = np.stack([first_adjacency, second_adjacency])
    laplacians = np.stack(
        [
            _symmetric_normalized_laplacian(first_adjacency),
            _symmetric_normalized_laplacian(second_adjacency),
        ]
    )
    baseline = np.linspace(0.05, 0.40, PILOT_N_REGIONS)
    return LongitudinalTrajectoryDataset(
        subject_ids=("Pilot Subject 1", "Pilot Subject 2"),
        roi_labels=make_pilot_roi_labels(),
        adjacencies=adjacencies,
        laplacians=laplacians,
        times_by_subject=(
            np.array([0.0, 1.0], dtype=float),
            np.array([0.0, 1.0, 3.0], dtype=float),
        ),
        trajectories=(
            np.stack([baseline, baseline + 0.2]),
            np.stack([baseline, baseline + 1.0, baseline + heldout_offset]),
        ),
        atlas=PILOT_ATLAS,
        roi_alignment=ROI_ALIGNMENT_DIRECT,
        tau_measurement=TAU_MEASUREMENT,
        tau_reference=TAU_REFERENCE_PENDING,
        sc_weight_type=SC_WEIGHT_TYPE,
        sc_static_over_time=SC_STATIC_OVER_TIME,
        laplacian_kind="symmetric_normalized",
        data_alignment_ready=DATA_ALIGNMENT_READY,
        formal_evaluation_ready=FORMAL_EVALUATION_READY,
    )


def _training_config() -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        hidden_width=4,
        initial_rho=0.05,
        learning_rate=1e-3,
        rho_learning_rate=1e-2,
        weight_decay=0.0,
        max_epochs=5,
        patience=3,
        min_delta=0.0,
        max_step=0.05,
        seed=42,
        dtype=torch.float64,
        device="cpu",
    )


class LinearForecastModel(ZeroAnchoredNeuralReactionDiffusion):
    def __init__(self) -> None:
        super().__init__(hidden_width=1, initial_rho=0.05)

    def integrate(
        self,
        y0: torch.Tensor,
        laplacian: torch.Tensor,
        times: torch.Tensor,
        *,
        max_step: float = 0.05,
    ) -> torch.Tensor:
        del laplacian, max_step
        return torch.stack([y0 + time * torch.ones_like(y0) for time in times])

    def reaction(self, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * y


def test_build_temporal_holdout_excludes_final_visit_and_keeps_laplacians() -> None:
    dataset = _fake_pilot_dataset()

    holdout = pilot.build_temporal_holdout(dataset)

    assert holdout.two_visit_subject_index == 0
    assert holdout.three_visit_subject_index == 1
    assert holdout.heldout_visit_index == 2
    assert holdout.training_visit_indices_by_subject == ((0, 1), (0, 1))
    assert holdout.training_dataset.trajectories[0].shape == (2, PILOT_N_REGIONS)
    assert holdout.training_dataset.trajectories[1].shape == (2, PILOT_N_REGIONS)
    np.testing.assert_array_equal(holdout.heldout_state, dataset.trajectories[1][2])
    assert not np.shares_memory(holdout.heldout_state, holdout.training_dataset.trajectories[1])
    np.testing.assert_array_equal(holdout.training_dataset.laplacians, dataset.laplacians)


def test_training_state_domain_uses_training_visits_only() -> None:
    holdout = pilot.build_temporal_holdout(_fake_pilot_dataset(heldout_offset=99.0))

    domain_min, domain_max = pilot.training_state_domain(
        holdout.training_dataset,
        holdout.train_subject_indices,
    )

    assert domain_min == pytest.approx(0.05)
    assert domain_max < 2.0


def test_pilot_qc_summary_and_validation_accept_expected_contract() -> None:
    summary = pilot.pilot_qc_summary(_fake_pilot_dataset())

    pilot.validate_pilot_qc(summary)

    assert summary["subject_count"] == 2
    assert summary["region_count"] == PILOT_N_REGIONS
    assert summary["visit_count_pattern"] == (2, 3)
    assert summary["missing_tau_values"] == 0
    assert summary["tau_values_missing"] is False
    assert summary["subject_specific_laplacians"] is True
    assert summary["final_12_nodes_subcortical"] is True


def test_pilot_qc_validation_rejects_missing_tau_values() -> None:
    summary = pilot.pilot_qc_summary(_fake_pilot_dataset(), missing_tau_values=3)

    assert summary["missing_tau_values"] == 3
    assert summary["tau_values_missing"] is True
    with pytest.raises(SchemaError, match="pilot QC failed"):
        pilot.validate_pilot_qc(summary)


def test_pilot_qc_validation_rejects_failed_contract() -> None:
    summary = pilot.pilot_qc_summary(_fake_pilot_dataset())
    summary["subject_specific_laplacians"] = False

    with pytest.raises(SchemaError, match="pilot QC failed"):
        pilot.validate_pilot_qc(summary)


def test_evaluate_forecasts_reports_temporal_holdout_and_baselines() -> None:
    holdout = pilot.build_temporal_holdout(_fake_pilot_dataset())
    model = LinearForecastModel().double()

    metrics = pilot.evaluate_forecasts(model, holdout, config=_training_config())

    assert metrics["primary_model"]["all"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["secondary_model"]["all"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["linear_extrapolation"]["all"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["persistence_long_horizon"]["all"]["rmse"] == pytest.approx(3.0)
    assert metrics["persistence_one_step"]["all"]["rmse"] == pytest.approx(2.0)
    assert metrics["primary_model"]["all"]["change"]["rmse"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_forecast_metrics_returns_region_groups_and_none_for_constant_correlation() -> None:
    predicted = np.ones(PILOT_N_REGIONS)
    observed = np.ones(PILOT_N_REGIONS) * 2.0
    previous = np.ones(PILOT_N_REGIONS)

    metrics = pilot.forecast_metrics(predicted, observed, previous)

    assert set(metrics) == {"all", "left_cortical", "right_cortical", "subcortical"}
    assert metrics["all"]["rmse"] == pytest.approx(1.0)
    assert metrics["subcortical"]["mae"] == pytest.approx(1.0)
    assert metrics["all"]["pearson_correlation"] is None
    assert metrics["all"]["change"]["pearson_correlation"] is None


def test_load_selected_pilot_training_config_uses_best_epoch_plus_one(
    tmp_path: Path,
) -> None:
    artifact = {
        "selected": {
            "zero_anchored_joint": {
                "candidate_name": "rho_lr_0.01",
                "model_type": "ZeroAnchoredNeuralReactionDiffusion",
                "best_epoch": 834,
                "training_config": {
                    "hidden_width": 16,
                    "initial_rho": 0.05,
                    "learning_rate": 0.001,
                    "rho_learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "max_epochs": 1000,
                    "patience": 100,
                    "min_delta": 1e-7,
                    "max_step": 0.05,
                    "seed": 42,
                    "dtype": "torch.float64",
                    "device": "cpu",
                },
            }
        }
    }
    artifact_path = tmp_path / "selected.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    config, n_epochs, source = pilot.load_selected_pilot_training_config(artifact_path)

    assert n_epochs == 835
    assert config.hidden_width == 16
    assert config.rho_learning_rate == pytest.approx(0.01)
    assert config.dtype is torch.float64
    assert source["epoch_conversion"] == "n_epochs = best_epoch + 1"


def test_validate_private_output_directory_rejects_unignored_repo_directory() -> None:
    target = pilot.REPO_ROOT / f"unignored_pilot_outputs_{uuid.uuid4().hex}"
    try:
        with pytest.raises(SchemaError, match="git-ignored"):
            pilot.validate_private_output_directory(target)
    finally:
        if target.exists():
            target.rmdir()


def test_private_output_directory_default_and_created_mode_are_private(
    tmp_path: Path,
) -> None:
    assert pilot.DEFAULT_PRIVATE_OUTPUT_DIR.is_relative_to(Path.home())

    target = tmp_path / "pilot_output"
    target.mkdir(mode=0o755)
    target.chmod(0o755)

    output_dir = pilot.validate_private_output_directory(target)

    assert output_dir == target.resolve()
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700


def test_run_symbolic_distillation_uses_training_domain_and_declared_pysr_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    holdout = pilot.build_temporal_holdout(_fake_pilot_dataset(heldout_offset=99.0))
    captured: dict[str, object] = {}

    def fake_worker(**kwargs: object) -> dict[str, object]:
        samples = kwargs["samples"]
        symbolic_config = kwargs["symbolic_config"]
        assert hasattr(samples, "input_values")
        assert hasattr(symbolic_config, "model_selection")
        captured["sample_count"] = samples.input_values.shape[0]
        captured["domain"] = (samples.domain_min, samples.domain_max)
        captured["niterations"] = symbolic_config.niterations
        captured["populations"] = symbolic_config.populations
        captured["population_size"] = symbolic_config.population_size
        captured["maxsize"] = symbolic_config.maxsize
        captured["seed"] = symbolic_config.seed
        captured["model_selection"] = symbolic_config.model_selection
        captured["output_directory"] = symbolic_config.output_directory
        return {
            "fit_domain": [samples.domain_min, samples.domain_max],
            "evaluation_domain": [samples.domain_min, samples.domain_max],
            "equation": "x0",
            "expanded_expression": "x0",
            "complexity": 1,
            "training_loss": 0.0,
            "artifact_directory": str(tmp_path / "pysr_artifacts"),
            "runtime_seconds": 0.1,
            "pysr_version": "test",
            "julia_version": "test",
            "metrics": {"symbolic_vs_neural": {"rmse": 0.0, "mae": 0.0}},
            "polynomial_interpretation": {
                "is_polynomial": True,
                "expression_string": "x0",
                "expanded_expression_string": "x0",
                "constant": 0.0,
                "linear": 1.0,
                "quadratic": 0.0,
                "highest_degree": 1,
                "symbolic_vs_true_rmse": 999.0,
            },
            "worker_exited_normally": True,
            "worker_cleanup_required": False,
            "worker_timed_out_after_result": False,
            "worker_returncode": 0,
        }

    monkeypatch.setattr(pilot, "_run_isolated_pysr_fit", fake_worker)

    summary = pilot.run_symbolic_distillation(
        LinearForecastModel().double(),
        holdout.training_dataset,
        holdout.train_subject_indices,
        config=_training_config(),
        output_dir=tmp_path,
        timeout_seconds=1.0,
        shutdown_grace_seconds=1.0,
    )

    assert captured["sample_count"] == 500
    assert captured["domain"][1] < 2.0
    assert captured["niterations"] == 100
    assert captured["populations"] == 15
    assert captured["population_size"] == 33
    assert captured["maxsize"] == 15
    assert captured["seed"] == 42
    assert captured["model_selection"] == "best"
    assert str(tmp_path / "pysr_artifacts") == captured["output_directory"]
    assert summary["artifact_directory"] == str(tmp_path / "pysr_artifacts")
    assert "symbolic_vs_true_rmse" not in summary["polynomial_interpretation"]


def test_run_pilot_experiment_uses_adapter_missing_tau_count_for_qc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _fake_pilot_dataset()
    original_qc_summary = pilot.pilot_qc_summary
    captured: dict[str, int] = {}

    monkeypatch.setattr(pilot, "load_pilot_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        pilot,
        "inspect_pilot_inputs",
        lambda *_args: {"relative_times_years": {}, "missing_tau_values": 2},
    )

    def qc_summary_spy(
        dataset_arg: LongitudinalTrajectoryDataset,
        *,
        missing_tau_values: int = 0,
    ) -> dict[str, object]:
        captured["missing_tau_values"] = missing_tau_values
        return original_qc_summary(
            dataset_arg,
            missing_tau_values=missing_tau_values,
        )

    monkeypatch.setattr(pilot, "pilot_qc_summary", qc_summary_spy)

    with pytest.raises(SchemaError, match="pilot QC failed"):
        pilot.run_pilot_experiment(
            "/tmp/sentinel_tau_input.xlsx",
            ("/tmp/sentinel_connectome_a.csv", "/tmp/sentinel_connectome_b.csv"),
            output_directory=tmp_path / "private_outputs",
        )

    assert captured["missing_tau_values"] == 2


def test_run_pilot_experiment_writes_private_summary_without_raw_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _fake_pilot_dataset()
    config = _training_config()
    model = LinearForecastModel().double()

    monkeypatch.setattr(pilot, "load_pilot_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        pilot,
        "inspect_pilot_inputs",
        lambda *_args: {
            "relative_times_years": {
                subject: tuple(float(value) for value in dataset.times_by_subject[index])
                for index, subject in enumerate(dataset.subject_ids)
            }
        },
    )
    monkeypatch.setattr(
        pilot,
        "load_selected_pilot_training_config",
        lambda *_args: (
            config,
            2,
            {
                "artifact": "experiments/results/identifiability_experiment.json",
                "selected_candidate": "rho_lr_0.01",
                "model_type": "ZeroAnchoredNeuralReactionDiffusion",
                "best_epoch_zero_indexed": 1,
                "fixed_epoch_count": 2,
                "epoch_conversion": "n_epochs = best_epoch + 1",
            },
        ),
    )
    monkeypatch.setattr(
        pilot,
        "fit_fixed_epoch_recovery_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            model=model,
            training_losses=(1.0, 0.5),
            epochs_run=2,
            numerical_instability=False,
        ),
    )
    monkeypatch.setattr(
        pilot,
        "evaluate_forecasts",
        lambda *_args, **_kwargs: {"primary_model": {"all": {"rmse": 0.0}}},
    )
    monkeypatch.setattr(
        pilot,
        "neural_reaction_diagnostics",
        lambda *_args, **_kwargs: {
            "reaction_at_zero": 0.0,
            "training_state_domain_min": 0.05,
            "training_state_domain_max": 1.4,
            "neural_reaction_min": 0.0,
            "neural_reaction_max": 0.7,
        },
    )
    monkeypatch.setattr(
        pilot,
        "run_symbolic_distillation",
        lambda *_args, **_kwargs: {"ran": True, "equation": "x0"},
    )
    monkeypatch.setattr(pilot, "environment_versions", lambda: {"python": "test"})

    summary = pilot.run_pilot_experiment(
        "/tmp/sentinel_tau_input.xlsx",
        ("/tmp/sentinel_connectome_a.csv", "/tmp/sentinel_connectome_b.csv"),
        output_directory=tmp_path / "private_outputs",
    )

    artifact_path = Path(str(summary["private_artifact_path"]))
    artifact_text = artifact_path.read_text(encoding="utf-8")
    artifact = json.loads(artifact_text)
    assert artifact_path.exists()
    assert artifact["dataset_summary"]["missing_tau_values"] == 0
    assert artifact["dataset_summary"]["tau_values_missing"] is False
    assert "/tmp/sentinel_tau_input.xlsx" not in artifact_text
    assert "/tmp/sentinel_connectome_a.csv" not in artifact_text
    assert "/tmp/sentinel_connectome_b.csv" not in artifact_text
    assert "Pilot Subject 1" in artifact_text
    assert "Pilot Subject 2" in artifact_text


def test_runner_output_does_not_echo_raw_input_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner_path = pilot.REPO_ROOT / "scripts" / "run_pilot_experiment.py"
    spec = importlib.util.spec_from_file_location("pilot_runner_for_test", runner_path)
    assert spec is not None
    assert spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def fake_run_pilot_experiment(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "dataset_summary": {
                "subject_count": 2,
                "subject_ids": ["Pilot Subject 1", "Pilot Subject 2"],
                "visit_counts": {"Pilot Subject 1": 2, "Pilot Subject 2": 3},
                "region_count": PILOT_N_REGIONS,
                "roi_alignment": ROI_ALIGNMENT_DIRECT,
                "data_alignment_ready": True,
                "formal_evaluation_ready": False,
            },
            "fixed_training_configuration": {
                "fixed_epoch_count": 2,
                "selected_candidate": "rho_lr_0.01",
            },
            "training_diagnostics": {
                "epochs_run": 2,
                "initial_training_loss": 1.0,
                "final_training_loss": 0.5,
                "learned_rho": 0.05,
                "numerical_instability": False,
            },
            "symbolic": {"worker_exited_normally": True},
            "private_artifact_path": "/tmp/private_outputs/pilot_experiment_summary.json",
        }

    monkeypatch.setattr(runner, "run_pilot_experiment", fake_run_pilot_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pilot_experiment.py",
            "--tau-workbook",
            "/tmp/sentinel_tau_input.xlsx",
            "--connectome",
            "/tmp/sentinel_connectome_a.csv",
            "--connectome",
            "/tmp/sentinel_connectome_b.csv",
        ],
    )

    assert runner.main() == 0
    output = capsys.readouterr().out
    assert "/tmp/sentinel_tau_input.xlsx" not in output
    assert "/tmp/sentinel_connectome_a.csv" not in output
    assert "/tmp/sentinel_connectome_b.csv" not in output
    assert "Pilot Subject 1" in output
    assert "Pilot Subject 2" in output


def test_runner_source_does_not_import_legacy_sr_models() -> None:
    source = (pilot.REPO_ROOT / "scripts" / "run_pilot_experiment.py").read_text(
        encoding="utf-8"
    )

    assert "sr_model_1" not in source
    assert "sr_model_2" not in source
    assert "sr_model_3" not in source


def test_forecast_metrics_rejects_nonfinite_vector() -> None:
    predicted = np.ones(PILOT_N_REGIONS)
    observed = np.ones(PILOT_N_REGIONS)
    previous = np.ones(PILOT_N_REGIONS)
    predicted[0] = math.inf

    with pytest.raises(SchemaError, match="finite"):
        pilot.forecast_metrics(predicted, observed, previous)
