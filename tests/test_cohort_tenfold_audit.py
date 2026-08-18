from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tau_sr.cohort_experiment import CohortExperimentConfig
from tau_sr.cohort_tenfold import (
    TenfoldCohort,
    TenfoldConfig,
    execute_tenfold_analysis,
    freeze_tenfold_protocol,
)
from tau_sr.cohort_tenfold_audit import audit_tenfold_experiment
from tau_sr.pilot_adapter import LongitudinalTrajectoryDataset


def _dataset(n_subjects: int = 20, n_regions: int = 2) -> LongitudinalTrajectoryDataset:
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


def _cohort() -> TenfoldCohort:
    return TenfoldCohort(
        dataset=_dataset(),
        qc_summary={
            "ready": True,
            "subject_count": 20,
            "matched_triplets": 20,
            "region_count": 2,
            "time_years": [0.0, 1.0],
        },
        cohort_fingerprint="synthetic-audit",
        source_file_match_count=20,
    )


def _config() -> TenfoldConfig:
    return TenfoldConfig(
        expected_subject_count=20,
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


def _completed_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cohort = _cohort()
    monkeypatch.setattr(
        "tau_sr.cohort_tenfold.load_verified_tenfold_cohort",
        lambda *_a, **_k: cohort,
    )
    output = tmp_path / "tenfold"
    freeze_tenfold_protocol("structure", "baseline", "followup", "prep", output, config=_config())
    execute_tenfold_analysis(
        "structure",
        "baseline",
        "followup",
        "prep",
        output,
        config=_config(),
        symbolic_runner=_fake_symbolic,
    )
    return output


def test_tenfold_audit_recomputes_completed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _completed_output(tmp_path, monkeypatch)

    audit = audit_tenfold_experiment(output)

    assert audit["audit_pass"] is True
    assert audit["metric_reproduction"]["max_aggregate_metric_discrepancy"] == 0.0
    assert audit["artifact_checks"]["artifact_count"] == 70


def test_tenfold_audit_detects_tampered_prediction_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _completed_output(tmp_path, monkeypatch)
    artifact = output / "fold_predictions" / "fold_00_quadratic.npz"
    with np.load(artifact) as arrays:
        payload = {name: arrays[name].copy() for name in arrays.files}
    payload["residual"][0, 0] += 1.0
    np.savez_compressed(artifact, **payload)
    artifact.chmod(0o600)

    audit = audit_tenfold_experiment(output)

    assert audit["audit_pass"] is False
    assert audit["metric_reproduction"]["max_aggregate_metric_discrepancy"] > 0.0
