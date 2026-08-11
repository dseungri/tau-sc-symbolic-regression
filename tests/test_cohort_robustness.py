from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tau_sr.cohort_adapter import _SubjectFiles
from tau_sr.cohort_experiment import CohortExperimentConfig
from tau_sr.cohort_robustness import (
    DevelopmentCohort,
    RobustnessConfig,
    _load_resume_checkpoint,
    _run_symbolic_fold,
    _write_robustness_checkpoint,
    build_outer_folds,
    development_labels_from_manifest,
    fit_ridge_baseline,
    inner_train_validation_split,
    load_development_cohort,
    paired_subject_bootstrap,
    run_development_cross_validation,
    select_ridge_alpha,
    validate_private_robustness_output_directory,
)
from tau_sr.identifiability import ZeroAnchoredNeuralReactionDiffusion
from tau_sr.pilot_adapter import LongitudinalTrajectoryDataset
from tau_sr.schemas import SchemaError


def _dataset(n_subjects: int = 8, n_regions: int = 4) -> LongitudinalTrajectoryDataset:
    subject_ids = tuple(f"Subject {index:03d}" for index in range(n_subjects))
    roi_labels = tuple(f"Region {index:03d}" for index in range(n_regions))
    adjacencies = np.zeros((n_subjects, n_regions, n_regions), dtype=float)
    laplacians = np.zeros_like(adjacencies)
    trajectories = []
    for subject_index in range(n_subjects):
        baseline = np.linspace(0.1, 0.4, n_regions) + subject_index * 0.001
        followup = baseline + 0.02 * baseline
        trajectories.append(np.stack([baseline, followup]))
    return LongitudinalTrajectoryDataset(
        subject_ids=subject_ids,
        roi_labels=roi_labels,
        adjacencies=adjacencies,
        laplacians=laplacians,
        times_by_subject=tuple(np.asarray([0.0, 1.0], dtype=float) for _ in subject_ids),
        trajectories=tuple(trajectories),
    )


def _split_manifest(n_subjects: int, test_indices: set[int]) -> dict[str, object]:
    labels = tuple(f"Cohort Subject {index:03d}" for index in range(1, n_subjects + 1))
    train = tuple(
        label
        for index, label in enumerate(labels)
        if index not in test_indices and index % 2 == 0
    )
    validation = tuple(
        label for index, label in enumerate(labels) if index not in test_indices and index % 2 == 1
    )
    test = tuple(label for index, label in enumerate(labels) if index in test_indices)
    return {
        "seed": 42,
        "subject_labels": {"train": train, "validation": validation, "test": test},
    }


def test_outer_folds_are_deterministic_and_cover_each_subject_once() -> None:
    first = build_outer_folds(130, n_folds=5, seed=42)
    second = build_outer_folds(130, n_folds=5, seed=42)

    assert first == second
    assert [len(fold) for fold in first] == [26, 26, 26, 26, 26]
    combined = [index for fold in first for index in fold]
    assert sorted(combined) == list(range(130))


def test_inner_split_excludes_outer_evaluation_subjects() -> None:
    outer = build_outer_folds(130, n_folds=5, seed=42)[0]
    outer_train = tuple(index for index in range(130) if index not in set(outer))

    train, validation, inner_seed = inner_train_validation_split(
        outer_train,
        outer,
        seed=42,
        fold_index=0,
    )

    assert inner_seed == 1051
    assert not (set(train) & set(validation))
    assert not (set(train) & set(outer))
    assert not (set(validation) & set(outer))


def test_development_labels_exclude_frozen_test_labels_in_original_order() -> None:
    manifest = _split_manifest(6, {1, 4})

    development, excluded = development_labels_from_manifest(manifest, 6)

    assert development == (
        "Cohort Subject 001",
        "Cohort Subject 003",
        "Cohort Subject 004",
        "Cohort Subject 006",
    )
    assert excluded == ("Cohort Subject 002", "Cohort Subject 005")


def test_development_loader_never_builds_test_triplets(monkeypatch: pytest.MonkeyPatch) -> None:
    triplets = tuple(
        _SubjectFiles(
            key=str(index),
            structure_path=Path(f"structure_{index}.csv"),
            baseline_tau_path=Path(f"baseline_{index}.csv"),
            followup_tau_path=Path(f"followup_{index}.csv"),
        )
        for index in range(1, 7)
    )
    file_counts = {
        "duplicate_key_count": 0,
        "unexpected_or_malformed_csv_count": 0,
        "missing_counterpart_count": 0,
    }
    manifest = _split_manifest(6, {1, 4})
    dataset = _dataset(n_subjects=4, n_regions=2)

    monkeypatch.setattr(
        "tau_sr.cohort_robustness._match_subject_files",
        lambda *_args: (triplets, file_counts),
    )

    def fake_build(selected, *, file_counts, laplacian_kind):
        del file_counts, laplacian_kind
        keys = {triplet.key for triplet in selected}
        if keys & {"2", "5"}:
            raise AssertionError("test triplet was numerically opened")
        return SimpleNamespace(dataset=dataset, summary={})

    monkeypatch.setattr("tau_sr.cohort_robustness._build_cohort_components", fake_build)

    cohort = load_development_cohort("structure", "baseline", "followup", manifest)

    assert cohort.development_label_count == 4
    assert cohort.excluded_test_label_count == 2
    assert cohort.test_numerical_records_opened == 0


def test_ridge_preprocessing_uses_training_statistics_only() -> None:
    x_train = np.asarray([[1.0, 10.0], [3.0, 14.0]])
    y_train = x_train + 1.0
    validation_like_extreme_values = np.asarray([[1000.0, -1000.0]])

    fit = fit_ridge_baseline(x_train, y_train, alpha=1.0)

    np.testing.assert_allclose(fit.x_mean, np.asarray([2.0, 12.0]))
    assert not np.allclose(fit.x_mean, validation_like_extreme_values.mean(axis=0))


def test_ridge_alpha_selection_breaks_ties_toward_larger_alpha() -> None:
    dataset = _dataset(n_subjects=4, n_regions=2)
    constant = np.full((2, 2), 0.2)
    dataset = LongitudinalTrajectoryDataset(
        subject_ids=dataset.subject_ids,
        roi_labels=dataset.roi_labels,
        adjacencies=dataset.adjacencies,
        laplacians=dataset.laplacians,
        times_by_subject=dataset.times_by_subject,
        trajectories=tuple(constant.copy() for _ in dataset.subject_ids),
    )

    selection = select_ridge_alpha(dataset, (0, 1), (2, 3), (0.1, 1.0, 10.0))

    assert selection["selected_alpha"] == 10.0


def test_symbolic_fold_domain_uses_outer_training_states_only(tmp_path: Path) -> None:
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
    model = ZeroAnchoredNeuralReactionDiffusion(hidden_width=2, initial_rho=0.05).double()

    def fake_symbolic(model, dataset, train_indices, config, output_dir, timeout, shutdown):
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
        }

    payload = _run_symbolic_fold(
        model,
        dataset,
        (0, 1, 2),
        (3,),
        0,
        {"models": {}},
        tmp_path,
        RobustnessConfig(model_config=CohortExperimentConfig(max_epochs=1, patience=1)),
        fake_symbolic,
    )

    assert payload["completed"] is True
    assert payload["fit_domain"][1] < 99.0


def test_paired_bootstrap_reports_both_metrics_deterministically() -> None:
    first = np.asarray([[0.0, 1.0], [0.0, 2.0]])
    second = np.asarray([[0.0, 2.0], [0.0, 3.0]])

    first_result = paired_subject_bootstrap(first, second, metric="rmse", n_resamples=200, seed=42)
    second_result = paired_subject_bootstrap(first, second, metric="rmse", n_resamples=200, seed=42)
    mae_result = paired_subject_bootstrap(first, second, metric="mae", n_resamples=200, seed=42)

    assert first_result == second_result
    assert first_result["difference_first_minus_second"] < 0.0
    assert mae_result["difference_first_minus_second"] < 0.0


def test_checkpoint_resume_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    fingerprints = {
        "repository_commit": "a",
        "development_fingerprint": "b",
        "dataset_fingerprint": "c",
        "outer_fold_fingerprint": "d",
        "configuration_fingerprint": "e",
    }
    _write_robustness_checkpoint(tmp_path, fingerprints, {}, "partial")
    changed = {**fingerprints, "configuration_fingerprint": "changed"}

    with pytest.raises(SchemaError, match="fingerprints"):
        _load_resume_checkpoint(tmp_path, changed)


def test_private_output_permissions_and_repo_refusal(tmp_path: Path) -> None:
    private = validate_private_robustness_output_directory(tmp_path / "private")

    assert oct(private.stat().st_mode & 0o777) == "0o700"
    with pytest.raises(SchemaError, match="outside the repository"):
        validate_private_robustness_output_directory(Path.cwd() / "robustness-output")


def test_tiny_end_to_end_cv_uses_development_only_subjects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(n_subjects=130, n_regions=4)
    cohort = DevelopmentCohort(
        dataset=dataset,
        original_split_manifest={},
        development_label_count=130,
        excluded_test_label_count=33,
        numerical_records_opened=390,
        test_numerical_records_opened=0,
        development_fingerprint="synthetic-development",
        source_file_match_count=163,
    )
    prep = tmp_path / "prep"
    prep.mkdir()
    (prep / "cohort_split_seed_42.json").write_text('{"seed": 42}', encoding="utf-8")
    monkeypatch.setattr(
        "tau_sr.cohort_robustness.load_development_cohort",
        lambda *_args: cohort,
    )

    def fake_symbolic(model, dataset, train_indices, config, output_dir, timeout, shutdown):
        del model, dataset, train_indices, config, output_dir, timeout, shutdown
        return {
            "completed": True,
            "fit_domain": [0.0, 1.0],
            "evaluation_domain": [0.0, 1.0],
            "equation": "0.0 * y",
            "expanded_expression": "0",
            "complexity": 1,
            "training_loss": 0.0,
            "metrics": {"symbolic_vs_neural_rmse": 0.0, "symbolic_vs_neural_mae": 0.0},
        }

    config = RobustnessConfig(
        model_config=CohortExperimentConfig(
            max_epochs=1,
            patience=1,
            neural_seeds=(41,),
            pysr_niterations=1,
            pysr_populations=1,
            pysr_population_size=5,
        ),
        bootstrap_resamples=20,
    )

    summary = run_development_cross_validation(
        "structure",
        "baseline",
        "followup",
        prep,
        tmp_path / "robustness",
        config=config,
        symbolic_runner=fake_symbolic,
    )

    assert summary["completion_status"] == "complete"
    assert summary["development_subjects"] == 130
    assert summary["excluded_frozen_test_subjects"] == 33
    assert summary["test_firewall"]["test_numerical_records_opened"] == 0
    assert set(summary["overall_out_of_fold_metrics"]) == {
        "persistence",
        "ridge",
        "diffusion",
        "linear",
        "quadratic",
        "neural",
        "symbolic",
    }
