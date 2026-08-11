from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tau_sr.schemas import SchemaError
from tau_sr.synthetic_robustness import (
    SweepCondition,
    SyntheticRobustnessConfig,
    aggregate_synthetic_robustness,
    build_paired_synthetic_run_data,
    build_sweep_manifest,
    classify_polynomial_recovery,
    count_to_binary_adjacency,
    derive_component_seeds,
    fixed_reference_reaction_grid,
    robustness_table_rows,
    run_synthetic_robustness_condition,
    training_fit_domain,
    write_publication_artifacts,
)


def _tiny_config(**overrides: object) -> SyntheticRobustnessConfig:
    values = {
        "n_subjects": 6,
        "n_regions": 4,
        "times": (0.0, 0.2, 0.4),
        "validation_fraction": 1 / 6,
        "test_fraction": 1 / 6,
        "hidden_width": 2,
        "max_epochs": 1,
        "patience": 1,
        "pysr_niterations": 1,
        "pysr_populations": 1,
        "pysr_population_size": 8,
        "pysr_maxsize": 7,
        "pysr_n_samples": 20,
        "reference_grid_points": 40,
        "dtype": torch.float64,
        "device": "cpu",
        "require_canonical_reproduction": False,
    }
    values.update(overrides)
    return SyntheticRobustnessConfig(**values)  # type: ignore[arg-type]


def _condition(**overrides: object) -> SweepCondition:
    values = {
        "run_id": "seed_master_42",
        "factor": "seed",
        "level": "master_seed_42",
        "master_seed": 42,
    }
    values.update(overrides)
    return SweepCondition(**values)  # type: ignore[arg-type]


def _fake_symbolic_runner(_model, run_data, config, _run_dir: Path) -> dict[str, object]:
    grid = fixed_reference_reaction_grid(config)
    truth = config.true_growth_rate * grid - config.true_saturation_rate * grid**2
    return {
        "fit_domain": list(training_fit_domain(run_data.observed_dataset, run_data.split)),
        "evaluation_domain": [float(grid.min()), float(grid.max())],
        "equation": "0.5*y - 0.5*y**2",
        "expanded_expression": "-0.5*y**2 + 0.5*y",
        "complexity": 5,
        "training_loss": 0.0,
        "runtime_seconds": 0.01,
        "pysr_version": "fake",
        "julia_version": "fake",
        "metrics": {
            "evaluation_domain_min": float(grid.min()),
            "evaluation_domain_max": float(grid.max()),
            "n_evaluation_points": int(grid.shape[0]),
            "neural_vs_true_rmse": 0.0,
            "neural_vs_true_mae": 0.0,
            "symbolic_vs_neural_rmse": 0.0,
            "symbolic_vs_neural_mae": 0.0,
            "symbolic_vs_true_rmse": float(np.sqrt(np.mean((truth - truth) ** 2))),
            "symbolic_vs_true_mae": 0.0,
            "true_reaction_range": float(np.ptp(truth)),
            "neural_vs_true_normalized_rmse": 0.0,
            "symbolic_vs_neural_normalized_rmse": 0.0,
            "symbolic_vs_true_normalized_rmse": 0.0,
            "symbolic_complexity": 5,
        },
        "polynomial_interpretation": {
            "is_polynomial": True,
            "expression_string": "0.5*y - 0.5*y**2",
            "expanded_expression_string": "-0.5*y**2 + 0.5*y",
            "constant": 0.0,
            "linear": 0.5,
            "quadratic": -0.5,
            "highest_degree": 2,
            "constant_error": 0.0,
            "linear_error": 0.0,
            "quadratic_error": 0.0,
        },
        "worker_exited_normally": True,
        "worker_cleanup_required": False,
        "worker_timed_out_after_result": False,
        "worker_returncode": 0,
    }


def test_manifest_contains_exactly_23_unique_prespecified_runs() -> None:
    manifest = build_sweep_manifest()

    assert len(manifest) == 23
    assert len({condition.run_id for condition in manifest}) == 23
    assert sum(condition.factor == "seed" for condition in manifest) == 5
    assert sum(condition.factor == "noise" for condition in manifest) == 9
    assert sum(condition.factor == "graph_representation" for condition in manifest) == 3
    assert sum(condition.factor == "laplacian" for condition in manifest) == 6


def test_component_seed_derivation_is_deterministic_and_master_driven() -> None:
    first = derive_component_seeds(42)
    second = derive_component_seeds(42)
    changed = derive_component_seeds(43)

    assert first == second
    assert first.graph_cohort == 42
    assert first.subject_split == 42
    assert first.neural_training == 42
    assert changed.graph_cohort == 43
    assert changed != first


def test_paired_binary_condition_reuses_weighted_topology_baseline_and_split() -> None:
    config = _tiny_config()
    weighted = build_paired_synthetic_run_data(_condition(), config)
    binary = build_paired_synthetic_run_data(
        _condition(
            run_id="graph_binary_seed_42",
            factor="graph_representation",
            level="binary",
            graph_representation="binary",
        ),
        config,
    )

    np.testing.assert_array_equal(weighted.weighted_adjacencies, binary.weighted_adjacencies)
    np.testing.assert_array_equal(
        weighted.clean_dataset.baseline_states,
        binary.clean_dataset.baseline_states,
    )
    assert weighted.split == binary.split
    np.testing.assert_array_equal(
        binary.clean_dataset.adjacencies,
        (weighted.weighted_adjacencies > 0.0).astype(float),
    )


def test_paired_laplacian_conditions_reuse_weighted_adjacencies_and_splits() -> None:
    config = _tiny_config()
    symmetric = build_paired_synthetic_run_data(_condition(), config)
    random_walk = build_paired_synthetic_run_data(
        _condition(
            run_id="laplacian_random_walk_seed_42",
            factor="laplacian",
            level="random_walk",
            laplacian_kind="random_walk",
        ),
        config,
    )

    np.testing.assert_array_equal(symmetric.weighted_adjacencies, random_walk.weighted_adjacencies)
    np.testing.assert_array_equal(
        symmetric.clean_dataset.baseline_states,
        random_walk.clean_dataset.baseline_states,
    )
    assert symmetric.split == random_walk.split
    assert not np.array_equal(
        symmetric.clean_dataset.laplacians,
        random_walk.clean_dataset.laplacians,
    )


def test_noise_fractions_share_base_noise_and_scale_training_only_std() -> None:
    config = _tiny_config()
    low = build_paired_synthetic_run_data(
        _condition(
            run_id="noise_0p01_seed_42",
            factor="noise",
            level="noise_0.01",
            noise_fraction=0.01,
        ),
        config,
    )
    high = build_paired_synthetic_run_data(
        _condition(
            run_id="noise_0p05_seed_42",
            factor="noise",
            level="noise_0.05",
            noise_fraction=0.05,
        ),
        config,
    )

    np.testing.assert_array_equal(low.base_noise, high.base_noise)
    assert high.noise_std == pytest.approx(5.0 * low.noise_std)
    clean = np.stack(low.clean_dataset.trajectories)
    low_delta = np.stack(low.observed_dataset.trajectories) - clean
    high_delta = np.stack(high.observed_dataset.trajectories) - clean
    np.testing.assert_allclose(high_delta, 5.0 * low_delta)
    expected_std = np.std(clean[list(low.split.train_indices)].ravel())
    assert low.clean_training_std == pytest.approx(expected_std)


def test_noisy_states_are_not_clipped() -> None:
    noisy = build_paired_synthetic_run_data(
        _condition(
            run_id="noise_10_seed_42",
            factor="noise",
            level="noise_10",
            noise_fraction=10.0,
        ),
        _tiny_config(),
    )

    observed = np.stack(noisy.observed_dataset.trajectories)
    assert np.any(observed < 0.0)
    assert noisy.negative_noisy_state_fraction > 0.0


def test_count_to_binary_preserves_zero_diagonal_symmetry_and_topology() -> None:
    adjacency = np.asarray([[0.0, 0.2, 0.0], [0.2, 0.0, 3.0], [0.0, 3.0, 0.0]])

    binary = count_to_binary_adjacency(adjacency)

    np.testing.assert_array_equal(
        binary,
        np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    )


def test_fixed_reference_grid_uses_prespecified_midpoint_domain() -> None:
    grid = fixed_reference_reaction_grid(_tiny_config(reference_grid_points=10))

    assert grid.shape == (10,)
    assert grid[0] > 0.05
    assert grid[-1] < 0.627
    assert np.all(np.diff(grid) > 0.0)


def test_training_fit_domain_uses_training_trajectories_only() -> None:
    data = build_paired_synthetic_run_data(_condition(), _tiny_config())
    trajectories = list(data.observed_dataset.trajectories)
    test_index = data.split.test_indices[0]
    trajectories[test_index] = np.full_like(trajectories[test_index], 99.0)
    poisoned = data.observed_dataset.__class__(
        subject_ids=data.observed_dataset.subject_ids,
        roi_labels=data.observed_dataset.roi_labels,
        baseline_states=data.observed_dataset.baseline_states,
        adjacencies=data.observed_dataset.adjacencies,
        laplacians=data.observed_dataset.laplacians,
        times_by_subject=data.observed_dataset.times_by_subject,
        trajectories=tuple(trajectories),
        representation=data.observed_dataset.representation,
        laplacian_kind=data.observed_dataset.laplacian_kind,
        rho=data.observed_dataset.rho,
        growth_rate=data.observed_dataset.growth_rate,
        saturation_rate=data.observed_dataset.saturation_rate,
        seed=data.observed_dataset.seed,
    )

    assert training_fit_domain(poisoned, data.split)[1] < 99.0


def test_run_condition_is_resumable_and_validates_fingerprints(tmp_path: Path) -> None:
    config = _tiny_config()
    calls = 0

    def fake_runner(model, run_data, config, run_dir):
        nonlocal calls
        calls += 1
        return _fake_symbolic_runner(model, run_data, config, run_dir)

    first = run_synthetic_robustness_condition(
        _condition(),
        tmp_path,
        config=config,
        symbolic_runner=fake_runner,
    )
    second = run_synthetic_robustness_condition(
        _condition(),
        tmp_path,
        config=config,
        symbolic_runner=fake_runner,
    )

    assert first["status"] == "complete"
    assert second["status"] == "complete"
    assert calls == 1
    with pytest.raises(SchemaError, match="fingerprints"):
        run_synthetic_robustness_condition(
            _condition(),
            tmp_path,
            config=_tiny_config(n_regions=5),
            symbolic_runner=fake_runner,
        )


def test_failed_symbolic_run_is_preserved(tmp_path: Path) -> None:
    def failing_runner(_model, _run_data, _config, _run_dir):
        raise RuntimeError("synthetic symbolic failure")

    result = run_synthetic_robustness_condition(
        _condition(),
        tmp_path,
        config=_tiny_config(),
        symbolic_runner=failing_runner,
    )

    assert result["status"] == "failed"
    assert result["failure_stage"] == "numerical"
    assert "synthetic symbolic failure" in result["error"]


def test_strict_symbolic_structure_and_threshold_classification() -> None:
    exact = {
        "is_polynomial": True,
        "constant": 0.0,
        "linear": 0.5,
        "quadratic": -0.5,
        "highest_degree": 2,
    }
    cubic = {**exact, "highest_degree": 3}

    exact_result = classify_polynomial_recovery(
        exact,
        true_linear=0.5,
        true_quadratic=-0.5,
    )
    cubic_result = classify_polynomial_recovery(
        cubic,
        true_linear=0.5,
        true_quadratic=-0.5,
    )

    assert exact_result["strict_symbolic_structure"] is True
    assert exact_result["joint_coefficient_within_10_percent"] is True
    assert exact_result["correct_sign"] is True
    assert cubic_result["strict_symbolic_structure"] is False
    assert cubic_result["has_higher_order_terms"] is True


def test_aggregation_and_table_rows_are_deterministic(tmp_path: Path) -> None:
    records = [
        run_synthetic_robustness_condition(
            _condition(run_id=f"seed_master_{seed}", master_seed=seed, level=f"master_seed_{seed}"),
            tmp_path / "results",
            config=_tiny_config(),
            symbolic_runner=_fake_symbolic_runner,
        )
        for seed in (41, 42)
    ]

    first = aggregate_synthetic_robustness(records, _tiny_config())
    second = aggregate_synthetic_robustness(records, _tiny_config())
    rows = robustness_table_rows(records)

    assert first == second
    assert first["completed_run_count"] == 2
    assert len(rows) == 2


def test_publication_artifacts_are_written_from_synthetic_rows(tmp_path: Path) -> None:
    record = run_synthetic_robustness_condition(
        _condition(),
        tmp_path / "results",
        config=_tiny_config(),
        symbolic_runner=_fake_symbolic_runner,
    )
    aggregate = aggregate_synthetic_robustness([record], _tiny_config())
    publication = tmp_path / "docs"

    write_publication_artifacts([record], aggregate, publication, config=_tiny_config())

    expected = {
        "synthetic_robustness_summary.json",
        "synthetic_robustness_table.csv",
        "synthetic_robustness_table.tex",
        "synthetic_robustness_report.md",
        "figure_1_noise_sensitivity.pdf",
        "figure_1_noise_sensitivity.png",
        "figure_2_graph_laplacian_sensitivity.pdf",
        "figure_2_graph_laplacian_sensitivity.png",
        "figure_3_coefficient_recovery.pdf",
        "figure_3_coefficient_recovery.png",
    }
    assert expected.issubset({path.name for path in publication.iterdir()})


def test_synthetic_robustness_module_does_not_import_real_data_paths() -> None:
    source = Path("src/tau_sr/synthetic_robustness.py").read_text(encoding="utf-8")

    forbidden = ["cohort_adapter", "pilot_adapter", "Tau_PET", "PrivateResearch", "Downloads"]
    assert not any(token in source for token in forbidden)
