from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import tau_sr.recovery as recovery
from tau_sr.reactions import make_linear_quadratic_reaction
from tau_sr.recovery import (
    SUPPORTED_CONNECTOME_REPRESENTATIONS,
    SUPPORTED_LAPLACIAN_KINDS,
    SyntheticRecoveryDataset,
    build_noise_free_recovery_dataset,
)
from tau_sr.schemas import SchemaError
from tau_sr.simulation import simulate_reaction_diffusion_cohort
from tau_sr.synthetic import SyntheticCohortConfig


def _small_config(**overrides: object) -> SyntheticCohortConfig:
    values = {
        "n_subjects": 3,
        "n_regions": 4,
        "visits_per_subject": 3,
        "seed": 42,
        "edge_probability": 0.5,
        "baseline_tau_min": 0.05,
        "baseline_tau_max": 0.4,
        "add_observation_noise": False,
        "observation_noise_std": 0.0,
    }
    values.update(overrides)
    return SyntheticCohortConfig(**values)  # type: ignore[arg-type]


def _unchecked_config(**overrides: object) -> SyntheticCohortConfig:
    values = {
        "n_subjects": 3,
        "n_regions": 4,
        "visits_per_subject": 3,
        "seed": 42,
        "edge_probability": 0.5,
        "baseline_tau_min": 0.05,
        "baseline_tau_max": 0.4,
        "add_observation_noise": False,
        "observation_noise_std": 0.0,
    }
    values.update(overrides)
    config = object.__new__(SyntheticCohortConfig)
    for name, value in values.items():
        object.__setattr__(config, name, value)
    return config


def _times_for_three_subjects() -> tuple[np.ndarray, ...]:
    return (
        np.array([0.0, 0.2, 0.5]),
        np.array([1.0, 1.3]),
        np.array([-0.5, 0.0, 0.25, 0.75]),
    )


def _build_small_dataset(**overrides: object) -> SyntheticRecoveryDataset:
    config = overrides.pop("config", _small_config())
    times_by_subject = overrides.pop("times_by_subject", _times_for_three_subjects())
    parameters = {
        "representation": "count",
        "laplacian_kind": "unnormalized",
        "rho": 0.1,
        "growth_rate": 0.5,
        "saturation_rate": 0.5,
        "max_step": 0.02,
    }
    parameters.update(overrides)
    return build_noise_free_recovery_dataset(
        config,  # type: ignore[arg-type]
        times_by_subject,  # type: ignore[arg-type]
        **parameters,  # type: ignore[arg-type]
    )


def _assert_dataset_equal(left: SyntheticRecoveryDataset, right: SyntheticRecoveryDataset) -> None:
    assert left.subject_ids == right.subject_ids
    assert left.roi_labels == right.roi_labels
    np.testing.assert_array_equal(left.baseline_states, right.baseline_states)
    np.testing.assert_array_equal(left.adjacencies, right.adjacencies)
    np.testing.assert_array_equal(left.laplacians, right.laplacians)
    for left_times, right_times in zip(left.times_by_subject, right.times_by_subject, strict=True):
        np.testing.assert_array_equal(left_times, right_times)
    for left_trajectory, right_trajectory in zip(
        left.trajectories,
        right.trajectories,
        strict=True,
    ):
        np.testing.assert_array_equal(left_trajectory, right_trajectory)
    assert left.representation == right.representation
    assert left.laplacian_kind == right.laplacian_kind
    assert left.rho == right.rho
    assert left.growth_rate == right.growth_rate
    assert left.saturation_rate == right.saturation_rate
    assert left.seed == right.seed


def test_build_noise_free_recovery_dataset_shapes_alignment_and_metadata() -> None:
    dataset = _build_small_dataset()

    assert isinstance(dataset, SyntheticRecoveryDataset)
    assert dataset.subject_ids == (
        "synthetic_subject_0000",
        "synthetic_subject_0001",
        "synthetic_subject_0002",
    )
    assert len(set(dataset.subject_ids)) == 3
    assert dataset.roi_labels == (
        "syn_roi_001",
        "syn_roi_002",
        "syn_roi_003",
        "syn_roi_004",
    )
    assert dataset.baseline_states.shape == (3, 4)
    assert dataset.adjacencies.shape == (3, 4, 4)
    assert dataset.laplacians.shape == (3, 4, 4)
    assert len(dataset.times_by_subject) == 3
    assert [trajectory.shape for trajectory in dataset.trajectories] == [
        (3, 4),
        (2, 4),
        (4, 4),
    ]
    for subject_index, trajectory in enumerate(dataset.trajectories):
        np.testing.assert_allclose(trajectory[0], dataset.baseline_states[subject_index])
        assert dataset.adjacencies[subject_index].shape == (4, 4)
        assert dataset.laplacians[subject_index].shape == (4, 4)
    assert dataset.representation == "count"
    assert dataset.laplacian_kind == "unnormalized"
    assert dataset.rho == 0.1
    assert dataset.growth_rate == 0.5
    assert dataset.saturation_rate == 0.5
    assert dataset.seed == 42


def test_recovery_dataset_container_is_frozen_by_attribute() -> None:
    dataset = _build_small_dataset()

    with pytest.raises(FrozenInstanceError):
        dataset.rho = 0.2  # type: ignore[misc]


def test_one_subject_dataset_works() -> None:
    dataset = _build_small_dataset(
        config=_small_config(n_subjects=1),
        times_by_subject=(np.array([0.0, 0.1]),),
    )

    assert dataset.subject_ids == ("synthetic_subject_0000",)
    assert dataset.baseline_states.shape == (1, 4)
    assert dataset.adjacencies.shape == (1, 4, 4)
    assert dataset.laplacians.shape == (1, 4, 4)
    assert len(dataset.trajectories) == 1
    assert dataset.trajectories[0].shape == (2, 4)


def test_recovery_dataset_148_region_smoke() -> None:
    dataset = build_noise_free_recovery_dataset(
        SyntheticCohortConfig(n_subjects=1, n_regions=148, visits_per_subject=2, seed=42),
        (np.array([0.0, 0.05]),),
        representation="count",
        laplacian_kind="symmetric_normalized",
        rho=0.1,
        growth_rate=0.5,
        saturation_rate=0.5,
        max_step=0.05,
    )

    assert dataset.baseline_states.shape == (1, 148)
    assert dataset.adjacencies.shape == (1, 148, 148)
    assert dataset.laplacians.shape == (1, 148, 148)
    assert dataset.trajectories[0].shape == (2, 148)
    assert np.isfinite(dataset.trajectories[0]).all()


def test_dataset_build_is_deterministic_for_identical_inputs() -> None:
    first = _build_small_dataset()
    second = _build_small_dataset()

    _assert_dataset_equal(first, second)


def test_changing_seed_changes_stochastic_components() -> None:
    first = _build_small_dataset(config=_small_config(seed=42))
    second = _build_small_dataset(config=_small_config(seed=43))

    assert not np.array_equal(first.baseline_states, second.baseline_states)
    assert not np.array_equal(first.adjacencies, second.adjacencies)


def test_dataset_build_does_not_mutate_numpy_legacy_global_rng_state() -> None:
    np.random.seed(123)
    expected_draws = np.random.random(5)
    np.random.seed(123)

    _build_small_dataset()

    np.testing.assert_array_equal(np.random.random(5), expected_draws)


def test_count_connectome_generation_is_invoked_once_per_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_generate_count_connectome(
        n_regions: int,
        rng: np.random.Generator,
        *,
        edge_probability: float,
    ) -> np.ndarray:
        calls.append(n_regions)
        matrix = np.zeros((n_regions, n_regions), dtype=float)
        scale = float(len(calls))
        for index in range(n_regions - 1):
            matrix[index, index + 1] = scale
            matrix[index + 1, index] = scale
        return matrix

    monkeypatch.setattr(recovery, "generate_count_connectome", fake_generate_count_connectome)

    dataset = _build_small_dataset()

    assert calls == [4, 4, 4]
    assert not np.array_equal(dataset.laplacians[0], dataset.laplacians[1])
    assert not np.array_equal(dataset.laplacians[1], dataset.laplacians[2])


def test_selected_adjacency_is_used_to_build_subject_laplacians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_adjacencies: list[np.ndarray] = []

    def fake_build_graph_laplacian(adjacency: np.ndarray, *, kind: str) -> np.ndarray:
        captured_adjacencies.append(adjacency.copy())
        return np.eye(adjacency.shape[0]) * len(captured_adjacencies)

    monkeypatch.setattr(recovery, "build_graph_laplacian", fake_build_graph_laplacian)

    dataset = _build_small_dataset(representation="fa_weighted")

    assert len(captured_adjacencies) == 3
    for subject_index, adjacency in enumerate(captured_adjacencies):
        np.testing.assert_array_equal(adjacency, dataset.adjacencies[subject_index])
        np.testing.assert_array_equal(
            dataset.laplacians[subject_index],
            np.eye(4) * (subject_index + 1),
        )


def test_cohort_simulator_receives_complete_laplacian_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_simulate_reaction_diffusion_cohort(
        baseline_states: np.ndarray,
        times_by_subject: tuple[np.ndarray, ...],
        laplacians: np.ndarray,
        rho: float,
        reaction_fn: object,
        *,
        max_step: float,
    ) -> tuple[np.ndarray, ...]:
        captured["baseline_states"] = baseline_states.copy()
        captured["laplacians"] = laplacians.copy()
        captured["rho"] = rho
        captured["max_step"] = max_step
        return tuple(
            np.tile(baseline_states[index], (times.shape[0], 1))
            for index, times in enumerate(times_by_subject)
        )

    monkeypatch.setattr(
        recovery,
        "simulate_reaction_diffusion_cohort",
        fake_simulate_reaction_diffusion_cohort,
    )

    dataset = _build_small_dataset(rho=0.25, max_step=0.03)

    np.testing.assert_array_equal(captured["laplacians"], dataset.laplacians)
    np.testing.assert_array_equal(captured["baseline_states"], dataset.baseline_states)
    assert captured["rho"] == 0.25
    assert captured["max_step"] == 0.03


@pytest.mark.parametrize("representation", SUPPORTED_CONNECTOME_REPRESENTATIONS)
def test_supported_representations_can_be_selected(representation: str) -> None:
    dataset = _build_small_dataset(representation=representation)

    assert dataset.representation == representation
    assert dataset.adjacencies.shape == (3, 4, 4)


def test_unsupported_representation_is_rejected_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_generate_count_connectome(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("generation should not start for invalid representation")

    monkeypatch.setattr(recovery, "generate_count_connectome", fail_generate_count_connectome)

    with pytest.raises(SchemaError, match="representation"):
        _build_small_dataset(representation="not_a_representation")


def test_representations_are_selected_from_existing_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_build_connectome_representations(
        count_matrix: np.ndarray,
        fa_matrix: np.ndarray,
        length_matrix: np.ndarray,
    ) -> dict[str, np.ndarray]:
        nonlocal calls
        calls += 1
        return {
            "count": np.full(count_matrix.shape, 1.0),
            "fa_weighted": np.full(count_matrix.shape, 2.0),
            "length_adjusted": np.full(count_matrix.shape, 3.0),
        }

    def fake_build_graph_laplacian(adjacency: np.ndarray, *, kind: str) -> np.ndarray:
        return adjacency.copy()

    monkeypatch.setattr(
        recovery,
        "build_connectome_representations",
        fake_build_connectome_representations,
    )
    monkeypatch.setattr(recovery, "build_graph_laplacian", fake_build_graph_laplacian)

    dataset = _build_small_dataset(representation="length_adjusted")

    assert calls == 3
    np.testing.assert_array_equal(dataset.adjacencies, np.full((3, 4, 4), 3.0))


@pytest.mark.parametrize("laplacian_kind", SUPPORTED_LAPLACIAN_KINDS)
def test_supported_laplacian_kinds_can_be_selected(laplacian_kind: str) -> None:
    dataset = _build_small_dataset(laplacian_kind=laplacian_kind)

    assert dataset.laplacian_kind == laplacian_kind
    assert dataset.laplacians.shape == (3, 4, 4)


def test_unsupported_laplacian_kind_is_rejected_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_generate_count_connectome(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("generation should not start for invalid laplacian kind")

    monkeypatch.setattr(recovery, "generate_count_connectome", fail_generate_count_connectome)

    with pytest.raises(SchemaError, match="laplacian_kind"):
        _build_small_dataset(laplacian_kind="not_a_laplacian")


def test_laplacians_are_built_through_existing_graph_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_build_graph_laplacian(adjacency: np.ndarray, *, kind: str) -> np.ndarray:
        calls.append(kind)
        return np.eye(adjacency.shape[0])

    monkeypatch.setattr(recovery, "build_graph_laplacian", fake_build_graph_laplacian)

    _build_small_dataset(laplacian_kind="symmetric_normalized")

    assert calls == ["symmetric_normalized", "symmetric_normalized", "symmetric_normalized"]


def test_reaction_factory_and_cohort_simulator_receive_exact_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_make_linear_quadratic_reaction(growth_rate: float, saturation_rate: float) -> object:
        captured["growth_rate"] = growth_rate
        captured["saturation_rate"] = saturation_rate
        return object()

    def fake_simulate_reaction_diffusion_cohort(
        baseline_states: np.ndarray,
        times_by_subject: tuple[np.ndarray, ...],
        laplacians: np.ndarray,
        rho: float,
        reaction_fn: object,
        *,
        max_step: float,
    ) -> tuple[np.ndarray, ...]:
        captured["rho"] = rho
        captured["max_step"] = max_step
        captured["reaction_fn"] = reaction_fn
        return tuple(
            np.tile(baseline_states[index], (times.shape[0], 1))
            for index, times in enumerate(times_by_subject)
        )

    monkeypatch.setattr(
        recovery,
        "make_linear_quadratic_reaction",
        fake_make_linear_quadratic_reaction,
    )
    monkeypatch.setattr(
        recovery,
        "simulate_reaction_diffusion_cohort",
        fake_simulate_reaction_diffusion_cohort,
    )

    _build_small_dataset(rho=0.25, growth_rate=0.6, saturation_rate=0.2, max_step=0.03)

    assert captured["growth_rate"] == 0.6
    assert captured["saturation_rate"] == 0.2
    assert captured["rho"] == 0.25
    assert captured["max_step"] == 0.03


def test_stored_trajectories_equal_direct_deterministic_simulator_output() -> None:
    dataset = _build_small_dataset()
    reaction_fn = make_linear_quadratic_reaction(dataset.growth_rate, dataset.saturation_rate)

    expected = simulate_reaction_diffusion_cohort(
        dataset.baseline_states,
        dataset.times_by_subject,
        dataset.laplacians,
        dataset.rho,
        reaction_fn,
        max_step=0.02,
    )

    for observed, expected_trajectory in zip(dataset.trajectories, expected, strict=True):
        np.testing.assert_allclose(observed, expected_trajectory)


def test_linear_control_dataset_works() -> None:
    dataset = _build_small_dataset(saturation_rate=0.0)

    assert dataset.saturation_rate == 0.0
    assert all(np.isfinite(trajectory).all() for trajectory in dataset.trajectories)


def test_dataset_owns_arrays_and_does_not_alias_caller_times() -> None:
    caller_times = list(_times_for_three_subjects())
    caller_before = [times.copy() for times in caller_times]

    dataset = _build_small_dataset(times_by_subject=caller_times)

    for observed, expected in zip(caller_times, caller_before, strict=True):
        np.testing.assert_array_equal(observed, expected)
    for stored, caller in zip(dataset.times_by_subject, caller_times, strict=True):
        assert not np.shares_memory(stored, caller)
    assert dataset.baseline_states.flags.owndata
    assert dataset.adjacencies.flags.owndata
    assert dataset.laplacians.flags.owndata
    assert all(trajectory.flags.owndata for trajectory in dataset.trajectories)
    assert not np.shares_memory(dataset.trajectories[0], dataset.trajectories[1])


def test_independently_built_datasets_do_not_alias_each_other() -> None:
    first = _build_small_dataset()
    second = _build_small_dataset()

    first.baseline_states[0, 0] = -999.0
    first.adjacencies[0, 0, 1] = -999.0
    first.laplacians[0, 0, 0] = -999.0
    first.trajectories[0][0, 0] = -999.0
    first.times_by_subject[0][0] = -999.0

    assert second.baseline_states[0, 0] != -999.0
    assert second.adjacencies[0, 0, 1] != -999.0
    assert second.laplacians[0, 0, 0] != -999.0
    assert second.trajectories[0][0, 0] != -999.0
    assert second.times_by_subject[0][0] != -999.0


def test_rejects_invalid_config_type() -> None:
    with pytest.raises(SchemaError, match="SyntheticCohortConfig"):
        build_noise_free_recovery_dataset(
            {"n_subjects": 3},  # type: ignore[arg-type]
            _times_for_three_subjects(),
            representation="count",
            laplacian_kind="unnormalized",
            rho=0.1,
            growth_rate=0.5,
            saturation_rate=0.5,
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_unchecked_config(n_subjects=0), "n_subjects"),
        (_unchecked_config(n_regions=1), "n_regions"),
        (_unchecked_config(n_regions=0), "n_regions"),
    ],
)
def test_rejects_invalid_config_values(config: SyntheticCohortConfig, message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        build_noise_free_recovery_dataset(
            config,
            _times_for_three_subjects(),
            representation="count",
            laplacian_kind="unnormalized",
            rho=0.1,
            growth_rate=0.5,
            saturation_rate=0.5,
        )


@pytest.mark.parametrize(
    ("times_by_subject", "message"),
    [
        ((np.array([0.0]),), "expected 3"),
        ((np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])), "expected 3"),
        ((np.array([0.0]), np.array([0.0, 0.0]), np.array([0.0])), r"times_by_subject\[1\]"),
    ],
)
def test_rejects_invalid_time_arrays(times_by_subject: object, message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        _build_small_dataset(times_by_subject=times_by_subject)


def test_invalid_later_time_array_is_rejected_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_generate_count_connectome(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("generation should not start when time arrays are invalid")

    monkeypatch.setattr(recovery, "generate_count_connectome", fail_generate_count_connectome)

    with pytest.raises(SchemaError, match=r"times_by_subject\[2\]"):
        _build_small_dataset(
            times_by_subject=(
                np.array([0.0, 0.1]),
                np.array([0.0, 0.2]),
                np.array([1.0, 1.0]),
            )
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"representation": "bad"}, "representation"),
        ({"laplacian_kind": "bad"}, "laplacian_kind"),
        ({"rho": -0.1}, "rho"),
        ({"rho": np.inf}, "rho"),
        ({"growth_rate": 0.0}, "growth_rate"),
        ({"growth_rate": np.nan}, "growth_rate"),
        ({"saturation_rate": -0.1}, "saturation_rate"),
        ({"saturation_rate": np.inf}, "saturation_rate"),
        ({"max_step": 0.0}, "max_step"),
        ({"max_step": np.nan}, "max_step"),
    ],
)
def test_rejects_invalid_public_parameters(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        _build_small_dataset(**kwargs)


def test_rejects_noise_enabled_configuration() -> None:
    config = _small_config(add_observation_noise=True, observation_noise_std=0.1)

    with pytest.raises(SchemaError, match="noise"):
        _build_small_dataset(config=config)


def test_known_validation_failure_does_not_mutate_numpy_legacy_rng_state() -> None:
    np.random.seed(123)
    expected_draws = np.random.random(5)
    np.random.seed(123)

    with pytest.raises(SchemaError, match="representation"):
        _build_small_dataset(representation="bad")

    np.testing.assert_array_equal(np.random.random(5), expected_draws)


def test_noise_free_builds_are_identical_and_do_not_call_global_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_normal(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("noise generation should not be called")

    monkeypatch.setattr(np.random, "normal", fail_normal)

    first = _build_small_dataset()
    second = _build_small_dataset()

    _assert_dataset_equal(first, second)
