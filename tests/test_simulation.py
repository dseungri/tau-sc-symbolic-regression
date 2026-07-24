from __future__ import annotations

import numpy as np
import pytest

from tau_sr.schemas import SchemaError
from tau_sr.simulation import (
    simulate_reaction_diffusion_cohort,
    simulate_reaction_diffusion_trajectory,
)


def _zero_reaction(state: np.ndarray) -> np.ndarray:
    return np.zeros_like(state)


def _two_node_laplacian() -> np.ndarray:
    return np.array([[1.0, -1.0], [-1.0, 1.0]])


def _scaled_two_node_laplacian(scale: float) -> np.ndarray:
    return scale * _two_node_laplacian()


def _two_node_diffusion_solution(times: np.ndarray, rho: float) -> np.ndarray:
    delta_t = times - times[0]
    decay = np.exp(-2.0 * rho * delta_t)
    return np.column_stack((0.5 + 0.5 * decay, 0.5 - 0.5 * decay))


def _scaled_two_node_diffusion_solution(
    times: np.ndarray,
    rho: float,
    scale: float,
) -> np.ndarray:
    delta_t = times - times[0]
    decay = np.exp(-2.0 * rho * scale * delta_t)
    return np.column_stack((0.5 + 0.5 * decay, 0.5 - 0.5 * decay))


def test_zero_dynamics_remains_constant_and_allocates_float_output() -> None:
    y0 = np.array([1, 2, 3])
    times = np.array([0, 1, 2])

    trajectory = simulate_reaction_diffusion_trajectory(
        y0,
        times,
        np.zeros((3, 3), dtype=int),
        0,
        _zero_reaction,
    )

    assert trajectory.shape == (3, 3)
    assert np.issubdtype(trajectory.dtype, np.floating)
    for row in trajectory:
        np.testing.assert_array_equal(row, y0)
    assert not np.shares_memory(trajectory[0], y0)
    assert not np.shares_memory(trajectory, y0)


def test_simulates_148_region_trajectory_shape() -> None:
    y0 = np.linspace(0.05, 0.4, 148)
    times = np.array([0.0, 0.25, 0.75])

    trajectory = simulate_reaction_diffusion_trajectory(
        y0,
        times,
        np.zeros((148, 148)),
        0.0,
        _zero_reaction,
    )

    assert trajectory.shape == (3, 148)
    np.testing.assert_array_equal(trajectory[0], y0)


def test_repeated_calls_return_equal_independent_arrays() -> None:
    y0 = np.array([0.2, 0.4])
    times = np.array([0.0, 0.1])
    laplacian = np.zeros((2, 2))

    first = simulate_reaction_diffusion_trajectory(y0, times, laplacian, 0.0, _zero_reaction)
    second = simulate_reaction_diffusion_trajectory(y0, times, laplacian, 0.0, _zero_reaction)

    np.testing.assert_array_equal(first, second)
    assert first is not second
    assert not np.shares_memory(first, second)


@pytest.mark.parametrize("growth_rate", [0.35, -0.45])
def test_reaction_only_linear_solution_matches_exponential(growth_rate: float) -> None:
    y0 = np.array([0.3, 1.2, -0.4])
    times = np.array([1.0, 1.3, 2.0])

    trajectory = simulate_reaction_diffusion_trajectory(
        y0,
        times,
        np.zeros((3, 3)),
        0.0,
        lambda state: growth_rate * state,
        max_step=0.005,
    )

    expected = y0 * np.exp(growth_rate * (times - times[0]))[:, None]
    np.testing.assert_allclose(trajectory, expected, rtol=1e-9, atol=1e-11)


def test_diffusion_only_two_node_solution_matches_analytical_formula() -> None:
    rho = 0.4
    times = np.array([0.0, 0.3, 0.8])

    trajectory = simulate_reaction_diffusion_trajectory(
        np.array([1.0, 0.0]),
        times,
        _two_node_laplacian(),
        rho,
        _zero_reaction,
        max_step=0.0025,
    )

    np.testing.assert_allclose(
        trajectory,
        _two_node_diffusion_solution(times, rho),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(trajectory.sum(axis=1), 1.0, atol=1e-12)


def test_spatially_constant_state_remains_constant_under_diffusion_only() -> None:
    y0 = np.array([0.7, 0.7])
    times = np.array([0.0, 0.5, 1.0])

    trajectory = simulate_reaction_diffusion_trajectory(
        y0,
        times,
        _two_node_laplacian(),
        0.8,
        _zero_reaction,
    )

    np.testing.assert_allclose(trajectory, np.tile(y0, (3, 1)), atol=1e-14)


def test_combined_linear_reaction_and_diffusion_matches_exact_two_node_solution() -> None:
    rho = 0.25
    growth_rate = 0.15
    times = np.array([2.0, 2.25, 2.75])

    trajectory = simulate_reaction_diffusion_trajectory(
        np.array([1.0, 0.0]),
        times,
        _two_node_laplacian(),
        rho,
        lambda state: growth_rate * state,
        max_step=0.0025,
    )

    reaction_scale = np.exp(growth_rate * (times - times[0]))[:, None]
    expected = _two_node_diffusion_solution(times, rho) * reaction_scale
    np.testing.assert_allclose(trajectory, expected, rtol=1e-10, atol=1e-12)


def test_irregular_nonzero_and_negative_requested_times_are_supported() -> None:
    y0 = np.array([1.0, -3.0])
    derivative = np.array([0.25, -0.5])
    times = np.array([-2.0, -1.25, 0.0, 0.4])

    trajectory = simulate_reaction_diffusion_trajectory(
        y0,
        times,
        np.zeros((2, 2)),
        0.0,
        lambda _state: derivative.copy(),
        max_step=0.05,
    )

    expected = y0 + (times - times[0])[:, None] * derivative
    np.testing.assert_allclose(trajectory, expected, atol=1e-12)


def test_one_element_time_array_returns_initial_state_without_reaction_call() -> None:
    calls = 0

    def reaction_fn(state: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros_like(state)

    y0 = np.array([0.1, 0.2, 0.3])
    trajectory = simulate_reaction_diffusion_trajectory(
        y0,
        np.array([5.0]),
        np.zeros((3, 3)),
        0.0,
        reaction_fn,
    )

    assert trajectory.shape == (1, 3)
    np.testing.assert_array_equal(trajectory[0], y0)
    assert not np.shares_memory(trajectory, y0)
    assert calls == 0


def test_internal_rk4_substeps_do_not_appear_as_output_rows() -> None:
    trajectory = simulate_reaction_diffusion_trajectory(
        np.array([0.0]),
        np.array([0.0, 0.12]),
        np.zeros((1, 1)),
        0.0,
        lambda _state: np.array([1.0]),
        max_step=0.05,
    )

    assert trajectory.shape == (2, 1)
    np.testing.assert_allclose(trajectory[:, 0], np.array([0.0, 0.12]))


def test_smaller_max_step_improves_accuracy_without_changing_output_shape() -> None:
    y0 = np.array([1.0])
    times = np.array([0.0, 0.4, 1.0])

    coarse = simulate_reaction_diffusion_trajectory(
        y0,
        times,
        np.zeros((1, 1)),
        0.0,
        lambda state: state.copy(),
        max_step=0.25,
    )
    fine = simulate_reaction_diffusion_trajectory(
        y0,
        times,
        np.zeros((1, 1)),
        0.0,
        lambda state: state.copy(),
        max_step=0.05,
    )

    expected = np.exp(times)[:, None]
    assert coarse.shape == fine.shape == (3, 1)
    np.testing.assert_array_equal(coarse[0], y0)
    np.testing.assert_array_equal(fine[0], y0)
    assert abs(fine[-1, 0] - expected[-1, 0]) < abs(coarse[-1, 0] - expected[-1, 0])


def test_simulation_applies_no_clipping_or_nonnegativity_projection() -> None:
    trajectory = simulate_reaction_diffusion_trajectory(
        np.array([0.3, 0.2]),
        np.array([0.0, 1.0]),
        np.zeros((2, 2)),
        0.0,
        lambda _state: np.array([1.1, -1.1]),
    )

    assert trajectory[-1, 0] > 1.0
    assert trajectory[-1, 1] < 0.0
    np.testing.assert_allclose(trajectory[-1], np.array([1.4, -0.9]))


def test_preserves_inputs_and_protects_against_reaction_state_mutation() -> None:
    y0 = np.array([1.0, 2.0])
    times = np.array([0.0, 0.1, 0.2])
    laplacian = np.zeros((2, 2))
    y0_before = y0.copy()
    times_before = times.copy()
    laplacian_before = laplacian.copy()

    def mutating_reaction(state: np.ndarray) -> np.ndarray:
        state[:] = -1000.0
        return np.array([1.0, -1.0])

    trajectory = simulate_reaction_diffusion_trajectory(
        y0,
        times,
        laplacian,
        0.0,
        mutating_reaction,
        max_step=0.05,
    )

    expected = y0_before + times[:, None] * np.array([1.0, -1.0])
    np.testing.assert_array_equal(y0, y0_before)
    np.testing.assert_array_equal(times, times_before)
    np.testing.assert_array_equal(laplacian, laplacian_before)
    np.testing.assert_allclose(trajectory, expected)
    assert all(not np.shares_memory(row, y0) for row in trajectory)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"y0": [1.0, 2.0]}, "y0"),
        ({"times": np.array([0.0, 0.0])}, "times"),
        ({"laplacian": [[0.0, 0.0], [0.0, 0.0]]}, "laplacian"),
        ({"rho": -0.1}, "rho"),
        ({"rho": np.inf}, "rho"),
        ({"reaction_fn": None}, "reaction_fn"),
        ({"max_step": 0.0}, "max_step"),
    ],
)
def test_representative_invalid_inputs_raise_schema_error(
    kwargs: dict[str, object],
    message: str,
) -> None:
    inputs = {
        "y0": np.array([1.0, 2.0]),
        "times": np.array([0.0, 0.1]),
        "laplacian": np.zeros((2, 2)),
        "rho": 0.0,
        "reaction_fn": _zero_reaction,
        "max_step": 0.05,
    }
    inputs.update(kwargs)

    with pytest.raises(SchemaError, match=message):
        simulate_reaction_diffusion_trajectory(**inputs)  # type: ignore[arg-type]


def test_state_laplacian_dimension_mismatch_raises_schema_error() -> None:
    with pytest.raises(SchemaError, match="expected 3"):
        simulate_reaction_diffusion_trajectory(
            np.array([1.0, 2.0]),
            np.array([0.0, 0.1]),
            np.eye(3),
            0.1,
            _zero_reaction,
        )


def test_invalid_reaction_output_raises_schema_error() -> None:
    with pytest.raises(SchemaError, match="reaction_fn output"):
        simulate_reaction_diffusion_trajectory(
            np.array([1.0, 2.0]),
            np.array([0.0, 0.1]),
            np.zeros((2, 2)),
            0.0,
            lambda _state: np.ones((1, 2)),
        )


def test_invalid_top_level_inputs_do_not_call_reaction_function() -> None:
    calls = 0

    def reaction_fn(state: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros_like(state)

    with pytest.raises(SchemaError, match="times"):
        simulate_reaction_diffusion_trajectory(
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
            np.zeros((2, 2)),
            0.0,
            reaction_fn,
        )

    assert calls == 0


def test_deterministic_inputs_produce_deterministic_outputs() -> None:
    y0 = np.array([1.0, 0.0])
    times = np.array([0.0, 0.2, 0.7])
    laplacian = _two_node_laplacian()

    def reaction_fn(state: np.ndarray) -> np.ndarray:
        return 0.1 * state

    first = simulate_reaction_diffusion_trajectory(
        y0,
        times,
        laplacian,
        0.2,
        reaction_fn,
        max_step=0.01,
    )
    second = simulate_reaction_diffusion_trajectory(
        y0,
        times,
        laplacian,
        0.2,
        reaction_fn,
        max_step=0.01,
    )

    np.testing.assert_array_equal(first, second)


def test_simulation_does_not_mutate_numpy_legacy_global_rng_state() -> None:
    np.random.seed(123)
    expected_draws = np.random.random(5)
    np.random.seed(123)

    simulate_reaction_diffusion_trajectory(
        np.array([1.0, 0.0]),
        np.array([0.0, 0.1]),
        _two_node_laplacian(),
        0.1,
        _zero_reaction,
    )

    np.testing.assert_array_equal(np.random.random(5), expected_draws)


def test_cohort_basic_behavior_preserves_subject_order_and_visit_counts() -> None:
    baseline_states = np.array(
        [
            [1.0, 0.0],
            [2.0, 2.0],
            [0.0, 1.0],
        ]
    )
    times_by_subject = (
        np.array([0.0, 0.2]),
        np.array([1.0, 1.5, 2.0]),
        np.array([-1.0]),
    )

    trajectories = simulate_reaction_diffusion_cohort(
        baseline_states,
        times_by_subject,
        np.zeros((3, 2, 2)),
        0.0,
        _zero_reaction,
    )

    assert len(trajectories) == 3
    assert [trajectory.shape for trajectory in trajectories] == [(2, 2), (3, 2), (1, 2)]
    for subject_index, trajectory in enumerate(trajectories):
        np.testing.assert_array_equal(trajectory[0], baseline_states[subject_index])
        np.testing.assert_allclose(
            trajectory,
            np.tile(baseline_states[subject_index], (trajectory.shape[0], 1)),
        )


def test_one_subject_cohort_works() -> None:
    trajectories = simulate_reaction_diffusion_cohort(
        np.array([[0.1, 0.2]]),
        (np.array([0.0, 0.5]),),
        np.zeros((1, 2, 2)),
        0.0,
        lambda state: np.ones_like(state),
    )

    assert len(trajectories) == 1
    np.testing.assert_allclose(trajectories[0], np.array([[0.1, 0.2], [0.6, 0.7]]))


def test_one_visit_cohort_subject_does_not_call_reaction_for_that_subject() -> None:
    calls: list[float] = []

    def reaction_fn(state: np.ndarray) -> np.ndarray:
        calls.append(float(state[0]))
        return np.zeros_like(state)

    trajectories = simulate_reaction_diffusion_cohort(
        np.array([[0.0, 0.0], [10.0, 10.0]]),
        (np.array([0.0, 0.1]), np.array([5.0])),
        np.zeros((2, 2, 2)),
        0.0,
        reaction_fn,
    )

    np.testing.assert_array_equal(trajectories[1], np.array([[10.0, 10.0]]))
    assert calls
    assert all(call == 0.0 for call in calls)


def test_cohort_accepts_fixed_visit_time_matrix_and_148_regions() -> None:
    baseline_states = np.vstack(
        [
            np.linspace(0.05, 0.40, 148),
            np.linspace(0.10, 0.45, 148),
        ]
    )

    trajectories = simulate_reaction_diffusion_cohort(
        baseline_states,
        np.array([[0.0, 0.25], [1.0, 1.25]]),
        np.zeros((2, 148, 148)),
        0.0,
        _zero_reaction,
    )

    assert len(trajectories) == 2
    assert trajectories[0].shape == (2, 148)
    assert trajectories[1].shape == (2, 148)
    np.testing.assert_array_equal(trajectories[0][0], baseline_states[0])
    np.testing.assert_array_equal(trajectories[1][0], baseline_states[1])


def test_cohort_uses_subject_specific_laplacians_in_order() -> None:
    baseline_states = np.tile(np.array([1.0, 0.0]), (3, 1))
    times_by_subject = (np.array([0.0, 1.0]),) * 3
    laplacians = np.stack(
        [
            _scaled_two_node_laplacian(1.0),
            _scaled_two_node_laplacian(2.0),
            _scaled_two_node_laplacian(3.0),
        ]
    )
    rho = 0.2

    trajectories = simulate_reaction_diffusion_cohort(
        baseline_states,
        times_by_subject,
        laplacians,
        rho,
        _zero_reaction,
        max_step=0.0025,
    )

    for scale, trajectory in zip((1.0, 2.0, 3.0), trajectories, strict=True):
        np.testing.assert_allclose(
            trajectory,
            _scaled_two_node_diffusion_solution(times_by_subject[0], rho, scale),
            rtol=1e-10,
            atol=1e-12,
        )
    assert not np.allclose(trajectories[0], trajectories[1])
    assert not np.allclose(trajectories[1], trajectories[2])


@pytest.mark.parametrize(
    ("laplacians", "message"),
    [
        (np.zeros((2, 2)), "three-dimensional"),
        (np.zeros((1, 2, 2)), "expected"),
        (np.zeros((2, 3, 3)), "expected"),
        (np.zeros((2, 2, 3)), "expected"),
        (np.zeros((2, 2, 2), dtype=bool), "real numeric"),
        (np.zeros((2, 2, 2), dtype=complex), "real numeric"),
        (np.array([[[0.0, np.nan], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]), "finite"),
    ],
)
def test_cohort_rejects_invalid_laplacian_inputs(
    laplacians: object,
    message: str,
) -> None:
    with pytest.raises(SchemaError, match=message):
        simulate_reaction_diffusion_cohort(
            np.ones((2, 2)),
            (np.array([0.0]), np.array([0.0])),
            laplacians,  # type: ignore[arg-type]
            0.0,
            _zero_reaction,
        )


def test_cohort_supports_different_visit_schedules_without_extra_rows() -> None:
    times_by_subject = (
        np.array([-1.0, -0.3, 0.4]),
        np.array([2.0, 2.12]),
        np.array([10.0, 10.03, 10.17, 10.40]),
    )

    trajectories = simulate_reaction_diffusion_cohort(
        np.zeros((3, 2)),
        times_by_subject,
        np.zeros((3, 2, 2)),
        0.0,
        lambda state: np.ones_like(state),
        max_step=0.05,
    )

    assert [trajectory.shape for trajectory in trajectories] == [(3, 2), (2, 2), (4, 2)]
    for times, trajectory in zip(times_by_subject, trajectories, strict=True):
        expected = (times - times[0])[:, None] * np.ones(2)
        np.testing.assert_allclose(trajectory, expected, atol=1e-12)


def test_cohort_reaction_only_linear_solution_matches_exponential() -> None:
    growth_rate = 0.25
    baseline_states = np.array([[1.0, 2.0], [0.5, 1.5], [3.0, -1.0]])
    times_by_subject = (
        np.array([0.0, 0.25, 1.0]),
        np.array([2.0, 2.5]),
        np.array([-1.0, -0.5, 0.25]),
    )

    trajectories = simulate_reaction_diffusion_cohort(
        baseline_states,
        times_by_subject,
        np.zeros((3, 2, 2)),
        0.0,
        lambda state: growth_rate * state,
        max_step=0.005,
    )

    for y0, times, trajectory in zip(baseline_states, times_by_subject, trajectories, strict=True):
        expected = y0 * np.exp(growth_rate * (times - times[0]))[:, None]
        np.testing.assert_allclose(trajectory, expected, rtol=1e-9, atol=1e-11)


def test_cohort_diffusion_only_matches_analytical_solution_and_conserves_mass() -> None:
    rho = 0.3
    times_by_subject = (np.array([0.0, 0.4]), np.array([1.0, 1.2, 1.8]))
    scales = (1.0, 0.5)

    trajectories = simulate_reaction_diffusion_cohort(
        np.tile(np.array([1.0, 0.0]), (2, 1)),
        times_by_subject,
        np.stack([_scaled_two_node_laplacian(scale) for scale in scales]),
        rho,
        _zero_reaction,
        max_step=0.0025,
    )

    for scale, times, trajectory in zip(scales, times_by_subject, trajectories, strict=True):
        np.testing.assert_allclose(
            trajectory,
            _scaled_two_node_diffusion_solution(times, rho, scale),
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(trajectory.sum(axis=1), 1.0, atol=1e-12)


def test_cohort_combined_linear_reaction_and_diffusion_matches_exact_solution() -> None:
    rho = 0.25
    growth_rate = 0.15
    times_by_subject = (np.array([0.0, 0.5]), np.array([2.0, 2.4, 2.8]))
    scales = (1.0, 2.0)

    trajectories = simulate_reaction_diffusion_cohort(
        np.tile(np.array([1.0, 0.0]), (2, 1)),
        times_by_subject,
        np.stack([_scaled_two_node_laplacian(scale) for scale in scales]),
        rho,
        lambda state: growth_rate * state,
        max_step=0.0025,
    )

    for scale, times, trajectory in zip(scales, times_by_subject, trajectories, strict=True):
        reaction_scale = np.exp(growth_rate * (times - times[0]))[:, None]
        expected = _scaled_two_node_diffusion_solution(times, rho, scale) * reaction_scale
        np.testing.assert_allclose(trajectory, expected, rtol=1e-10, atol=1e-12)


def test_cohort_uses_shared_reaction_callable_for_every_subject() -> None:
    seen_baselines: list[float] = []

    def reaction_fn(state: np.ndarray) -> np.ndarray:
        seen_baselines.append(float(state[0]))
        return np.zeros_like(state)

    simulate_reaction_diffusion_cohort(
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        (np.array([0.0, 0.1]), np.array([0.0, 0.1])),
        np.zeros((2, 2, 2)),
        0.0,
        reaction_fn,
    )

    assert 1.0 in seen_baselines
    assert 2.0 in seen_baselines


def test_cohort_preserves_inputs_and_allocates_independent_outputs_under_mutation() -> None:
    baseline_states = np.array([[1.0, 2.0], [3.0, 4.0]])
    laplacians = np.zeros((2, 2, 2))
    times_by_subject = [np.array([0.0, 0.1]), np.array([1.0, 1.2])]
    baseline_before = baseline_states.copy()
    laplacians_before = laplacians.copy()
    times_before = [times.copy() for times in times_by_subject]

    def mutating_reaction(state: np.ndarray) -> np.ndarray:
        state[:] = -1000.0
        return np.array([1.0, -1.0])

    trajectories = simulate_reaction_diffusion_cohort(
        baseline_states,
        times_by_subject,
        laplacians,
        0.0,
        mutating_reaction,
        max_step=0.05,
    )

    np.testing.assert_array_equal(baseline_states, baseline_before)
    np.testing.assert_array_equal(laplacians, laplacians_before)
    for times, expected_times in zip(times_by_subject, times_before, strict=True):
        np.testing.assert_array_equal(times, expected_times)
    assert all(not np.shares_memory(trajectory, baseline_states) for trajectory in trajectories)
    assert not np.shares_memory(trajectories[0], trajectories[1])
    for y0, times, trajectory in zip(baseline_before, times_before, trajectories, strict=True):
        expected = y0 + (times - times[0])[:, None] * np.array([1.0, -1.0])
        np.testing.assert_allclose(trajectory, expected)


def test_cohort_is_deterministic_and_does_not_mutate_numpy_legacy_rng_state() -> None:
    baseline_states = np.array([[1.0, 0.0], [0.0, 1.0]])
    times_by_subject = (np.array([0.0, 0.2]), np.array([1.0, 1.4]))
    laplacians = np.stack([_two_node_laplacian(), _scaled_two_node_laplacian(2.0)])

    np.random.seed(456)
    expected_draws = np.random.random(5)
    np.random.seed(456)

    first = simulate_reaction_diffusion_cohort(
        baseline_states,
        times_by_subject,
        laplacians,
        0.2,
        lambda state: 0.1 * state,
        max_step=0.01,
    )
    second = simulate_reaction_diffusion_cohort(
        baseline_states,
        times_by_subject,
        laplacians,
        0.2,
        lambda state: 0.1 * state,
        max_step=0.01,
    )

    for first_trajectory, second_trajectory in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_trajectory, second_trajectory)
    np.testing.assert_array_equal(np.random.random(5), expected_draws)


def test_cohort_applies_no_clipping_nonnegativity_projection_or_noise() -> None:
    inputs = (
        np.array([[0.3, 0.2], [0.4, 0.1]]),
        (np.array([0.0, 1.0]), np.array([2.0, 3.0])),
        np.zeros((2, 2, 2)),
        0.0,
        lambda _state: np.array([1.1, -1.1]),
    )

    first = simulate_reaction_diffusion_cohort(*inputs)
    second = simulate_reaction_diffusion_cohort(*inputs)

    for first_trajectory, second_trajectory in zip(first, second, strict=True):
        assert first_trajectory[-1, 0] > 1.0
        assert first_trajectory[-1, 1] < 0.0
        np.testing.assert_array_equal(first_trajectory, second_trajectory)


@pytest.mark.parametrize(
    ("baseline_states", "message"),
    [
        ([[1.0, 2.0]], "NumPy"),
        (np.array([1.0, 2.0]), "two-dimensional"),
        (np.zeros((0, 2)), "one subject"),
        (np.zeros((2, 1)), "two regions"),
        (np.array([[True, False], [False, True]]), "real numeric"),
        (np.array([["1.0", "2.0"], ["3.0", "4.0"]]), "real numeric"),
        (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=object), "real numeric"),
        (np.array([[1.0 + 0.0j, 2.0], [3.0, 4.0]]), "real numeric"),
        (np.array([[1.0, np.nan], [3.0, 4.0]]), "finite"),
        (np.array([[1.0, np.inf], [3.0, 4.0]]), "finite"),
    ],
)
def test_cohort_rejects_invalid_baseline_states(
    baseline_states: object,
    message: str,
) -> None:
    with pytest.raises(SchemaError, match=message):
        simulate_reaction_diffusion_cohort(
            baseline_states,  # type: ignore[arg-type]
            (np.array([0.0]), np.array([0.0])),
            np.zeros((2, 2, 2)),
            0.0,
            _zero_reaction,
        )


@pytest.mark.parametrize(
    ("times_by_subject", "message"),
    [
        ((np.array([0.0]),), "expected 2"),
        ((np.array([0.0]), np.array([0.0]), np.array([0.0])), "expected 2"),
        ("not-times", "times_by_subject"),
        (np.array([0.0, 1.0]), "two-dimensional"),
        ((np.array([0.0]), [0.0]), r"times_by_subject\[1\]"),
        ((np.array([0.0]), np.array([[0.0, 1.0]])), r"times_by_subject\[1\]"),
        ((np.array([0.0]), np.array([], dtype=float)), r"times_by_subject\[1\]"),
        ((np.array([0.0]), np.array([False, True])), r"times_by_subject\[1\]"),
        ((np.array([0.0]), np.array([0.0, np.nan])), r"times_by_subject\[1\]"),
        ((np.array([0.0]), np.array([0.0, 0.0])), r"times_by_subject\[1\]"),
        ((np.array([0.0]), np.array([1.0, 0.0])), r"times_by_subject\[1\]"),
    ],
)
def test_cohort_rejects_invalid_times_by_subject(
    times_by_subject: object,
    message: str,
) -> None:
    with pytest.raises(SchemaError, match=message):
        simulate_reaction_diffusion_cohort(
            np.ones((2, 2)),
            times_by_subject,  # type: ignore[arg-type]
            np.zeros((2, 2, 2)),
            0.0,
            _zero_reaction,
        )


def test_cohort_outer_validation_happens_before_reaction_calls() -> None:
    calls = 0

    def reaction_fn(state: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros_like(state)

    with pytest.raises(SchemaError, match=r"times_by_subject\[1\]"):
        simulate_reaction_diffusion_cohort(
            np.ones((2, 2)),
            (np.array([0.0, 0.1]), np.array([0.0, 0.0])),
            np.zeros((2, 2, 2)),
            0.0,
            reaction_fn,
        )

    assert calls == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rho": -0.1}, "rho"),
        ({"rho": np.inf}, "rho"),
        ({"rho": True}, "rho"),
        ({"reaction_fn": None}, "reaction_fn"),
        ({"max_step": 0.0}, "max_step"),
        ({"max_step": np.nan}, "max_step"),
        ({"max_step": False}, "max_step"),
    ],
)
def test_cohort_rejects_invalid_shared_parameters(
    kwargs: dict[str, object],
    message: str,
) -> None:
    inputs = {
        "baseline_states": np.ones((2, 2)),
        "times_by_subject": (np.array([0.0]), np.array([0.0])),
        "laplacians": np.zeros((2, 2, 2)),
        "rho": 0.0,
        "reaction_fn": _zero_reaction,
        "max_step": 0.05,
    }
    inputs.update(kwargs)

    with pytest.raises(SchemaError, match=message):
        simulate_reaction_diffusion_cohort(**inputs)  # type: ignore[arg-type]


def test_cohort_invalid_reaction_output_reports_subject_index() -> None:
    def reaction_fn(state: np.ndarray) -> np.ndarray:
        if state[0] > 5.0:
            return np.ones((1, 2))
        return np.zeros_like(state)

    with pytest.raises(SchemaError, match=r"subject index 1.*reaction_fn output"):
        simulate_reaction_diffusion_cohort(
            np.array([[1.0, 0.0], [10.0, 0.0]]),
            (np.array([0.0, 0.1]), np.array([0.0, 0.1])),
            np.zeros((2, 2, 2)),
            0.0,
            reaction_fn,
        )
