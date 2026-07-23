from __future__ import annotations

import numpy as np
import pytest

from tau_sr.schemas import SchemaError
from tau_sr.simulation import simulate_reaction_diffusion_trajectory


def _zero_reaction(state: np.ndarray) -> np.ndarray:
    return np.zeros_like(state)


def _two_node_laplacian() -> np.ndarray:
    return np.array([[1.0, -1.0], [-1.0, 1.0]])


def _two_node_diffusion_solution(times: np.ndarray, rho: float) -> np.ndarray:
    delta_t = times - times[0]
    decay = np.exp(-2.0 * rho * delta_t)
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
