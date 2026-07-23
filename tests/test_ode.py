from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

import tau_sr.ode as ode
from tau_sr.ode import integrate_rk4, rk4_step
from tau_sr.schemas import SchemaError


def _constant_rhs(derivative: np.ndarray) -> Callable[[float, np.ndarray], np.ndarray]:
    def rhs(_t: float, _y: np.ndarray) -> np.ndarray:
        return derivative.copy()

    return rhs


def test_rk4_step_constant_vector_derivative_is_exact() -> None:
    y = np.array([0.0, 1.0, 2.0])
    derivative = np.array([1.0, -2.0, 0.5])

    result = rk4_step(_constant_rhs(derivative), 2.0, y, 0.25)

    np.testing.assert_allclose(result, y + 0.25 * derivative)


def test_rk4_step_time_dependent_polynomial_derivative_is_exact() -> None:
    def rhs(t: float, _y: np.ndarray) -> np.ndarray:
        return np.array([t**3, 2.0 * t])

    t = 1.5
    h = 0.2
    y = np.array([3.0, -1.0])
    expected = y + np.array(
        [
            ((t + h) ** 4 - t**4) / 4.0,
            (t + h) ** 2 - t**2,
        ]
    )

    result = rk4_step(rhs, t, y, h)

    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


def test_rk4_step_exponential_growth_is_accurate() -> None:
    y = np.array([1.0, 2.0])
    h = 0.1

    result = rk4_step(lambda _t, state: state.copy(), 0.0, y, h)

    np.testing.assert_allclose(result, y * math.exp(h), rtol=1e-6, atol=1e-8)


def test_rk4_step_multidimensional_components_evolve_independently() -> None:
    rates = np.array([1.0, -0.5, 2.0])
    y = np.array([1.0, 4.0, 0.25])

    result = rk4_step(lambda _t, state: rates * state, 0.0, y, 0.05)

    np.testing.assert_allclose(result, y * np.exp(rates * 0.05), rtol=1e-6, atol=1e-8)


def test_rk4_step_allocates_float_output_and_preserves_input() -> None:
    y = np.array([1, 2, 3])
    y_before = y.copy()

    result = rk4_step(_constant_rhs(np.array([0.5, 0.5, 0.5])), 0.0, y, 0.2)

    assert result.shape == (3,)
    assert np.issubdtype(result.dtype, np.floating)
    assert not np.shares_memory(result, y)
    np.testing.assert_array_equal(y, y_before)
    np.testing.assert_allclose(result, np.array([1.1, 2.1, 3.1]))


def test_rk4_step_uses_classical_stage_times_and_states() -> None:
    calls: list[tuple[float, np.ndarray]] = []

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        calls.append((t, state.copy()))
        return np.array([1.0, 2.0])

    y = np.array([10.0, 20.0])

    result = rk4_step(rhs, 3.0, y, 0.4)

    assert len(calls) == 4
    np.testing.assert_allclose([call[0] for call in calls], [3.0, 3.2, 3.2, 3.4])
    expected_states = [
        np.array([10.0, 20.0]),
        np.array([10.2, 20.4]),
        np.array([10.2, 20.4]),
        np.array([10.4, 20.8]),
    ]
    for (_time, state), expected in zip(calls, expected_states, strict=True):
        np.testing.assert_allclose(state, expected)
    np.testing.assert_allclose(result, np.array([10.4, 20.8]))


def test_rk4_step_uses_prior_stage_derivatives_for_stage_states() -> None:
    calls: list[tuple[float, np.ndarray]] = []

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        calls.append((t, state.copy()))
        return state.copy()

    rk4_step(rhs, t=1.0, y=np.array([2.0]), step_size=0.2)

    np.testing.assert_allclose([call[0] for call in calls], [1.0, 1.1, 1.1, 1.2])
    expected_states = [
        np.array([2.0]),
        np.array([2.2]),
        np.array([2.22]),
        np.array([2.444]),
    ]
    for (_time, state), expected in zip(calls, expected_states, strict=True):
        np.testing.assert_allclose(state, expected, atol=1e-14)


def test_rk4_step_calls_rhs_exactly_four_times() -> None:
    calls = 0

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros_like(state)

    rk4_step(rhs, 0.0, np.array([1.0, 2.0]), 0.1)

    assert calls == 4


def test_integrate_rk4_constant_derivative_irregular_and_negative_times() -> None:
    y0 = np.array([1.0, -2.0])
    derivative = np.array([2.0, -1.0])
    times = np.array([-1.0, -0.25, 0.25, 1.0])

    trajectory = integrate_rk4(_constant_rhs(derivative), y0, times, max_step=0.2)

    expected = y0 + (times - times[0])[:, None] * derivative
    assert trajectory.shape == (4, 2)
    np.testing.assert_allclose(trajectory, expected)
    assert not np.shares_memory(trajectory[0], y0)


def test_integrate_rk4_exponential_growth_is_accurate() -> None:
    y0 = np.array([1.0])
    times = np.array([0.0, 0.1, 0.4, 1.0])

    trajectory = integrate_rk4(lambda _t, state: state.copy(), y0, times, max_step=0.01)

    np.testing.assert_allclose(trajectory[:, 0], np.exp(times), rtol=1e-9, atol=1e-11)


def test_integrate_rk4_multidimensional_vector_ode() -> None:
    rates = np.array([0.5, -1.0, 2.0])
    y0 = np.array([2.0, 3.0, 0.5])
    times = np.array([1.5, 1.7, 2.1])

    trajectory = integrate_rk4(lambda _t, state: rates * state, y0, times, max_step=0.005)

    expected = y0 * np.exp((times - times[0])[:, None] * rates)
    np.testing.assert_allclose(trajectory, expected, rtol=1e-9, atol=1e-11)


def test_integrate_rk4_harmonic_oscillator_returns_near_initial_state() -> None:
    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -state[0]])

    trajectory = integrate_rk4(
        rhs,
        np.array([1.0, 0.0]),
        np.array([0.0, 2.0 * math.pi]),
        max_step=0.01,
    )

    np.testing.assert_allclose(trajectory[-1], np.array([1.0, 0.0]), rtol=0.0, atol=1e-8)


def test_integrate_rk4_error_decreases_at_fourth_order_rate() -> None:
    times = np.array([0.0, 1.0])
    y0 = np.array([1.0])

    coarse = integrate_rk4(lambda _t, state: state.copy(), y0, times, max_step=0.25)
    fine = integrate_rk4(lambda _t, state: state.copy(), y0, times, max_step=0.125)

    coarse_error = abs(coarse[-1, 0] - math.e)
    fine_error = abs(fine[-1, 0] - math.e)

    assert fine_error < coarse_error
    assert coarse_error / fine_error > 10.0


def test_integrate_rk4_single_time_returns_initial_without_rhs_call() -> None:
    def rhs(_t: float, _y: np.ndarray) -> np.ndarray:
        raise AssertionError("rhs should not be called for a single requested time")

    y0 = np.array([1.0, 2.0])

    trajectory = integrate_rk4(rhs, y0, np.array([2.5]))

    assert trajectory.shape == (1, 2)
    np.testing.assert_array_equal(trajectory[0], y0)
    assert not np.shares_memory(trajectory, y0)


def test_integrate_rk4_initial_row_equals_but_does_not_alias_y0() -> None:
    y0 = np.array([1.0, 2.0])

    trajectory = integrate_rk4(
        lambda _t, state: np.zeros_like(state),
        y0,
        np.array([0.0, 1.0]),
    )

    np.testing.assert_array_equal(trajectory[0], y0)
    assert not np.shares_memory(trajectory, y0)


def test_integrate_rk4_subdivides_requested_intervals_equally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, float, np.ndarray]] = []

    def fake_rk4_step(
        _rhs: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        y: np.ndarray,
        step_size: float,
    ) -> np.ndarray:
        calls.append((t, step_size, y.copy()))
        return y + np.full_like(y, step_size)

    monkeypatch.setattr(ode, "rk4_step", fake_rk4_step)

    times = np.array([0.0, 0.0625, 0.1875, 0.4875])
    trajectory = ode.integrate_rk4(lambda _t, state: state, np.array([0.0]), times, max_step=0.125)

    np.testing.assert_allclose(
        [call[1] for call in calls],
        [0.0625, 0.125, 0.1, 0.1, 0.1],
    )
    np.testing.assert_allclose(
        [call[0] for call in calls],
        [0.0, 0.0625, 0.1875, 0.2875, 0.3875],
    )
    assert trajectory.shape == (4, 1)
    np.testing.assert_allclose(trajectory[:, 0], times)


def test_integrate_rk4_uses_equal_rhs_stage_times_for_nondivisible_interval() -> None:
    observed_stage_times: list[float] = []

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        observed_stage_times.append(t)
        return np.zeros_like(state)

    trajectory = integrate_rk4(
        rhs,
        np.array([1.0]),
        np.array([0.0, 0.12]),
        max_step=0.05,
    )

    expected_stage_times = [
        0.00,
        0.02,
        0.02,
        0.04,
        0.04,
        0.06,
        0.06,
        0.08,
        0.08,
        0.10,
        0.10,
        0.12,
    ]
    assert len(observed_stage_times) == 12
    np.testing.assert_allclose(observed_stage_times, expected_stage_times, atol=1e-14)
    assert trajectory.shape == (2, 1)


@pytest.mark.parametrize(
    ("interval_end", "max_step"),
    [
        (0.03, 0.05),
        (0.05, 0.05),
    ],
)
def test_integrate_rk4_short_or_equal_interval_uses_one_step(
    interval_end: float,
    max_step: float,
) -> None:
    calls = 0

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros_like(state)

    integrate_rk4(rhs, np.array([1.0]), np.array([0.0, interval_end]), max_step=max_step)

    assert calls == 4


def test_integrate_rk4_supports_nonzero_and_negative_initial_times() -> None:
    times = np.array([-2.0, -1.5, 0.25])
    y0 = np.array([4.0])

    trajectory = integrate_rk4(
        lambda _t, _state: np.array([2.0]),
        y0,
        times,
        max_step=0.04,
    )

    expected = 4.0 + 2.0 * (times - times[0])
    np.testing.assert_allclose(trajectory[:, 0], expected, atol=1e-12)


def test_integrate_rk4_preserves_inputs_and_allocates_distinct_outputs() -> None:
    y0 = np.array([1.0, 2.0])
    times = np.array([0.0, 0.1])
    y0_before = y0.copy()
    times_before = times.copy()

    trajectory_a = integrate_rk4(_constant_rhs(np.ones(2)), y0, times)
    trajectory_b = integrate_rk4(_constant_rhs(np.ones(2)), y0, times)

    np.testing.assert_array_equal(y0, y0_before)
    np.testing.assert_array_equal(times, times_before)
    np.testing.assert_allclose(trajectory_a, trajectory_b)
    assert trajectory_a is not trajectory_b
    assert not np.shares_memory(trajectory_a, trajectory_b)
    assert not np.shares_memory(trajectory_a, y0)


@pytest.mark.parametrize("bad_rhs", [None, 1, "rhs", object(), np.array([1.0])])
def test_rk4_step_and_integrate_rk4_reject_invalid_rhs(bad_rhs: object) -> None:
    with pytest.raises(SchemaError, match="rhs"):
        rk4_step(bad_rhs, 0.0, np.array([1.0]), 0.1)  # type: ignore[arg-type]

    with pytest.raises(SchemaError, match="rhs"):
        integrate_rk4(bad_rhs, np.array([1.0]), np.array([0.0, 0.1]))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "t",
    [
        True,
        False,
        "0.0",
        None,
        1.0 + 0.0j,
        1.0 + 2.0j,
        np.array(0.0),
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rk4_step_rejects_invalid_t_values(t: object) -> None:
    with pytest.raises(SchemaError, match="t"):
        rk4_step(_constant_rhs(np.array([1.0])), t, np.array([1.0]), 0.1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "step_size",
    [0.0, -0.1, True, False, "0.1", None, 1.0 + 2.0j, np.nan, np.inf, -np.inf],
)
def test_rk4_step_rejects_invalid_step_sizes(step_size: object) -> None:
    with pytest.raises(SchemaError, match="step_size"):
        rk4_step(
            _constant_rhs(np.array([1.0])),
            0.0,
            np.array([1.0]),
            step_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "state_factory",
    [
        lambda: [1.0, 2.0],
        lambda: np.array(1.0),
        lambda: np.array([[1.0, 2.0]]),
        lambda: np.array([], dtype=float),
        lambda: np.array([True, False]),
        lambda: np.array([1.0, 2.0], dtype=object),
        lambda: np.array(["1.0", "2.0"]),
        lambda: np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        lambda: np.array([1.0, np.nan]),
        lambda: np.array([1.0, np.inf]),
        lambda: np.array([1.0, -np.inf]),
    ],
)
def test_rk4_step_rejects_invalid_state_arrays(state_factory: Callable[[], object]) -> None:
    with pytest.raises(SchemaError, match="y"):
        rk4_step(_constant_rhs(np.ones(2)), 0.0, state_factory(), 0.1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "state_factory",
    [
        lambda: [1.0, 2.0],
        lambda: np.array(1.0),
        lambda: np.array([[1.0, 2.0]]),
        lambda: np.array([], dtype=float),
        lambda: np.array([True, False]),
        lambda: np.array([1.0, 2.0], dtype=object),
        lambda: np.array(["1.0", "2.0"]),
        lambda: np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        lambda: np.array([1.0, np.nan]),
        lambda: np.array([1.0, np.inf]),
        lambda: np.array([1.0, -np.inf]),
    ],
)
def test_integrate_rk4_rejects_invalid_initial_states(
    state_factory: Callable[[], object],
) -> None:
    with pytest.raises(SchemaError, match="y0"):
        integrate_rk4(_constant_rhs(np.ones(2)), state_factory(), np.array([0.0, 0.1]))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "times_factory",
    [
        lambda: [0.0, 0.1],
        lambda: np.array(0.0),
        lambda: np.array([[0.0, 0.1]]),
        lambda: np.array([], dtype=float),
        lambda: np.array([False, True]),
        lambda: np.array([0.0, 0.1], dtype=object),
        lambda: np.array(["0.0", "0.1"]),
        lambda: np.array([0.0 + 0.0j, 0.1 + 0.0j]),
        lambda: np.array([0.0, np.nan]),
        lambda: np.array([0.0, np.inf]),
        lambda: np.array([0.0, -np.inf]),
        lambda: np.array([0.0, 0.0]),
        lambda: np.array([0.1, 0.0]),
        lambda: np.array([0.0, 0.2, 0.1]),
    ],
)
def test_integrate_rk4_rejects_invalid_times(times_factory: Callable[[], object]) -> None:
    with pytest.raises(SchemaError, match="times"):
        integrate_rk4(_constant_rhs(np.ones(2)), np.array([1.0, 2.0]), times_factory())  # type: ignore[arg-type]


def test_integrate_rk4_invalid_top_level_input_does_not_call_rhs() -> None:
    calls = 0

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros_like(state)

    with pytest.raises(SchemaError):
        integrate_rk4(rhs, np.array([1.0]), np.array([0.0, 0.0]))

    assert calls == 0


@pytest.mark.parametrize(
    "max_step",
    [0.0, -0.1, True, False, "0.1", None, 1.0 + 2.0j, np.nan, np.inf, -np.inf],
)
def test_integrate_rk4_rejects_invalid_max_steps(max_step: object) -> None:
    with pytest.raises(SchemaError, match="max_step"):
        integrate_rk4(
            _constant_rhs(np.array([1.0])),
            np.array([1.0]),
            np.array([0.0, 0.1]),
            max_step=max_step,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "derivative_factory",
    [
        lambda: 1.0,
        lambda: [1.0, 1.0],
        lambda: (1.0, 1.0),
        lambda: np.array([1.0]),
        lambda: np.array([[1.0, 1.0]]),
        lambda: np.array([1.0, 2.0], dtype=object),
        lambda: np.array(["1.0", "2.0"]),
        lambda: np.array([True, False]),
        lambda: np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        lambda: np.array([1.0, np.nan]),
        lambda: np.array([1.0, np.inf]),
        lambda: np.array([1.0, -np.inf]),
    ],
)
def test_rk4_step_rejects_invalid_rhs_outputs(
    derivative_factory: Callable[[], object],
) -> None:
    def rhs(_t: float, _y: np.ndarray) -> object:
        return derivative_factory()

    with pytest.raises(SchemaError, match="rhs output"):
        rk4_step(rhs, 0.0, np.array([1.0, 2.0]), 0.1)  # type: ignore[arg-type]


def test_rk4_step_validates_later_rhs_stages_independently() -> None:
    calls = 0

    def rhs(_t: float, _y: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 3:
            return np.array([np.nan])
        return np.array([1.0])

    with pytest.raises(SchemaError, match="k3"):
        rk4_step(rhs, 0.0, np.array([1.0]), 0.1)

    assert calls == 3


def test_rk4_step_does_not_hide_rhs_exceptions() -> None:
    class ExpectedFailure(RuntimeError):
        pass

    def rhs(_t: float, _state: np.ndarray) -> np.ndarray:
        raise ExpectedFailure("deliberate RHS failure")

    with pytest.raises(ExpectedFailure, match="deliberate RHS failure"):
        rk4_step(rhs, 0.0, np.array([1.0]), 0.1)


def test_rk4_step_rejects_nonfinite_intermediate_states() -> None:
    def rhs(_t: float, _y: np.ndarray) -> np.ndarray:
        return np.array([np.finfo(float).max])

    with pytest.raises(SchemaError, match="finite"):
        rk4_step(rhs, 0.0, np.array([1.0]), 2.0)


def test_rk4_step_rhs_mutation_cannot_mutate_input_or_internal_state() -> None:
    y = np.array([1.0, 2.0])
    y_before = y.copy()
    stage_arguments: list[np.ndarray] = []

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        stage_arguments.append(state)
        state[:] = -1000.0
        return np.ones(2)

    result = rk4_step(rhs, 0.0, y, 0.2)

    np.testing.assert_array_equal(y, y_before)
    np.testing.assert_allclose(result, y_before + 0.2)
    assert all(not np.shares_memory(stage_state, y) for stage_state in stage_arguments)


def test_rk4_step_each_rhs_stage_receives_independent_state_array() -> None:
    received_states: list[np.ndarray] = []
    y = np.array([1.0, 2.0])

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        received_states.append(state)
        return np.zeros_like(state)

    rk4_step(rhs, 0.0, y, 0.1)

    assert len(received_states) == 4
    assert len({id(state) for state in received_states}) == 4
    assert all(not np.shares_memory(state, y) for state in received_states)


def test_integrate_rk4_rhs_mutation_cannot_mutate_inputs_or_trajectory_rows() -> None:
    y0 = np.array([1.0, 2.0])
    times = np.array([0.0, 0.1, 0.2])
    y0_before = y0.copy()
    times_before = times.copy()
    derivative = np.array([1.0, -1.0])

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        state[:] = 999.0
        return derivative.copy()

    trajectory = integrate_rk4(rhs, y0, times, max_step=0.05)

    expected = y0_before + times[:, None] * derivative
    np.testing.assert_array_equal(y0, y0_before)
    np.testing.assert_array_equal(times, times_before)
    np.testing.assert_allclose(trajectory, expected)
    assert not np.shares_memory(trajectory, y0)
