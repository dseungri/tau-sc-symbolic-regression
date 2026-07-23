from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from tau_sr.dynamics import make_reaction_diffusion_rhs
from tau_sr.ode import integrate_rk4, rk4_step
from tau_sr.schemas import SchemaError


def _zero_reaction(shape: tuple[int, ...]) -> Callable[[np.ndarray], np.ndarray]:
    def reaction_fn(_state: np.ndarray) -> np.ndarray:
        return np.zeros(shape)

    return reaction_fn


def test_diffusion_only_two_node_graph() -> None:
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    rhs = make_reaction_diffusion_rhs(
        laplacian,
        rho=0.5,
        reaction_fn=lambda state: np.zeros_like(state),
    )

    derivative = rhs(0.0, np.array([1.0, 0.0]))

    np.testing.assert_allclose(derivative, np.array([-0.5, 0.5]))


def test_spatially_constant_state_has_zero_unnormalized_diffusion() -> None:
    laplacian = np.array(
        [
            [2.0, -1.0, -1.0],
            [-1.0, 2.0, -1.0],
            [-1.0, -1.0, 2.0],
        ]
    )
    rhs = make_reaction_diffusion_rhs(laplacian, 0.7, lambda state: np.zeros_like(state))

    derivative = rhs(0.0, np.array([3.0, 3.0, 3.0]))

    np.testing.assert_allclose(derivative, np.zeros(3), atol=1e-14)


def test_diffusion_only_conserves_state_sum_for_unnormalized_laplacian() -> None:
    laplacian = np.array(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0],
        ]
    )
    rhs = make_reaction_diffusion_rhs(laplacian, 0.25, lambda state: np.zeros_like(state))

    derivative = rhs(0.0, np.array([1.0, 4.0, -2.0]))

    np.testing.assert_allclose(derivative.sum(), 0.0, atol=1e-14)


def test_zero_rho_disables_transport() -> None:
    laplacian = np.array([[2.0, -2.0], [-2.0, 2.0]])
    reaction = np.array([0.1, -0.2])
    rhs = make_reaction_diffusion_rhs(laplacian, 0.0, lambda _state: reaction.copy())

    derivative = rhs(1.5, np.array([100.0, -100.0]))

    np.testing.assert_array_equal(derivative, reaction)


def test_diffusion_orientation_uses_laplacian_times_state() -> None:
    laplacian = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([5.0, 7.0])
    rhs = make_reaction_diffusion_rhs(laplacian, 1.0, lambda state: np.zeros_like(state))

    derivative = rhs(0.0, y)

    np.testing.assert_allclose(derivative, -(laplacian @ y))
    assert not np.allclose(derivative, -(laplacian.T @ y))


def test_reaction_only_linear_dynamics() -> None:
    rhs = make_reaction_diffusion_rhs(
        np.eye(3),
        0.0,
        reaction_fn=lambda state: 0.25 * state,
    )

    derivative = rhs(0.0, np.array([2.0, -4.0, 8.0]))

    np.testing.assert_allclose(derivative, np.array([0.5, -1.0, 2.0]))


def test_polynomial_reaction_defined_in_test() -> None:
    rhs = make_reaction_diffusion_rhs(
        np.zeros((2, 2)),
        0.0,
        reaction_fn=lambda state: 0.5 * state * (1.0 - state),
    )

    derivative = rhs(0.0, np.array([0.25, 1.5]))

    np.testing.assert_allclose(derivative, np.array([0.09375, -0.375]))


def test_reaction_function_receives_complete_regional_vector() -> None:
    received: list[np.ndarray] = []

    def reaction_fn(state: np.ndarray) -> np.ndarray:
        received.append(state.copy())
        return np.array([state.sum(), state[0] - state[1]])

    rhs = make_reaction_diffusion_rhs(np.zeros((2, 2)), 0.0, reaction_fn)

    derivative = rhs(0.0, np.array([2.0, 3.5]))

    np.testing.assert_array_equal(received[0], np.array([2.0, 3.5]))
    np.testing.assert_allclose(derivative, np.array([5.5, -1.5]))


def test_combined_reaction_diffusion_matches_manual_formula() -> None:
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    y = np.array([2.0, 0.5])
    rho = 0.2
    rhs = make_reaction_diffusion_rhs(laplacian, rho, lambda state: 0.1 * state)

    derivative = rhs(10.0, y)

    expected = -rho * (laplacian @ y) + 0.1 * y
    np.testing.assert_allclose(derivative, expected)


def test_output_shape_dtype_and_allocation_with_integer_inputs() -> None:
    laplacian = np.array([[1, -1], [-1, 1]])
    y = np.array([1, 0])
    rhs = make_reaction_diffusion_rhs(laplacian, 1, lambda state: np.ones_like(state))

    derivative = rhs(0.0, y)

    assert derivative.shape == (2,)
    assert np.issubdtype(derivative.dtype, np.floating)
    assert not np.shares_memory(derivative, y)
    np.testing.assert_array_equal(y, np.array([1, 0]))


def test_repeated_calls_return_independent_arrays_and_ignore_time() -> None:
    rhs = make_reaction_diffusion_rhs(
        np.array([[1.0, -1.0], [-1.0, 1.0]]),
        0.5,
        lambda state: state**2,
    )
    y = np.array([1.0, 3.0])

    first = rhs(0.0, y)
    second = rhs(99.0, y)

    np.testing.assert_allclose(first, second)
    assert first is not second
    assert not np.shares_memory(first, second)


def test_rhs_is_compatible_with_rk4_step() -> None:
    rhs = make_reaction_diffusion_rhs(
        np.zeros((1, 1)),
        0.0,
        lambda state: 2.0 * state,
    )

    result = rk4_step(rhs, 0.0, np.array([1.0]), 0.1)

    np.testing.assert_allclose(result, np.array([math.exp(0.2)]), rtol=1e-5)


def test_rhs_is_compatible_with_integrate_rk4() -> None:
    growth_rate = 0.3
    times = np.array([1.0, 1.25, 2.0])
    y0 = np.array([2.0])
    rhs = make_reaction_diffusion_rhs(
        np.zeros((1, 1)),
        0.0,
        lambda state: growth_rate * state,
    )

    trajectory = integrate_rk4(rhs, y0, times, max_step=0.01)

    expected = y0 * np.exp(growth_rate * (times - times[0]))[:, None]
    np.testing.assert_allclose(trajectory, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("reaction_fn", [None, 1, "reaction", object()])
def test_rejects_noncallable_reaction_function(reaction_fn: object) -> None:
    with pytest.raises(SchemaError, match="reaction_fn"):
        make_reaction_diffusion_rhs(np.eye(2), 0.1, reaction_fn)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "laplacian",
    [
        [[1.0, -1.0], [-1.0, 1.0]],
        np.array(1.0),
        np.array([1.0, -1.0]),
        np.zeros((0, 0)),
        np.zeros((2, 2, 1)),
        np.zeros((2, 3)),
        np.array([[True, False], [False, True]]),
        np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=object),
        np.array([["1", "-1"], ["-1", "1"]]),
        np.array([[1.0 + 0.0j, -1.0], [-1.0, 1.0]]),
        np.array([[1.0, np.nan], [0.0, 1.0]]),
        np.array([[1.0, np.inf], [0.0, 1.0]]),
        np.array([[1.0, -np.inf], [0.0, 1.0]]),
    ],
)
def test_rejects_invalid_laplacians(laplacian: object) -> None:
    with pytest.raises(SchemaError, match="laplacian"):
        make_reaction_diffusion_rhs(laplacian, 0.1, lambda state: state)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "rho",
    [-0.1, True, False, "0.1", None, 1.0 + 0.0j, np.array(0.1), np.nan, np.inf, -np.inf],
)
def test_rejects_invalid_rho(rho: object) -> None:
    with pytest.raises(SchemaError, match="rho"):
        make_reaction_diffusion_rhs(np.eye(2), rho, lambda state: state)  # type: ignore[arg-type]


def test_accepts_zero_rho() -> None:
    rhs = make_reaction_diffusion_rhs(np.eye(2), 0.0, lambda state: state)

    np.testing.assert_allclose(rhs(0.0, np.array([1.0, 2.0])), np.array([1.0, 2.0]))


@pytest.mark.parametrize("t", [True, False, "0.0", None, 1.0 + 0.0j, np.nan, np.inf, -np.inf])
def test_rhs_rejects_invalid_dynamic_times(t: object) -> None:
    rhs = make_reaction_diffusion_rhs(np.eye(2), 0.1, lambda state: state)

    with pytest.raises(SchemaError, match="t"):
        rhs(t, np.array([1.0, 2.0]))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "state",
    [
        [1.0, 2.0],
        np.array(1.0),
        np.array([], dtype=float),
        np.ones((2, 2)),
        np.array([True, False]),
        np.array([1.0, 2.0], dtype=object),
        np.array(["1.0", "2.0"]),
        np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0, -np.inf]),
    ],
)
def test_rhs_rejects_invalid_states(state: object) -> None:
    rhs = make_reaction_diffusion_rhs(np.eye(2), 0.1, lambda state: state)

    with pytest.raises(SchemaError, match="y"):
        rhs(0.0, state)  # type: ignore[arg-type]


def test_rhs_rejects_state_length_mismatch_before_reaction_fn_is_called() -> None:
    calls = 0

    def reaction_fn(state: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return state

    rhs = make_reaction_diffusion_rhs(np.eye(3), 0.1, reaction_fn)

    with pytest.raises(SchemaError, match="expected 3"):
        rhs(0.0, np.array([1.0, 2.0]))

    assert calls == 0


def test_rhs_rejects_invalid_t_and_y_before_reaction_fn_is_called() -> None:
    calls = 0

    def reaction_fn(state: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return state

    rhs = make_reaction_diffusion_rhs(np.eye(2), 0.1, reaction_fn)

    with pytest.raises(SchemaError, match="t"):
        rhs(np.nan, np.array([1.0, 2.0]))
    with pytest.raises(SchemaError, match="y"):
        rhs(0.0, np.array([1.0, np.nan]))

    assert calls == 0


@pytest.mark.parametrize(
    "reaction_factory",
    [
        lambda: 1.0,
        lambda: [1.0, 2.0],
        lambda: (1.0, 2.0),
        lambda: np.array([1.0]),
        lambda: np.ones((1, 2)),
        lambda: np.array([1.0, 2.0], dtype=object),
        lambda: np.array(["1.0", "2.0"]),
        lambda: np.array([True, False]),
        lambda: np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        lambda: np.array([1.0, np.nan]),
        lambda: np.array([1.0, np.inf]),
        lambda: np.array([1.0, -np.inf]),
    ],
)
def test_rhs_rejects_invalid_reaction_outputs(
    reaction_factory: Callable[[], object],
) -> None:
    rhs = make_reaction_diffusion_rhs(np.eye(2), 0.1, lambda _state: reaction_factory())

    with pytest.raises(SchemaError, match="reaction_fn output"):
        rhs(0.0, np.array([1.0, 2.0]))


def test_rhs_validates_reaction_output_on_every_call() -> None:
    calls = 0

    def reaction_fn(state: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            return np.array([np.nan, 0.0])
        return np.zeros_like(state)

    rhs = make_reaction_diffusion_rhs(np.eye(2), 0.1, reaction_fn)

    np.testing.assert_allclose(rhs(0.0, np.array([1.0, 2.0])), np.array([-0.1, -0.2]))
    with pytest.raises(SchemaError, match="reaction_fn output"):
        rhs(0.0, np.array([1.0, 2.0]))


def test_reaction_exception_is_not_hidden() -> None:
    class ExpectedFailure(RuntimeError):
        pass

    def reaction_fn(_state: np.ndarray) -> np.ndarray:
        raise ExpectedFailure("deliberate reaction failure")

    rhs = make_reaction_diffusion_rhs(np.eye(1), 0.0, reaction_fn)

    with pytest.raises(ExpectedFailure, match="deliberate reaction failure"):
        rhs(0.0, np.array([1.0]))


def test_laplacian_is_snapshotted_and_caller_inputs_are_preserved() -> None:
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    laplacian_before = laplacian.copy()
    rhs = make_reaction_diffusion_rhs(laplacian, 0.5, lambda state: np.zeros_like(state))

    laplacian[:, :] = 0.0
    y = np.array([1.0, 0.0])
    y_before = y.copy()
    derivative = rhs(0.0, y)

    np.testing.assert_array_equal(y, y_before)
    np.testing.assert_array_equal(laplacian_before, np.array([[1.0, -1.0], [-1.0, 1.0]]))
    np.testing.assert_allclose(derivative, np.array([-0.5, 0.5]))


def test_factory_and_rhs_do_not_mutate_original_laplacian() -> None:
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    laplacian_before = laplacian.copy()
    rhs = make_reaction_diffusion_rhs(laplacian, 0.5, lambda state: np.zeros_like(state))

    np.testing.assert_array_equal(laplacian, laplacian_before)
    rhs(0.0, np.array([1.0, 0.0]))
    np.testing.assert_array_equal(laplacian, laplacian_before)


def test_reaction_fn_mutating_its_input_cannot_mutate_caller_state() -> None:
    y = np.array([1.0, 2.0])
    y_before = y.copy()

    def reaction_fn(state: np.ndarray) -> np.ndarray:
        state[:] = 999.0
        return np.array([0.1, 0.2])

    rhs = make_reaction_diffusion_rhs(np.zeros((2, 2)), 0.0, reaction_fn)

    derivative = rhs(0.0, y)

    np.testing.assert_array_equal(y, y_before)
    np.testing.assert_allclose(derivative, np.array([0.1, 0.2]))


def test_returned_derivative_does_not_alias_y_or_reaction_output() -> None:
    y = np.array([1.0, 2.0])
    reaction_output = np.array([0.5, 0.25])
    rhs = make_reaction_diffusion_rhs(np.zeros((2, 2)), 0.0, lambda _state: reaction_output)

    derivative = rhs(0.0, y)
    derivative[0] = 99.0

    assert not np.shares_memory(derivative, y)
    assert not np.shares_memory(derivative, reaction_output)
    np.testing.assert_array_equal(reaction_output, np.array([0.5, 0.25]))


def test_nonfinite_transport_or_derivative_raises_schema_error() -> None:
    huge = np.finfo(float).max

    rhs = make_reaction_diffusion_rhs(np.array([[huge]]), huge, lambda state: np.zeros_like(state))
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(
        SchemaError,
        match="transport",
    ):
        rhs(0.0, np.array([huge]))

    rhs = make_reaction_diffusion_rhs(np.array([[-1.0]]), 1.0, lambda _state: np.array([huge]))
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(
        SchemaError,
        match="derivative",
    ):
        rhs(0.0, np.array([huge]))
