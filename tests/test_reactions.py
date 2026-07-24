from __future__ import annotations

import numpy as np
import pytest

from tau_sr.dynamics import make_reaction_diffusion_rhs
from tau_sr.reactions import make_linear_quadratic_reaction
from tau_sr.schemas import SchemaError
from tau_sr.simulation import (
    simulate_reaction_diffusion_cohort,
    simulate_reaction_diffusion_trajectory,
)


def _logistic_exact_solution(
    y0: np.ndarray,
    times: np.ndarray,
    *,
    growth_rate: float,
    saturation_rate: float,
) -> np.ndarray:
    carrying_capacity = growth_rate / saturation_rate
    delta_t = times - times[0]
    denominator = 1.0 + ((carrying_capacity / y0) - 1.0) * np.exp(-growth_rate * delta_t)[
        :, None
    ]
    return carrying_capacity / denominator


def test_linear_quadratic_reaction_matches_exact_formula() -> None:
    growth_rate = 0.5
    saturation_rate = 0.5
    reaction_fn = make_linear_quadratic_reaction(growth_rate, saturation_rate)
    y = np.array([-0.5, 0.0, 0.25, 1.0, 1.5])

    reaction = reaction_fn(y)

    expected = growth_rate * y - saturation_rate * y**2
    np.testing.assert_allclose(reaction, expected)


def test_linear_quadratic_reaction_equilibrium_and_signs() -> None:
    growth_rate = 0.5
    saturation_rate = 0.5
    carrying_capacity = growth_rate / saturation_rate
    reaction_fn = make_linear_quadratic_reaction(growth_rate, saturation_rate)

    y = np.array([0.0, 0.25 * carrying_capacity, carrying_capacity, 1.5 * carrying_capacity])
    reaction = reaction_fn(y)

    assert reaction[0] == 0.0
    np.testing.assert_allclose(reaction[2], 0.0, atol=1e-15)
    assert reaction[1] > 0.0
    assert reaction[3] < 0.0


def test_zero_saturation_rate_is_linear_control() -> None:
    reaction_fn = make_linear_quadratic_reaction(0.25, 0.0)
    y = np.array([-2.0, 0.0, 1.5, 4.0])

    reaction = reaction_fn(y)

    np.testing.assert_allclose(reaction, 0.25 * y)


def test_shape_dtype_and_ownership_with_integer_state() -> None:
    reaction_fn = make_linear_quadratic_reaction(0.5, 0.25)
    y = np.array([0, 1, 2])
    y_before = y.copy()

    reaction = reaction_fn(y)

    assert reaction.shape == y.shape
    assert np.issubdtype(reaction.dtype, np.floating)
    np.testing.assert_array_equal(y, y_before)
    assert not np.shares_memory(reaction, y)
    np.testing.assert_allclose(reaction, np.array([0.0, 0.25, 0.0]))


def test_repeated_reaction_calls_are_deterministic_and_independent() -> None:
    reaction_fn = make_linear_quadratic_reaction(0.3, 0.1)
    y = np.array([0.2, 0.4, 0.8])

    first = reaction_fn(y)
    second = reaction_fn(y)

    np.testing.assert_array_equal(first, second)
    assert first is not second
    assert not np.shares_memory(first, second)


def test_factory_and_reaction_do_not_mutate_numpy_legacy_global_rng_state() -> None:
    np.random.seed(123)
    expected_draws = np.random.random(5)
    np.random.seed(123)

    reaction_fn = make_linear_quadratic_reaction(0.5, 0.5)
    reaction_fn(np.array([0.2, 0.4]))

    np.testing.assert_array_equal(np.random.random(5), expected_draws)


@pytest.mark.parametrize(
    "growth_rate",
    [
        0.0,
        -0.1,
        True,
        False,
        "0.5",
        None,
        0.5 + 0.0j,
        np.array(0.5),
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rejects_invalid_growth_rates(growth_rate: object) -> None:
    with pytest.raises(SchemaError, match="growth_rate"):
        make_linear_quadratic_reaction(growth_rate, 0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "saturation_rate",
    [
        -0.1,
        True,
        False,
        "0.5",
        None,
        0.5 + 0.0j,
        np.array(0.5),
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rejects_invalid_saturation_rates(saturation_rate: object) -> None:
    with pytest.raises(SchemaError, match="saturation_rate"):
        make_linear_quadratic_reaction(0.5, saturation_rate)  # type: ignore[arg-type]


def test_accepts_numpy_real_scalar_coefficients() -> None:
    reaction_fn = make_linear_quadratic_reaction(np.float64(0.5), np.float32(0.25))

    np.testing.assert_allclose(reaction_fn(np.array([2.0])), np.array([0.0]))


@pytest.mark.parametrize(
    "state",
    [
        [0.0, 1.0],
        np.array(1.0),
        np.array([], dtype=float),
        np.ones((1, 2)),
        np.array([1.0, 2.0], dtype=object),
        np.array(["1.0", "2.0"]),
        np.array([True, False]),
        np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0, -np.inf]),
    ],
)
def test_reaction_rejects_invalid_states(state: object) -> None:
    reaction_fn = make_linear_quadratic_reaction(0.5, 0.5)

    with pytest.raises(SchemaError, match="y"):
        reaction_fn(state)  # type: ignore[arg-type]


def test_reaction_rejects_nonfinite_computed_values() -> None:
    reaction_fn = make_linear_quadratic_reaction(1.0, 1.0)

    with pytest.raises(SchemaError, match="reaction values"):
        reaction_fn(np.array([np.finfo(float).max]))


def test_reaction_is_compatible_with_reaction_diffusion_rhs_factory() -> None:
    reaction_fn = make_linear_quadratic_reaction(0.5, 0.5)
    rhs = make_reaction_diffusion_rhs(np.zeros((2, 2)), 0.0, reaction_fn)
    y = np.array([0.25, 1.5])

    derivative = rhs(0.0, y)

    expected = 0.5 * y - 0.5 * y**2
    np.testing.assert_allclose(derivative, expected)


def test_reaction_is_compatible_with_single_subject_simulation() -> None:
    growth_rate = 0.5
    saturation_rate = 0.5
    y0 = np.array([0.2, 0.6])
    times = np.array([0.0, 0.4, 1.0])
    reaction_fn = make_linear_quadratic_reaction(growth_rate, saturation_rate)

    trajectory = simulate_reaction_diffusion_trajectory(
        y0,
        times,
        np.zeros((2, 2)),
        0.0,
        reaction_fn,
        max_step=0.001,
    )

    expected = _logistic_exact_solution(
        y0,
        times,
        growth_rate=growth_rate,
        saturation_rate=saturation_rate,
    )
    np.testing.assert_allclose(trajectory, expected, rtol=1e-12, atol=1e-12)


def test_reaction_is_compatible_with_cohort_simulation() -> None:
    reaction_fn = make_linear_quadratic_reaction(0.5, 0.5)

    trajectories = simulate_reaction_diffusion_cohort(
        np.array([[0.2, 0.4], [0.6, 0.8]]),
        (np.array([0.0, 0.5]), np.array([1.0, 1.3, 1.8])),
        np.zeros((2, 2, 2)),
        0.0,
        reaction_fn,
        max_step=0.001,
    )

    assert len(trajectories) == 2
    assert trajectories[0].shape == (2, 2)
    assert trajectories[1].shape == (3, 2)
    for trajectory in trajectories:
        assert np.isfinite(trajectory).all()
        assert np.all(trajectory >= 0.0)
        assert np.all(trajectory <= 1.0)
