"""Generic NumPy ODE integration utilities."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real

import numpy as np

from tau_sr.schemas import SchemaError

_RhsFunction = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(
    rhs: _RhsFunction,
    t: float,
    y: np.ndarray,
    step_size: float,
) -> np.ndarray:
    """Advance a vector-valued ODE state by one classical RK4 step."""

    _require_callable(rhs)
    t_value = _require_real_scalar(t, "t")
    h = _require_real_scalar(step_size, "step_size", positive=True)
    y_current = _validate_vector_array(y, "y")

    k1 = _evaluate_rhs(rhs, t_value, y_current, expected_shape=y_current.shape, stage_name="k1")
    k2_state = _stage_state(y_current, 0.5 * h, k1)
    k2 = _evaluate_rhs(
        rhs,
        t_value + 0.5 * h,
        k2_state,
        expected_shape=y_current.shape,
        stage_name="k2",
    )
    k3_state = _stage_state(y_current, 0.5 * h, k2)
    k3 = _evaluate_rhs(
        rhs,
        t_value + 0.5 * h,
        k3_state,
        expected_shape=y_current.shape,
        stage_name="k3",
    )
    k4_state = _stage_state(y_current, h, k3)
    k4 = _evaluate_rhs(
        rhs,
        t_value + h,
        k4_state,
        expected_shape=y_current.shape,
        stage_name="k4",
    )

    with np.errstate(over="ignore", invalid="ignore"):
        y_next = y_current + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return _validate_vector_array(y_next, "RK4 step result")


def integrate_rk4(
    rhs: _RhsFunction,
    y0: np.ndarray,
    times: np.ndarray,
    *,
    max_step: float = 0.05,
) -> np.ndarray:
    """Integrate a vector-valued ODE at requested observation times."""

    _require_callable(rhs)
    y_current = _validate_vector_array(y0, "y0")
    times_array = _validate_times(times)
    max_step_value = _require_real_scalar(max_step, "max_step", positive=True)

    trajectory = np.empty((times_array.shape[0], y_current.shape[0]), dtype=float)
    trajectory[0] = y_current

    for index in range(times_array.shape[0] - 1):
        interval_start = float(times_array[index])
        interval_end = float(times_array[index + 1])
        delta_t = interval_end - interval_start
        n_steps = math.ceil(delta_t / max_step_value)
        h = delta_t / n_steps

        step_time = interval_start
        for _ in range(n_steps):
            y_current = rk4_step(rhs, step_time, y_current, h)
            step_time += h

        trajectory[index + 1] = y_current

    return trajectory


def _require_callable(rhs: object) -> None:
    if not callable(rhs):
        raise SchemaError("rhs must be callable.")


def _require_real_scalar(value: object, field_name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise SchemaError(f"{field_name} must be a real numeric scalar.")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise SchemaError(f"{field_name} must be finite.")
    if positive and value_float <= 0.0:
        raise SchemaError(f"{field_name} must be positive.")
    return value_float


def _validate_vector_array(array: object, field_name: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise SchemaError(f"{field_name} must be a NumPy array.")
    if array.ndim != 1:
        raise SchemaError(f"{field_name} must be one-dimensional.")
    if array.size == 0:
        raise SchemaError(f"{field_name} must be nonempty.")
    if not _is_real_numeric_dtype(array):
        raise SchemaError(f"{field_name} must have a real numeric dtype.")
    if not np.isfinite(array).all():
        raise SchemaError(f"{field_name} must contain only finite values.")
    return array.astype(float, copy=True)


def _validate_times(times: object) -> np.ndarray:
    times_array = _validate_vector_array(times, "times")
    if times_array.shape[0] > 1 and not np.all(np.diff(times_array) > 0.0):
        raise SchemaError("times must be strictly increasing.")
    return times_array


def _evaluate_rhs(
    rhs: _RhsFunction,
    t: float,
    stage_state: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    stage_name: str,
) -> np.ndarray:
    derivative = rhs(float(t), stage_state.copy())
    return _validate_rhs_output(derivative, expected_shape=expected_shape, stage_name=stage_name)


def _validate_rhs_output(
    derivative: object,
    *,
    expected_shape: tuple[int, ...],
    stage_name: str,
) -> np.ndarray:
    if not isinstance(derivative, np.ndarray):
        raise SchemaError(f"rhs output for {stage_name} must be a NumPy array.")
    if derivative.ndim != 1:
        raise SchemaError(f"rhs output for {stage_name} must be one-dimensional.")
    if derivative.shape != expected_shape:
        raise SchemaError(
            f"rhs output for {stage_name} has shape {derivative.shape}; "
            f"expected {expected_shape}."
        )
    if not _is_real_numeric_dtype(derivative):
        raise SchemaError(f"rhs output for {stage_name} must have a real numeric dtype.")
    if not np.isfinite(derivative).all():
        raise SchemaError(f"rhs output for {stage_name} must contain only finite values.")
    return derivative.astype(float, copy=True)


def _stage_state(y: np.ndarray, scaled_step: float, derivative: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        stage_state = y + scaled_step * derivative
    return _validate_vector_array(stage_state, "RK4 stage state")


def _is_real_numeric_dtype(array: np.ndarray) -> bool:
    if array.dtype.kind in {"b", "c", "O", "S", "U"}:
        return False
    return bool(np.issubdtype(array.dtype, np.number))
