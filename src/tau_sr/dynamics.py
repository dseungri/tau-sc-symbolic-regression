"""NumPy reaction-diffusion dynamics helpers"""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real

import numpy as np

from tau_sr.schemas import SchemaError

_ReactionFunction = Callable[[np.ndarray], np.ndarray]
_RhsFunction = Callable[[float, np.ndarray], np.ndarray]


def make_reaction_diffusion_rhs(
    laplacian: np.ndarray,
    rho: float,
    reaction_fn: _ReactionFunction,
) -> _RhsFunction:
    """Build a NumPy reaction-diffusion RHS for one subject."""

    laplacian_snapshot = _validate_laplacian(laplacian)
    rho_value = _require_real_scalar(rho, "rho", minimum=0.0)
    if not callable(reaction_fn):
        raise SchemaError("reaction_fn must be callable.")

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        _require_real_scalar(t, "t")
        state = _validate_state(y, expected_length=laplacian_snapshot.shape[0])
        reaction = _evaluate_reaction(
            reaction_fn,
            state,
            expected_shape=state.shape,
        )

        with np.errstate(over="ignore", invalid="ignore"):
            transport = -rho_value * (laplacian_snapshot @ state)
        transport = _validate_computed_vector(
            transport,
            "transport",
            expected_shape=state.shape,
        )

        with np.errstate(over="ignore", invalid="ignore"):
            derivative = transport + reaction
        return _validate_computed_vector(
            derivative,
            "derivative",
            expected_shape=state.shape,
        )

    return rhs


def _validate_laplacian(laplacian: object) -> np.ndarray:
    if not isinstance(laplacian, np.ndarray):
        raise SchemaError("laplacian must be a NumPy array.")
    if laplacian.ndim != 2:
        raise SchemaError("laplacian must be two-dimensional.")
    if laplacian.size == 0:
        raise SchemaError("laplacian must be nonempty.")
    if laplacian.shape[0] != laplacian.shape[1]:
        raise SchemaError("laplacian must be square.")
    if not _is_real_numeric_dtype(laplacian):
        raise SchemaError("laplacian must have a real numeric dtype.")
    if not np.isfinite(laplacian).all():
        raise SchemaError("laplacian must contain only finite values.")
    return laplacian.astype(float, copy=True)


def _validate_state(y: object, *, expected_length: int) -> np.ndarray:
    if not isinstance(y, np.ndarray):
        raise SchemaError("y must be a NumPy array.")
    if y.ndim != 1:
        raise SchemaError("y must be one-dimensional.")
    if y.size == 0:
        raise SchemaError("y must be nonempty.")
    if not _is_real_numeric_dtype(y):
        raise SchemaError("y must have a real numeric dtype.")
    if not np.isfinite(y).all():
        raise SchemaError("y must contain only finite values.")
    if y.shape[0] != expected_length:
        raise SchemaError(
            f"y has length {y.shape[0]}; expected {expected_length} for laplacian."
        )
    return y.astype(float, copy=True)


def _evaluate_reaction(
    reaction_fn: _ReactionFunction,
    state: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    reaction = reaction_fn(state.copy())
    return _validate_reaction_output(reaction, expected_shape=expected_shape)


def _validate_reaction_output(
    reaction: object,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(reaction, np.ndarray):
        raise SchemaError("reaction_fn output must be a NumPy array.")
    if reaction.ndim != 1:
        raise SchemaError("reaction_fn output must be one-dimensional.")
    if reaction.shape != expected_shape:
        raise SchemaError(
            f"reaction_fn output has shape {reaction.shape}; expected {expected_shape}."
        )
    if not _is_real_numeric_dtype(reaction):
        raise SchemaError("reaction_fn output must have a real numeric dtype.")
    if not np.isfinite(reaction).all():
        raise SchemaError("reaction_fn output must contain only finite values.")
    return reaction.astype(float, copy=True)


def _validate_computed_vector(
    values: np.ndarray,
    field_name: str,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if values.shape != expected_shape:
        raise SchemaError(f"{field_name} has shape {values.shape}; expected {expected_shape}.")
    if not np.isfinite(values).all():
        raise SchemaError(f"{field_name} must contain only finite values.")
    return values.astype(float, copy=True)


def _require_real_scalar(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise SchemaError(f"{field_name} must be a real numeric scalar.")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise SchemaError(f"{field_name} must be finite.")
    if minimum is not None and value_float < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")
    return value_float


def _is_real_numeric_dtype(array: np.ndarray) -> bool:
    if array.dtype.kind in {"b", "c", "O", "S", "U"}:
        return False
    return bool(np.issubdtype(array.dtype, np.number))
