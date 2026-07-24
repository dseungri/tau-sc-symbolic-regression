"""Synthetic reaction-law helpers."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real

import numpy as np

from tau_sr.schemas import SchemaError

_ReactionFunction = Callable[[np.ndarray], np.ndarray]


def make_linear_quadratic_reaction(
    growth_rate: float,
    saturation_rate: float,
) -> _ReactionFunction:
    """Construct the synthetic reaction law r(y) = a*y - b*y**2."""

    growth_rate_value = _require_real_scalar(
        growth_rate,
        "growth_rate",
        positive=True,
    )
    saturation_rate_value = _require_real_scalar(
        saturation_rate,
        "saturation_rate",
        minimum=0.0,
    )

    def reaction_fn(y: np.ndarray) -> np.ndarray:
        state = _validate_state(y)
        with np.errstate(over="ignore", invalid="ignore"):
            reaction = growth_rate_value * state - saturation_rate_value * state**2
        if not np.isfinite(reaction).all():
            raise SchemaError("reaction values must contain only finite values.")
        return reaction.astype(float, copy=True)

    return reaction_fn


def _validate_state(y: object) -> np.ndarray:
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
    return y.astype(float, copy=True)


def _require_real_scalar(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise SchemaError(f"{field_name} must be a real numeric scalar.")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise SchemaError(f"{field_name} must be finite.")
    if positive and value_float <= 0.0:
        raise SchemaError(f"{field_name} must be positive.")
    if minimum is not None and value_float < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")
    return value_float


def _is_real_numeric_dtype(array: np.ndarray) -> bool:
    if array.dtype.kind in {"b", "c", "O", "S", "U"}:
        return False
    return bool(np.issubdtype(array.dtype, np.number))
