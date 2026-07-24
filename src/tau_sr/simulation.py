"""Single-subject reaction-diffusion simulation helper"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from numbers import Real

import numpy as np

from tau_sr.dynamics import make_reaction_diffusion_rhs
from tau_sr.ode import integrate_rk4
from tau_sr.schemas import SchemaError

_ReactionFunction = Callable[[np.ndarray], np.ndarray]


def simulate_reaction_diffusion_trajectory(
    y0: np.ndarray,
    times: np.ndarray,
    laplacian: np.ndarray,
    rho: float,
    reaction_fn: _ReactionFunction,
    *,
    max_step: float = 0.05,
) -> np.ndarray:
    """Simulate one subject's reaction-diffusion trajectory"""

    rhs = make_reaction_diffusion_rhs(
        laplacian=laplacian,
        rho=rho,
        reaction_fn=reaction_fn,
    )
    return integrate_rk4(
        rhs,
        y0,
        times,
        max_step=max_step,
    )


def simulate_reaction_diffusion_cohort(
    baseline_states: np.ndarray,
    times_by_subject: Sequence[np.ndarray],
    laplacians: np.ndarray,
    rho: float,
    reaction_fn: _ReactionFunction,
    *,
    max_step: float = 0.05,
) -> tuple[np.ndarray, ...]:
    """Simulate noise-free reaction-diffusion trajectories for a cohort."""

    checked_baseline_states = _validate_baseline_states(baseline_states)
    n_subjects, n_regions = checked_baseline_states.shape
    checked_laplacians = _validate_laplacians(
        laplacians,
        n_subjects=n_subjects,
        n_regions=n_regions,
    )
    checked_times = _validate_times_by_subject(
        times_by_subject,
        n_subjects=n_subjects,
    )
    rho_value = _require_real_scalar(rho, "rho", minimum=0.0)
    if not callable(reaction_fn):
        raise SchemaError("reaction_fn must be callable.")
    max_step_value = _require_real_scalar(max_step, "max_step", positive=True)

    trajectories: list[np.ndarray] = []
    for subject_index, (y0, times, laplacian) in enumerate(
        zip(checked_baseline_states, checked_times, checked_laplacians, strict=True)
    ):
        try:
            trajectory = simulate_reaction_diffusion_trajectory(
                y0=y0,
                times=times,
                laplacian=laplacian,
                rho=rho_value,
                reaction_fn=reaction_fn,
                max_step=max_step_value,
            )
        except SchemaError as exc:
            raise SchemaError(
                f"invalid simulation input for subject index {subject_index}: {exc}"
            ) from exc
        trajectories.append(trajectory)

    return tuple(trajectories)


def _validate_baseline_states(baseline_states: object) -> np.ndarray:
    if not isinstance(baseline_states, np.ndarray):
        raise SchemaError("baseline_states must be a NumPy array.")
    if baseline_states.ndim != 2:
        raise SchemaError("baseline_states must be two-dimensional.")
    if baseline_states.shape[0] < 1:
        raise SchemaError("baseline_states must contain at least one subject.")
    if baseline_states.shape[1] < 2:
        raise SchemaError("baseline_states must contain at least two regions.")
    if not _is_real_numeric_dtype(baseline_states):
        raise SchemaError("baseline_states must have a real numeric dtype.")
    if not np.isfinite(baseline_states).all():
        raise SchemaError("baseline_states must contain only finite values.")
    return baseline_states


def _validate_laplacians(
    laplacians: object,
    *,
    n_subjects: int,
    n_regions: int,
) -> np.ndarray:
    if not isinstance(laplacians, np.ndarray):
        raise SchemaError("laplacians must be a NumPy array.")
    if laplacians.ndim != 3:
        raise SchemaError("laplacians must be three-dimensional.")
    expected_shape = (n_subjects, n_regions, n_regions)
    if laplacians.shape != expected_shape:
        raise SchemaError(f"laplacians has shape {laplacians.shape}; expected {expected_shape}.")
    if not _is_real_numeric_dtype(laplacians):
        raise SchemaError("laplacians must have a real numeric dtype.")
    if not np.isfinite(laplacians).all():
        raise SchemaError("laplacians must contain only finite values.")
    return laplacians


def _validate_times_by_subject(
    times_by_subject: object,
    *,
    n_subjects: int,
) -> tuple[np.ndarray, ...]:
    if isinstance(times_by_subject, (str, bytes)):
        raise SchemaError("times_by_subject must be a sequence of NumPy time arrays.")
    if isinstance(times_by_subject, np.ndarray):
        if times_by_subject.ndim != 2:
            raise SchemaError("times_by_subject as a NumPy array must be two-dimensional.")
        raw_times = tuple(times_by_subject[index] for index in range(times_by_subject.shape[0]))
    elif isinstance(times_by_subject, Sequence):
        raw_times = tuple(times_by_subject)
    else:
        raise SchemaError("times_by_subject must be a sequence of NumPy time arrays.")

    if len(raw_times) != n_subjects:
        raise SchemaError(
            f"times_by_subject has {len(raw_times)} entries; expected {n_subjects}."
        )

    return tuple(
        _validate_subject_times(times, subject_index=subject_index)
        for subject_index, times in enumerate(raw_times)
    )


def _validate_subject_times(times: object, *, subject_index: int) -> np.ndarray:
    field_name = f"times_by_subject[{subject_index}]"
    if not isinstance(times, np.ndarray):
        raise SchemaError(f"{field_name} must be a NumPy array.")
    if times.ndim != 1:
        raise SchemaError(f"{field_name} must be one-dimensional.")
    if times.size == 0:
        raise SchemaError(f"{field_name} must be nonempty.")
    if not _is_real_numeric_dtype(times):
        raise SchemaError(f"{field_name} must have a real numeric dtype.")
    if not np.isfinite(times).all():
        raise SchemaError(f"{field_name} must contain only finite values.")
    times_float = times.astype(float, copy=False)
    if times_float.shape[0] > 1 and not np.all(np.diff(times_float) > 0.0):
        raise SchemaError(f"{field_name} must be strictly increasing.")
    return times


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
    if minimum is not None and value_float < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")
    if positive and value_float <= 0.0:
        raise SchemaError(f"{field_name} must be positive.")
    return value_float


def _is_real_numeric_dtype(array: np.ndarray) -> bool:
    if array.dtype.kind in {"b", "c", "O", "S", "U"}:
        return False
    return bool(np.issubdtype(array.dtype, np.number))
