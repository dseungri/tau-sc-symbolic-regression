"""Canonical PyTorch neural reaction-diffusion model"""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn as nn
import torch.nn.functional as F

from tau_sr.schemas import SchemaError

__all__ = ("NeuralReactionDiffusion",)


class NeuralReactionDiffusion(nn.Module):
    """Differentiable neural reaction-diffusion model with fixed external graphs."""

    def __init__(
        self,
        *,
        hidden_width: int = 16,
        initial_rho: float = 0.1,
    ) -> None:
        super().__init__()

        _require_positive_int(hidden_width, "hidden_width")
        initial_rho_value = _require_real_scalar(initial_rho, "initial_rho", positive=True)

        self.raw_rho = nn.Parameter(
            torch.tensor(
                _inverse_softplus(initial_rho_value),
                dtype=torch.get_default_dtype(),
            )
        )
        self.reaction_eq = nn.Sequential(
            nn.Linear(1, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
        )

    @property
    def rho(self) -> torch.Tensor:
        """Positive physical diffusion coefficient."""

        return F.softplus(self.raw_rho)

    def reaction(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the shared scalar neural reaction law independently to each ROI."""

        state = self._validate_state(y)
        reaction = self.reaction_eq(state.unsqueeze(-1)).squeeze(-1)
        return _validate_computed_tensor(
            reaction,
            "reaction",
            expected_shape=state.shape,
        )

    def forward(
        self,
        y: torch.Tensor,
        laplacian: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dy/dt = -rho * L*y + r_theta(y)."""

        state = self._validate_state(y)
        graph = self._validate_laplacian(laplacian, state=state)

        if state.ndim == 1:
            transport_base = graph @ state
        elif graph.ndim == 2:
            transport_base = state @ graph.T
        else:
            transport_base = torch.einsum("bij,bj->bi", graph, state)

        transport = -self.rho * transport_base
        transport = _validate_computed_tensor(
            transport,
            "transport",
            expected_shape=state.shape,
        )
        reaction = self.reaction(state)
        derivative = transport + reaction
        return _validate_computed_tensor(
            derivative,
            "derivative",
            expected_shape=state.shape,
        )

    def integrate(
        self,
        y0: torch.Tensor,
        laplacian: torch.Tensor,
        times: torch.Tensor,
        *,
        max_step: float = 0.05,
    ) -> torch.Tensor:
        """Integrate requested times with differentiable PyTorch RK4 operations.

        Gradients with respect to observation times are not a supported objective.
        Gradients with respect to model parameters and the initial state are
        preserved.
        """

        y_current = self._validate_state(y0)
        graph = self._validate_laplacian(laplacian, state=y_current)
        checked_times = _validate_times(times, state=y_current)
        max_step_value = _require_real_scalar(max_step, "max_step", positive=True)

        trajectory = [y_current.clone()]
        for index in range(checked_times.shape[0] - 1):
            interval = checked_times[index + 1] - checked_times[index]
            n_steps = math.ceil(float((interval / max_step_value).item()))
            h = interval / n_steps
            for _ in range(n_steps):
                y_current = self._torch_rk4_step(y_current, graph, h)
                _validate_computed_tensor(
                    y_current,
                    "integrated state",
                    expected_shape=trajectory[0].shape,
                )
            trajectory.append(y_current.clone())

        return torch.stack(trajectory, dim=0)

    def _torch_rk4_step(
        self,
        y: torch.Tensor,
        laplacian: torch.Tensor,
        step_size: torch.Tensor,
    ) -> torch.Tensor:
        k1 = self.forward(y, laplacian)
        k2 = self.forward(y + 0.5 * step_size * k1, laplacian)
        k3 = self.forward(y + 0.5 * step_size * k2, laplacian)
        k4 = self.forward(y + step_size * k3, laplacian)

        y_next = y + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return _validate_computed_tensor(
            y_next,
            "RK4 step result",
            expected_shape=y.shape,
        )

    def _validate_state(self, y: object) -> torch.Tensor:
        state = _validate_state_tensor(y, "y")
        reference = self.raw_rho
        if state.device != reference.device:
            raise SchemaError("y must be on the same device as model parameters.")
        if state.dtype != reference.dtype:
            raise SchemaError("y dtype must match model parameters.")
        return state

    def _validate_laplacian(
        self,
        laplacian: object,
        *,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return _validate_laplacian_tensor(laplacian, state=state)


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


def _require_positive_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field_name} must be an integer.")
    if value < 1:
        raise SchemaError(f"{field_name} must be positive.")


def _require_real_scalar(value: object, field_name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SchemaError(f"{field_name} must be a real numeric scalar.")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise SchemaError(f"{field_name} must be finite.")
    if positive and value_float <= 0.0:
        raise SchemaError(f"{field_name} must be positive.")
    return value_float


def _validate_state_tensor(value: object, field_name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise SchemaError(f"{field_name} must be a torch.Tensor.")
    if value.layout != torch.strided:
        raise SchemaError(f"{field_name} must be a dense strided tensor.")
    if value.ndim not in (1, 2):
        raise SchemaError(f"{field_name} must have shape (R,) or (B, R).")
    if value.numel() == 0 or value.shape[-1] < 1:
        raise SchemaError(f"{field_name} must be nonempty.")
    if not torch.is_floating_point(value):
        raise SchemaError(f"{field_name} must have a real floating dtype.")
    if not torch.isfinite(value).all().item():
        raise SchemaError(f"{field_name} must contain only finite values.")
    return value


def _validate_laplacian_tensor(
    value: object,
    *,
    state: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise SchemaError("laplacian must be a torch.Tensor.")
    if value.layout != torch.strided:
        raise SchemaError("laplacian must be a dense strided tensor.")
    if value.ndim not in (2, 3):
        raise SchemaError("laplacian must have shape (R, R) or (B, R, R).")
    if not torch.is_floating_point(value):
        raise SchemaError("laplacian must have a real floating dtype.")
    if value.device != state.device:
        raise SchemaError("laplacian must be on the same device as y.")
    if value.dtype != state.dtype:
        raise SchemaError("laplacian dtype must match y dtype.")
    if not torch.isfinite(value).all().item():
        raise SchemaError("laplacian must contain only finite values.")

    n_regions = state.shape[-1]
    if value.ndim == 2:
        if value.shape != (n_regions, n_regions):
            raise SchemaError(
                f"laplacian has shape {tuple(value.shape)}; expected "
                f"({n_regions}, {n_regions})."
            )
    elif state.ndim == 1:
        raise SchemaError("batched laplacian requires batched y.")
    else:
        expected_shape = (state.shape[0], n_regions, n_regions)
        if value.shape != expected_shape:
            raise SchemaError(
                f"laplacian has shape {tuple(value.shape)}; expected {expected_shape}."
            )

    return value


def _validate_times(value: object, *, state: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise SchemaError("times must be a torch.Tensor.")
    if value.layout != torch.strided:
        raise SchemaError("times must be a dense strided tensor.")
    if value.ndim != 1:
        raise SchemaError("times must be one-dimensional.")
    if value.numel() == 0:
        raise SchemaError("times must be nonempty.")
    if not torch.is_floating_point(value):
        raise SchemaError("times must have a real floating dtype.")
    if value.device != state.device:
        raise SchemaError("times must be on the same device as y0.")
    if value.dtype != state.dtype:
        raise SchemaError("times dtype must match y0 dtype.")
    if not torch.isfinite(value).all().item():
        raise SchemaError("times must contain only finite values.")
    if value.shape[0] > 1 and not torch.all(value[1:] > value[:-1]).item():
        raise SchemaError("times must be strictly increasing.")
    return value


def _validate_computed_tensor(
    value: torch.Tensor,
    field_name: str,
    *,
    expected_shape: torch.Size,
) -> torch.Tensor:
    if value.shape != expected_shape:
        raise SchemaError(
            f"{field_name} has shape {tuple(value.shape)}; expected {tuple(expected_shape)}."
        )
    if not torch.isfinite(value).all().item():
        raise SchemaError(f"{field_name} must contain only finite values.")
    return value
