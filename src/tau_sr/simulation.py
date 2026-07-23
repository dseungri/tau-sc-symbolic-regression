"""Single-subject reaction-diffusion simulation helper"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tau_sr.dynamics import make_reaction_diffusion_rhs
from tau_sr.ode import integrate_rk4

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
