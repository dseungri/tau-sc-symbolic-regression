from __future__ import annotations

import inspect
import math
import subprocess
import sys

import pytest
import torch

import tau_sr.neural as neural
from tau_sr.neural import NeuralReactionDiffusion
from tau_sr.schemas import SchemaError


def _chain_laplacian(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


def _set_rho(model: NeuralReactionDiffusion, value: float) -> None:
    with torch.no_grad():
        model.raw_rho.copy_(
            torch.tensor(
                _inverse_softplus(value),
                dtype=model.raw_rho.dtype,
                device=model.raw_rho.device,
            )
        )


def _zero_reaction(model: NeuralReactionDiffusion) -> None:
    with torch.no_grad():
        for parameter in model.reaction_eq.parameters():
            parameter.zero_()


def _constant_reaction(model: NeuralReactionDiffusion, value: float) -> None:
    with torch.no_grad():
        for parameter in model.reaction_eq.parameters():
            parameter.zero_()
        model.reaction_eq[-1].bias.fill_(value)


def test_model_constructs_with_defaults() -> None:
    model = NeuralReactionDiffusion()

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model.raw_rho, torch.nn.Parameter)
    assert len(list(model.reaction_eq.parameters())) > 0


@pytest.mark.parametrize("hidden_width", [0, -1, True, 1.5, "16"])
def test_invalid_hidden_width_values_are_rejected(hidden_width: object) -> None:
    with pytest.raises(SchemaError, match="hidden_width"):
        NeuralReactionDiffusion(hidden_width=hidden_width)  # type: ignore[arg-type]


@pytest.mark.parametrize("initial_rho", [0.0, -0.1, math.inf, math.nan, True, "0.1"])
def test_invalid_initial_rho_values_are_rejected(initial_rho: object) -> None:
    with pytest.raises(SchemaError, match="initial_rho"):
        NeuralReactionDiffusion(initial_rho=initial_rho)  # type: ignore[arg-type]


def test_rho_is_positive_and_matches_requested_initial_value() -> None:
    model = NeuralReactionDiffusion(initial_rho=0.25)

    assert model.rho.item() > 0.0
    assert model.rho.item() == pytest.approx(0.25, rel=1e-6, abs=1e-7)


def test_rho_gradient_flows_to_raw_rho() -> None:
    model = NeuralReactionDiffusion(initial_rho=0.25)

    model.rho.backward()

    assert model.raw_rho.grad is not None
    assert torch.isfinite(model.raw_rho.grad)
    assert model.raw_rho.grad.item() > 0.0


def test_reaction_single_state_shape() -> None:
    model = NeuralReactionDiffusion(hidden_width=4)
    y = torch.tensor([0.1, 0.2, 0.3])

    reaction = model.reaction(y)

    assert reaction.shape == y.shape


def test_reaction_batched_state_shape() -> None:
    model = NeuralReactionDiffusion(hidden_width=4)
    y = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    reaction = model.reaction(y)

    assert reaction.shape == y.shape


def test_reaction_dtype_and_device_match_model_and_state() -> None:
    model = NeuralReactionDiffusion(hidden_width=4).double()
    y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

    reaction = model.reaction(y)

    assert reaction.dtype == y.dtype
    assert reaction.device == y.device


def test_reaction_is_shared_scalar_function_applied_independently() -> None:
    model = NeuralReactionDiffusion(hidden_width=2)
    with torch.no_grad():
        first_layer = model.reaction_eq[0]
        second_layer = model.reaction_eq[2]
        output_layer = model.reaction_eq[4]
        first_layer.weight.fill_(0.7)
        first_layer.bias.zero_()
        second_layer.weight.fill_(0.3)
        second_layer.bias.zero_()
        output_layer.weight.fill_(0.5)
        output_layer.bias.zero_()
    y = torch.tensor([0.1, 0.2, 0.3])
    changed = y.clone()
    changed[1] = 1.2

    reaction = model.reaction(y)
    changed_reaction = model.reaction(changed)

    torch.testing.assert_close(reaction[[0, 2]], changed_reaction[[0, 2]])
    assert not torch.allclose(reaction[1], changed_reaction[1])


def test_reaction_does_not_mutate_input() -> None:
    model = NeuralReactionDiffusion(hidden_width=4)
    y = torch.tensor([0.1, 0.2, 0.3])
    before = y.clone()

    model.reaction(y)

    torch.testing.assert_close(y, before)


def test_forward_single_state_matches_manual_transport_when_reaction_is_zero() -> None:
    model = NeuralReactionDiffusion(hidden_width=4)
    _zero_reaction(model)
    _set_rho(model, 0.25)
    y = torch.tensor([1.0, 2.0, 4.0])
    laplacian = _chain_laplacian()

    derivative = model(y, laplacian)

    torch.testing.assert_close(derivative, -0.25 * (laplacian @ y))


def test_forward_batched_states_with_shared_laplacian() -> None:
    model = NeuralReactionDiffusion(hidden_width=4)
    _zero_reaction(model)
    _set_rho(model, 0.5)
    y = torch.tensor([[1.0, 2.0, 4.0], [0.5, 1.0, 1.5]])
    laplacian = _chain_laplacian()

    derivative = model(y, laplacian)

    torch.testing.assert_close(derivative, -0.5 * (y @ laplacian.T))


def test_forward_batched_states_with_subject_specific_laplacians() -> None:
    model = NeuralReactionDiffusion(hidden_width=4)
    _zero_reaction(model)
    _set_rho(model, 0.5)
    y = torch.tensor([[1.0, 2.0, 4.0], [0.5, 1.0, 1.5]])
    first_laplacian = _chain_laplacian()
    second_laplacian = 2.0 * first_laplacian
    laplacians = torch.stack([first_laplacian, second_laplacian])

    derivative = model(y, laplacians)
    expected = -0.5 * torch.einsum("bij,bj->bi", laplacians, y)

    torch.testing.assert_close(derivative, expected)
    assert not torch.allclose(derivative[0], derivative[1])


def test_forward_output_shape_matches_state_shape() -> None:
    model = NeuralReactionDiffusion(hidden_width=4)
    y = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])
    laplacian = _chain_laplacian()

    assert model(y, laplacian).shape == y.shape


def test_forward_does_not_mutate_state_or_laplacian() -> None:
    model = NeuralReactionDiffusion(hidden_width=4)
    y = torch.tensor([0.1, 0.2, 0.3])
    laplacian = _chain_laplacian()
    y_before = y.clone()
    laplacian_before = laplacian.clone()

    model(y, laplacian)

    torch.testing.assert_close(y, y_before)
    torch.testing.assert_close(laplacian, laplacian_before)


@pytest.mark.parametrize(
    "state",
    [
        torch.tensor(1.0),
        torch.zeros((1, 2, 3)),
        torch.tensor([1, 2, 3]),
        torch.tensor([True, False]),
        torch.tensor([1.0 + 1.0j]),
        torch.empty(0),
        torch.tensor([1.0, math.inf]),
        torch.tensor([1.0, math.nan]),
    ],
)
def test_rejects_invalid_states(state: torch.Tensor) -> None:
    model = NeuralReactionDiffusion()

    with pytest.raises(SchemaError):
        model.reaction(state)


@pytest.mark.parametrize(
    "laplacian",
    [
        torch.tensor(1.0),
        torch.zeros(3),
        torch.zeros((1, 3, 3, 1)),
        torch.zeros((2, 3)),
        torch.zeros((2, 2)),
        torch.tensor([[0.0, math.inf, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ],
)
def test_rejects_invalid_laplacian_dimensions_values_and_shapes(
    laplacian: torch.Tensor,
) -> None:
    model = NeuralReactionDiffusion()
    y = torch.tensor([0.1, 0.2, 0.3])

    with pytest.raises(SchemaError):
        model(y, laplacian)


def test_rejects_batch_size_mismatch() -> None:
    model = NeuralReactionDiffusion()
    y = torch.zeros((2, 3))
    laplacians = torch.zeros((3, 3, 3))

    with pytest.raises(SchemaError, match="expected"):
        model(y, laplacians)


def test_rejects_regional_dimension_mismatch() -> None:
    model = NeuralReactionDiffusion()
    y = torch.zeros((2, 3))
    laplacian = torch.zeros((4, 4))

    with pytest.raises(SchemaError, match="expected"):
        model(y, laplacian)


def test_rejects_laplacian_dtype_mismatch() -> None:
    model = NeuralReactionDiffusion().double()
    y = torch.zeros(3, dtype=torch.float64)
    laplacian = torch.zeros((3, 3), dtype=torch.float32)

    with pytest.raises(SchemaError, match="dtype"):
        model(y, laplacian)


def test_rejects_state_dtype_mismatch_with_model() -> None:
    model = NeuralReactionDiffusion()
    y = torch.zeros(3, dtype=torch.float64)

    with pytest.raises(SchemaError, match="dtype"):
        model.reaction(y)


def test_rejects_batched_laplacian_for_single_state() -> None:
    model = NeuralReactionDiffusion()
    y = torch.zeros(3)
    laplacian = torch.zeros((1, 3, 3))

    with pytest.raises(SchemaError, match="batched"):
        model(y, laplacian)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rejects_laplacian_device_mismatch_when_cuda_is_available() -> None:
    model = NeuralReactionDiffusion().cuda()
    y = torch.zeros(3, device="cuda")
    laplacian = torch.zeros((3, 3), device="cpu")

    with pytest.raises(SchemaError, match="device"):
        model(y, laplacian)


@pytest.mark.parametrize(
    "times",
    [
        torch.tensor(0.0),
        torch.zeros((1, 2)),
        torch.tensor([0, 1]),
        torch.empty(0),
        torch.tensor([0.0, math.inf]),
        torch.tensor([0.0, math.nan]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.2, 0.1]),
    ],
)
def test_integrate_rejects_invalid_times(times: torch.Tensor) -> None:
    model = NeuralReactionDiffusion()

    with pytest.raises(SchemaError):
        model.integrate(torch.zeros(3), _chain_laplacian(), times)


def test_integrate_rejects_time_dtype_mismatch() -> None:
    model = NeuralReactionDiffusion().double()
    y0 = torch.zeros(3, dtype=torch.float64)
    laplacian = _chain_laplacian(dtype=torch.float64)
    times = torch.tensor([0.0, 1.0], dtype=torch.float32)

    with pytest.raises(SchemaError, match="times dtype"):
        model.integrate(y0, laplacian, times)


@pytest.mark.parametrize("max_step", [0.0, -0.1, math.inf, math.nan, True, "0.05"])
def test_integrate_rejects_invalid_max_step(max_step: object) -> None:
    model = NeuralReactionDiffusion()

    with pytest.raises(SchemaError, match="max_step"):
        model.integrate(
            torch.zeros(3),
            _chain_laplacian(),
            torch.tensor([0.0, 1.0]),
            max_step=max_step,  # type: ignore[arg-type]
        )


class ConstantDerivativeModel(NeuralReactionDiffusion):
    def __init__(self, derivative: float) -> None:
        super().__init__(hidden_width=1)
        self.derivative = derivative
        self.calls = 0

    def forward(self, y: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.full_like(y, self.derivative)


class LinearRateModel(NeuralReactionDiffusion):
    def __init__(self, rate: float) -> None:
        super().__init__(hidden_width=1)
        self.rate = rate

    def forward(self, y: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        return self.rate * y


class RecordingConstantModel(ConstantDerivativeModel):
    def __init__(self, derivative: float) -> None:
        super().__init__(derivative)
        self.seen_states: list[torch.Tensor] = []

    def forward(self, y: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        self.seen_states.append(y.detach().clone())
        return super().forward(y, laplacian)


def test_rk4_integrates_constant_derivative_exactly() -> None:
    model = ConstantDerivativeModel(derivative=2.0)
    y0 = torch.tensor([1.0, -1.0, 0.5])
    times = torch.tensor([0.0, 0.12, 0.5])

    trajectory = model.integrate(y0, _chain_laplacian(), times, max_step=0.05)

    expected = y0.unsqueeze(0) + 2.0 * (times - times[0]).unsqueeze(1)
    torch.testing.assert_close(trajectory, expected)


def test_rk4_integrates_exponential_growth_accurately_with_irregular_times() -> None:
    model = LinearRateModel(rate=0.4)
    y0 = torch.tensor([1.0, 2.0, 3.0])
    times = torch.tensor([1.0, 1.1, 1.35, 1.7])

    trajectory = model.integrate(y0, _chain_laplacian(), times, max_step=0.01)

    expected = y0.unsqueeze(0) * torch.exp(0.4 * (times - times[0])).unsqueeze(1)
    torch.testing.assert_close(trajectory, expected, rtol=1e-5, atol=1e-6)


def test_integrate_single_requested_time_returns_initial_clone_without_rhs_call() -> None:
    model = ConstantDerivativeModel(derivative=2.0)
    y0 = torch.tensor([1.0, 2.0, 3.0])

    trajectory = model.integrate(y0, _chain_laplacian(), torch.tensor([2.5]))

    assert trajectory.shape == (1, 3)
    torch.testing.assert_close(trajectory[0], y0)
    assert trajectory.data_ptr() != y0.data_ptr()
    assert model.calls == 0


def test_internal_substeps_do_not_become_output_rows() -> None:
    model = ConstantDerivativeModel(derivative=1.0)

    trajectory = model.integrate(
        torch.zeros(3),
        _chain_laplacian(),
        torch.tensor([0.0, 0.2]),
        max_step=0.05,
    )

    assert trajectory.shape == (2, 3)


def test_nondivisible_interval_uses_equal_subdivision() -> None:
    model = RecordingConstantModel(derivative=1.0)

    model.integrate(
        torch.zeros(1),
        torch.zeros((1, 1)),
        torch.tensor([0.0, 0.12]),
        max_step=0.05,
    )

    first_step_states = torch.stack(model.seen_states[:4]).squeeze(-1)
    torch.testing.assert_close(
        first_step_states,
        torch.tensor([0.0, 0.02, 0.02, 0.04]),
        rtol=1e-6,
        atol=1e-7,
    )
    assert model.calls == 12


def test_integrate_batched_trajectory_shape() -> None:
    model = ConstantDerivativeModel(derivative=1.0)
    y0 = torch.zeros((2, 3))
    times = torch.tensor([0.0, 0.1, 0.2])

    trajectory = model.integrate(y0, _chain_laplacian(), times, max_step=0.05)

    assert trajectory.shape == (3, 2, 3)


def test_smaller_max_step_reduces_error_for_smooth_system() -> None:
    model = LinearRateModel(rate=1.0)
    y0 = torch.tensor([1.0])
    laplacian = torch.zeros((1, 1))
    times = torch.tensor([0.0, 1.0])
    expected = y0 * math.e

    coarse = model.integrate(y0, laplacian, times, max_step=0.5)[-1]
    fine = model.integrate(y0, laplacian, times, max_step=0.1)[-1]

    assert torch.abs(fine - expected).item() < torch.abs(coarse - expected).item()


def test_integration_preserves_parameter_and_initial_state_gradients() -> None:
    model = NeuralReactionDiffusion(hidden_width=3, initial_rho=0.2)
    with torch.no_grad():
        for parameter in model.reaction_eq.parameters():
            parameter.fill_(0.1)
    y0 = torch.tensor([0.2, 0.4, 0.7], requires_grad=True)
    times = torch.tensor([0.0, 0.1, 0.2])

    trajectory = model.integrate(y0, _chain_laplacian(), times, max_step=0.05)
    loss = (trajectory[-1] ** 2).sum()
    loss.backward()

    assert trajectory.grad_fn is not None
    assert model.raw_rho.grad is not None
    assert torch.isfinite(model.raw_rho.grad)
    assert model.raw_rho.grad.abs().item() > 0.0
    assert y0.grad is not None
    assert torch.isfinite(y0.grad).all()
    for parameter in model.reaction_eq.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_integrate_source_does_not_use_numpy_detach_or_no_grad() -> None:
    source = inspect.getsource(NeuralReactionDiffusion.integrate)
    source += inspect.getsource(NeuralReactionDiffusion._torch_rk4_step)

    assert ".detach(" not in source
    assert ".numpy(" not in source
    assert "no_grad" not in source


def test_forward_raises_when_computed_derivative_becomes_nonfinite() -> None:
    model = NeuralReactionDiffusion(hidden_width=4)
    _constant_reaction(model, math.inf)

    with pytest.raises(SchemaError, match="reaction"):
        model(torch.zeros(3), _chain_laplacian())


def test_import_has_no_output_pysr_dependency_or_global_laplacian() -> None:
    code = (
        "import sys\n"
        "import tau_sr.neural as neural\n"
        "assert 'pysr' not in sys.modules\n"
        "assert not hasattr(neural, 'L')\n"
        "assert not hasattr(neural, 'W')\n"
        "assert not hasattr(neural, 'model')\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_neural_module_exports_only_canonical_public_model() -> None:
    assert neural.__all__ == ("NeuralReactionDiffusion",)
    assert neural.NeuralReactionDiffusion is NeuralReactionDiffusion
