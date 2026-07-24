"""Subject-level neural training for noise-free synthetic recovery datasets."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch

from tau_sr.neural import NeuralReactionDiffusion
from tau_sr.reactions import make_linear_quadratic_reaction
from tau_sr.recovery import SyntheticRecoveryDataset
from tau_sr.schemas import SchemaError

__all__ = (
    "NeuralRecoveryMetrics",
    "NeuralTrainingConfig",
    "NeuralTrainingResult",
    "SubjectSplit",
    "evaluate_neural_recovery",
    "fit_noise_free_recovery_model",
    "make_subject_split",
)


@dataclass(frozen=True)
class SubjectSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        train_indices = _validate_index_tuple(self.train_indices, "train_indices")
        validation_indices = _validate_index_tuple(
            self.validation_indices,
            "validation_indices",
        )
        test_indices = _validate_index_tuple(self.test_indices, "test_indices")

        seen: set[int] = set()
        for field_name, indices in (
            ("train_indices", train_indices),
            ("validation_indices", validation_indices),
            ("test_indices", test_indices),
        ):
            overlap = seen.intersection(indices)
            if overlap:
                raise SchemaError(f"{field_name} overlap with another split.")
            seen.update(indices)

        object.__setattr__(self, "train_indices", train_indices)
        object.__setattr__(self, "validation_indices", validation_indices)
        object.__setattr__(self, "test_indices", test_indices)


@dataclass(frozen=True)
class NeuralTrainingConfig:
    hidden_width: int = 16
    initial_rho: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 500
    patience: int = 50
    min_delta: float = 1e-7
    max_step: float = 0.05
    seed: int = 42
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self) -> None:
        _require_int(self.hidden_width, "hidden_width", minimum=1)
        _require_int(self.max_epochs, "max_epochs", minimum=1)
        _require_int(self.patience, "patience", minimum=1)
        _require_int(self.seed, "seed", minimum=0)
        initial_rho = _require_real_scalar(self.initial_rho, "initial_rho", positive=True)
        learning_rate = _require_real_scalar(
            self.learning_rate,
            "learning_rate",
            positive=True,
        )
        weight_decay = _require_real_scalar(
            self.weight_decay,
            "weight_decay",
            minimum=0.0,
        )
        min_delta = _require_real_scalar(self.min_delta, "min_delta", minimum=0.0)
        max_step = _require_real_scalar(self.max_step, "max_step", positive=True)
        dtype = _validate_torch_dtype(self.dtype)
        device = _validate_torch_device(self.device)

        object.__setattr__(self, "initial_rho", initial_rho)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(self, "min_delta", min_delta)
        object.__setattr__(self, "max_step", max_step)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "device", str(device))


@dataclass(frozen=True)
class NeuralTrainingResult:
    model: NeuralReactionDiffusion
    split: SubjectSplit
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    epochs_run: int

    def __post_init__(self) -> None:
        if not isinstance(self.model, NeuralReactionDiffusion):
            raise SchemaError("model must be a NeuralReactionDiffusion.")
        if not isinstance(self.split, SubjectSplit):
            raise SchemaError("split must be a SubjectSplit.")
        training_losses = _validate_loss_history(self.training_losses, "training_losses")
        validation_losses = _validate_loss_history(self.validation_losses, "validation_losses")
        if len(training_losses) != len(validation_losses):
            raise SchemaError("training_losses and validation_losses must have equal length.")
        _require_int(self.epochs_run, "epochs_run", minimum=1)
        if self.epochs_run != len(training_losses):
            raise SchemaError("epochs_run must match loss-history length.")
        _require_int(self.best_epoch, "best_epoch", minimum=0)
        if self.best_epoch >= self.epochs_run:
            raise SchemaError("best_epoch must be less than epochs_run.")

        object.__setattr__(self, "training_losses", training_losses)
        object.__setattr__(self, "validation_losses", validation_losses)


@dataclass(frozen=True)
class NeuralRecoveryMetrics:
    subject_indices: tuple[int, ...]
    trajectory_mse: float
    trajectory_rmse: float
    trajectory_mae: float
    estimated_rho: float
    true_rho: float
    rho_absolute_error: float
    rho_relative_error: float
    reaction_curve_rmse: float
    reaction_curve_mae: float

    def __post_init__(self) -> None:
        subject_indices = _validate_index_tuple(self.subject_indices, "subject_indices")
        if not subject_indices:
            raise SchemaError("subject_indices must be nonempty.")
        for field_name in (
            "trajectory_mse",
            "trajectory_rmse",
            "trajectory_mae",
            "estimated_rho",
            "true_rho",
            "rho_absolute_error",
            "rho_relative_error",
            "reaction_curve_rmse",
            "reaction_curve_mae",
        ):
            value = _require_real_scalar(getattr(self, field_name), field_name)
            if value < 0.0:
                raise SchemaError(f"{field_name} must be nonnegative.")
        object.__setattr__(self, "subject_indices", subject_indices)


def make_subject_split(
    n_subjects: int,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> SubjectSplit:
    """Create a deterministic complete subject-level train/validation/test split."""

    _require_int(n_subjects, "n_subjects", minimum=3)
    validation_fraction_value = _require_real_scalar(
        validation_fraction,
        "validation_fraction",
        positive=True,
    )
    test_fraction_value = _require_real_scalar(test_fraction, "test_fraction", positive=True)
    _require_int(seed, "seed", minimum=0)

    train_fraction = 1.0 - validation_fraction_value - test_fraction_value
    if train_fraction <= 0.0:
        raise SchemaError("train_fraction must be positive.")

    validation_count = max(1, math.floor(n_subjects * validation_fraction_value))
    test_count = max(1, math.floor(n_subjects * test_fraction_value))
    while validation_count + test_count >= n_subjects:
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise SchemaError("n_subjects is too small for nonempty splits.")

    rng = np.random.default_rng(seed)
    shuffled = tuple(int(index) for index in rng.permutation(n_subjects))
    validation_indices = tuple(sorted(shuffled[:validation_count]))
    test_start = validation_count
    test_stop = validation_count + test_count
    test_indices = tuple(sorted(shuffled[test_start:test_stop]))
    train_indices = tuple(sorted(shuffled[test_stop:]))

    return SubjectSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def fit_noise_free_recovery_model(
    dataset: SyntheticRecoveryDataset,
    split: SubjectSplit,
    config: NeuralTrainingConfig,
) -> NeuralTrainingResult:
    """Fit a new neural reaction-diffusion model using train and validation subjects only."""

    _require_dataset(dataset)
    checked_split = _validate_split_for_dataset(split, dataset)
    checked_config = _validate_training_config(config)
    device = torch.device(checked_config.device)
    dtype = checked_config.dtype

    _validate_fit_subjects(dataset, checked_split.train_indices)
    _validate_fit_subjects(dataset, checked_split.validation_indices)

    model = _new_model(checked_config, device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=checked_config.learning_rate,
        weight_decay=checked_config.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    training_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(checked_config.max_epochs):
        model.train()
        optimizer.zero_grad()
        training_loss = _split_trajectory_loss(
            model,
            dataset,
            checked_split.train_indices,
            max_step=checked_config.max_step,
            device=device,
            dtype=dtype,
        )
        _validate_finite_tensor(training_loss, "training loss")
        training_loss.backward()
        _validate_finite_gradients(model)
        optimizer.step()
        _validate_finite_parameters(model)

        model.eval()
        with torch.no_grad():
            validation_loss = _split_trajectory_loss(
                model,
                dataset,
                checked_split.validation_indices,
                max_step=checked_config.max_step,
                device=device,
                dtype=dtype,
            )
        _validate_finite_tensor(validation_loss, "validation loss")

        training_loss_value = float(training_loss.detach().cpu().item())
        validation_loss_value = float(validation_loss.detach().cpu().item())
        training_losses.append(training_loss_value)
        validation_losses.append(validation_loss_value)

        if validation_loss_value < best_validation_loss - checked_config.min_delta:
            best_validation_loss = validation_loss_value
            best_epoch = epoch
            best_state = _clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= checked_config.patience:
                break

    if best_state is None:
        raise SchemaError("training did not produce a finite validation state.")

    model.load_state_dict(best_state)
    model.eval()

    return NeuralTrainingResult(
        model=model,
        split=checked_split,
        training_losses=tuple(training_losses),
        validation_losses=tuple(validation_losses),
        best_epoch=best_epoch,
        epochs_run=len(training_losses),
    )


def evaluate_neural_recovery(
    model: NeuralReactionDiffusion,
    dataset: SyntheticRecoveryDataset,
    subject_indices: Sequence[int],
    reaction_grid: np.ndarray,
    *,
    max_step: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> NeuralRecoveryMetrics:
    """Evaluate held-out trajectories, diffusion coefficient, and reaction curve."""

    if not isinstance(model, NeuralReactionDiffusion):
        raise SchemaError("model must be a NeuralReactionDiffusion.")
    _require_dataset(dataset)
    checked_indices = _validate_indices_for_dataset(
        subject_indices,
        "subject_indices",
        dataset,
    )
    if not checked_indices:
        raise SchemaError("subject_indices must be nonempty.")
    max_step_value = _require_real_scalar(max_step, "max_step", positive=True)
    torch_device = _validate_torch_device(device)
    torch_dtype = _validate_torch_dtype(dtype)
    _validate_model_device_dtype(model, device=torch_device, dtype=torch_dtype)
    grid = _validate_reaction_grid(reaction_grid)
    true_rho = _require_real_scalar(dataset.rho, "dataset.rho", positive=True)

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            subject_mse: list[float] = []
            subject_mae: list[float] = []
            for subject_index in checked_indices:
                loss_inputs = _subject_tensors(
                    dataset,
                    subject_index,
                    device=torch_device,
                    dtype=torch_dtype,
                    require_two_visits=True,
                )
                prediction = model.integrate(
                    loss_inputs.y0,
                    loss_inputs.laplacian,
                    loss_inputs.times,
                    max_step=max_step_value,
                )
                _validate_prediction_tensor(
                    prediction,
                    f"prediction for subject index {subject_index}",
                    expected_shape=loss_inputs.target.shape,
                )
                residual = prediction[1:] - loss_inputs.target[1:]
                subject_mse.append(float(torch.mean(residual**2).detach().cpu().item()))
                subject_mae.append(float(torch.mean(torch.abs(residual)).detach().cpu().item()))

            grid_tensor = torch.as_tensor(
                grid.reshape(-1, 1),
                device=torch_device,
                dtype=torch_dtype,
            )
            learned_reaction = model.reaction(grid_tensor)
            _validate_prediction_tensor(
                learned_reaction,
                "learned reaction curve",
                expected_shape=grid_tensor.shape,
            )
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    true_reaction = make_linear_quadratic_reaction(
        dataset.growth_rate,
        dataset.saturation_rate,
    )
    learned_reaction_np = learned_reaction.detach().cpu().numpy().reshape(-1).copy()
    true_reaction_np = true_reaction(grid.copy())
    if true_reaction_np.shape != grid.shape:
        raise SchemaError(
            "true reaction curve has shape "
            f"{true_reaction_np.shape}; expected {grid.shape}."
        )
    if not np.isfinite(true_reaction_np).all():
        raise SchemaError("true reaction curve must contain only finite values.")

    trajectory_mse = _mean_checked(subject_mse, "trajectory_mse")
    trajectory_mae = _mean_checked(subject_mae, "trajectory_mae")
    trajectory_rmse = math.sqrt(trajectory_mse)
    estimated_rho = float(model.rho.detach().cpu().item())
    rho_absolute_error = abs(estimated_rho - true_rho)
    rho_relative_error = rho_absolute_error / true_rho
    reaction_residual = learned_reaction_np - true_reaction_np
    reaction_curve_rmse = math.sqrt(float(np.mean(reaction_residual**2)))
    reaction_curve_mae = float(np.mean(np.abs(reaction_residual)))

    return NeuralRecoveryMetrics(
        subject_indices=checked_indices,
        trajectory_mse=trajectory_mse,
        trajectory_rmse=trajectory_rmse,
        trajectory_mae=trajectory_mae,
        estimated_rho=estimated_rho,
        true_rho=true_rho,
        rho_absolute_error=rho_absolute_error,
        rho_relative_error=rho_relative_error,
        reaction_curve_rmse=reaction_curve_rmse,
        reaction_curve_mae=reaction_curve_mae,
    )


@dataclass(frozen=True)
class _SubjectTensors:
    y0: torch.Tensor
    laplacian: torch.Tensor
    times: torch.Tensor
    target: torch.Tensor


def _new_model(
    config: NeuralTrainingConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> NeuralReactionDiffusion:
    devices: list[int] = []
    if device.type == "cuda":
        devices = [0 if device.index is None else device.index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        model = NeuralReactionDiffusion(
            hidden_width=config.hidden_width,
            initial_rho=config.initial_rho,
        )
    return model.to(device=device, dtype=dtype)


def _split_trajectory_loss(
    model: NeuralReactionDiffusion,
    dataset: SyntheticRecoveryDataset,
    subject_indices: Sequence[int],
    *,
    max_step: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    subject_losses = [
        _subject_trajectory_loss(
            model,
            dataset,
            subject_index,
            max_step=max_step,
            device=device,
            dtype=dtype,
        )
        for subject_index in subject_indices
    ]
    if not subject_losses:
        raise SchemaError("subject_indices must be nonempty.")
    return torch.stack(subject_losses).mean()


def _subject_trajectory_loss(
    model: NeuralReactionDiffusion,
    dataset: SyntheticRecoveryDataset,
    subject_index: int,
    *,
    max_step: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    subject = _subject_tensors(
        dataset,
        subject_index,
        device=device,
        dtype=dtype,
        require_two_visits=True,
    )
    prediction = model.integrate(
        subject.y0,
        subject.laplacian,
        subject.times,
        max_step=max_step,
    )
    _validate_prediction_tensor(
        prediction,
        f"prediction for subject index {subject_index}",
        expected_shape=subject.target.shape,
    )
    return torch.mean((prediction[1:] - subject.target[1:]) ** 2)


def _subject_tensors(
    dataset: SyntheticRecoveryDataset,
    subject_index: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    require_two_visits: bool,
) -> _SubjectTensors:
    subject_count = _dataset_subject_count(dataset)
    if subject_index < 0 or subject_index >= subject_count:
        raise SchemaError(f"subject index {subject_index} is outside the dataset range.")
    n_regions = _dataset_region_count(dataset)

    y0 = _numpy_vector(dataset.baseline_states[subject_index], f"baseline_states[{subject_index}]")
    laplacian = _numpy_matrix(
        dataset.laplacians[subject_index],
        f"laplacians[{subject_index}]",
        expected_shape=(n_regions, n_regions),
    )
    times = _numpy_vector(
        dataset.times_by_subject[subject_index],
        f"times_by_subject[{subject_index}]",
    )
    target = _numpy_matrix(
        dataset.trajectories[subject_index],
        f"trajectories[{subject_index}]",
        expected_shape=(times.shape[0], n_regions),
    )
    if y0.shape != (n_regions,):
        raise SchemaError(
            f"baseline_states[{subject_index}] has shape {y0.shape}; expected ({n_regions},)."
        )
    if require_two_visits and times.shape[0] < 2:
        raise SchemaError(f"subject index {subject_index} must have at least two visits.")

    return _SubjectTensors(
        y0=torch.tensor(y0.copy(), device=device, dtype=dtype),
        laplacian=torch.tensor(laplacian.copy(), device=device, dtype=dtype),
        times=torch.tensor(times.copy(), device=device, dtype=dtype),
        target=torch.tensor(target.copy(), device=device, dtype=dtype),
    )


def _validate_fit_subjects(
    dataset: SyntheticRecoveryDataset,
    subject_indices: Sequence[int],
) -> None:
    for subject_index in subject_indices:
        _subject_tensors(
            dataset,
            subject_index,
            device=torch.device("cpu"),
            dtype=torch.float64,
            require_two_visits=True,
        )


def _validate_split_for_dataset(
    split: SubjectSplit,
    dataset: SyntheticRecoveryDataset,
) -> SubjectSplit:
    if not isinstance(split, SubjectSplit):
        raise SchemaError("split must be a SubjectSplit.")
    checked_split = SubjectSplit(
        train_indices=split.train_indices,
        validation_indices=split.validation_indices,
        test_indices=split.test_indices,
    )
    subject_count = _dataset_subject_count(dataset)
    all_indices = (
        checked_split.train_indices
        + checked_split.validation_indices
        + checked_split.test_indices
    )
    if set(all_indices) != set(range(subject_count)):
        raise SchemaError("split indices must cover every subject exactly once.")
    return checked_split


def _validate_indices_for_dataset(
    indices: object,
    field_name: str,
    dataset: SyntheticRecoveryDataset,
) -> tuple[int, ...]:
    checked = _validate_index_tuple(indices, field_name)
    subject_count = _dataset_subject_count(dataset)
    for index in checked:
        if index >= subject_count:
            raise SchemaError(f"{field_name} contains index outside the dataset range.")
    return checked


def _validate_index_tuple(indices: object, field_name: str) -> tuple[int, ...]:
    if isinstance(indices, np.ndarray):
        raw_values = tuple(indices.tolist())
    elif isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise SchemaError(f"{field_name} must be a sequence of subject indices.")
    else:
        raw_values = tuple(indices)
    if not raw_values:
        raise SchemaError(f"{field_name} must be nonempty.")

    checked: list[int] = []
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError(f"{field_name} must contain integer subject indices.")
        if value < 0:
            raise SchemaError(f"{field_name} must contain nonnegative subject indices.")
        checked.append(value)
    if len(set(checked)) != len(checked):
        raise SchemaError(f"{field_name} must not contain duplicate indices.")
    return tuple(checked)


def _validate_training_config(config: NeuralTrainingConfig) -> NeuralTrainingConfig:
    if not isinstance(config, NeuralTrainingConfig):
        raise SchemaError("config must be a NeuralTrainingConfig.")
    return NeuralTrainingConfig(
        hidden_width=config.hidden_width,
        initial_rho=config.initial_rho,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        max_step=config.max_step,
        seed=config.seed,
        dtype=config.dtype,
        device=config.device,
    )


def _require_dataset(dataset: object) -> SyntheticRecoveryDataset:
    if not isinstance(dataset, SyntheticRecoveryDataset):
        raise SchemaError("dataset must be a SyntheticRecoveryDataset.")
    if _dataset_subject_count(dataset) < 1:
        raise SchemaError("dataset must contain at least one subject.")
    if _dataset_region_count(dataset) < 1:
        raise SchemaError("dataset must contain at least one ROI label.")
    return dataset


def _dataset_subject_count(dataset: SyntheticRecoveryDataset) -> int:
    return len(dataset.subject_ids)


def _dataset_region_count(dataset: SyntheticRecoveryDataset) -> int:
    return len(dataset.roi_labels)


def _numpy_vector(value: object, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise SchemaError(f"{field_name} must be a NumPy array.")
    if value.ndim != 1:
        raise SchemaError(f"{field_name} must be one-dimensional.")
    if value.size == 0:
        raise SchemaError(f"{field_name} must be nonempty.")
    if not _is_real_numeric_dtype(value):
        raise SchemaError(f"{field_name} must have a real numeric dtype.")
    if not np.isfinite(value).all():
        raise SchemaError(f"{field_name} must contain only finite values.")
    return value.astype(float, copy=True)


def _numpy_matrix(
    value: object,
    field_name: str,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise SchemaError(f"{field_name} must be a NumPy array.")
    if value.ndim != 2:
        raise SchemaError(f"{field_name} must be two-dimensional.")
    if expected_shape is not None and value.shape != expected_shape:
        raise SchemaError(f"{field_name} has shape {value.shape}; expected {expected_shape}.")
    if not _is_real_numeric_dtype(value):
        raise SchemaError(f"{field_name} must have a real numeric dtype.")
    if not np.isfinite(value).all():
        raise SchemaError(f"{field_name} must contain only finite values.")
    return value.astype(float, copy=True)


def _validate_reaction_grid(reaction_grid: np.ndarray) -> np.ndarray:
    if isinstance(reaction_grid, np.ndarray) and not np.issubdtype(
        reaction_grid.dtype,
        np.floating,
    ):
        raise SchemaError("reaction_grid must have a real floating dtype.")
    grid = _numpy_vector(reaction_grid, "reaction_grid")
    if grid.shape[0] > 1 and not np.all(np.diff(grid) > 0.0):
        raise SchemaError("reaction_grid must be strictly increasing.")
    return grid


def _validate_prediction_tensor(
    value: torch.Tensor,
    field_name: str,
    *,
    expected_shape: torch.Size | tuple[int, ...],
) -> None:
    if not isinstance(value, torch.Tensor):
        raise SchemaError(f"{field_name} must be a torch.Tensor.")
    if tuple(value.shape) != tuple(expected_shape):
        raise SchemaError(
            f"{field_name} has shape {tuple(value.shape)}; expected {tuple(expected_shape)}."
        )
    _validate_finite_tensor(value, field_name)


def _validate_finite_tensor(value: torch.Tensor, field_name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise SchemaError(f"{field_name} must contain only finite values.")


def _validate_finite_gradients(model: NeuralReactionDiffusion) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise SchemaError(f"gradient for {name} is missing.")
        _validate_finite_tensor(parameter.grad, f"gradient for {name}")


def _validate_finite_parameters(model: NeuralReactionDiffusion) -> None:
    for name, parameter in model.named_parameters():
        _validate_finite_tensor(parameter.detach(), f"model parameter {name}")


def _validate_model_device_dtype(
    model: NeuralReactionDiffusion,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    reference = next(model.parameters())
    if reference.device != device:
        raise SchemaError("model parameters must be on the requested device.")
    if reference.dtype != dtype:
        raise SchemaError("model parameter dtype must match requested dtype.")


def _clone_state_dict(model: NeuralReactionDiffusion) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def _validate_loss_history(values: object, field_name: str) -> tuple[float, ...]:
    if isinstance(values, np.ndarray):
        raw_values = tuple(values.tolist())
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SchemaError(f"{field_name} must be a sequence of losses.")
    else:
        raw_values = tuple(values)
    if not raw_values:
        raise SchemaError(f"{field_name} must be nonempty.")
    checked = tuple(_require_real_scalar(value, field_name, minimum=0.0) for value in raw_values)
    return checked


def _mean_checked(values: Sequence[float], field_name: str) -> float:
    if not values:
        raise SchemaError(f"{field_name} requires at least one value.")
    value = float(np.mean(np.asarray(values, dtype=float)))
    if not math.isfinite(value):
        raise SchemaError(f"{field_name} must be finite.")
    if value < 0.0:
        raise SchemaError(f"{field_name} must be nonnegative.")
    return value


def _require_int(value: object, field_name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field_name} must be an integer.")
    if value < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")


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


def _validate_torch_dtype(dtype: object) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise SchemaError("dtype must be a torch.dtype.")
    try:
        sample = torch.empty((), dtype=dtype)
    except (TypeError, RuntimeError) as exc:
        raise SchemaError("dtype must be supported by PyTorch.") from exc
    if not torch.is_floating_point(sample):
        raise SchemaError("dtype must be a real floating torch dtype.")
    if dtype.is_complex:
        raise SchemaError("dtype must be a real floating torch dtype.")
    return dtype


def _validate_torch_device(device: object) -> torch.device:
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise SchemaError("device must be a valid torch device.") from exc
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise SchemaError("requested CUDA device is not available.")
    if torch_device.type == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise SchemaError("requested MPS device is not available.")
    try:
        torch.empty((), device=torch_device)
    except (RuntimeError, TypeError) as exc:
        raise SchemaError("requested torch device is not available.") from exc
    return torch_device


def _is_real_numeric_dtype(array: np.ndarray) -> bool:
    if array.dtype.kind in {"b", "O", "S", "U", "c"}:
        return False
    return bool(np.issubdtype(array.dtype, np.number))
