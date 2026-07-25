from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import tau_sr.training as training
from tau_sr.identifiability import (
    FixedRhoZeroAnchoredReactionDiffusion,
    OracleReactionDiffusion,
    ZeroAnchoredNeuralReactionDiffusion,
    controlled_reaction_grid,
    environment_versions,
    fit_identifiability_model,
    summarize_identifiability_result,
)
from tau_sr.pysr_lifecycle import (
    PySRWorkerError,
    parse_json_file,
    run_json_worker,
    sha256_file,
    write_json_atomically,
)
from tau_sr.recovery import SyntheticRecoveryDataset
from tau_sr.schemas import SchemaError
from tau_sr.training import (
    NeuralTrainingConfig,
    SubjectSplit,
    fit_noise_free_recovery_model,
)


def _dataset(
    *,
    trajectories: tuple[np.ndarray, ...] | None = None,
    laplacians: np.ndarray | None = None,
) -> SyntheticRecoveryDataset:
    times = tuple(np.array([0.0, 0.1, 0.2], dtype=float) for _ in range(4))
    baseline_states = np.array(
        [
            [0.10, 0.20],
            [0.20, 0.30],
            [0.30, 0.40],
            [0.40, 0.50],
        ],
        dtype=float,
    )
    if trajectories is None:
        trajectories = tuple(
            baseline_states[index][None, :] + time[:, None] * (0.1 + baseline_states[index])
            for index, time in enumerate(times)
        )
    if laplacians is None:
        base = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        laplacians = np.stack([(index + 1) * base for index in range(4)])
    return SyntheticRecoveryDataset(
        subject_ids=tuple(f"synthetic_subject_{index:04d}" for index in range(4)),
        roi_labels=("syn_roi_001", "syn_roi_002"),
        baseline_states=baseline_states.copy(),
        adjacencies=np.zeros((4, 2, 2), dtype=float),
        laplacians=laplacians.copy(),
        times_by_subject=tuple(time.copy() for time in times),
        trajectories=tuple(trajectory.copy() for trajectory in trajectories),
        representation="count",
        laplacian_kind="unnormalized",
        rho=0.1,
        growth_rate=0.5,
        saturation_rate=0.5,
        seed=42,
    )


def _split() -> SubjectSplit:
    return SubjectSplit(train_indices=(0, 1), validation_indices=(2,), test_indices=(3,))


def _config(**overrides: object) -> NeuralTrainingConfig:
    values = {
        "hidden_width": 2,
        "initial_rho": 0.05,
        "learning_rate": 1e-2,
        "rho_learning_rate": None,
        "max_epochs": 3,
        "patience": 3,
        "max_step": 0.1,
        "seed": 42,
        "dtype": torch.float64,
        "device": "cpu",
    }
    values.update(overrides)
    return NeuralTrainingConfig(**values)  # type: ignore[arg-type]


def _zero_anchored_model(seed: int = 42) -> ZeroAnchoredNeuralReactionDiffusion:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ZeroAnchoredNeuralReactionDiffusion(hidden_width=2).double()


def test_fixed_rho_model_does_not_expose_rho_as_trainable_parameter() -> None:
    model = FixedRhoZeroAnchoredReactionDiffusion(fixed_rho=0.1, hidden_width=2)

    assert "raw_rho" not in dict(model.named_parameters())
    assert "raw_rho" in dict(model.named_buffers())
    assert model.rho.item() == pytest.approx(0.1)


def test_oracle_reaction_model_exposes_only_rho_as_trainable_parameter() -> None:
    model = OracleReactionDiffusion(growth_rate=0.5, saturation_rate=0.5)

    assert tuple(name for name, _parameter in model.named_parameters()) == ("raw_rho",)
    assert not hasattr(model, "reaction_eq")


def test_zero_anchored_reaction_is_zero_at_origin_across_initializations() -> None:
    for seed in (1, 2, 3):
        model = _zero_anchored_model(seed)
        y = torch.zeros((2, 3), dtype=torch.float64)

        reaction = model.reaction(y)

        torch.testing.assert_close(reaction, torch.zeros_like(y), atol=0.0, rtol=0.0)


def test_optimizer_groups_use_requested_learning_rates() -> None:
    model = _zero_anchored_model()
    grouped_config = _config(learning_rate=1e-3, rho_learning_rate=5e-3)

    optimizer = training._make_optimizer(model, grouped_config)

    assert [group["name"] for group in optimizer.param_groups] == ["reaction", "rho"]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 5e-3]


def test_default_optimizer_behavior_remains_single_learning_rate() -> None:
    model = _zero_anchored_model()

    optimizer = training._make_optimizer(model, _config(learning_rate=2e-3))

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 2e-3


@pytest.mark.parametrize("rho_learning_rate", [0.0, -1e-3, math.inf, math.nan, True])
def test_invalid_rho_learning_rates_are_rejected(rho_learning_rate: object) -> None:
    with pytest.raises(SchemaError, match="rho_learning_rate"):
        _config(rho_learning_rate=rho_learning_rate)


def test_supplied_model_training_and_validation_best_restore_rho_and_reaction() -> None:
    dataset = _dataset()
    split = _split()
    config = _config(max_epochs=4, rho_learning_rate=5e-3)
    model = _zero_anchored_model()

    result = fit_noise_free_recovery_model(dataset, split, config, model=model)

    assert result.model is model
    assert "raw_rho" in result.model.state_dict()
    assert any(name.startswith("reaction_eq") for name in result.model.state_dict())
    with torch.no_grad():
        restored_validation_loss = training._split_trajectory_loss(
            result.model,
            dataset,
            split.validation_indices,
            max_step=config.max_step,
            device=torch.device(config.device),
            dtype=config.dtype,
        )
    assert restored_validation_loss.item() == pytest.approx(min(result.validation_losses))


def test_repeated_identifiability_runs_are_deterministic() -> None:
    dataset = _dataset()
    split = _split()
    config = _config(max_epochs=2, rho_learning_rate=5e-3)

    first, _first_runtime = fit_identifiability_model(
        dataset,
        split,
        config,
        _zero_anchored_model(seed=7),
    )
    second, _second_runtime = fit_identifiability_model(
        dataset,
        split,
        config,
        _zero_anchored_model(seed=7),
    )

    assert first.training_losses == pytest.approx(second.training_losses)
    assert first.validation_losses == pytest.approx(second.validation_losses)
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name])


def test_test_subjects_do_not_enter_training_or_validation() -> None:
    trajectories = list(_dataset().trajectories)
    poisoned_test = trajectories[3].copy()
    poisoned_test[-1, 0] = math.inf
    trajectories[3] = poisoned_test
    dataset = _dataset(trajectories=tuple(trajectories))

    result = fit_noise_free_recovery_model(
        dataset,
        _split(),
        _config(max_epochs=2),
        model=_zero_anchored_model(),
    )

    assert result.epochs_run == 2


def test_subject_specific_laplacians_remain_isolated_in_batched_loss() -> None:
    class RecordingModel(ZeroAnchoredNeuralReactionDiffusion):
        def __init__(self) -> None:
            super().__init__(hidden_width=2)
            self.seen_laplacian: torch.Tensor | None = None

        def integrate(
            self,
            y0: torch.Tensor,
            laplacian: torch.Tensor,
            times: torch.Tensor,
            *,
            max_step: float = 0.05,
        ) -> torch.Tensor:
            del max_step
            self.seen_laplacian = laplacian.detach().clone()
            return torch.zeros(
                (times.shape[0], y0.shape[0], y0.shape[1]),
                dtype=y0.dtype,
                device=y0.device,
            )

    model = RecordingModel().double()
    dataset = _dataset()

    losses = training._subject_group_trajectory_losses(
        model,
        dataset,
        (0, 1),
        max_step=0.1,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert len(losses) == 2
    assert model.seen_laplacian is not None
    assert model.seen_laplacian.shape == (2, 2, 2)
    torch.testing.assert_close(model.seen_laplacian[0], torch.tensor(dataset.laplacians[0]))
    torch.testing.assert_close(model.seen_laplacian[1], torch.tensor(dataset.laplacians[1]))


def test_oracle_models_work_with_rk4_integration() -> None:
    y0 = torch.tensor([0.2, 0.4], dtype=torch.float64)
    laplacian = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
    times = torch.tensor([0.0, 0.1], dtype=torch.float64)

    for model in (
        OracleReactionDiffusion().double(),
        FixedRhoZeroAnchoredReactionDiffusion(hidden_width=2).double(),
    ):
        trajectory = model.integrate(y0, laplacian, times, max_step=0.05)
        assert trajectory.shape == (2, 2)
        assert torch.isfinite(trajectory).all()


def test_result_records_distinguish_validation_and_test_evaluation() -> None:
    dataset = _dataset()
    split = _split()
    config = _config(max_epochs=2)
    training_result, runtime = fit_identifiability_model(
        dataset,
        split,
        config,
        _zero_anchored_model(),
    )
    grid = controlled_reaction_grid(dataset, split, n_points=10)

    validation_only = summarize_identifiability_result(
        experiment_name="zero_anchored",
        candidate_name="rho_lr_1e-3",
        training_result=training_result,
        runtime_seconds=runtime,
        dataset=dataset,
        split=split,
        config=config,
        reaction_grid=grid,
        validation_selected=False,
        evaluate_test=False,
    )
    final = summarize_identifiability_result(
        experiment_name="zero_anchored",
        candidate_name="rho_lr_1e-3",
        training_result=training_result,
        runtime_seconds=runtime,
        dataset=dataset,
        split=split,
        config=config,
        reaction_grid=grid,
        validation_selected=True,
        evaluate_test=True,
    )

    assert validation_only.test_evaluated is False
    assert validation_only.test_trajectory_rmse is None
    assert final.validation_selected is True
    assert final.test_evaluated is True
    assert final.test_trajectory_rmse is not None
    assert final.to_dict()["test_evaluated"] is True


def test_environment_versions_are_machine_readable() -> None:
    versions = environment_versions()

    assert versions["python"]
    assert versions["numpy"]
    assert versions["torch"]


def _write_worker(tmp_path: Path, source: str) -> Path:
    worker_path = tmp_path / "worker.py"
    worker_path.write_text(textwrap.dedent(source), encoding="utf-8")
    return worker_path


def _run_worker(tmp_path: Path, worker_path: Path, *, timeout: float = 2.0):
    result_path = tmp_path / "worker_result.json"
    return run_json_worker(
        [sys.executable, str(worker_path), str(result_path)],
        result_path,
        timeout_seconds=timeout,
        shutdown_grace_seconds=0.2,
        poll_interval_seconds=0.01,
    )


def test_json_worker_normal_completion_returns_payload_and_cleans_temp_logs(
    tmp_path: Path,
) -> None:
    worker_path = _write_worker(
        tmp_path,
        """
        import json
        import sys
        from pathlib import Path

        result = Path(sys.argv[1])
        tmp = result.with_suffix(".tmp")
        tmp.write_text(json.dumps({"status": "ok", "payload": {"value": 3}}))
        tmp.replace(result)
        """,
    )

    result = _run_worker(tmp_path, worker_path)

    assert result.payload == {"value": 3}
    assert result.process_exited_normally is True
    assert result.cleanup_required is False
    assert not list(tmp_path.glob(".worker_*.log"))


def test_json_worker_failure_before_result_is_reported(tmp_path: Path) -> None:
    worker_path = _write_worker(
        tmp_path,
        """
        import sys

        raise SystemExit(7)
        """,
    )

    with pytest.raises(PySRWorkerError, match="failed before producing a result") as exc_info:
        _run_worker(tmp_path, worker_path)

    assert exc_info.value.returncode == 7
    assert exc_info.value.cleanup_required is False


def test_json_worker_rejects_malformed_output(tmp_path: Path) -> None:
    worker_path = _write_worker(
        tmp_path,
        """
        import sys
        from pathlib import Path

        Path(sys.argv[1]).write_text("{")
        """,
    )

    with pytest.raises(PySRWorkerError, match="malformed or incomplete"):
        _run_worker(tmp_path, worker_path)


def test_json_worker_timeout_before_result_cleans_owned_worker(tmp_path: Path) -> None:
    worker_path = _write_worker(
        tmp_path,
        """
        import time

        time.sleep(10)
        """,
    )

    with pytest.raises(PySRWorkerError, match="timed out") as exc_info:
        _run_worker(tmp_path, worker_path, timeout=0.1)

    assert exc_info.value.timed_out is True
    assert exc_info.value.cleanup_required is True
    assert not list(tmp_path.glob(".worker_*.log"))


def test_json_worker_valid_result_then_hung_shutdown_is_contained(tmp_path: Path) -> None:
    worker_path = _write_worker(
        tmp_path,
        """
        import json
        import sys
        import time
        from pathlib import Path

        result = Path(sys.argv[1])
        tmp = result.with_suffix(".tmp")
        tmp.write_text(json.dumps({"status": "ok", "payload": {"complete": True}}))
        tmp.replace(result)
        time.sleep(10)
        """,
    )

    result = _run_worker(tmp_path, worker_path, timeout=2.0)

    assert result.payload == {"complete": True}
    assert result.process_exited_normally is False
    assert result.cleanup_required is True
    assert result.timed_out is True


def test_json_worker_cleanup_targets_only_owned_worker(tmp_path: Path) -> None:
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        start_new_session=True,
    )
    try:
        worker_path = _write_worker(
            tmp_path,
            """
            import json
            import sys
            import time
            from pathlib import Path

            result = Path(sys.argv[1])
            tmp = result.with_suffix(".tmp")
            tmp.write_text(json.dumps({"status": "ok", "payload": {"complete": True}}))
            tmp.replace(result)
            time.sleep(10)
            """,
        )

        result = _run_worker(tmp_path, worker_path, timeout=2.0)

        assert result.cleanup_required is True
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        try:
            sentinel.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            sentinel.kill()
            sentinel.wait(timeout=1.0)


def test_worker_failure_does_not_overwrite_previous_valid_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    write_json_atomically(artifact_path, {"status": "complete", "value": 1})
    original_checksum = sha256_file(artifact_path)
    worker_path = _write_worker(
        tmp_path,
        """
        raise SystemExit(3)
        """,
    )

    with pytest.raises(PySRWorkerError):
        _run_worker(tmp_path, worker_path)

    assert sha256_file(artifact_path) == original_checksum
    assert parse_json_file(artifact_path) == {"status": "complete", "value": 1}


def test_atomic_json_write_is_parseable_and_preserves_existing_on_failure(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "summary.json"
    write_json_atomically(artifact_path, {"numbers": [1, 2, 3]})
    original_checksum = sha256_file(artifact_path)

    with pytest.raises(TypeError):
        write_json_atomically(artifact_path, {"bad": object()})

    assert sha256_file(artifact_path) == original_checksum
    assert parse_json_file(artifact_path) == {"numbers": [1, 2, 3]}
