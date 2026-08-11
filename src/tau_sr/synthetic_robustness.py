"""Controlled synthetic robustness sweep for recovery and symbolic distillation."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from tau_sr.graphs import build_graph_laplacian
from tau_sr.identifiability import (
    CONTROLLED_N_REGIONS,
    CONTROLLED_N_SUBJECTS,
    CONTROLLED_TIMES,
    CONTROLLED_TRUE_GROWTH_RATE,
    CONTROLLED_TRUE_RHO,
    CONTROLLED_TRUE_SATURATION_RATE,
    ZeroAnchoredNeuralReactionDiffusion,
    environment_versions,
)
from tau_sr.pysr_lifecycle import PySRWorkerError, run_json_worker, write_json_atomically
from tau_sr.reactions import make_linear_quadratic_reaction
from tau_sr.recovery import SyntheticRecoveryDataset
from tau_sr.schemas import SchemaError
from tau_sr.simulation import simulate_reaction_diffusion_cohort
from tau_sr.symbolic import (
    ReactionSamples,
    SymbolicRegressionConfig,
    make_midpoint_evaluation_grid,
    sample_neural_reaction,
    training_state_domain,
)
from tau_sr.synthetic import (
    SyntheticCohortConfig,
    build_connectome_representations,
    generate_baseline_tau,
    generate_count_connectome,
    generate_edge_attributes,
    generate_synthetic_roi_labels,
)
from tau_sr.training import (
    NeuralTrainingConfig,
    NeuralTrainingResult,
    SubjectSplit,
    fit_noise_free_recovery_model,
    make_subject_split,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DOMAIN: tuple[float, float] = (0.05, 0.627)
CANONICAL_MASTER_SEED = 42
SEED_SWEEP: tuple[int, ...] = (40, 41, 42, 43, 44)
PAIRED_FACTOR_SEEDS: tuple[int, ...] = (41, 42, 43)
NOISE_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10)
RIDGE_NOT_USED_NOTE = "This synthetic sweep evaluates only the recovery pipeline models."
DERIVED_SEED_FORMULA = "component_seed = master_seed"

FactorName = Literal["seed", "noise", "graph_representation", "laplacian"]
GraphRepresentation = Literal["count", "binary"]
RobustnessLaplacianKind = Literal[
    "symmetric_normalized",
    "random_walk",
    "combinatorial",
]

__all__ = (
    "CanonicalTolerance",
    "DerivedSeeds",
    "FactorName",
    "GraphRepresentation",
    "RobustnessLaplacianKind",
    "SweepCondition",
    "SyntheticRobustnessConfig",
    "SyntheticRunData",
    "aggregate_synthetic_robustness",
    "build_paired_synthetic_run_data",
    "build_sweep_manifest",
    "classify_polynomial_recovery",
    "count_to_binary_adjacency",
    "derive_component_seeds",
    "fixed_reference_reaction_grid",
    "run_synthetic_robustness_condition",
    "run_synthetic_robustness_sweep",
    "training_fit_domain",
    "write_publication_artifacts",
)


@dataclass(frozen=True)
class DerivedSeeds:
    graph_cohort: int
    initial_state: int
    subject_split: int
    neural_training: int
    observation_noise: int
    pysr: int
    formula: str = DERIVED_SEED_FORMULA

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_cohort": self.graph_cohort,
            "initial_state": self.initial_state,
            "subject_split": self.subject_split,
            "neural_training": self.neural_training,
            "observation_noise": self.observation_noise,
            "pysr": self.pysr,
            "formula": self.formula,
        }


@dataclass(frozen=True)
class SweepCondition:
    run_id: str
    factor: FactorName
    level: str
    master_seed: int
    graph_representation: GraphRepresentation = "count"
    laplacian_kind: RobustnessLaplacianKind = "symmetric_normalized"
    noise_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise SchemaError("run_id must be a nonempty string.")
        if self.factor not in ("seed", "noise", "graph_representation", "laplacian"):
            raise SchemaError("factor is not supported.")
        _require_int(self.master_seed, "master_seed", minimum=0)
        if self.graph_representation not in ("count", "binary"):
            raise SchemaError("graph_representation must be count or binary.")
        if self.laplacian_kind not in (
            "symmetric_normalized",
            "random_walk",
            "combinatorial",
        ):
            raise SchemaError("laplacian_kind is not supported.")
        _require_float(self.noise_fraction, "noise_fraction", minimum=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "factor": self.factor,
            "level": self.level,
            "master_seed": self.master_seed,
            "graph_representation": self.graph_representation,
            "laplacian_kind": self.laplacian_kind,
            "noise_fraction": self.noise_fraction,
        }


@dataclass(frozen=True)
class CanonicalTolerance:
    test_trajectory_rmse: float = 1.0e-4
    learned_rho: float = 1.0e-3
    neural_vs_true_rmse: float = 2.0e-4
    symbolic_vs_neural_rmse: float = 1.0e-4
    symbolic_vs_true_rmse: float = 2.0e-4
    linear_coefficient: float = 3.0e-3
    quadratic_coefficient: float = 3.0e-3


@dataclass(frozen=True)
class SyntheticRobustnessConfig:
    n_subjects: int = CONTROLLED_N_SUBJECTS
    n_regions: int = CONTROLLED_N_REGIONS
    times: tuple[float, ...] = CONTROLLED_TIMES
    edge_probability: float = 0.08
    baseline_tau_min: float = 0.05
    baseline_tau_max: float = 0.40
    true_rho: float = CONTROLLED_TRUE_RHO
    true_growth_rate: float = CONTROLLED_TRUE_GROWTH_RATE
    true_saturation_rate: float = CONTROLLED_TRUE_SATURATION_RATE
    max_step: float = 0.05
    validation_fraction: float = 4 / CONTROLLED_N_SUBJECTS
    test_fraction: float = 4 / CONTROLLED_N_SUBJECTS
    hidden_width: int = 16
    initial_rho: float = 0.05
    learning_rate: float = 1e-3
    rho_learning_rate: float | None = 1e-2
    weight_decay: float = 0.0
    max_epochs: int = 1000
    patience: int = 100
    min_delta: float = 1e-7
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    pysr_niterations: int = 100
    pysr_populations: int = 15
    pysr_population_size: int = 33
    pysr_maxsize: int = 15
    pysr_n_samples: int = 500
    reference_domain: tuple[float, float] = REFERENCE_DOMAIN
    reference_grid_points: int = 1000
    pysr_worker_timeout_seconds: float = 1200.0
    pysr_worker_shutdown_grace_seconds: float = 10.0
    require_canonical_reproduction: bool = True
    canonical_tolerance: CanonicalTolerance = CanonicalTolerance()

    def __post_init__(self) -> None:
        _require_int(self.n_subjects, "n_subjects", minimum=3)
        _require_int(self.n_regions, "n_regions", minimum=2)
        if len(self.times) < 2:
            raise SchemaError("times must contain at least two visits.")
        time_values = tuple(_require_float(value, "times") for value in self.times)
        if any(right <= left for left, right in zip(time_values, time_values[1:], strict=False)):
            raise SchemaError("times must be strictly increasing.")
        _require_float(self.edge_probability, "edge_probability", minimum=0.0)
        if self.edge_probability > 1.0:
            raise SchemaError("edge_probability must be between 0 and 1.")
        _require_float(self.baseline_tau_min, "baseline_tau_min", minimum=0.0)
        _require_float(self.baseline_tau_max, "baseline_tau_max", minimum=0.0)
        if self.baseline_tau_max <= self.baseline_tau_min:
            raise SchemaError("baseline_tau_max must be greater than baseline_tau_min.")
        _require_float(self.true_rho, "true_rho", minimum=0.0)
        _require_float(self.true_growth_rate, "true_growth_rate", positive=True)
        _require_float(self.true_saturation_rate, "true_saturation_rate", minimum=0.0)
        _require_float(self.max_step, "max_step", positive=True)
        _require_float(self.validation_fraction, "validation_fraction", positive=True)
        _require_float(self.test_fraction, "test_fraction", positive=True)
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise SchemaError("validation and test fractions must leave training subjects.")
        _require_int(self.hidden_width, "hidden_width", minimum=1)
        _require_float(self.initial_rho, "initial_rho", positive=True)
        _require_float(self.learning_rate, "learning_rate", positive=True)
        if self.rho_learning_rate is not None:
            _require_float(self.rho_learning_rate, "rho_learning_rate", positive=True)
        _require_float(self.weight_decay, "weight_decay", minimum=0.0)
        _require_int(self.max_epochs, "max_epochs", minimum=1)
        _require_int(self.patience, "patience", minimum=1)
        _require_float(self.min_delta, "min_delta", minimum=0.0)
        if self.dtype not in (torch.float32, torch.float64):
            raise SchemaError("dtype must be torch.float32 or torch.float64.")
        if str(torch.device(self.device)) != self.device:
            raise SchemaError("device must be a canonical torch device string.")
        _require_int(self.pysr_niterations, "pysr_niterations", minimum=1)
        _require_int(self.pysr_populations, "pysr_populations", minimum=1)
        _require_int(self.pysr_population_size, "pysr_population_size", minimum=1)
        _require_int(self.pysr_maxsize, "pysr_maxsize", minimum=1)
        _require_int(self.pysr_n_samples, "pysr_n_samples", minimum=2)
        _require_int(self.reference_grid_points, "reference_grid_points", minimum=2)
        domain_min, domain_max = self.reference_domain
        _require_float(domain_min, "reference_domain_min")
        _require_float(domain_max, "reference_domain_max")
        if domain_min >= domain_max:
            raise SchemaError("reference_domain min must be less than max.")
        _require_float(
            self.pysr_worker_timeout_seconds,
            "pysr_worker_timeout_seconds",
            positive=True,
        )
        _require_float(
            self.pysr_worker_shutdown_grace_seconds,
            "pysr_worker_shutdown_grace_seconds",
            positive=True,
        )
        if not isinstance(self.require_canonical_reproduction, bool):
            raise SchemaError("require_canonical_reproduction must be a boolean.")
        if not isinstance(self.canonical_tolerance, CanonicalTolerance):
            raise SchemaError("canonical_tolerance must be CanonicalTolerance.")
        object.__setattr__(self, "times", time_values)

    def to_training_config(self, *, seed: int) -> NeuralTrainingConfig:
        return NeuralTrainingConfig(
            hidden_width=self.hidden_width,
            initial_rho=self.initial_rho,
            learning_rate=self.learning_rate,
            rho_learning_rate=self.rho_learning_rate,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            patience=self.patience,
            min_delta=self.min_delta,
            max_step=self.max_step,
            seed=seed,
            dtype=self.dtype,
            device=self.device,
        )

    def to_symbolic_config(self, *, seed: int, output_directory: Path) -> SymbolicRegressionConfig:
        return SymbolicRegressionConfig(
            niterations=self.pysr_niterations,
            populations=self.pysr_populations,
            population_size=self.pysr_population_size,
            maxsize=self.pysr_maxsize,
            seed=seed,
            model_selection="best",
            output_directory=str(output_directory),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_subjects": self.n_subjects,
            "n_regions": self.n_regions,
            "times": list(self.times),
            "edge_probability": self.edge_probability,
            "baseline_tau_min": self.baseline_tau_min,
            "baseline_tau_max": self.baseline_tau_max,
            "true_rho": self.true_rho,
            "true_growth_rate": self.true_growth_rate,
            "true_saturation_rate": self.true_saturation_rate,
            "max_step": self.max_step,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "hidden_width": self.hidden_width,
            "initial_rho": self.initial_rho,
            "learning_rate": self.learning_rate,
            "rho_learning_rate": self.rho_learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "dtype": str(self.dtype),
            "device": self.device,
            "pysr_niterations": self.pysr_niterations,
            "pysr_populations": self.pysr_populations,
            "pysr_population_size": self.pysr_population_size,
            "pysr_maxsize": self.pysr_maxsize,
            "pysr_n_samples": self.pysr_n_samples,
            "reference_domain": list(self.reference_domain),
            "reference_grid_points": self.reference_grid_points,
            "pysr_worker_timeout_seconds": self.pysr_worker_timeout_seconds,
            "pysr_worker_shutdown_grace_seconds": self.pysr_worker_shutdown_grace_seconds,
            "require_canonical_reproduction": self.require_canonical_reproduction,
        }


@dataclass(frozen=True)
class SyntheticRunData:
    condition: SweepCondition
    seeds: DerivedSeeds
    clean_dataset: SyntheticRecoveryDataset
    observed_dataset: SyntheticRecoveryDataset
    split: SubjectSplit
    weighted_adjacencies: np.ndarray
    base_noise: np.ndarray
    clean_training_std: float
    noise_std: float
    negative_noisy_state_fraction: float


SymbolicRunner = Callable[
    [
        ZeroAnchoredNeuralReactionDiffusion,
        SyntheticRunData,
        SyntheticRobustnessConfig,
        Path,
    ],
    dict[str, Any],
]


def derive_component_seeds(master_seed: int) -> DerivedSeeds:
    """Derive all component seeds from one master seed.

    The current canonical synthetic interface uses a single seed argument for
    each stochastic component. Keeping each component seed equal to the master
    seed preserves the already recorded seed-42 canonical result while still
    changing the complete pipeline when the master seed changes.
    """

    seed = _require_int(master_seed, "master_seed", minimum=0)
    return DerivedSeeds(
        graph_cohort=seed,
        initial_state=seed,
        subject_split=seed,
        neural_training=seed,
        observation_noise=seed,
        pysr=seed,
    )


def build_sweep_manifest() -> tuple[SweepCondition, ...]:
    """Build the prespecified 23-run one-factor-at-a-time manifest."""

    conditions: list[SweepCondition] = []
    for seed in SEED_SWEEP:
        conditions.append(
            SweepCondition(
                run_id=f"seed_master_{seed}",
                factor="seed",
                level=f"master_seed_{seed}",
                master_seed=seed,
            )
        )
    for seed in PAIRED_FACTOR_SEEDS:
        for noise_fraction in NOISE_FRACTIONS:
            conditions.append(
                SweepCondition(
                    run_id=f"noise_{_level_token(noise_fraction)}_seed_{seed}",
                    factor="noise",
                    level=f"noise_{noise_fraction:g}",
                    master_seed=seed,
                    noise_fraction=noise_fraction,
                )
            )
    for seed in PAIRED_FACTOR_SEEDS:
        conditions.append(
            SweepCondition(
                run_id=f"graph_binary_seed_{seed}",
                factor="graph_representation",
                level="binary",
                master_seed=seed,
                graph_representation="binary",
            )
        )
    for seed in PAIRED_FACTOR_SEEDS:
        for laplacian_kind in ("random_walk", "combinatorial"):
            conditions.append(
                SweepCondition(
                    run_id=f"laplacian_{laplacian_kind}_seed_{seed}",
                    factor="laplacian",
                    level=laplacian_kind,
                    master_seed=seed,
                    laplacian_kind=laplacian_kind,  # type: ignore[arg-type]
                )
            )
    _validate_manifest(conditions)
    return tuple(conditions)


def fixed_reference_reaction_grid(
    config: SyntheticRobustnessConfig | None = None,
) -> np.ndarray:
    checked = SyntheticRobustnessConfig() if config is None else SyntheticRobustnessConfig(
        **config.__dict__
    )
    return make_midpoint_evaluation_grid(
        checked.reference_domain[0],
        checked.reference_domain[1],
        n_points=checked.reference_grid_points,
    )


def count_to_binary_adjacency(adjacency: np.ndarray) -> np.ndarray:
    matrix = _validate_adjacency_matrix(adjacency, "adjacency")
    binary = (matrix > 0.0).astype(float)
    np.fill_diagonal(binary, 0.0)
    if not np.array_equal(binary, binary.T):
        raise SchemaError("binary adjacency must preserve symmetry.")
    return binary


def build_paired_synthetic_run_data(
    condition: SweepCondition,
    config: SyntheticRobustnessConfig | None = None,
) -> SyntheticRunData:
    """Build paired clean and observed datasets for one sweep condition."""

    checked_config = SyntheticRobustnessConfig() if config is None else SyntheticRobustnessConfig(
        **config.__dict__
    )
    checked_condition = SweepCondition(**condition.to_dict())
    seeds = derive_component_seeds(checked_condition.master_seed)
    split = make_subject_split(
        checked_config.n_subjects,
        validation_fraction=checked_config.validation_fraction,
        test_fraction=checked_config.test_fraction,
        seed=seeds.subject_split,
    )
    clean_dataset, weighted_adjacencies = _build_clean_dataset(
        checked_condition,
        checked_config,
        seeds=seeds,
    )
    observed_dataset, base_noise, clean_std, noise_std, negative_fraction = (
        _build_observed_dataset(
            clean_dataset,
            checked_condition,
            checked_config,
            split=split,
            seeds=seeds,
        )
    )
    return SyntheticRunData(
        condition=checked_condition,
        seeds=seeds,
        clean_dataset=clean_dataset,
        observed_dataset=observed_dataset,
        split=split,
        weighted_adjacencies=weighted_adjacencies,
        base_noise=base_noise,
        clean_training_std=clean_std,
        noise_std=noise_std,
        negative_noisy_state_fraction=negative_fraction,
    )


def training_fit_domain(
    dataset: SyntheticRecoveryDataset,
    split: SubjectSplit,
) -> tuple[float, float]:
    return training_state_domain(dataset, split)


def run_synthetic_robustness_condition(
    condition: SweepCondition,
    results_dir: str | Path,
    *,
    config: SyntheticRobustnessConfig | None = None,
    symbolic_runner: SymbolicRunner | None = None,
) -> dict[str, Any]:
    """Run or resume one prespecified synthetic robustness condition."""

    checked_config = SyntheticRobustnessConfig() if config is None else SyntheticRobustnessConfig(
        **config.__dict__
    )
    checked_condition = SweepCondition(**condition.to_dict())
    root = Path(results_dir)
    run_dir = root / "conditions" / checked_condition.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    start = time.perf_counter()
    run_data: SyntheticRunData | None = None

    try:
        run_data = build_paired_synthetic_run_data(checked_condition, checked_config)
        fingerprints = _run_fingerprints(run_data, checked_config)
        existing = _load_reusable_run(summary_path, fingerprints)
        if existing is not None:
            return existing

        _write_incomplete_checkpoint(
            summary_path,
            run_data,
            checked_config,
            fingerprints,
            stage="started",
            start_time=start,
        )
        runner = _run_symbolic_distillation if symbolic_runner is None else symbolic_runner
        model, training_result, training_config, train_runtime = _fit_neural_model(
            run_data,
            checked_config,
        )
        _write_incomplete_checkpoint(
            summary_path,
            run_data,
            checked_config,
            fingerprints,
            stage="neural_complete",
            start_time=start,
        )
        clean_metrics = _evaluate_trajectory_prediction(
            model,
            run_data.observed_dataset,
            run_data.clean_dataset.trajectories,
            run_data.split.test_indices,
            max_step=checked_config.max_step,
            device=checked_config.device,
            dtype=checked_config.dtype,
        )
        observed_metrics = _evaluate_trajectory_prediction(
            model,
            run_data.observed_dataset,
            run_data.observed_dataset.trajectories,
            run_data.split.test_indices,
            max_step=checked_config.max_step,
            device=checked_config.device,
            dtype=checked_config.dtype,
        )
        neural_metrics = _evaluate_neural_reaction(
            model,
            run_data.clean_dataset,
            checked_config,
        )
        symbolic_payload = runner(model, run_data, checked_config, run_dir)
        polynomial = _polynomial_from_symbolic_payload(symbolic_payload)
        classification = classify_polynomial_recovery(
            polynomial,
            true_linear=checked_config.true_growth_rate,
            true_quadratic=-checked_config.true_saturation_rate,
        )
        summary = _complete_run_summary(
            run_data,
            checked_config,
            fingerprints,
            training_result=training_result,
            training_config=training_config,
            train_runtime_seconds=train_runtime,
            clean_trajectory_metrics=clean_metrics,
            observed_trajectory_metrics=observed_metrics,
            neural_metrics=neural_metrics,
            symbolic_payload=symbolic_payload,
            polynomial=polynomial,
            classification=classification,
            runtime_seconds=time.perf_counter() - start,
        )
        write_json_atomically(summary_path, summary)
        return summary
    except Exception as exc:
        if (
            isinstance(exc, SchemaError)
            and "checkpoint fingerprints do not match" in str(exc)
        ):
            raise
        if run_data is None:
            fallback_seeds = derive_component_seeds(checked_condition.master_seed)
            failure = _failed_run_summary_without_data(
                checked_condition,
                checked_config,
                fallback_seeds,
                exc,
                runtime_seconds=time.perf_counter() - start,
            )
        else:
            failure = _failed_run_summary(
                run_data,
                checked_config,
                _run_fingerprints(run_data, checked_config),
                exc,
                runtime_seconds=time.perf_counter() - start,
            )
        write_json_atomically(summary_path, failure)
        return failure
    finally:
        gc.collect()


def run_synthetic_robustness_sweep(
    results_dir: str | Path = "results/synthetic_robustness",
    publication_dir: str | Path = "docs/spie/synthetic_robustness",
    *,
    config: SyntheticRobustnessConfig | None = None,
    symbolic_runner: SymbolicRunner | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the complete 23-condition synthetic robustness sweep sequentially."""

    checked_config = SyntheticRobustnessConfig() if config is None else SyntheticRobustnessConfig(
        **config.__dict__
    )
    results_root = Path(results_dir)
    publication_root = Path(publication_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    publication_root.mkdir(parents=True, exist_ok=True)
    conditions = build_sweep_manifest()
    manifest = _manifest_payload(conditions, checked_config)
    write_json_atomically(results_root / "manifest.json", manifest)
    write_json_atomically(publication_root / "synthetic_robustness_manifest.json", manifest)

    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for index, condition in enumerate(conditions, start=1):
        record = run_synthetic_robustness_condition(
            condition,
            results_root,
            config=checked_config,
            symbolic_runner=symbolic_runner,
        )
        records.append(record)
        elapsed = time.perf_counter() - start
        average = elapsed / index
        remaining = average * (len(conditions) - index)
        if progress is not None:
            progress(
                f"completed {index}/{len(conditions)} "
                f"({record['run_id']}: {record['status']}); "
                f"elapsed {elapsed / 60.0:.1f} min; "
                f"estimated remaining {remaining / 60.0:.1f} min"
            )

    aggregate = aggregate_synthetic_robustness(records, checked_config)
    aggregate["runtime_seconds"] = time.perf_counter() - start
    write_json_atomically(results_root / "synthetic_robustness_summary.json", aggregate)
    write_publication_artifacts(
        records,
        aggregate,
        publication_root,
        config=checked_config,
    )
    return aggregate


def aggregate_synthetic_robustness(
    records: Sequence[Mapping[str, Any]],
    config: SyntheticRobustnessConfig | None = None,
) -> dict[str, Any]:
    checked_config = SyntheticRobustnessConfig() if config is None else SyntheticRobustnessConfig(
        **config.__dict__
    )
    copied_records = [_json_safe(record) for record in records]
    run_ids = [record["run_id"] for record in copied_records]
    if len(run_ids) != len(set(run_ids)):
        raise SchemaError("run records must not contain duplicate run_id values.")
    completed = [record for record in copied_records if record.get("status") == "complete"]
    failed = [record for record in copied_records if record.get("status") != "complete"]
    canonical = next(
        (
            record
            for record in completed
            if record["run_id"] == f"seed_master_{CANONICAL_MASTER_SEED}"
        ),
        None,
    )
    canonical_check = (
        _canonical_reproduction_check(canonical, checked_config)
        if canonical is not None
        else {"passed": False, "reason": "canonical seed-42 run is missing or failed"}
    )
    if checked_config.require_canonical_reproduction and not canonical_check["passed"]:
        raise SchemaError("canonical seed-42 reproduction check failed.")

    summary = {
        "experiment": "controlled_synthetic_robustness_sweep",
        "repository_commit": _git_commit(),
        "manifest_run_count": len(build_sweep_manifest()),
        "completed_run_count": len(completed),
        "failed_run_count": len(failed),
        "configuration": checked_config.to_dict(),
        "derived_seed_formula": DERIVED_SEED_FORMULA,
        "canonical_reproduction": canonical_check,
        "recovery_rates": _recovery_rates(completed),
        "seed_robustness": _seed_robustness_summary(completed),
        "noise_robustness": _noise_robustness_summary(completed),
        "graph_representation": _graph_representation_summary(completed),
        "laplacian": _laplacian_summary(completed),
        "symbolic_structure": _symbolic_structure_summary(completed),
        "failure_cases": [
            {
                "run_id": record["run_id"],
                "factor": record["factor"],
                "level": record["level"],
                "master_seed": record["master_seed"],
                "failure_stage": record.get("failure_stage"),
                "error_type": record.get("error_type"),
                "error": record.get("error"),
            }
            for record in failed
        ],
        "numerical_failure_count": sum(
            1 for record in failed if record.get("failure_stage") == "numerical"
        ),
        "pysr_worker_cleanup": _pysr_cleanup_summary(completed),
        "software_versions": environment_versions(),
        "records": copied_records,
    }
    summary["strongest_defensible_conclusion"] = _defensible_conclusion(summary)
    return summary


def classify_polynomial_recovery(
    polynomial: Mapping[str, Any],
    *,
    true_linear: float,
    true_quadratic: float,
    zero_tolerance: float = 1e-8,
) -> dict[str, Any]:
    true_linear_value = _require_float(true_linear, "true_linear")
    true_quadratic_value = _require_float(true_quadratic, "true_quadratic")
    if true_linear_value == 0.0 or true_quadratic_value == 0.0:
        raise SchemaError("true coefficients must be nonzero for relative recovery criteria.")
    is_polynomial = bool(polynomial.get("is_polynomial"))
    constant = _optional_float(polynomial.get("constant"))
    linear = _optional_float(polynomial.get("linear"))
    quadratic = _optional_float(polynomial.get("quadratic"))
    highest_degree = polynomial.get("highest_degree")
    if highest_degree is not None:
        highest_degree = _require_int(highest_degree, "highest_degree", minimum=0)

    has_linear = linear is not None and abs(linear) > zero_tolerance
    has_quadratic = quadratic is not None and abs(quadratic) > zero_tolerance
    no_constant = constant is not None and abs(constant) <= zero_tolerance
    no_higher = highest_degree is not None and highest_degree <= 2
    strict = bool(is_polynomial and has_linear and has_quadratic and no_constant and no_higher)
    correct_sign = bool(
        linear is not None
        and quadratic is not None
        and linear > 0.0
        and quadratic < 0.0
    )
    linear_relative_error = (
        None if linear is None else abs(linear - true_linear_value) / abs(true_linear_value)
    )
    quadratic_relative_error = (
        None
        if quadratic is None
        else abs(quadratic - true_quadratic_value) / abs(true_quadratic_value)
    )
    linear_within_10_percent = (
        False if linear_relative_error is None else linear_relative_error <= 0.10
    )
    quadratic_within_10_percent = (
        False if quadratic_relative_error is None else quadratic_relative_error <= 0.10
    )
    return {
        "is_polynomial": is_polynomial,
        "strict_symbolic_structure": strict,
        "correct_sign": correct_sign,
        "linear_within_10_percent": linear_within_10_percent,
        "quadratic_within_10_percent": quadratic_within_10_percent,
        "joint_coefficient_within_10_percent": bool(
            linear_within_10_percent and quadratic_within_10_percent
        ),
        "linear_relative_error": linear_relative_error,
        "quadratic_relative_error": quadratic_relative_error,
        "has_constant_term": bool(constant is not None and abs(constant) > zero_tolerance),
        "has_higher_order_terms": bool(highest_degree is not None and highest_degree > 2),
    }


def write_publication_artifacts(
    records: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    publication_dir: str | Path,
    *,
    config: SyntheticRobustnessConfig | None = None,
) -> None:
    checked_config = SyntheticRobustnessConfig() if config is None else SyntheticRobustnessConfig(
        **config.__dict__
    )
    output = Path(publication_dir)
    output.mkdir(parents=True, exist_ok=True)
    serializable_records = [_json_safe(record) for record in records]
    serializable_aggregate = _json_safe(aggregate)
    write_json_atomically(output / "synthetic_robustness_summary.json", serializable_aggregate)
    rows = robustness_table_rows(serializable_records)
    _write_csv(output / "synthetic_robustness_table.csv", rows)
    _write_latex_table(output / "synthetic_robustness_table.tex", rows)
    _write_report(
        output / "synthetic_robustness_report.md",
        serializable_records,
        serializable_aggregate,
        checked_config,
    )
    _write_figures(output, serializable_records)


def robustness_table_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: str(row["run_id"])):
        symbolic = _mapping(record.get("symbolic_recovery"), "symbolic_recovery", default={})
        success = _mapping(record.get("success"), "success", default={})
        trajectory = _mapping(record.get("trajectory_metrics"), "trajectory_metrics", default={})
        clean_test = _mapping(trajectory.get("clean_test"), "clean_test", default={})
        noisy_test = _mapping(trajectory.get("noisy_test"), "noisy_test", default={})
        neural = _mapping(record.get("neural_recovery"), "neural_recovery", default={})
        coefficients = _mapping(symbolic.get("coefficients"), "coefficients", default={})
        rows.append(
            {
                "run_id": record.get("run_id"),
                "factor": record.get("factor"),
                "level": record.get("level"),
                "master_seed": record.get("master_seed"),
                "graph_representation": record.get("graph_representation"),
                "laplacian_kind": record.get("laplacian_kind"),
                "noise_fraction": record.get("noise_fraction"),
                "status": record.get("status"),
                "clean_test_rmse": clean_test.get("rmse"),
                "noisy_test_rmse": noisy_test.get("rmse"),
                "learned_rho": neural.get("learned_rho"),
                "rho_relative_error": neural.get("rho_relative_error"),
                "neural_vs_true_rmse": neural.get("neural_vs_true_rmse"),
                "symbolic_vs_true_rmse": symbolic.get("symbolic_vs_true_rmse"),
                "symbolic_vs_neural_rmse": symbolic.get("symbolic_vs_neural_rmse"),
                "equation": symbolic.get("equation"),
                "complexity": symbolic.get("complexity"),
                "linear_coefficient": coefficients.get("linear"),
                "quadratic_coefficient": coefficients.get("quadratic"),
                "strict_structure": success.get("strict_symbolic_structure"),
                "correct_sign": success.get("correct_sign"),
                "joint_coefficient_10pct": success.get("joint_coefficient_within_10_percent"),
                "runtime_seconds": record.get("runtime_seconds"),
            }
        )
    return rows


def _build_clean_dataset(
    condition: SweepCondition,
    config: SyntheticRobustnessConfig,
    *,
    seeds: DerivedSeeds,
) -> tuple[SyntheticRecoveryDataset, np.ndarray]:
    synthetic_config = SyntheticCohortConfig(
        n_subjects=config.n_subjects,
        n_regions=config.n_regions,
        visits_per_subject=len(config.times),
        seed=seeds.graph_cohort,
        edge_probability=config.edge_probability,
        baseline_tau_min=config.baseline_tau_min,
        baseline_tau_max=config.baseline_tau_max,
        add_observation_noise=False,
        observation_noise_std=0.0,
    )
    rng = np.random.default_rng(seeds.graph_cohort)
    roi_labels = generate_synthetic_roi_labels(synthetic_config)
    subject_ids = tuple(
        f"synthetic_subject_{subject_index:04d}"
        for subject_index in range(config.n_subjects)
    )
    times_by_subject = tuple(
        np.asarray(config.times, dtype=float).copy() for _ in range(config.n_subjects)
    )
    adjacencies: list[np.ndarray] = []
    weighted_adjacencies: list[np.ndarray] = []
    laplacians: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    for _subject_index in range(config.n_subjects):
        count_matrix = generate_count_connectome(
            config.n_regions,
            rng,
            edge_probability=config.edge_probability,
        )
        fa_matrix, length_matrix = generate_edge_attributes(count_matrix, rng)
        representations = build_connectome_representations(
            count_matrix,
            fa_matrix,
            length_matrix,
        )
        weighted = representations["count"].astype(float, copy=True)
        adjacency = (
            count_to_binary_adjacency(weighted)
            if condition.graph_representation == "binary"
            else weighted.copy()
        )
        laplacian = build_graph_laplacian(adjacency, kind=condition.laplacian_kind)
        baseline = generate_baseline_tau(
            config.n_regions,
            rng,
            tau_min=config.baseline_tau_min,
            tau_max=config.baseline_tau_max,
        )
        weighted_adjacencies.append(weighted)
        adjacencies.append(adjacency.astype(float, copy=True))
        laplacians.append(laplacian.astype(float, copy=True))
        baselines.append(baseline.astype(float, copy=True))

    baseline_states = np.stack(baselines).astype(float, copy=True)
    laplacian_array = np.stack(laplacians).astype(float, copy=True)
    reaction_fn = make_linear_quadratic_reaction(
        config.true_growth_rate,
        config.true_saturation_rate,
    )
    trajectories = simulate_reaction_diffusion_cohort(
        baseline_states,
        times_by_subject,
        laplacian_array,
        config.true_rho,
        reaction_fn,
        max_step=config.max_step,
    )
    dataset = SyntheticRecoveryDataset(
        subject_ids=subject_ids,
        roi_labels=roi_labels,
        baseline_states=baseline_states.copy(),
        adjacencies=np.stack(adjacencies).astype(float, copy=True),
        laplacians=laplacian_array.copy(),
        times_by_subject=tuple(times.copy() for times in times_by_subject),
        trajectories=tuple(trajectory.copy() for trajectory in trajectories),
        representation=condition.graph_representation,
        laplacian_kind=condition.laplacian_kind,
        rho=config.true_rho,
        growth_rate=config.true_growth_rate,
        saturation_rate=config.true_saturation_rate,
        seed=seeds.graph_cohort,
    )
    return dataset, np.stack(weighted_adjacencies).astype(float, copy=True)


def _build_observed_dataset(
    clean_dataset: SyntheticRecoveryDataset,
    condition: SweepCondition,
    config: SyntheticRobustnessConfig,
    *,
    split: SubjectSplit,
    seeds: DerivedSeeds,
) -> tuple[SyntheticRecoveryDataset, np.ndarray, float, float, float]:
    clean_stack = np.stack(clean_dataset.trajectories).astype(float, copy=True)
    train_values = clean_stack[list(split.train_indices)].ravel()
    clean_training_std = float(np.std(train_values))
    if not math.isfinite(clean_training_std) or clean_training_std <= 0.0:
        raise SchemaError("clean training trajectory standard deviation must be positive.")
    noise_rng = np.random.default_rng(seeds.observation_noise)
    base_noise = noise_rng.standard_normal(clean_stack.shape)
    noise_std = float(condition.noise_fraction) * clean_training_std
    observed_stack = clean_stack + noise_std * base_noise
    baseline_states = observed_stack[:, 0, :].astype(float, copy=True)
    negative_fraction = float(np.mean(observed_stack < 0.0))
    observed_dataset = SyntheticRecoveryDataset(
        subject_ids=clean_dataset.subject_ids,
        roi_labels=clean_dataset.roi_labels,
        baseline_states=baseline_states,
        adjacencies=clean_dataset.adjacencies.copy(),
        laplacians=clean_dataset.laplacians.copy(),
        times_by_subject=tuple(times.copy() for times in clean_dataset.times_by_subject),
        trajectories=tuple(
            observed_stack[index].copy() for index in range(observed_stack.shape[0])
        ),
        representation=clean_dataset.representation,
        laplacian_kind=clean_dataset.laplacian_kind,
        rho=clean_dataset.rho,
        growth_rate=clean_dataset.growth_rate,
        saturation_rate=clean_dataset.saturation_rate,
        seed=clean_dataset.seed,
    )
    return observed_dataset, base_noise, clean_training_std, noise_std, negative_fraction


def _fit_neural_model(
    run_data: SyntheticRunData,
    config: SyntheticRobustnessConfig,
) -> tuple[
    ZeroAnchoredNeuralReactionDiffusion,
    NeuralTrainingResult,
    NeuralTrainingConfig,
    float,
]:
    training_config = config.to_training_config(seed=run_data.seeds.neural_training)
    device = torch.device(training_config.device)
    with torch.random.fork_rng(devices=[] if device.type != "cuda" else [device.index or 0]):
        torch.manual_seed(run_data.seeds.neural_training)
        model = ZeroAnchoredNeuralReactionDiffusion(
            hidden_width=config.hidden_width,
            initial_rho=config.initial_rho,
        )
    model = model.to(device=device, dtype=training_config.dtype)
    start = time.perf_counter()
    training_result = fit_noise_free_recovery_model(
        run_data.observed_dataset,
        run_data.split,
        training_config,
        model=model,
    )
    return (
        training_result.model,  # type: ignore[return-value]
        training_result,
        training_config,
        time.perf_counter() - start,
    )


def _run_symbolic_distillation(
    model: ZeroAnchoredNeuralReactionDiffusion,
    run_data: SyntheticRunData,
    config: SyntheticRobustnessConfig,
    run_dir: Path,
) -> dict[str, Any]:
    domain_min, domain_max = training_fit_domain(run_data.observed_dataset, run_data.split)
    samples = sample_neural_reaction(
        model,
        domain_min=domain_min,
        domain_max=domain_max,
        n_samples=config.pysr_n_samples,
        device=config.device,
        dtype=config.dtype,
    )
    evaluation_grid = fixed_reference_reaction_grid(config)
    neural_prediction = _neural_reaction_values(
        model,
        evaluation_grid,
        device=config.device,
        dtype=config.dtype,
    )
    true_reaction = make_linear_quadratic_reaction(
        config.true_growth_rate,
        config.true_saturation_rate,
    )
    true_prediction = true_reaction(evaluation_grid.copy())
    symbolic_config = config.to_symbolic_config(
        seed=run_data.seeds.pysr,
        output_directory=run_dir / "pysr_artifacts",
    )
    return _run_isolated_pysr_fit(
        samples=samples,
        symbolic_config=symbolic_config,
        evaluation_grid=evaluation_grid,
        neural_prediction=neural_prediction,
        true_prediction=true_prediction,
        true_growth_rate=config.true_growth_rate,
        true_saturation_rate=config.true_saturation_rate,
        run_dir=run_dir,
        timeout_seconds=config.pysr_worker_timeout_seconds,
        shutdown_grace_seconds=config.pysr_worker_shutdown_grace_seconds,
    )


def _run_isolated_pysr_fit(
    *,
    samples: ReactionSamples,
    symbolic_config: SymbolicRegressionConfig,
    evaluation_grid: np.ndarray,
    neural_prediction: np.ndarray,
    true_prediction: np.ndarray,
    true_growth_rate: float,
    true_saturation_rate: float,
    run_dir: Path,
    timeout_seconds: float,
    shutdown_grace_seconds: float,
) -> dict[str, Any]:
    worker_dir = run_dir / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="request_", dir=worker_dir) as temporary_dir:
        temporary_root = Path(temporary_dir)
        request_path = temporary_root / "request.json"
        result_path = temporary_root / "result.json"
        write_json_atomically(
            request_path,
            _pysr_worker_request(
                samples=samples,
                symbolic_config=symbolic_config,
                evaluation_grid=evaluation_grid,
                neural_prediction=neural_prediction,
                true_prediction=true_prediction,
                true_growth_rate=true_growth_rate,
                true_saturation_rate=true_saturation_rate,
            ),
        )
        worker_result = run_json_worker(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_pysr_worker.py"),
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
            result_path,
            timeout_seconds=timeout_seconds,
            shutdown_grace_seconds=shutdown_grace_seconds,
        )
    return {
        **worker_result.payload,
        "worker_exited_normally": worker_result.process_exited_normally,
        "worker_cleanup_required": worker_result.cleanup_required,
        "worker_timed_out_after_result": worker_result.timed_out,
        "worker_returncode": worker_result.returncode,
    }


def _pysr_worker_request(
    *,
    samples: ReactionSamples,
    symbolic_config: SymbolicRegressionConfig,
    evaluation_grid: np.ndarray,
    neural_prediction: np.ndarray,
    true_prediction: np.ndarray,
    true_growth_rate: float,
    true_saturation_rate: float,
) -> dict[str, Any]:
    return {
        "samples": {
            "input_values": samples.input_values.tolist(),
            "neural_reaction_values": samples.neural_reaction_values.tolist(),
            "domain_min": samples.domain_min,
            "domain_max": samples.domain_max,
        },
        "config": {
            "niterations": symbolic_config.niterations,
            "populations": symbolic_config.populations,
            "population_size": symbolic_config.population_size,
            "maxsize": symbolic_config.maxsize,
            "seed": symbolic_config.seed,
            "model_selection": symbolic_config.model_selection,
            "output_directory": symbolic_config.output_directory,
        },
        "evaluation": {
            "grid": np.asarray(evaluation_grid, dtype=float).tolist(),
            "neural_prediction": np.asarray(neural_prediction, dtype=float).tolist(),
            "true_prediction": np.asarray(true_prediction, dtype=float).tolist(),
            "true_growth_rate": float(true_growth_rate),
            "true_saturation_rate": float(true_saturation_rate),
        },
    }


def _evaluate_trajectory_prediction(
    model: ZeroAnchoredNeuralReactionDiffusion,
    driver_dataset: SyntheticRecoveryDataset,
    target_trajectories: Sequence[np.ndarray],
    subject_indices: Sequence[int],
    *,
    max_step: float,
    device: str,
    dtype: torch.dtype,
) -> dict[str, float]:
    torch_device = torch.device(device)
    subject_mse: list[float] = []
    subject_mae: list[float] = []
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for subject_index in subject_indices:
                y0 = torch.tensor(
                    driver_dataset.baseline_states[subject_index].copy(),
                    device=torch_device,
                    dtype=dtype,
                )
                laplacian = torch.tensor(
                    driver_dataset.laplacians[subject_index].copy(),
                    device=torch_device,
                    dtype=dtype,
                )
                times = torch.tensor(
                    driver_dataset.times_by_subject[subject_index].copy(),
                    device=torch_device,
                    dtype=dtype,
                )
                prediction = model.integrate(y0, laplacian, times, max_step=max_step)
                target = torch.tensor(
                    np.asarray(target_trajectories[subject_index], dtype=float).copy(),
                    device=torch_device,
                    dtype=dtype,
                )
                residual = prediction[1:] - target[1:]
                subject_mse.append(float(torch.mean(residual**2).detach().cpu().item()))
                subject_mae.append(float(torch.mean(torch.abs(residual)).detach().cpu().item()))
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
    mse = _mean(subject_mse)
    mae = _mean(subject_mae)
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": mae}


def _evaluate_neural_reaction(
    model: ZeroAnchoredNeuralReactionDiffusion,
    dataset: SyntheticRecoveryDataset,
    config: SyntheticRobustnessConfig,
) -> dict[str, Any]:
    grid = fixed_reference_reaction_grid(config)
    neural_prediction = _neural_reaction_values(
        model,
        grid,
        device=config.device,
        dtype=config.dtype,
    )
    true_reaction = make_linear_quadratic_reaction(
        dataset.growth_rate,
        dataset.saturation_rate,
    )
    true_prediction = true_reaction(grid.copy())
    residual = neural_prediction - true_prediction
    curvature = np.diff(neural_prediction, n=2)
    learned_rho = float(model.rho.detach().cpu().item())
    rho_abs = abs(learned_rho - dataset.rho)
    return {
        "learned_rho": learned_rho,
        "rho_absolute_error": rho_abs,
        "rho_relative_error": None if dataset.rho == 0.0 else rho_abs / dataset.rho,
        "neural_vs_true_rmse": float(np.sqrt(np.mean(residual**2))),
        "neural_vs_true_mae": float(np.mean(np.abs(residual))),
        "curvature_sign_summary": {
            "negative_second_difference_fraction": float(np.mean(curvature < 0.0)),
            "positive_second_difference_fraction": float(np.mean(curvature > 0.0)),
            "median_second_difference": float(np.median(curvature)),
        },
    }


def _neural_reaction_values(
    model: ZeroAnchoredNeuralReactionDiffusion,
    grid: np.ndarray,
    *,
    device: str,
    dtype: torch.dtype,
) -> np.ndarray:
    y_values = torch.as_tensor(
        np.asarray(grid, dtype=float).reshape(-1, 1),
        device=torch.device(device),
        dtype=dtype,
    )
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            prediction = model.reaction(y_values)
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
    values = prediction.detach().cpu().numpy().reshape(-1)
    if values.shape != np.asarray(grid).shape or not np.isfinite(values).all():
        raise SchemaError("neural reaction evaluation produced invalid values.")
    return values.astype(float, copy=True)


def _complete_run_summary(
    run_data: SyntheticRunData,
    config: SyntheticRobustnessConfig,
    fingerprints: Mapping[str, Any],
    *,
    training_result: NeuralTrainingResult,
    training_config: NeuralTrainingConfig,
    train_runtime_seconds: float,
    clean_trajectory_metrics: Mapping[str, float],
    observed_trajectory_metrics: Mapping[str, float],
    neural_metrics: Mapping[str, Any],
    symbolic_payload: Mapping[str, Any],
    polynomial: Mapping[str, Any],
    classification: Mapping[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    symbolic_metrics = _mapping(symbolic_payload.get("metrics"), "symbolic metrics")
    success = {
        "rho_within_10_percent": (
            False
            if neural_metrics.get("rho_relative_error") is None
            else neural_metrics["rho_relative_error"] <= 0.10
        ),
        **dict(classification),
    }
    return {
        **_condition_metadata(run_data, config),
        "status": "complete",
        "fingerprints": dict(fingerprints),
        "training": {
            "epochs_run": training_result.epochs_run,
            "best_epoch": training_result.best_epoch,
            "initial_training_loss": training_result.training_losses[0],
            "final_training_loss": training_result.training_losses[-1],
            "best_validation_loss": min(training_result.validation_losses),
            "final_validation_loss": training_result.validation_losses[-1],
            "runtime_seconds": train_runtime_seconds,
            "config": _training_config_dict(training_config),
        },
        "trajectory_metrics": {
            "clean_test": dict(clean_trajectory_metrics),
            "noisy_test": (
                None
                if run_data.condition.noise_fraction == 0.0
                else dict(observed_trajectory_metrics)
            ),
        },
        "neural_recovery": dict(neural_metrics),
        "symbolic_recovery": _symbolic_summary(symbolic_payload, polynomial, symbolic_metrics),
        "success": success,
        "numerical_instability_count": 0,
        "runtime_seconds": runtime_seconds,
        "software_versions": environment_versions(),
    }


def _condition_metadata(
    run_data: SyntheticRunData,
    config: SyntheticRobustnessConfig,
) -> dict[str, Any]:
    clean_stack = np.stack(run_data.clean_dataset.trajectories)
    observed_stack = np.stack(run_data.observed_dataset.trajectories)
    return {
        "run_id": run_data.condition.run_id,
        "factor": run_data.condition.factor,
        "level": run_data.condition.level,
        "master_seed": run_data.condition.master_seed,
        "derived_seeds": run_data.seeds.to_dict(),
        "graph_representation": run_data.condition.graph_representation,
        "laplacian_kind": run_data.condition.laplacian_kind,
        "noise_fraction": run_data.condition.noise_fraction,
        "noise_std": run_data.noise_std,
        "clean_training_trajectory_std": run_data.clean_training_std,
        "negative_noisy_state_fraction": run_data.negative_noisy_state_fraction,
        "split_counts": {
            "train": len(run_data.split.train_indices),
            "validation": len(run_data.split.validation_indices),
            "test": len(run_data.split.test_indices),
        },
        "state_domains": {
            "clean": _state_summary(clean_stack),
            "observed": _state_summary(observed_stack),
            "fixed_reference": {
                "domain_min": config.reference_domain[0],
                "domain_max": config.reference_domain[1],
                "n_points": config.reference_grid_points,
            },
        },
    }


def _symbolic_summary(
    symbolic_payload: Mapping[str, Any],
    polynomial: Mapping[str, Any],
    symbolic_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    coefficients = {
        "constant": polynomial.get("constant"),
        "linear": polynomial.get("linear"),
        "quadratic": polynomial.get("quadratic"),
        "highest_degree": polynomial.get("highest_degree"),
    }
    return {
        "equation": symbolic_payload.get("equation"),
        "expanded_expression": symbolic_payload.get("expanded_expression"),
        "complexity": symbolic_payload.get("complexity"),
        "pysr_loss": symbolic_payload.get("training_loss"),
        "pysr_runtime_seconds": symbolic_payload.get("runtime_seconds"),
        "fit_domain": symbolic_payload.get("fit_domain"),
        "evaluation_domain": symbolic_payload.get("evaluation_domain"),
        "symbolic_vs_neural_rmse": symbolic_metrics.get("symbolic_vs_neural_rmse"),
        "symbolic_vs_neural_mae": symbolic_metrics.get("symbolic_vs_neural_mae"),
        "symbolic_vs_true_rmse": symbolic_metrics.get("symbolic_vs_true_rmse"),
        "symbolic_vs_true_mae": symbolic_metrics.get("symbolic_vs_true_mae"),
        "neural_vs_true_rmse": symbolic_metrics.get("neural_vs_true_rmse"),
        "neural_vs_true_mae": symbolic_metrics.get("neural_vs_true_mae"),
        "coefficients": coefficients,
        "coefficient_errors": {
            "constant": polynomial.get("constant_error"),
            "linear": polynomial.get("linear_error"),
            "quadratic": polynomial.get("quadratic_error"),
        },
        "is_polynomial": polynomial.get("is_polynomial"),
        "worker_exited_normally": symbolic_payload.get("worker_exited_normally"),
        "worker_cleanup_required": symbolic_payload.get("worker_cleanup_required"),
        "worker_timed_out_after_result": symbolic_payload.get("worker_timed_out_after_result"),
        "worker_returncode": symbolic_payload.get("worker_returncode"),
        "pysr_version": symbolic_payload.get("pysr_version"),
        "julia_version": symbolic_payload.get("julia_version"),
    }


def _failed_run_summary(
    run_data: SyntheticRunData,
    config: SyntheticRobustnessConfig,
    fingerprints: Mapping[str, Any],
    exc: Exception,
    *,
    runtime_seconds: float,
) -> dict[str, Any]:
    failure_stage = "symbolic" if isinstance(exc, PySRWorkerError) else "numerical"
    return {
        **_condition_metadata(run_data, config),
        "status": "failed",
        "fingerprints": dict(fingerprints),
        "failure_stage": failure_stage,
        "error_type": type(exc).__name__,
        "error": _safe_error(exc),
        "runtime_seconds": runtime_seconds,
        "numerical_instability_count": 1 if failure_stage == "numerical" else 0,
        "software_versions": environment_versions(),
    }


def _failed_run_summary_without_data(
    condition: SweepCondition,
    config: SyntheticRobustnessConfig,
    seeds: DerivedSeeds,
    exc: Exception,
    *,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "run_id": condition.run_id,
        "factor": condition.factor,
        "level": condition.level,
        "master_seed": condition.master_seed,
        "derived_seeds": seeds.to_dict(),
        "graph_representation": condition.graph_representation,
        "laplacian_kind": condition.laplacian_kind,
        "noise_fraction": condition.noise_fraction,
        "status": "failed",
        "configuration": config.to_dict(),
        "failure_stage": "setup",
        "error_type": type(exc).__name__,
        "error": _safe_error(exc),
        "runtime_seconds": runtime_seconds,
        "numerical_instability_count": 0,
        "software_versions": environment_versions(),
    }


def _write_incomplete_checkpoint(
    summary_path: Path,
    run_data: SyntheticRunData,
    config: SyntheticRobustnessConfig,
    fingerprints: Mapping[str, Any],
    *,
    stage: str,
    start_time: float,
) -> None:
    write_json_atomically(
        summary_path,
        {
            **_condition_metadata(run_data, config),
            "status": "incomplete",
            "stage": stage,
            "fingerprints": dict(fingerprints),
            "runtime_seconds": time.perf_counter() - start_time,
        },
    )


def _load_reusable_run(
    summary_path: Path,
    fingerprints: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    payload = _read_json_object(summary_path)
    if payload.get("status") != "complete" and payload.get("status") != "failed":
        return None
    saved = _mapping(payload.get("fingerprints"), "fingerprints", default={})
    for key, value in fingerprints.items():
        if saved.get(key) != value:
            raise SchemaError("synthetic robustness checkpoint fingerprints do not match.")
    if payload.get("status") == "complete":
        symbolic = _mapping(payload.get("symbolic_recovery"), "symbolic_recovery", default={})
        if symbolic.get("equation") is None:
            raise SchemaError("completed checkpoint is missing symbolic equation.")
    return payload


def _run_fingerprints(
    run_data: SyntheticRunData,
    config: SyntheticRobustnessConfig,
) -> dict[str, Any]:
    return {
        "repository_commit": _git_commit(),
        "manifest_fingerprint": _manifest_fingerprint(build_sweep_manifest(), config),
        "condition_fingerprint": _fingerprint(run_data.condition.to_dict()),
        "dataset_fingerprint": _dataset_fingerprint(run_data),
        "split_fingerprint": _split_fingerprint(run_data.split),
        "model_config_fingerprint": _fingerprint(config.to_dict()),
        "pysr_config_fingerprint": _fingerprint(
            {
                "niterations": config.pysr_niterations,
                "populations": config.pysr_populations,
                "population_size": config.pysr_population_size,
                "maxsize": config.pysr_maxsize,
                "n_samples": config.pysr_n_samples,
                "model_selection": "best",
            }
        ),
    }


def _dataset_fingerprint(run_data: SyntheticRunData) -> str:
    digest = hashlib.sha256()
    for array in (
        run_data.clean_dataset.baseline_states,
        run_data.clean_dataset.adjacencies,
        run_data.clean_dataset.laplacians,
        np.stack(run_data.clean_dataset.trajectories),
        np.stack(run_data.observed_dataset.trajectories),
    ):
        digest.update(np.asarray(array, dtype=float).tobytes())
    digest.update(run_data.condition.run_id.encode())
    return digest.hexdigest()


def _split_fingerprint(split: SubjectSplit) -> str:
    return _fingerprint(
        {
            "train": split.train_indices,
            "validation": split.validation_indices,
            "test": split.test_indices,
        }
    )


def _manifest_fingerprint(
    conditions: Sequence[SweepCondition],
    config: SyntheticRobustnessConfig,
) -> str:
    return _fingerprint(
        {
            "conditions": [condition.to_dict() for condition in conditions],
            "config": config.to_dict(),
        }
    )


def _canonical_reproduction_check(
    record: Mapping[str, Any] | None,
    config: SyntheticRobustnessConfig,
) -> dict[str, Any]:
    if record is None:
        return {"passed": False, "reason": "canonical record is missing"}
    symbolic = _mapping(record.get("symbolic_recovery"), "symbolic_recovery", default={})
    trajectory = _mapping(record.get("trajectory_metrics"), "trajectory_metrics", default={})
    clean_test = _mapping(trajectory.get("clean_test"), "clean_test", default={})
    neural = _mapping(record.get("neural_recovery"), "neural_recovery", default={})
    coefficients = _mapping(symbolic.get("coefficients"), "coefficients", default={})
    expected = {
        "test_trajectory_rmse": 2.7434e-4,
        "learned_rho": 0.102621,
        "neural_vs_true_rmse": 8.452e-4,
        "symbolic_vs_neural_rmse": 1.595e-4,
        "symbolic_vs_true_rmse": 8.317e-4,
        "linear_coefficient": 0.495391,
        "quadratic_coefficient": -0.486935,
    }
    observed = {
        "test_trajectory_rmse": clean_test.get("rmse"),
        "learned_rho": neural.get("learned_rho"),
        "neural_vs_true_rmse": neural.get("neural_vs_true_rmse"),
        "symbolic_vs_neural_rmse": symbolic.get("symbolic_vs_neural_rmse"),
        "symbolic_vs_true_rmse": symbolic.get("symbolic_vs_true_rmse"),
        "linear_coefficient": coefficients.get("linear"),
        "quadratic_coefficient": coefficients.get("quadratic"),
    }
    tolerances = {
        "test_trajectory_rmse": config.canonical_tolerance.test_trajectory_rmse,
        "learned_rho": config.canonical_tolerance.learned_rho,
        "neural_vs_true_rmse": config.canonical_tolerance.neural_vs_true_rmse,
        "symbolic_vs_neural_rmse": config.canonical_tolerance.symbolic_vs_neural_rmse,
        "symbolic_vs_true_rmse": config.canonical_tolerance.symbolic_vs_true_rmse,
        "linear_coefficient": config.canonical_tolerance.linear_coefficient,
        "quadratic_coefficient": config.canonical_tolerance.quadratic_coefficient,
    }
    comparisons: dict[str, Any] = {}
    passed = True
    for key, expected_value in expected.items():
        value = observed[key]
        ok = value is not None and abs(float(value) - expected_value) <= tolerances[key]
        comparisons[key] = {
            "observed": value,
            "expected": expected_value,
            "absolute_error": None if value is None else abs(float(value) - expected_value),
            "tolerance": tolerances[key],
            "passed": bool(ok),
        }
        passed = passed and bool(ok)
    return {"passed": passed, "comparisons": comparisons}


def _recovery_rates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "rho_within_10_percent",
        "linear_within_10_percent",
        "quadratic_within_10_percent",
        "strict_symbolic_structure",
        "correct_sign",
        "joint_coefficient_within_10_percent",
    )
    return {key: _success_count(records, key) for key in keys}


def _seed_robustness_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seed_records = [record for record in records if record.get("factor") == "seed"]
    return {
        "run_count": len(seed_records),
        "metrics": _metric_stats(seed_records),
        "individual_seed_points": [
            _compact_point(record)
            for record in sorted(seed_records, key=lambda row: row["master_seed"])
        ],
        "success_counts": _recovery_rates(seed_records),
    }


def _noise_robustness_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[float, list[Mapping[str, Any]]] = {0.0: []}
    for fraction in NOISE_FRACTIONS:
        grouped[fraction] = []
    for record in records:
        if (
            record.get("master_seed") in PAIRED_FACTOR_SEEDS
            and record.get("graph_representation") == "count"
            and record.get("laplacian_kind") == "symmetric_normalized"
        ):
            grouped.setdefault(float(record.get("noise_fraction", 0.0)), []).append(record)
    by_noise = {
        f"{fraction:g}": {
            "run_count": len(points),
            "metrics": _metric_stats(points),
            "individual_seed_points": [
                _compact_point(record)
                for record in sorted(points, key=lambda row: row["master_seed"])
            ],
            "success_counts": _recovery_rates(points),
        }
        for fraction, points in sorted(grouped.items())
    }
    return {
        "by_noise_fraction": by_noise,
        "monotonic_trends": _noise_monotonic_trends(by_noise),
        "paired_design_note": "Zero-noise points are reused from the seed sweep.",
    }


def _graph_representation_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    weighted = _records_by_seed(
        records,
        graph_representation="count",
        laplacian_kind="symmetric_normalized",
        noise_fraction=0.0,
    )
    binary = _records_by_seed(
        records,
        graph_representation="binary",
        laplacian_kind="symmetric_normalized",
        noise_fraction=0.0,
    )
    return _paired_variant_summary(
        baseline=weighted,
        variants={"binary": binary},
        baseline_name="count",
    )


def _laplacian_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    symmetric = _records_by_seed(
        records,
        graph_representation="count",
        laplacian_kind="symmetric_normalized",
        noise_fraction=0.0,
    )
    random_walk = _records_by_seed(
        records,
        graph_representation="count",
        laplacian_kind="random_walk",
        noise_fraction=0.0,
    )
    combinatorial = _records_by_seed(
        records,
        graph_representation="count",
        laplacian_kind="combinatorial",
        noise_fraction=0.0,
    )
    return _paired_variant_summary(
        baseline=symmetric,
        variants={"random_walk": random_walk, "combinatorial": combinatorial},
        baseline_name="symmetric_normalized",
    )


def _symbolic_structure_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "strict_recovery": _success_count(records, "strict_symbolic_structure"),
        "correct_sign": _success_count(records, "correct_sign"),
        "linear_coefficient": _coefficient_stats(records, "linear"),
        "quadratic_coefficient": _coefficient_stats(records, "quadratic"),
    }


def _metric_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_paths = {
        "clean_test_rmse": ("trajectory_metrics", "clean_test", "rmse"),
        "clean_test_mae": ("trajectory_metrics", "clean_test", "mae"),
        "rho_relative_error": ("neural_recovery", "rho_relative_error"),
        "neural_vs_true_rmse": ("neural_recovery", "neural_vs_true_rmse"),
        "symbolic_vs_true_rmse": ("symbolic_recovery", "symbolic_vs_true_rmse"),
        "symbolic_vs_neural_rmse": ("symbolic_recovery", "symbolic_vs_neural_rmse"),
    }
    return {
        name: _numeric_summary(_path_values(records, path))
        for name, path in metric_paths.items()
    }


def _paired_variant_summary(
    *,
    baseline: Mapping[int, Mapping[str, Any]],
    variants: Mapping[str, Mapping[int, Mapping[str, Any]]],
    baseline_name: str,
) -> dict[str, Any]:
    metric_path = ("trajectory_metrics", "clean_test", "rmse")
    reaction_path = ("symbolic_recovery", "symbolic_vs_true_rmse")
    rho_path = ("neural_recovery", "rho_relative_error")
    result: dict[str, Any] = {
        baseline_name: {
            "run_count": len(baseline),
            "metrics": _metric_stats(list(baseline.values())),
            "success_counts": _recovery_rates(list(baseline.values())),
        }
    }
    for variant_name, variant_records in variants.items():
        seeds = sorted(set(baseline) & set(variant_records))
        result[variant_name] = {
            "run_count": len(variant_records),
            "metrics": _metric_stats(list(variant_records.values())),
            "success_counts": _recovery_rates(list(variant_records.values())),
            "paired_differences_relative_to_baseline": {
                "clean_test_rmse": _paired_difference_summary(
                    baseline,
                    variant_records,
                    seeds,
                    metric_path,
                ),
                "symbolic_vs_true_rmse": _paired_difference_summary(
                    baseline,
                    variant_records,
                    seeds,
                    reaction_path,
                ),
                "rho_relative_error": _paired_difference_summary(
                    baseline,
                    variant_records,
                    seeds,
                    rho_path,
                ),
            },
            "numerical_failure_count": sum(
                1 for record in variant_records.values() if record.get("status") != "complete"
            ),
        }
    return result


def _records_by_seed(
    records: Sequence[Mapping[str, Any]],
    *,
    graph_representation: str,
    laplacian_kind: str,
    noise_fraction: float,
) -> dict[int, Mapping[str, Any]]:
    by_seed: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if (
            record.get("master_seed") in PAIRED_FACTOR_SEEDS
            and record.get("graph_representation") == graph_representation
            and record.get("laplacian_kind") == laplacian_kind
            and math.isclose(float(record.get("noise_fraction", math.nan)), noise_fraction)
        ):
            by_seed[int(record["master_seed"])] = record
    return by_seed


def _paired_difference_summary(
    baseline: Mapping[int, Mapping[str, Any]],
    variant: Mapping[int, Mapping[str, Any]],
    seeds: Sequence[int],
    path: tuple[str, ...],
) -> dict[str, Any]:
    differences: list[float] = []
    points: list[dict[str, Any]] = []
    for seed in seeds:
        base_value = _path_value(baseline[seed], path)
        variant_value = _path_value(variant[seed], path)
        if base_value is None or variant_value is None:
            continue
        diff = float(variant_value) - float(base_value)
        differences.append(diff)
        points.append({"master_seed": seed, "difference": diff})
    summary = _numeric_summary(differences)
    summary["individual_seed_points"] = points
    return summary


def _noise_monotonic_trends(by_noise: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted((float(level), payload) for level, payload in by_noise.items())
    trends: dict[str, Any] = {}
    for metric in (
        "clean_test_rmse",
        "rho_relative_error",
        "neural_vs_true_rmse",
        "symbolic_vs_true_rmse",
    ):
        means = [
            payload["metrics"][metric]["mean"]
            for _level, payload in ordered
            if payload["metrics"][metric]["mean"] is not None
        ]
        trends[metric] = {
            "means_by_increasing_noise": means,
            "monotonic_non_decreasing": bool(
                len(means) >= 2
                and all(
                    right >= left
                    for left, right in zip(means, means[1:], strict=False)
                )
            ),
        }
    return trends


def _success_count(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    successes = 0
    eligible = 0
    for record in records:
        success = _mapping(record.get("success"), "success", default={})
        if key in success:
            eligible += 1
            successes += int(bool(success[key]))
    return {"successes": successes, "eligible": eligible}


def _coefficient_stats(records: Sequence[Mapping[str, Any]], coefficient: str) -> dict[str, Any]:
    values = []
    for record in records:
        symbolic = _mapping(record.get("symbolic_recovery"), "symbolic_recovery", default={})
        coefficients = _mapping(symbolic.get("coefficients"), "coefficients", default={})
        value = coefficients.get(coefficient)
        if value is not None:
            values.append(float(value))
    return _numeric_summary(values)


def _compact_point(record: Mapping[str, Any]) -> dict[str, Any]:
    symbolic = _mapping(record.get("symbolic_recovery"), "symbolic_recovery", default={})
    trajectory = _mapping(record.get("trajectory_metrics"), "trajectory_metrics", default={})
    clean = _mapping(trajectory.get("clean_test"), "clean_test", default={})
    neural = _mapping(record.get("neural_recovery"), "neural_recovery", default={})
    success = _mapping(record.get("success"), "success", default={})
    return {
        "run_id": record.get("run_id"),
        "master_seed": record.get("master_seed"),
        "noise_fraction": record.get("noise_fraction"),
        "clean_test_rmse": clean.get("rmse"),
        "rho_relative_error": neural.get("rho_relative_error"),
        "neural_vs_true_rmse": neural.get("neural_vs_true_rmse"),
        "symbolic_vs_true_rmse": symbolic.get("symbolic_vs_true_rmse"),
        "strict_symbolic_structure": success.get("strict_symbolic_structure"),
    }


def _path_values(records: Sequence[Mapping[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _path_value(record, path)
        if value is not None:
            values.append(float(value))
    return values


def _path_value(record: Mapping[str, Any], path: tuple[str, ...]) -> object | None:
    current: object = record
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "range": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "range": [float(np.min(array)), float(np.max(array))],
    }


def _pysr_cleanup_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    symbolic_records = [
        _mapping(record.get("symbolic_recovery"), "symbolic_recovery", default={})
        for record in records
    ]
    return {
        "worker_count": len(symbolic_records),
        "exited_normally_count": sum(
            1 for record in symbolic_records if record.get("worker_exited_normally") is True
        ),
        "cleanup_required_count": sum(
            1 for record in symbolic_records if record.get("worker_cleanup_required") is True
        ),
        "timed_out_after_result_count": sum(
            1 for record in symbolic_records if record.get("worker_timed_out_after_result") is True
        ),
    }


def _defensible_conclusion(summary: Mapping[str, Any]) -> str:
    completed = int(summary.get("completed_run_count", 0))
    failed = int(summary.get("failed_run_count", 0))
    rates = _mapping(summary.get("recovery_rates"), "recovery_rates", default={})
    strict = _mapping(rates.get("strict_symbolic_structure"), "strict", default={})
    joint = _mapping(rates.get("joint_coefficient_within_10_percent"), "joint", default={})
    return (
        "Synthetic robustness was evaluated under a paired one-factor-at-a-time "
        f"design with {completed} completed runs and {failed} failed runs. "
        f"Strict symbolic structure recovery succeeded in {strict.get('successes', 0)} "
        f"of {strict.get('eligible', 0)} eligible completed runs; joint linear/quadratic "
        f"coefficient recovery succeeded in {joint.get('successes', 0)} of "
        f"{joint.get('eligible', 0)} eligible completed runs. These controlled "
        "synthetic results support only the tested simulation regime and do not "
        "establish clinical validity or higher-order factor interactions."
    )


def _polynomial_from_symbolic_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    polynomial = payload.get("polynomial_interpretation")
    if not isinstance(polynomial, Mapping):
        raise SchemaError("symbolic payload is missing polynomial_interpretation.")
    return polynomial


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SchemaError("CSV table requires at least one row.")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_latex_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "run_id",
        "status",
        "clean_test_rmse",
        "rho_relative_error",
        "symbolic_vs_true_rmse",
        "strict_structure",
    ]
    lines = [
        "\\begin{tabular}{llllll}",
        "\\hline",
        "Run & Status & Test RMSE & Rho rel. err. & Sym.-true RMSE & Strict \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(_latex_cell(row.get(column)) for column in columns) + " \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    config: SyntheticRobustnessConfig,
) -> None:
    completed = aggregate.get("completed_run_count")
    failed = aggregate.get("failed_run_count")
    canonical = _mapping(aggregate.get("canonical_reproduction"), "canonical", default={})
    rates = _mapping(aggregate.get("recovery_rates"), "recovery_rates", default={})
    lines = [
        "# Synthetic Robustness Sweep",
        "",
        "## Objective",
        "",
        "This synthetic-only robustness sweep evaluates whether the controlled "
        "tau reaction-diffusion recovery pipeline remains stable across pipeline "
        "seeds, observation noise, graph representation, and Laplacian definition.",
        "",
        "## Prespecified Design",
        "",
        "The study uses 23 unique one-factor-at-a-time runs: five seed runs, nine "
        "additional observation-noise runs, three binary-graph runs, and six "
        "alternative-Laplacian runs. Higher-order interactions were not tested.",
        "",
        "Pairing is by master seed. Within a seed, the weighted topology, clean "
        "initial states, subject split, true reaction law, true rho, and base "
        "standard-normal noise draw are reused; only the declared factor changes.",
        "",
        "## Fixed Configuration",
        "",
        f"- Subjects: {config.n_subjects}",
        f"- Regions: {config.n_regions}",
        f"- Times: {list(config.times)}",
        f"- True rho: {config.true_rho}",
        f"- True reaction: {config.true_growth_rate}*y - "
        f"{config.true_saturation_rate}*y**2",
        f"- Neural hidden width: {config.hidden_width}",
        f"- Max epochs/patience: {config.max_epochs}/{config.patience}",
        f"- PySR samples/maxsize: {config.pysr_n_samples}/{config.pysr_maxsize}",
        "",
        "## Canonical Reproduction Check",
        "",
        f"Passed: {canonical.get('passed')}. The check compares the seed-42 "
        "weighted symmetric-normalized zero-noise run against the previously "
        "reported approximate canonical values with documented deterministic "
        "tolerances.",
        "",
        "## Run Status",
        "",
        f"Completed runs: {completed}. Failed runs: {failed}.",
        "",
        "## Recovery Rates",
        "",
    ]
    for key, payload in rates.items():
        mapping = _mapping(payload, key, default={})
        lines.append(
            f"- {key}: {mapping.get('successes', 0)} / {mapping.get('eligible', 0)}"
        )
    lines.extend(
        [
            "",
            "## Seed Results",
            "",
            _summary_sentence("Seed robustness", aggregate.get("seed_robustness")),
            "",
            "## Noise Results",
            "",
            _summary_sentence("Noise robustness", aggregate.get("noise_robustness")),
            "",
            "## Graph Representation Results",
            "",
            _summary_sentence(
                "Graph representation sensitivity",
                aggregate.get("graph_representation"),
            ),
            "",
            "## Laplacian Results",
            "",
            _summary_sentence("Laplacian sensitivity", aggregate.get("laplacian")),
            "",
            "## Symbolic Structure And Coefficients",
            "",
            "The selected symbolic equations and coefficient interpretations are "
            "listed in the CSV and JSON artifacts. Structurally incorrect or "
            "non-polynomial equations are retained as failures for strict recovery.",
            "",
            "## Failure Cases",
            "",
        ]
    )
    failures = aggregate.get("failure_cases", [])
    if failures:
        for failure in failures:
            lines.append(
                f"- {failure.get('run_id')}: {failure.get('error_type')} during "
                f"{failure.get('failure_stage')}"
            )
    else:
        lines.append("No failed runs were recorded.")
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"Total runtime seconds: {aggregate.get('runtime_seconds')}.",
            "",
            "## Limitations",
            "",
            "This is a synthetic study generated from the same model family being "
            "fitted. It is paired and one-factor-at-a-time, so interactions were "
            "not estimated. Controlled synthetic robustness cannot replace "
            "longitudinal real-data validation and does not establish clinical "
            "validity.",
            "",
            "## Strongest Defensible Conclusion",
            "",
            str(aggregate.get("strongest_defensible_conclusion")),
            "",
            "## Run Inventory",
            "",
        ]
    )
    for record in records:
        lines.append(
            f"- {record.get('run_id')}: {record.get('status')}, "
            f"factor={record.get('factor')}, level={record.get('level')}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figures(output: Path, records: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )
    _figure_noise_sensitivity(output, records, plt)
    _figure_graph_laplacian_sensitivity(output, records, plt)
    _figure_coefficient_recovery(output, records, plt)


def _figure_noise_sensitivity(output: Path, records: Sequence[Mapping[str, Any]], plt: Any) -> None:
    noise_records = [
        record
        for record in records
        if record.get("master_seed") in PAIRED_FACTOR_SEEDS
        and record.get("graph_representation") == "count"
        and record.get("laplacian_kind") == "symmetric_normalized"
        and record.get("status") == "complete"
    ]
    metrics = [
        ("Clean test RMSE", ("trajectory_metrics", "clean_test", "rmse")),
        ("Rho relative error", ("neural_recovery", "rho_relative_error")),
        ("Neural-true RMSE", ("neural_recovery", "neural_vs_true_rmse")),
        ("Symbolic-true RMSE", ("symbolic_recovery", "symbolic_vs_true_rmse")),
        ("Strict recovery", ("success", "strict_symbolic_structure")),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.1), constrained_layout=True)
    for ax, (title, path) in zip(axes, metrics, strict=True):
        for seed in PAIRED_FACTOR_SEEDS:
            seed_records = sorted(
                [record for record in noise_records if record["master_seed"] == seed],
                key=lambda row: float(row["noise_fraction"]),
            )
            x = [float(record["noise_fraction"]) for record in seed_records]
            y = [
                float(bool(_path_value(record, path)))
                if path[-1] == "strict_symbolic_structure"
                else _path_value(record, path)
                for record in seed_records
            ]
            ax.plot(x, y, marker="o", linewidth=1.4, label=f"seed {seed}")
        ax.set_title(title)
        ax.set_xlabel("Noise fraction")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[-1].set_ylim(-0.05, 1.05)
    axes[-1].legend(loc="best", fontsize=7)
    _save_figure(fig, output / "figure_1_noise_sensitivity")


def _figure_graph_laplacian_sensitivity(
    output: Path,
    records: Sequence[Mapping[str, Any]],
    plt: Any,
) -> None:
    selected = [
        record
        for record in records
        if record.get("master_seed") in PAIRED_FACTOR_SEEDS
        and float(record.get("noise_fraction", 0.0)) == 0.0
    ]
    categories = [
        ("count", "symmetric_normalized"),
        ("binary", "symmetric_normalized"),
        ("count", "random_walk"),
        ("count", "combinatorial"),
    ]
    labels = ["count/sym", "binary/sym", "count/rw", "count/comb"]
    metrics = [
        ("Rho rel. error", ("neural_recovery", "rho_relative_error")),
        ("Sym.-true RMSE", ("symbolic_recovery", "symbolic_vs_true_rmse")),
        ("Linear rel. error", ("success", "linear_relative_error")),
        ("Quadratic rel. error", ("success", "quadratic_relative_error")),
        ("Strict recovery", ("success", "strict_symbolic_structure")),
        ("Numerical failure", ("numerical_instability_count",)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for ax, (title, path) in zip(axes.ravel(), metrics, strict=True):
        for index, (graph, laplacian) in enumerate(categories):
            points = [
                record
                for record in selected
                if record.get("graph_representation") == graph
                and record.get("laplacian_kind") == laplacian
            ]
            for record in points:
                value = _path_value(record, path)
                if path[-1] == "strict_symbolic_structure":
                    value = float(bool(value))
                if value is not None:
                    ax.scatter(index, float(value), color="#0072B2", s=24)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, output / "figure_2_graph_laplacian_sensitivity")


def _figure_coefficient_recovery(
    output: Path,
    records: Sequence[Mapping[str, Any]],
    plt: Any,
) -> None:
    completed = [record for record in records if record.get("status") == "complete"]
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    palette = {
        "seed": "#0072B2",
        "noise": "#D55E00",
        "graph_representation": "#009E73",
        "laplacian": "#CC79A7",
    }
    for index, record in enumerate(completed):
        symbolic = _mapping(record.get("symbolic_recovery"), "symbolic_recovery", default={})
        coefficients = _mapping(symbolic.get("coefficients"), "coefficients", default={})
        color = palette.get(str(record.get("factor")), "#333333")
        if coefficients.get("linear") is not None:
            ax.scatter(index, float(coefficients["linear"]), color=color, marker="o", s=28)
        if coefficients.get("quadratic") is not None:
            ax.scatter(index, float(coefficients["quadratic"]), color=color, marker="s", s=28)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="true linear")
    ax.axhline(-0.5, color="gray", linestyle="--", linewidth=1.0, label="true quadratic")
    ax.set_xticks(
        range(len(completed)),
        [str(record["run_id"]) for record in completed],
        rotation=90,
    )
    ax.set_ylabel("Recovered coefficient")
    ax.set_title("Symbolic coefficient recovery across all completed synthetic runs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _save_figure(fig, output / "figure_3_coefficient_recovery")


def _save_figure(fig: Any, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _manifest_payload(
    conditions: Sequence[SweepCondition],
    config: SyntheticRobustnessConfig,
) -> dict[str, Any]:
    return {
        "experiment": "controlled_synthetic_robustness_sweep",
        "repository_commit": _git_commit(),
        "run_count": len(conditions),
        "one_factor_at_a_time": True,
        "interactions_tested": False,
        "conditions": [condition.to_dict() for condition in conditions],
        "configuration": config.to_dict(),
        "derived_seed_formula": DERIVED_SEED_FORMULA,
    }


def _validate_manifest(conditions: Sequence[SweepCondition]) -> None:
    if len(conditions) != 23:
        raise SchemaError("synthetic robustness manifest must contain exactly 23 runs.")
    run_ids = [condition.run_id for condition in conditions]
    if len(run_ids) != len(set(run_ids)):
        raise SchemaError("synthetic robustness manifest contains duplicate run IDs.")
    keys = [
        (
            condition.master_seed,
            condition.graph_representation,
            condition.laplacian_kind,
            condition.noise_fraction,
        )
        for condition in conditions
    ]
    if len(keys) != len(set(keys)):
        raise SchemaError("synthetic robustness manifest contains duplicate conditions.")


def _level_token(value: float) -> str:
    return str(value).replace(".", "p")


def _state_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise SchemaError("state values must be finite.")
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _training_config_dict(config: NeuralTrainingConfig) -> dict[str, Any]:
    return {
        "hidden_width": config.hidden_width,
        "initial_rho": config.initial_rho,
        "learning_rate": config.learning_rate,
        "rho_learning_rate": config.rho_learning_rate,
        "weight_decay": config.weight_decay,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "min_delta": config.min_delta,
        "max_step": config.max_step,
        "seed": config.seed,
        "dtype": str(config.dtype),
        "device": config.device,
    }


def _mapping(
    value: object,
    field_name: str,
    *,
    default: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if value is None and default is not None:
        return default
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field_name} must be a mapping.")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SchemaError("JSON payload must be an object.")
    return payload


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return ""
    return value


def _latex_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _summary_sentence(label: str, payload: object) -> str:
    if not isinstance(payload, Mapping):
        return f"{label}: unavailable."
    return f"{label}: see JSON summary for per-seed points, paired differences, and rates."


def _validate_adjacency_matrix(adjacency: object, field_name: str) -> np.ndarray:
    if not isinstance(adjacency, np.ndarray):
        raise SchemaError(f"{field_name} must be a NumPy array.")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise SchemaError(f"{field_name} must be a square matrix.")
    if not np.issubdtype(adjacency.dtype, np.number) or adjacency.dtype.kind in {
        "b",
        "c",
        "O",
        "S",
        "U",
    }:
        raise SchemaError(f"{field_name} must have a real numeric dtype.")
    if not np.isfinite(adjacency).all():
        raise SchemaError(f"{field_name} must contain only finite values.")
    if (adjacency < 0.0).any():
        raise SchemaError(f"{field_name} must be nonnegative.")
    if not np.allclose(adjacency, adjacency.T, rtol=1e-7, atol=1e-12):
        raise SchemaError(f"{field_name} must be symmetric.")
    if not np.array_equal(np.diag(adjacency), np.zeros(adjacency.shape[0])):
        raise SchemaError(f"{field_name} diagonal must be exactly zero.")
    return adjacency.astype(float, copy=True)


def _require_int(value: object, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field_name} must be an integer.")
    if value < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")
    return value


def _require_float(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise SchemaError(f"{field_name} must be numeric.")
    checked = float(value)
    if not math.isfinite(checked):
        raise SchemaError(f"{field_name} must be finite.")
    if minimum is not None and checked < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")
    if positive and checked <= 0.0:
        raise SchemaError(f"{field_name} must be positive.")
    return checked


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _require_float(value, "coefficient")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise SchemaError("mean requires at least one value.")
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_error(exc: Exception) -> str:
    text = str(exc)
    replacements = [str(REPO_ROOT), os.getcwd()]
    for item in replacements:
        if item:
            text = text.replace(item, "<repo>")
    return text


def _git_commit() -> str:
    try:
        result = os.popen("git rev-parse HEAD").read().strip()
    except Exception:
        return "unavailable"
    return result or "unavailable"
