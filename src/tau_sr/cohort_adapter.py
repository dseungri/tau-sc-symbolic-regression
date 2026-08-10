"""Privacy-preserving adapter for the full two-timepoint real cohort."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tau_sr.graphs import LaplacianKind, build_graph_laplacian
from tau_sr.pilot_adapter import LongitudinalTrajectoryDataset, make_pilot_roi_labels
from tau_sr.pysr_lifecycle import write_json_atomically
from tau_sr.schemas import SchemaError
from tau_sr.training import SubjectSplit

COHORT_N_REGIONS = 160
COHORT_TIMES: tuple[float, float] = (0.0, 1.0)
COHORT_ATLAS = "Destrieux cortical regions plus 12 subcortical regions"
COHORT_ROI_ALIGNMENT = "confirmed_direct_index_alignment"
COHORT_TAU_MEASUREMENT = "SUVR"
COHORT_TAU_REFERENCE = "brainstem"
COHORT_SC_WEIGHT_TYPE = "whole-brain-normalized structural connectivity"
COHORT_SC_STATIC_OVER_TIME = True
COHORT_DATA_ALIGNMENT_READY = True
COHORT_FORMAL_EVALUATION_READY = False
DEFAULT_COHORT_LAPLACIAN_KIND: LaplacianKind = "symmetric_normalized"
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_TEST_FRACTION = 0.20

REPO_ROOT = Path(__file__).resolve().parents[2]
_TERMINAL_NUMERIC_KEY_RE = re.compile(r"(?:^|[^0-9])([0-9]+)\s*$")

__all__ = (
    "COHORT_ATLAS",
    "COHORT_DATA_ALIGNMENT_READY",
    "COHORT_FORMAL_EVALUATION_READY",
    "COHORT_N_REGIONS",
    "COHORT_ROI_ALIGNMENT",
    "COHORT_SC_STATIC_OVER_TIME",
    "COHORT_SC_WEIGHT_TYPE",
    "COHORT_TAU_MEASUREMENT",
    "COHORT_TAU_REFERENCE",
    "COHORT_TIMES",
    "DEFAULT_COHORT_LAPLACIAN_KIND",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_TEST_FRACTION",
    "DEFAULT_TRAIN_FRACTION",
    "DEFAULT_VALIDATION_FRACTION",
    "inspect_two_timepoint_cohort",
    "load_two_timepoint_cohort",
    "make_frozen_cohort_split",
    "split_subject_labels",
    "write_private_cohort_artifacts",
)


@dataclass(frozen=True)
class _SubjectFiles:
    key: str
    structure_path: Path
    baseline_tau_path: Path
    followup_tau_path: Path


@dataclass(frozen=True)
class _RoleFiles:
    files_by_key: dict[str, Path]
    file_count: int
    duplicate_key_count: int
    malformed_csv_count: int


@dataclass(frozen=True)
class _CohortComponents:
    dataset: LongitudinalTrajectoryDataset
    summary: dict[str, Any]


def load_two_timepoint_cohort(
    structure_dir: str | Path,
    baseline_tau_dir: str | Path,
    followup_tau_dir: str | Path,
    *,
    laplacian_kind: LaplacianKind = DEFAULT_COHORT_LAPLACIAN_KIND,
) -> LongitudinalTrajectoryDataset:
    """Load the validated two-timepoint cohort into the training dataset contract."""

    triplets, file_counts = _match_subject_files(
        structure_dir,
        baseline_tau_dir,
        followup_tau_dir,
    )
    return _build_cohort_components(
        triplets,
        file_counts=file_counts,
        laplacian_kind=laplacian_kind,
    ).dataset


def inspect_two_timepoint_cohort(
    structure_dir: str | Path,
    baseline_tau_dir: str | Path,
    followup_tau_dir: str | Path,
) -> dict[str, object]:
    """Return aggregate privacy-safe QC for the two-timepoint cohort."""

    triplets, file_counts = _match_subject_files(
        structure_dir,
        baseline_tau_dir,
        followup_tau_dir,
    )
    components = _build_cohort_components(
        triplets,
        file_counts=file_counts,
        laplacian_kind=DEFAULT_COHORT_LAPLACIAN_KIND,
    )
    return components.summary


def make_frozen_cohort_split(
    subject_ids: Sequence[str],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> SubjectSplit:
    """Create the frozen subject-level 70/10/20 cohort split."""

    labels = _validate_subject_ids(subject_ids)
    seed_value = _require_int(seed, "seed", minimum=0)
    train_fraction_value = _require_fraction(train_fraction, "train_fraction")
    validation_fraction_value = _require_fraction(
        validation_fraction,
        "validation_fraction",
    )
    test_fraction_value = _require_fraction(test_fraction, "test_fraction")
    if not math.isclose(
        train_fraction_value + validation_fraction_value + test_fraction_value,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SchemaError("split fractions must sum to 1.0.")

    n_subjects = len(labels)
    train_count = math.floor(n_subjects * train_fraction_value)
    validation_count = math.floor(n_subjects * validation_fraction_value)
    test_count = n_subjects - train_count - validation_count
    if min(train_count, validation_count, test_count) < 1:
        raise SchemaError("cohort must be large enough for nonempty split groups.")

    rng = np.random.default_rng(seed_value)
    shuffled = tuple(int(index) for index in rng.permutation(n_subjects))
    train_indices = tuple(sorted(shuffled[:train_count]))
    validation_stop = train_count + validation_count
    validation_indices = tuple(sorted(shuffled[train_count:validation_stop]))
    test_indices = tuple(sorted(shuffled[validation_stop:]))
    return SubjectSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def split_subject_labels(
    subject_ids: Sequence[str],
    split: SubjectSplit,
) -> dict[str, tuple[str, ...]]:
    """Return neutral subject labels for each split group."""

    labels = _validate_subject_ids(subject_ids)
    if not isinstance(split, SubjectSplit):
        raise SchemaError("split must be a SubjectSplit.")
    return {
        "train": tuple(labels[index] for index in split.train_indices),
        "validation": tuple(labels[index] for index in split.validation_indices),
        "test": tuple(labels[index] for index in split.test_indices),
    }


def write_private_cohort_artifacts(
    dataset: LongitudinalTrajectoryDataset,
    qc_summary: dict[str, object],
    split: SubjectSplit,
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> dict[str, Path]:
    """Write privacy-safe QC and split JSON to an explicit private directory."""

    if not isinstance(dataset, LongitudinalTrajectoryDataset):
        raise SchemaError("dataset must be a LongitudinalTrajectoryDataset.")
    if not isinstance(qc_summary, dict):
        raise SchemaError("qc_summary must be a dictionary.")
    if not isinstance(split, SubjectSplit):
        raise SchemaError("split must be a SubjectSplit.")

    private_dir = _prepare_private_output_dir(output_dir)
    qc_path = private_dir / "cohort_qc_summary.json"
    split_path = private_dir / f"cohort_split_seed_{int(seed)}.json"

    write_json_atomically(qc_path, _json_safe(qc_summary))
    os.chmod(qc_path, 0o600)
    write_json_atomically(
        split_path,
        _split_manifest(
            dataset,
            split,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        ),
    )
    os.chmod(split_path, 0o600)
    return {"qc_summary": qc_path, "split_manifest": split_path}


def _match_subject_files(
    structure_dir: str | Path,
    baseline_tau_dir: str | Path,
    followup_tau_dir: str | Path,
) -> tuple[tuple[_SubjectFiles, ...], dict[str, int]]:
    structure = _role_files(structure_dir, "structural_connectivity")
    baseline = _role_files(baseline_tau_dir, "baseline_tau")
    followup = _role_files(followup_tau_dir, "followup_tau")

    all_keys = set(structure.files_by_key) | set(baseline.files_by_key) | set(followup.files_by_key)
    missing_triplets = sum(
        int(key not in structure.files_by_key)
        + int(key not in baseline.files_by_key)
        + int(key not in followup.files_by_key)
        for key in all_keys
    )
    duplicate_key_count = (
        structure.duplicate_key_count
        + baseline.duplicate_key_count
        + followup.duplicate_key_count
    )
    malformed_csv_count = (
        structure.malformed_csv_count
        + baseline.malformed_csv_count
        + followup.malformed_csv_count
    )
    if duplicate_key_count or malformed_csv_count or missing_triplets:
        raise SchemaError(
            "cohort file matching failed: "
            f"duplicate_keys={duplicate_key_count}, "
            f"unexpected_or_malformed_csv={malformed_csv_count}, "
            f"missing_counterparts={missing_triplets}."
        )
    if not all_keys:
        raise SchemaError("cohort directories contain no matched CSV triplets.")

    triplets = tuple(
        _SubjectFiles(
            key=key,
            structure_path=structure.files_by_key[key],
            baseline_tau_path=baseline.files_by_key[key],
            followup_tau_path=followup.files_by_key[key],
        )
        for key in sorted(all_keys, key=_sort_key)
    )
    return triplets, {
        "structural_connectivity_files": structure.file_count,
        "baseline_tau_files": baseline.file_count,
        "followup_tau_files": followup.file_count,
        "duplicate_key_count": duplicate_key_count,
        "unexpected_or_malformed_csv_count": malformed_csv_count,
        "missing_counterpart_count": missing_triplets,
    }


def _role_files(path: str | Path, role_name: str) -> _RoleFiles:
    directory = _require_directory(path, role_name)
    files = tuple(
        sorted(
            (
                child
                for child in directory.iterdir()
                if child.is_file() and not child.name.startswith(".")
            ),
            key=lambda item: item.name,
        )
    )
    csv_files = tuple(child for child in files if child.suffix.casefold() == ".csv")
    files_by_key: dict[str, Path] = {}
    duplicate_keys: set[str] = set()
    malformed_count = 0
    for csv_file in csv_files:
        key = _subject_key(csv_file)
        if key is None:
            malformed_count += 1
            continue
        if key in files_by_key:
            duplicate_keys.add(key)
            continue
        files_by_key[key] = csv_file
    return _RoleFiles(
        files_by_key=files_by_key,
        file_count=len(csv_files),
        duplicate_key_count=len(duplicate_keys),
        malformed_csv_count=malformed_count,
    )


def _build_cohort_components(
    triplets: Sequence[_SubjectFiles],
    *,
    file_counts: dict[str, int],
    laplacian_kind: LaplacianKind,
) -> _CohortComponents:
    labels = make_pilot_roi_labels()
    subject_ids: list[str] = []
    adjacencies: list[np.ndarray] = []
    laplacians: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    times_by_subject: list[np.ndarray] = []
    degree_values: list[np.ndarray] = []
    baseline_values: list[np.ndarray] = []
    followup_values: list[np.ndarray] = []
    diagonal_removal_count = 0

    for position, triplet in enumerate(triplets, start=1):
        neutral_id = f"Cohort Subject {position:03d}"
        try:
            adjacency, laplacian, sc_stats = _load_structural_connectivity(
                triplet.structure_path,
                laplacian_kind=laplacian_kind,
            )
            baseline_tau = _load_tau_vector(triplet.baseline_tau_path, "baseline_tau")
            followup_tau = _load_tau_vector(triplet.followup_tau_path, "followup_tau")
        except SchemaError as exc:
            raise SchemaError(f"invalid cohort input for {neutral_id}: {exc}") from exc

        subject_ids.append(neutral_id)
        adjacencies.append(adjacency)
        laplacians.append(laplacian)
        trajectory = np.stack([baseline_tau, followup_tau]).astype(float, copy=True)
        trajectories.append(trajectory)
        times_by_subject.append(np.asarray(COHORT_TIMES, dtype=float))
        degree_values.append(sc_stats["degree"])
        baseline_values.append(baseline_tau)
        followup_values.append(followup_tau)
        diagonal_removal_count += int(sc_stats["removed_diagonal"])

    if not subject_ids:
        raise SchemaError("cohort must contain at least one matched subject.")

    dataset = LongitudinalTrajectoryDataset(
        subject_ids=tuple(subject_ids),
        roi_labels=labels,
        adjacencies=np.stack(adjacencies),
        laplacians=np.stack(laplacians),
        times_by_subject=tuple(times_by_subject),
        trajectories=tuple(trajectories),
        atlas=COHORT_ATLAS,
        roi_alignment=COHORT_ROI_ALIGNMENT,
        tau_measurement=COHORT_TAU_MEASUREMENT,
        tau_reference=COHORT_TAU_REFERENCE,
        sc_weight_type=COHORT_SC_WEIGHT_TYPE,
        sc_static_over_time=COHORT_SC_STATIC_OVER_TIME,
        laplacian_kind=laplacian_kind,
        data_alignment_ready=COHORT_DATA_ALIGNMENT_READY,
        formal_evaluation_ready=COHORT_FORMAL_EVALUATION_READY,
    )
    degrees = np.concatenate(degree_values)
    baseline = np.concatenate(baseline_values)
    followup = np.concatenate(followup_values)
    summary = {
        "ready": True,
        "subject_count": len(subject_ids),
        "matched_triplets": len(subject_ids),
        **file_counts,
        "expected_sc_shape": (COHORT_N_REGIONS, COHORT_N_REGIONS),
        "expected_tau_shape": (COHORT_N_REGIONS,),
        "time_years": COHORT_TIMES,
        "atlas": COHORT_ATLAS,
        "region_count": COHORT_N_REGIONS,
        "roi_alignment": COHORT_ROI_ALIGNMENT,
        "tau_measurement": COHORT_TAU_MEASUREMENT,
        "tau_reference": COHORT_TAU_REFERENCE,
        "sc_weight_type": COHORT_SC_WEIGHT_TYPE,
        "sc_static_over_time": COHORT_SC_STATIC_OVER_TIME,
        "laplacian_kind": laplacian_kind,
        "nonfinite_tau_values": 0,
        "negative_tau_values": 0,
        "nonfinite_sc_values": 0,
        "negative_sc_values": 0,
        "sc_matrices_requiring_diagonal_removal": diagonal_removal_count,
        "all_processed_matrices_symmetric": True,
        "all_processed_diagonals_zero": True,
        "minimum_processed_node_degree": float(np.min(degrees)),
        "median_processed_node_degree": float(np.median(degrees)),
        "zero_degree_nodes": int(np.count_nonzero(degrees <= 0.0)),
        "baseline_tau": _summary_stats(baseline),
        "followup_tau": _summary_stats(followup),
        "mean_tau_change": float(np.mean(followup - baseline)),
    }
    return _CohortComponents(dataset=dataset, summary=summary)


def _load_structural_connectivity(
    path: Path,
    *,
    laplacian_kind: LaplacianKind,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = _load_numeric_csv(path, "structural_connectivity")
    if raw.shape != (COHORT_N_REGIONS, COHORT_N_REGIONS):
        raise SchemaError("structural_connectivity has unexpected shape.")
    if (raw < 0.0).any():
        raise SchemaError("structural_connectivity must be nonnegative.")
    adjacency = raw.astype(float, copy=True)
    removed_diagonal = bool(np.any(np.diag(adjacency) != 0.0))
    adjacency = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    if not np.array_equal(adjacency, adjacency.T):
        raise SchemaError("processed structural_connectivity must be symmetric.")
    if not (np.isfinite(degree).all() and np.all(degree > 0.0)):
        raise SchemaError(
            "processed structural_connectivity must have positive finite node degree."
        )
    laplacian = build_graph_laplacian(adjacency, kind=laplacian_kind)
    return (
        adjacency.astype(float, copy=True),
        laplacian.astype(float, copy=True),
        {"degree": degree.astype(float, copy=True), "removed_diagonal": removed_diagonal},
    )


def _load_tau_vector(path: Path, field_name: str) -> np.ndarray:
    raw = _load_numeric_csv(path, field_name)
    if raw.shape == (COHORT_N_REGIONS,):
        vector = raw
    elif raw.shape == (COHORT_N_REGIONS, 1):
        vector = raw[:, 0]
    elif raw.shape == (1, COHORT_N_REGIONS):
        vector = raw[0, :]
    else:
        raise SchemaError(f"{field_name} has unexpected shape.")
    if (vector < 0.0).any():
        raise SchemaError(f"{field_name} must be nonnegative.")
    return vector.astype(float, copy=True)


def _load_numeric_csv(path: Path, field_name: str) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            array = np.genfromtxt(path, delimiter=",", dtype=float)
    except Exception as exc:
        raise SchemaError(f"{field_name} must be a numeric CSV.") from exc
    if array.dtype.kind in {"b", "O", "S", "U", "c"} or not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise SchemaError(f"{field_name} must be real numeric.")
    if not np.isfinite(array).all():
        raise SchemaError(f"{field_name} must contain finite values.")
    return np.asarray(array, dtype=float)


def _split_manifest(
    dataset: LongitudinalTrajectoryDataset,
    split: SubjectSplit,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, object]:
    labels = split_subject_labels(dataset.subject_ids, split)
    return {
        "seed": int(seed),
        "proportions": {
            "train": float(train_fraction),
            "validation": float(validation_fraction),
            "test": float(test_fraction),
        },
        "counts": {name: len(values) for name, values in labels.items()},
        "subject_labels": labels,
        "n_subjects": len(dataset.subject_ids),
        "n_regions": len(dataset.roi_labels),
        "time_years": COHORT_TIMES,
        "cohort_fingerprint": _cohort_fingerprint(dataset),
        "test_set_role": "held_out_final_evaluation_only",
    }


def _cohort_fingerprint(dataset: LongitudinalTrajectoryDataset) -> str:
    payload = {
        "subject_ids": dataset.subject_ids,
        "n_regions": len(dataset.roi_labels),
        "adjacency_sum": float(np.sum(dataset.adjacencies)),
        "trajectory_sum": float(sum(np.sum(trajectory) for trajectory in dataset.trajectories)),
        "time_years": tuple(
            tuple(float(value) for value in times)
            for times in dataset.times_by_subject
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_private_output_dir(output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if _is_relative_to(output, REPO_ROOT) and not _is_git_ignored(output):
        raise SchemaError("repository-contained private output directory must be ignored by Git.")
    if output.exists() and not output.is_dir():
        raise SchemaError("private output location must be a directory.")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    return output


def _is_git_ignored(path: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _subject_key(path: Path) -> str | None:
    match = _TERMINAL_NUMERIC_KEY_RE.search(path.stem)
    if match is None:
        return None
    return match.group(1)


def _sort_key(value: str) -> tuple[int, str]:
    return int(value), value


def _require_directory(path: str | Path, field_name: str) -> Path:
    try:
        directory = Path(path).expanduser().resolve()
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{field_name} directory is invalid.") from exc
    if not directory.exists() or not directory.is_dir():
        raise SchemaError(f"{field_name} directory is unavailable.")
    return directory


def _validate_subject_ids(subject_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(subject_ids, (str, bytes)) or not isinstance(subject_ids, Sequence):
        raise SchemaError("subject_ids must be a sequence.")
    labels = tuple(str(value) for value in subject_ids)
    if not labels or any(not label.strip() for label in labels):
        raise SchemaError("subject_ids must contain nonempty labels.")
    if len(set(labels)) != len(labels):
        raise SchemaError("subject_ids must be unique.")
    return labels


def _require_int(value: object, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field_name} must be an integer.")
    if value < minimum:
        raise SchemaError(f"{field_name} must be at least {minimum}.")
    return value


def _require_fraction(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{field_name} must be numeric.")
    output = float(value)
    if not math.isfinite(output) or output <= 0.0 or output >= 1.0:
        raise SchemaError(f"{field_name} must be between 0 and 1.")
    return output


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
