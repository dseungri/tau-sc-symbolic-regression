from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tau_sr.cohort_adapter import (
    COHORT_ATLAS,
    COHORT_N_REGIONS,
    COHORT_ROI_ALIGNMENT,
    COHORT_SC_WEIGHT_TYPE,
    COHORT_TAU_REFERENCE,
    DEFAULT_SPLIT_SEED,
    inspect_two_timepoint_cohort,
    load_two_timepoint_cohort,
    make_frozen_cohort_split,
    split_subject_labels,
    write_private_cohort_artifacts,
)
from tau_sr.graphs import build_graph_laplacian
from tau_sr.schemas import SchemaError


def _matrix(scale: float = 1.0, *, diagonal: float = 0.0) -> np.ndarray:
    matrix = np.zeros((COHORT_N_REGIONS, COHORT_N_REGIONS), dtype=float)
    for index in range(COHORT_N_REGIONS - 1):
        matrix[index, index + 1] = scale * (0.2 + 0.001 * index)
        matrix[index + 1, index] = scale * (0.1 + 0.001 * index)
    if diagonal:
        np.fill_diagonal(matrix, diagonal)
    return matrix


def _tau(offset: float = 0.0) -> np.ndarray:
    return np.linspace(0.1 + offset, 0.5 + offset, COHORT_N_REGIONS, dtype=float)


def _write_csv(path: Path, array: np.ndarray) -> None:
    np.savetxt(path, array, delimiter=",")


def _write_subject(
    root: Path,
    key: int,
    *,
    scale: float = 1.0,
    diagonal: float = 0.0,
    baseline_row: bool = False,
    followup_column: bool = False,
) -> None:
    _write_csv(root / "structure" / f"structure_fake_{key}.csv", _matrix(scale, diagonal=diagonal))
    baseline = _tau(0.01 * key)
    followup = baseline + 0.05
    _write_csv(
        root / "baseline" / f"baseline_fake_{key}.csv",
        baseline.reshape(1, -1) if baseline_row else baseline.reshape(-1, 1),
    )
    _write_csv(
        root / "followup" / f"followup_fake_{key}.csv",
        followup.reshape(-1, 1) if followup_column else followup.reshape(1, -1),
    )


def _cohort_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    structure = tmp_path / "structure"
    baseline = tmp_path / "baseline"
    followup = tmp_path / "followup"
    for directory in (structure, baseline, followup):
        directory.mkdir(parents=True)
        (directory / ".DS_Store").write_text("ignored", encoding="utf-8")
    return structure, baseline, followup


def _valid_cohort(tmp_path: Path, *, n_subjects: int = 3) -> tuple[Path, Path, Path]:
    structure, baseline, followup = _cohort_dirs(tmp_path)
    for offset, key in enumerate(range(1001, 1001 + n_subjects), start=1):
        _write_subject(
            tmp_path,
            key,
            scale=1.0 + 0.1 * offset,
            diagonal=0.5,
            baseline_row=offset % 2 == 0,
            followup_column=offset % 2 == 1,
        )
    return structure, baseline, followup


def test_load_two_timepoint_cohort_uses_neutral_order_and_metadata(tmp_path: Path) -> None:
    structure, baseline, followup = _valid_cohort(tmp_path)

    dataset = load_two_timepoint_cohort(structure, baseline, followup)

    assert dataset.subject_ids == (
        "Cohort Subject 001",
        "Cohort Subject 002",
        "Cohort Subject 003",
    )
    assert len(dataset.roi_labels) == COHORT_N_REGIONS
    assert dataset.atlas == COHORT_ATLAS
    assert dataset.roi_alignment == COHORT_ROI_ALIGNMENT
    assert dataset.tau_measurement == "SUVR"
    assert dataset.tau_reference == COHORT_TAU_REFERENCE
    assert dataset.sc_weight_type == COHORT_SC_WEIGHT_TYPE
    assert dataset.sc_static_over_time is True
    assert dataset.laplacian_kind == "symmetric_normalized"
    assert dataset.data_alignment_ready is True
    assert dataset.formal_evaluation_ready is False
    for times in dataset.times_by_subject:
        np.testing.assert_array_equal(times, np.array([0.0, 1.0]))
    assert [trajectory.shape for trajectory in dataset.trajectories] == [
        (2, COHORT_N_REGIONS),
        (2, COHORT_N_REGIONS),
        (2, COHORT_N_REGIONS),
    ]
    with pytest.raises(ValueError):
        dataset.adjacencies[0, 0, 1] = 9.0
    with pytest.raises(ValueError):
        dataset.trajectories[0][0, 0] = 9.0
    assert "1001" not in repr(dataset)


def test_structural_connectivity_is_symmetrized_without_thresholding_or_normalization(
    tmp_path: Path,
) -> None:
    structure, baseline, followup = _valid_cohort(tmp_path, n_subjects=1)
    raw = _matrix(1.1, diagonal=0.5)
    expected = 0.5 * (raw + raw.T)
    np.fill_diagonal(expected, 0.0)

    dataset = load_two_timepoint_cohort(structure, baseline, followup)

    np.testing.assert_allclose(dataset.adjacencies[0], expected)
    np.testing.assert_allclose(
        dataset.laplacians[0],
        build_graph_laplacian(expected, kind="symmetric_normalized"),
    )
    assert np.array_equal(np.diag(dataset.adjacencies[0]), np.zeros(COHORT_N_REGIONS))
    assert np.count_nonzero(dataset.adjacencies[0]) == np.count_nonzero(expected)


def test_inspect_two_timepoint_cohort_is_aggregate_and_privacy_safe(tmp_path: Path) -> None:
    structure, baseline, followup = _valid_cohort(tmp_path)

    summary = inspect_two_timepoint_cohort(structure, baseline, followup)

    assert summary["ready"] is True
    assert summary["subject_count"] == 3
    assert summary["matched_triplets"] == 3
    assert summary["expected_sc_shape"] == (COHORT_N_REGIONS, COHORT_N_REGIONS)
    assert summary["expected_tau_shape"] == (COHORT_N_REGIONS,)
    assert summary["sc_matrices_requiring_diagonal_removal"] == 3
    assert summary["all_processed_matrices_symmetric"] is True
    assert summary["all_processed_diagonals_zero"] is True
    assert summary["minimum_processed_node_degree"] > 0.0
    assert summary["zero_degree_nodes"] == 0
    assert summary["mean_tau_change"] == pytest.approx(0.05)
    text = repr(summary)
    for private in ("1001", "structure_fake", "baseline_fake", str(structure), str(baseline)):
        assert private not in text


@pytest.mark.parametrize(
    ("role", "array", "message"),
    [
        ("structure", np.zeros((159, 160)), "structural_connectivity has unexpected shape"),
        ("baseline", np.zeros((2, 80)), "baseline_tau has unexpected shape"),
        ("followup", np.full((COHORT_N_REGIONS,), np.nan), "followup_tau must contain finite"),
        ("baseline", np.full((COHORT_N_REGIONS,), np.inf), "baseline_tau must contain finite"),
        ("followup", -np.ones((COHORT_N_REGIONS,)), "followup_tau must be nonnegative"),
    ],
)
def test_loader_rejects_invalid_shapes_and_tau_values(
    tmp_path: Path,
    role: str,
    array: np.ndarray,
    message: str,
) -> None:
    structure, baseline, followup = _valid_cohort(tmp_path, n_subjects=1)
    paths = {
        "structure": next(structure.glob("*.csv")),
        "baseline": next(baseline.glob("*.csv")),
        "followup": next(followup.glob("*.csv")),
    }
    _write_csv(paths[role], array)

    with pytest.raises(SchemaError, match=message) as error:
        load_two_timepoint_cohort(structure, baseline, followup)
    assert "1001" not in str(error.value)


def test_loader_rejects_nonnumeric_cells_negative_sc_and_zero_degree(tmp_path: Path) -> None:
    structure, baseline, followup = _valid_cohort(tmp_path, n_subjects=1)
    baseline_file = next(baseline.glob("*.csv"))
    baseline_file.write_text("not,numeric\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="baseline_tau must contain finite"):
        load_two_timepoint_cohort(structure, baseline, followup)

    _write_csv(baseline_file, _tau())
    structure_file = next(structure.glob("*.csv"))
    bad_sc = _matrix()
    bad_sc[0, 1] = -1.0
    _write_csv(structure_file, bad_sc)
    with pytest.raises(SchemaError, match="structural_connectivity must be nonnegative"):
        load_two_timepoint_cohort(structure, baseline, followup)

    _write_csv(structure_file, np.eye(COHORT_N_REGIONS))
    with pytest.raises(SchemaError, match="positive finite node degree"):
        load_two_timepoint_cohort(structure, baseline, followup)


def test_matching_rejects_missing_duplicate_and_unexpected_csv(tmp_path: Path) -> None:
    structure, baseline, followup = _valid_cohort(tmp_path, n_subjects=1)
    next(followup.glob("*.csv")).unlink()
    with pytest.raises(SchemaError, match="missing_counterparts=1"):
        load_two_timepoint_cohort(structure, baseline, followup)

    structure, baseline, followup = _valid_cohort(tmp_path / "dup", n_subjects=1)
    _write_csv(structure / "other_fake_1001.csv", _matrix())
    with pytest.raises(SchemaError, match="duplicate_keys=1"):
        load_two_timepoint_cohort(structure, baseline, followup)

    structure, baseline, followup = _valid_cohort(tmp_path / "bad", n_subjects=1)
    _write_csv(baseline / "unexpected_subject.csv", _tau())
    with pytest.raises(SchemaError, match="unexpected_or_malformed_csv=1"):
        load_two_timepoint_cohort(structure, baseline, followup)


def test_frozen_split_is_deterministic_subject_level_and_preserves_global_rng() -> None:
    subject_ids = tuple(f"Cohort Subject {index:03d}" for index in range(1, 161))
    np.random.seed(123)
    before = np.random.get_state()

    first = make_frozen_cohort_split(subject_ids, seed=DEFAULT_SPLIT_SEED)
    second = make_frozen_cohort_split(subject_ids, seed=DEFAULT_SPLIT_SEED)
    after = np.random.get_state()

    assert first == second
    assert len(first.train_indices) == 112
    assert len(first.validation_indices) == 16
    assert len(first.test_indices) == 32
    all_indices = set(first.train_indices) | set(first.validation_indices) | set(first.test_indices)
    assert all_indices == set(range(160))
    assert set(first.train_indices).isdisjoint(first.validation_indices)
    assert set(first.train_indices).isdisjoint(first.test_indices)
    assert set(first.validation_indices).isdisjoint(first.test_indices)
    assert all(np.array_equal(left, right) for left, right in zip(before, after, strict=True))

    labels = split_subject_labels(subject_ids, first)
    assert set(labels) == {"train", "validation", "test"}
    assert all(label.startswith("Cohort Subject ") for group in labels.values() for label in group)

    split_150 = make_frozen_cohort_split(subject_ids[:150], seed=DEFAULT_SPLIT_SEED)
    assert len(split_150.train_indices) == 105
    assert len(split_150.validation_indices) == 15
    assert len(split_150.test_indices) == 30


def test_private_artifact_writer_rejects_unignored_repo_output_and_writes_neutral_manifest(
    tmp_path: Path,
) -> None:
    structure, baseline, followup = _valid_cohort(tmp_path, n_subjects=10)
    summary = inspect_two_timepoint_cohort(structure, baseline, followup)
    dataset = load_two_timepoint_cohort(structure, baseline, followup)
    split = make_frozen_cohort_split(dataset.subject_ids)

    with pytest.raises(SchemaError, match="repository-contained"):
        write_private_cohort_artifacts(
            dataset,
            summary,
            split,
            Path.cwd() / "private_cohort_output",
        )

    output_dir = tmp_path / "private_output"
    paths = write_private_cohort_artifacts(dataset, summary, split, output_dir)

    assert set(paths) == {"qc_summary", "split_manifest"}
    assert oct(output_dir.stat().st_mode & 0o777) == "0o700"
    for path in paths.values():
        assert path.exists()
        assert oct(path.stat().st_mode & 0o777) == "0o600"
    manifest = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    assert manifest["counts"] == {"train": 7, "validation": 1, "test": 2}
    assert "cohort_fingerprint" in manifest
    assert "1001" not in repr(manifest)
