from __future__ import annotations

import copy
import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pytest
import torch

from tau_sr.neural import NeuralReactionDiffusion
from tau_sr.pilot_adapter import (
    DATA_ALIGNMENT_READY,
    FORMAL_EVALUATION_READY,
    PILOT_N_LEFT_CORTICAL,
    PILOT_N_REGIONS,
    PILOT_N_RIGHT_CORTICAL,
    PILOT_N_SUBCORTICAL,
    PILOT_SUBCORTICAL_START,
    ROI_ALIGNMENT_DIRECT,
    ROI_LABELS_INVALID,
    UNKNOWN_PROVENANCE,
    LongitudinalTrajectoryDataset,
    inspect_pilot_inputs,
    load_pilot_dataset,
    make_pilot_roi_labels,
)
from tau_sr.recovery import SyntheticRecoveryDataset
from tau_sr.schemas import SchemaError
from tau_sr.training import (
    NeuralTrainingConfig,
    SubjectSplit,
    fit_noise_free_recovery_model,
    fit_recovery_model,
)


def _xlsx(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    sheet_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]
    for row_index, row in enumerate([headers, *rows], start=1):
        sheet_xml.append(f'<row r="{row_index}">')
        for column_index, value in enumerate(row, start=1):
            if value is None:
                continue
            cell = f"{_column(column_index)}{row_index}"
            if isinstance(value, str):
                sheet_xml.append(
                    f'<c r="{cell}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
                )
            else:
                sheet_xml.append(f'<c r="{cell}"><v>{value}</v></c>')
        sheet_xml.append("</row>")
    sheet_xml.extend(["</sheetData>", "</worksheet>"])
    with zipfile.ZipFile(path, "w") as workbook:
        workbook.writestr("[Content_Types].xml", "<Types/>")
        workbook.writestr("xl/worksheets/sheet1.xml", "".join(sheet_xml))


def _column(index: int) -> str:
    letters = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _node_values(offset: float) -> list[float]:
    return [0.1 + offset + 0.0001 * index for index in range(PILOT_N_REGIONS)]


def _tau_workbook(path: Path, *, rows: list[list[object]] | None = None) -> Path:
    headers = ["PTID", "SCAN", "EXAMDATE", "DX", "AGE"]
    headers.extend(f"Node {index}" for index in range(1, PILOT_N_REGIONS + 1))
    if rows is None:
        rows = [
            ["SUBJECT_A", "SCAN_A_2", 10366, "drop", 70, *_node_values(0.02)],
            ["SUBJECT_A", "SCAN_A_1", 10000, "drop", 70, *_node_values(0.00)],
            ["SUBJECT_B", "SCAN_B_1", 20000, "drop", 72, *_node_values(0.04)],
            ["SUBJECT_B", "SCAN_B_2", 20183, "drop", 72, *_node_values(0.05)],
            ["SUBJECT_B", "SCAN_B_3", 20365, "drop", 72, *_node_values(0.06)],
        ]
    _xlsx(path, headers, rows)
    return path


def _connectome_matrix() -> np.ndarray:
    matrix = np.zeros((PILOT_N_REGIONS, PILOT_N_REGIONS), dtype=float)
    for index in range(PILOT_N_REGIONS):
        next_index = (index + 1) % PILOT_N_REGIONS
        matrix[index, next_index] = 0.2 + 0.001 * index
        matrix[next_index, index] = 0.1 + 0.001 * index
    return matrix


def _connectomes(tmp_path: Path) -> tuple[Path, Path]:
    first = tmp_path / "SUBJECT_A_matrix.csv"
    second = tmp_path / "SUBJECT_B_matrix.csv"
    np.savetxt(first, _connectome_matrix(), delimiter=",")
    np.savetxt(second, _connectome_matrix() * 1.1, delimiter=",")
    return first, second


def _roi_labels() -> tuple[str, ...]:
    return tuple(f"ROI {index + 1}" for index in range(PILOT_N_REGIONS))


def _fixture(tmp_path: Path) -> tuple[Path, tuple[Path, Path]]:
    return _tau_workbook(tmp_path / "tau.xlsx"), _connectomes(tmp_path)


def test_make_pilot_roi_labels_uses_confirmed_structural_order() -> None:
    labels = make_pilot_roi_labels()

    assert len(labels) == PILOT_N_REGIONS
    assert labels[:PILOT_N_LEFT_CORTICAL] == tuple(
        f"LH_Destrieux_{index:02d}"
        for index in range(1, PILOT_N_LEFT_CORTICAL + 1)
    )
    assert labels[PILOT_N_LEFT_CORTICAL:PILOT_SUBCORTICAL_START] == tuple(
        f"RH_Destrieux_{index:02d}"
        for index in range(1, PILOT_N_RIGHT_CORTICAL + 1)
    )
    assert labels[PILOT_SUBCORTICAL_START:] == tuple(
        f"Subcortical_{index:02d}"
        for index in range(1, PILOT_N_SUBCORTICAL + 1)
    )
    assert PILOT_N_LEFT_CORTICAL + PILOT_N_RIGHT_CORTICAL + PILOT_N_SUBCORTICAL == 160
    assert sum(label.startswith("Subcortical_") for label in labels) == 12
    assert "Subcortical_13" not in labels


def test_inspect_pilot_inputs_is_privacy_safe_and_reports_readiness(tmp_path: Path) -> None:
    tau, connectomes = _fixture(tmp_path)

    summary = inspect_pilot_inputs(tau, connectomes)

    assert summary["subject_count"] == 2
    assert summary["subject_ids"] == ("Pilot Subject 1", "Pilot Subject 2")
    assert summary["visit_counts"] == {"Pilot Subject 1": 2, "Pilot Subject 2": 3}
    assert summary["region_count"] == 160
    assert summary["roi_alignment"] == ROI_ALIGNMENT_DIRECT
    assert summary["data_alignment_ready"] is DATA_ALIGNMENT_READY
    assert summary["formal_evaluation_ready"] is FORMAL_EVALUATION_READY
    assert summary["missing_tau_values"] == 0
    assert summary["adjacency_shapes"] == (
        (PILOT_N_REGIONS, PILOT_N_REGIONS),
        (PILOT_N_REGIONS, PILOT_N_REGIONS),
    )
    assert summary["laplacian_shapes"] == (
        (PILOT_N_REGIONS, PILOT_N_REGIONS),
        (PILOT_N_REGIONS, PILOT_N_REGIONS),
    )
    assert summary["symmetrization_valid"] is True
    text = repr(summary)
    for private in ("SUBJECT_A", "SUBJECT_B", "SCAN_A", "SCAN_B", "10366", str(tau)):
        assert private not in text


def test_load_pilot_dataset_defaults_to_neutral_roi_labels(tmp_path: Path) -> None:
    tau, connectomes = _fixture(tmp_path)

    dataset = load_pilot_dataset(tau, connectomes)

    assert dataset.roi_labels == make_pilot_roi_labels()
    assert dataset.roi_alignment == ROI_ALIGNMENT_DIRECT
    assert dataset.data_alignment_ready is True
    assert dataset.formal_evaluation_ready is False


def test_supplied_roi_labels_change_names_only_and_are_validated(tmp_path: Path) -> None:
    tau, connectomes = _fixture(tmp_path)
    default = load_pilot_dataset(tau, connectomes)
    custom_labels = _roi_labels()

    custom = load_pilot_dataset(tau, connectomes, roi_labels=custom_labels)

    assert custom.roi_labels == custom_labels
    np.testing.assert_array_equal(custom.adjacencies, default.adjacencies)
    np.testing.assert_array_equal(custom.laplacians, default.laplacians)
    for left, right in zip(custom.trajectories, default.trajectories, strict=True):
        np.testing.assert_array_equal(left, right)

    with pytest.raises(SchemaError, match=ROI_LABELS_INVALID):
        load_pilot_dataset(tau, connectomes, roi_labels=_roi_labels()[:-1])
    with pytest.raises(SchemaError, match=ROI_LABELS_INVALID):
        load_pilot_dataset(tau, connectomes, roi_labels=(*_roi_labels()[:-1], "ROI 1"))
    with pytest.raises(SchemaError, match="nonempty"):
        load_pilot_dataset(tau, connectomes, roi_labels=(*_roi_labels()[:-1], ""))


def test_load_pilot_dataset_sorts_visits_symmetrizes_and_uses_neutral_labels(
    tmp_path: Path,
) -> None:
    tau, connectomes = _fixture(tmp_path)

    dataset = load_pilot_dataset(tau, connectomes)

    assert dataset.subject_ids == ("Pilot Subject 1", "Pilot Subject 2")
    assert dataset.atlas == "Destrieux cortical plus 12 subcortical regions"
    assert dataset.roi_alignment == "verified_direct_index"
    assert dataset.tau_measurement == "SUVR"
    assert dataset.tau_reference == "pending_exact_confirmation"
    assert dataset.sc_weight_type == "normalized_fiber_count"
    assert dataset.sc_static_over_time is True
    assert dataset.laplacian_kind == "symmetric_normalized"
    assert dataset.times_by_subject[0][0] == pytest.approx(0.0)
    assert dataset.times_by_subject[0][1] == pytest.approx(366.0 / 365.25)
    expected_first = 0.5 * (_connectome_matrix() + _connectome_matrix().T)
    expected_second = 0.5 * ((_connectome_matrix() * 1.1) + (_connectome_matrix() * 1.1).T)
    np.fill_diagonal(expected_first, 0.0)
    np.fill_diagonal(expected_second, 0.0)
    np.testing.assert_allclose(dataset.adjacencies[0], expected_first)
    np.testing.assert_allclose(dataset.adjacencies[1], expected_second)
    assert not np.array_equal(dataset.adjacencies[0], dataset.adjacencies[1])
    assert np.array_equal(np.diag(dataset.adjacencies[0]), np.zeros(PILOT_N_REGIONS))
    assert np.min(dataset.adjacencies[0].sum(axis=1)) > 0.0
    np.testing.assert_allclose(dataset.trajectories[0][0], _node_values(0.0))
    assert dataset.trajectories[0][0, 0] == pytest.approx(_node_values(0.0)[0])
    assert dataset.trajectories[0][0, 73] == pytest.approx(_node_values(0.0)[73])
    assert dataset.trajectories[0][0, 74] == pytest.approx(_node_values(0.0)[74])
    assert dataset.trajectories[0][0, 147] == pytest.approx(_node_values(0.0)[147])
    assert dataset.trajectories[0][0, 148] == pytest.approx(_node_values(0.0)[148])
    assert dataset.trajectories[0][0, 159] == pytest.approx(_node_values(0.0)[159])
    assert [trajectory.shape[0] for trajectory in dataset.trajectories] == [2, 3]
    with pytest.raises(ValueError):
        dataset.trajectories[0][0, 0] = 9.0
    assert "SUBJECT_A" not in repr(dataset)


def test_connectome_matching_uses_complete_subject_tokens(tmp_path: Path) -> None:
    rows = [
        ["S1", "SCAN_1", 10000, "drop", 70, *_node_values(0.0)],
        ["S1", "SCAN_2", 10366, "drop", 70, *_node_values(0.1)],
        ["S10", "SCAN_1", 20000, "drop", 70, *_node_values(0.2)],
        ["S10", "SCAN_2", 20366, "drop", 70, *_node_values(0.3)],
    ]
    tau = _tau_workbook(tmp_path / "tau.xlsx", rows=rows)
    s1_connectome = tmp_path / "S1_matrix.csv"
    s10_connectome = tmp_path / "S10_matrix.csv"
    np.savetxt(s1_connectome, _connectome_matrix(), delimiter=",")
    np.savetxt(s10_connectome, _connectome_matrix() * 1.1, delimiter=",")

    summary = inspect_pilot_inputs(tau, (s10_connectome, s1_connectome))

    assert summary["subject_count"] == 2
    assert summary["connectome_shapes"] == (
        (PILOT_N_REGIONS, PILOT_N_REGIONS),
        (PILOT_N_REGIONS, PILOT_N_REGIONS),
    )

    embedded_s10_connectome = tmp_path / "S10matrix.csv"
    np.savetxt(embedded_s10_connectome, _connectome_matrix() * 1.1, delimiter=",")
    with pytest.raises(SchemaError, match="exactly one connectome"):
        inspect_pilot_inputs(tau, (s1_connectome, embedded_s10_connectome))


def test_tau_validation_counts_missing_node_values_and_rejects_duplicate_visits(
    tmp_path: Path,
) -> None:
    connectomes = _connectomes(tmp_path)
    bad_rows = [
        ["SUBJECT_A", "SCAN_A_1", 10000, "drop", 70, *_node_values(0.0)],
        ["SUBJECT_A", "SCAN_A_1", 10366, "drop", 70, *_node_values(0.1)],
    ]
    with pytest.raises(SchemaError, match="duplicate subject/visit"):
        inspect_pilot_inputs(
            _tau_workbook(tmp_path / "duplicate.xlsx", rows=bad_rows),
            connectomes[:1],
        )

    missing_rows = copy.deepcopy(bad_rows)
    missing_rows[1][1] = "SCAN_A_2"
    missing_rows[0][5] = math.nan
    missing_rows[0][6] = math.inf
    missing_rows[1][7] = None
    summary = inspect_pilot_inputs(
        _tau_workbook(tmp_path / "missing.xlsx", rows=missing_rows),
        connectomes[:1],
    )
    assert summary["missing_tau_values"] == 3

    nonnumeric_rows = copy.deepcopy(missing_rows)
    nonnumeric_rows[0][8] = "not numeric"
    with pytest.raises(SchemaError, match="numeric"):
        inspect_pilot_inputs(
            _tau_workbook(tmp_path / "nonnumeric.xlsx", rows=nonnumeric_rows),
            connectomes[:1],
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda matrix: matrix[:159, :159], "unexpected shape"),
        (lambda matrix: matrix.__setitem__((0, 1), -1.0) or matrix, "nonnegative"),
        (lambda matrix: matrix.__setitem__((0, 1), math.inf) or matrix, "finite"),
        (lambda matrix: matrix.__setitem__((0, 0), 1.0) or matrix, "diagonal"),
    ],
)
def test_connectome_validation_is_privacy_safe(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    tau = _tau_workbook(
        tmp_path / "tau.xlsx",
        rows=[
            ["SUBJECT_A", "SCAN_A_1", 10000, "drop", 70, *_node_values(0.0)],
            ["SUBJECT_A", "SCAN_A_2", 10366, "drop", 70, *_node_values(0.1)],
        ],
    )
    connectome = tmp_path / "SUBJECT_A_matrix.csv"
    np.savetxt(connectome, mutate(_connectome_matrix()), delimiter=",")

    with pytest.raises(SchemaError, match=message) as excinfo:
        inspect_pilot_inputs(tau, (connectome,))
    assert "SUBJECT_A" not in str(excinfo.value)
    assert str(connectome) not in str(excinfo.value)


def test_adapter_reuses_existing_laplacian_builder(tmp_path: Path, monkeypatch) -> None:
    tau, connectomes = _fixture(tmp_path)
    calls: list[tuple[tuple[int, int], str]] = []

    def fake_laplacian(adjacency: np.ndarray, *, kind: str) -> np.ndarray:
        calls.append((adjacency.shape, kind))
        return np.eye(adjacency.shape[0], dtype=float)

    monkeypatch.setattr("tau_sr.pilot_adapter.build_graph_laplacian", fake_laplacian)
    dataset = load_pilot_dataset(tau, connectomes, roi_labels=_roi_labels())

    assert calls == [
        ((PILOT_N_REGIONS, PILOT_N_REGIONS), "symmetric_normalized"),
        ((PILOT_N_REGIONS, PILOT_N_REGIONS), "symmetric_normalized"),
    ]
    np.testing.assert_array_equal(dataset.laplacians[0], np.eye(PILOT_N_REGIONS))


def test_generic_dataset_defaults_to_unknown_unready_provenance() -> None:
    synthetic = _small_synthetic_dataset()

    generic = LongitudinalTrajectoryDataset(
        subject_ids=synthetic.subject_ids,
        roi_labels=synthetic.roi_labels,
        adjacencies=synthetic.adjacencies,
        laplacians=synthetic.laplacians,
        times_by_subject=synthetic.times_by_subject,
        trajectories=synthetic.trajectories,
    )

    assert generic.atlas == UNKNOWN_PROVENANCE
    assert generic.roi_alignment == UNKNOWN_PROVENANCE
    assert generic.tau_measurement == UNKNOWN_PROVENANCE
    assert generic.tau_reference == UNKNOWN_PROVENANCE
    assert generic.sc_weight_type == UNKNOWN_PROVENANCE
    assert generic.sc_static_over_time is False
    assert generic.laplacian_kind == UNKNOWN_PROVENANCE
    assert generic.data_alignment_ready is False
    assert generic.formal_evaluation_ready is False


def test_ready_dataset_requires_explicit_provenance() -> None:
    synthetic = _small_synthetic_dataset()

    with pytest.raises(SchemaError, match="explicit provenance"):
        LongitudinalTrajectoryDataset(
            subject_ids=synthetic.subject_ids,
            roi_labels=synthetic.roi_labels,
            adjacencies=synthetic.adjacencies,
            laplacians=synthetic.laplacians,
            times_by_subject=synthetic.times_by_subject,
            trajectories=synthetic.trajectories,
            data_alignment_ready=True,
        )

    with pytest.raises(SchemaError, match="explicit provenance"):
        LongitudinalTrajectoryDataset(
            subject_ids=synthetic.subject_ids,
            roi_labels=synthetic.roi_labels,
            adjacencies=synthetic.adjacencies,
            laplacians=synthetic.laplacians,
            times_by_subject=synthetic.times_by_subject,
            trajectories=synthetic.trajectories,
            data_alignment_ready=True,
            formal_evaluation_ready=True,
        )


def test_formal_evaluation_readiness_requires_data_alignment() -> None:
    synthetic = _small_synthetic_dataset()

    with pytest.raises(SchemaError, match="data_alignment_ready"):
        LongitudinalTrajectoryDataset(
            subject_ids=synthetic.subject_ids,
            roi_labels=synthetic.roi_labels,
            adjacencies=synthetic.adjacencies,
            laplacians=synthetic.laplacians,
            times_by_subject=synthetic.times_by_subject,
            trajectories=synthetic.trajectories,
            formal_evaluation_ready=True,
        )


def test_synthetic_and_generic_training_paths_are_equivalent() -> None:
    synthetic = _small_synthetic_dataset()
    generic = LongitudinalTrajectoryDataset(
        subject_ids=synthetic.subject_ids,
        roi_labels=synthetic.roi_labels,
        adjacencies=synthetic.adjacencies,
        laplacians=synthetic.laplacians,
        times_by_subject=synthetic.times_by_subject,
        trajectories=synthetic.trajectories,
    )
    split = SubjectSplit(train_indices=(0, 1), validation_indices=(2,), test_indices=(3,))
    config = NeuralTrainingConfig(
        hidden_width=3,
        initial_rho=0.05,
        learning_rate=1e-2,
        max_epochs=3,
        patience=3,
        seed=123,
        dtype=torch.float64,
    )
    torch.manual_seed(999)
    model = NeuralReactionDiffusion(hidden_width=3, initial_rho=0.05).double()
    synthetic_model = copy.deepcopy(model)
    generic_model = copy.deepcopy(model)

    synthetic_result = fit_noise_free_recovery_model(
        synthetic,
        split,
        config,
        model=synthetic_model,
    )
    generic_result = fit_recovery_model(generic, split, config, model=generic_model)

    assert generic.baseline_states.shape == synthetic.baseline_states.shape
    assert synthetic_result.training_losses == pytest.approx(generic_result.training_losses)
    assert synthetic_result.validation_losses == pytest.approx(generic_result.validation_losses)
    assert synthetic_result.best_epoch == generic_result.best_epoch
    assert synthetic_result.epochs_run == generic_result.epochs_run
    assert synthetic_result.model.rho.item() == pytest.approx(generic_result.model.rho.item())
    for name, synthetic_value in synthetic_result.model.state_dict().items():
        torch.testing.assert_close(synthetic_value, generic_result.model.state_dict()[name])
    _assert_same_prediction(synthetic_result.model, generic_result.model, synthetic)


def _small_synthetic_dataset() -> SyntheticRecoveryDataset:
    times = (
        np.array([0.0, 0.1], dtype=float),
        np.array([0.0, 0.1], dtype=float),
        np.array([0.0, 0.2], dtype=float),
        np.array([0.0, 0.2], dtype=float),
    )
    baseline = np.array(
        [[0.10, 0.20], [0.12, 0.22], [0.14, 0.24], [0.16, 0.26]],
        dtype=float,
    )
    trajectories = tuple(
        baseline[index][None, :] + subject_times[:, None] * 0.03
        for index, subject_times in enumerate(times)
    )
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    return SyntheticRecoveryDataset(
        subject_ids=tuple(f"synthetic_{index}" for index in range(4)),
        roi_labels=("ROI 1", "ROI 2"),
        baseline_states=baseline.copy(),
        adjacencies=np.stack([np.ones((2, 2)) - np.eye(2) for _ in range(4)]),
        laplacians=np.stack([laplacian.copy() for _ in range(4)]),
        times_by_subject=tuple(subject_times.copy() for subject_times in times),
        trajectories=tuple(trajectory.copy() for trajectory in trajectories),
        representation="count",
        laplacian_kind="unnormalized",
        rho=0.1,
        growth_rate=0.5,
        saturation_rate=0.5,
        seed=1,
    )


def _assert_same_prediction(
    synthetic_model: NeuralReactionDiffusion,
    generic_model: NeuralReactionDiffusion,
    dataset: SyntheticRecoveryDataset,
) -> None:
    y0 = torch.tensor(dataset.baseline_states[0], dtype=torch.float64)
    laplacian = torch.tensor(dataset.laplacians[0], dtype=torch.float64)
    times = torch.tensor(dataset.times_by_subject[0], dtype=torch.float64)
    with torch.no_grad():
        synthetic_prediction = synthetic_model.integrate(y0, laplacian, times)
        generic_prediction = generic_model.integrate(y0, laplacian, times)
    torch.testing.assert_close(synthetic_prediction, generic_prediction)
