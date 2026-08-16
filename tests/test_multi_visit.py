from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tau_sr.multi_visit import (
    MULTI_VISIT_N_REGIONS,
    cohort_visit_count_summary,
    compute_protocol_hash,
    evaluate_final_visit_forecasts,
    forecast_linear_extrapolation,
    inspect_multi_visit_manifest,
    load_multi_visit_manifest,
    make_multi_visit_outer_folds,
    paired_participant_bootstrap,
    prepare_private_multi_visit_output_dir,
    run_synthetic_multi_visit_rehearsal,
    validate_multi_visit_protocol,
    verify_analysis_freeze_protocol,
    write_private_multi_visit_json,
)
from tau_sr.schemas import SchemaError


def _roi_labels() -> list[str]:
    return [f"Region {index:03d}" for index in range(1, MULTI_VISIT_N_REGIONS + 1)]


def _adjacency(scale: float = 1.0) -> np.ndarray:
    matrix = np.zeros((MULTI_VISIT_N_REGIONS, MULTI_VISIT_N_REGIONS), dtype=float)
    for index in range(MULTI_VISIT_N_REGIONS - 1):
        weight = scale * (0.1 + 0.001 * index)
        matrix[index, index + 1] = weight
        matrix[index + 1, index] = weight
    np.fill_diagonal(matrix, 0.25)
    return matrix


def _tau(offset: float) -> np.ndarray:
    return np.linspace(0.1 + offset, 0.5 + offset, MULTI_VISIT_N_REGIONS, dtype=float)


def _write_labeled_tau(path: Path, values: np.ndarray, labels: list[str] | None = None) -> None:
    labels = _roi_labels() if labels is None else labels
    with path.open("w", encoding="utf-8") as handle:
        handle.write("roi_label,tau_value\n")
        for label, value in zip(labels, values, strict=True):
            handle.write(f"{label},{float(value):.12g}\n")


def _write_manifest(
    root: Path,
    *,
    visit_counts: tuple[int, ...] = (3, 4, 3),
    shuffled_visits: bool = False,
    duplicate_sc: bool = False,
) -> Path:
    sc_dir = root / "sc"
    tau_dir = root / "tau"
    sc_dir.mkdir(parents=True)
    tau_dir.mkdir(parents=True)
    participants = []
    shared_sc_name = "participant_shared_sc.csv"
    for participant_index, visit_count in enumerate(visit_counts, start=1):
        sc_name = shared_sc_name if duplicate_sc else f"participant_{participant_index:03d}_sc.csv"
        np.savetxt(sc_dir / sc_name, _adjacency(1.0 + 0.01 * participant_index), delimiter=",")
        visits = []
        times = [0.0, 0.7, 1.6, 2.5][:visit_count]
        if visit_count == 3:
            times = [0.0, 1.1, 2.3]
        for visit_index, elapsed in enumerate(times):
            tau_name = f"participant_{participant_index:03d}_visit_{visit_index:02d}.csv"
            _write_labeled_tau(
                tau_dir / tau_name,
                _tau(0.02 * participant_index + 0.03 * visit_index),
            )
            visits.append({"tau_file": f"tau/{tau_name}", "elapsed_years": elapsed})
        if shuffled_visits:
            visits = list(reversed(visits))
        participants.append(
            {
                "participant_label": f"Participant {participant_index:03d}",
                "sc_file": f"sc/{sc_name}",
                "visits": visits,
            }
        )
    payload = {
        "schema_version": "tau_sr_multi_visit_manifest_v1",
        "n_regions": MULTI_VISIT_N_REGIONS,
        "roi_labels": _roi_labels(),
        "metadata": {
            "atlas": "synthetic 160-region atlas",
            "roi_alignment": "synthetic_direct_index_alignment",
            "tau_measurement": "synthetic_tau",
            "tau_reference": "synthetic_reference",
            "sc_weight_type": "synthetic_weight",
            "sc_static_over_time": True,
            "data_alignment_ready": True,
        },
        "participants": participants,
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


def _manifest_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_documents_validate_and_hash_is_stable() -> None:
    first = validate_multi_visit_protocol()
    second = validate_multi_visit_protocol()

    assert first["protocol_hash"] == second["protocol_hash"]
    assert first["protocol_hash"] == compute_protocol_hash()


def test_load_multi_visit_manifest_sorts_visits_and_keeps_privacy(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, visit_counts=(3, 4), shuffled_visits=True)

    result = load_multi_visit_manifest(manifest)

    assert result.dataset.subject_ids == ("Participant 001", "Participant 002")
    assert cohort_visit_count_summary(result.dataset)["visit_count_histogram"] == {
        "3": 1,
        "4": 1,
    }
    np.testing.assert_array_equal(result.dataset.times_by_subject[0], np.asarray([0.0, 1.1, 2.3]))
    assert result.qc_summary["minimum_visits_per_participant"] == 3
    assert result.qc_summary["maximum_visits_per_participant"] == 4
    assert result.qc_summary["zero_degree_nodes"] == 0
    assert result.qc_summary["irregular_interval_participant_count"] == 2
    assert "participant_001" not in repr(result.qc_summary)
    with pytest.raises(ValueError):
        result.dataset.adjacencies[0, 0, 1] = 9.0
    with pytest.raises(ValueError):
        result.dataset.trajectories[0][0, 0] = 9.0


def test_inspect_multi_visit_manifest_is_aggregate(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, visit_counts=(3, 4, 3))

    summary = inspect_multi_visit_manifest(manifest)

    assert summary["ready"] is True
    assert summary["subject_count"] == 3
    assert summary["total_tau_visits"] == 10
    assert summary["cohort_fingerprint"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload, root: payload["participants"][0]["visits"].pop(),
            "at least three visits",
        ),
        (
            lambda payload, root: Path(
                root / payload["participants"][0]["visits"][0]["tau_file"]
            ).unlink(),
            "missing tau visit file",
        ),
        (
            lambda payload, root: payload["participants"][0]["visits"][1].update(
                {"elapsed_years": 0.0}
            ),
            "elapsed_years must be unique",
        ),
        (
            lambda payload, root: payload["participants"][0]["visits"][1].update(
                {"elapsed_years": float("nan")}
            ),
            "elapsed_years must be finite",
        ),
        (
            lambda payload, root: payload.update({"n_regions": 159}),
            "n_regions must be 160",
        ),
        (
            lambda payload, root: payload["participants"].append(payload["participants"][0].copy()),
            "duplicate participant labels",
        ),
    ],
)
def test_manifest_rejects_invalid_manifest_level_inputs(
    tmp_path: Path,
    mutator: object,
    message: str,
) -> None:
    manifest = _write_manifest(tmp_path, visit_counts=(3,))
    payload = _manifest_payload(manifest)
    mutator(payload, tmp_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SchemaError, match=message):
        load_multi_visit_manifest(manifest)


def test_manifest_rejects_duplicate_sc_wrong_region_order_nonfinite_and_zero_degree(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path / "duplicate_sc", visit_counts=(3, 3), duplicate_sc=True)
    with pytest.raises(SchemaError, match="duplicate structural connectivity"):
        load_multi_visit_manifest(manifest)

    manifest = _write_manifest(tmp_path / "labels", visit_counts=(3,))
    payload = _manifest_payload(manifest)
    tau_file = tmp_path / "labels" / payload["participants"][0]["visits"][0]["tau_file"]
    labels = _roi_labels()
    labels[0], labels[1] = labels[1], labels[0]
    _write_labeled_tau(tau_file, _tau(0.0), labels=labels)
    with pytest.raises(SchemaError, match="region-order mismatch"):
        load_multi_visit_manifest(manifest)

    manifest = _write_manifest(tmp_path / "nonfinite", visit_counts=(3,))
    payload = _manifest_payload(manifest)
    tau_file = tmp_path / "nonfinite" / payload["participants"][0]["visits"][0]["tau_file"]
    bad_tau = _tau(0.0)
    bad_tau[0] = np.inf
    _write_labeled_tau(tau_file, bad_tau)
    with pytest.raises(SchemaError, match="tau_value must be finite"):
        load_multi_visit_manifest(manifest)

    manifest = _write_manifest(tmp_path / "zero_degree", visit_counts=(3,))
    payload = _manifest_payload(manifest)
    sc_file = tmp_path / "zero_degree" / payload["participants"][0]["sc_file"]
    np.savetxt(sc_file, np.zeros((MULTI_VISIT_N_REGIONS, MULTI_VISIT_N_REGIONS)), delimiter=",")
    with pytest.raises(SchemaError, match="finite positive degree"):
        load_multi_visit_manifest(manifest)


def test_folds_are_subject_level_and_linear_forecast_uses_only_permissible_visits(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path, visit_counts=(3, 3, 3, 3, 3, 3))
    dataset = load_multi_visit_manifest(manifest).dataset
    folds = make_multi_visit_outer_folds(dataset.subject_ids, n_folds=3, seed=42)

    all_test_labels = [
        label
        for fold in folds
        for label in fold["outer_test_labels"]
    ]
    assert sorted(all_test_labels) == sorted(dataset.subject_ids)
    assert len(set(all_test_labels)) == len(dataset.subject_ids)
    for fold in folds:
        assert set(fold["outer_test_labels"]).isdisjoint(fold["fitting_labels"])
        assert set(fold["outer_test_labels"]).isdisjoint(fold["inner_validation_labels"])

    forecast = forecast_linear_extrapolation(dataset, (0,))["Participant 001"]
    original = forecast.copy()
    modified = list(dataset.trajectories)
    changed = modified[0].copy()
    changed[-1] = 999.0
    modified[0] = changed
    changed_dataset = type(dataset)(
        subject_ids=dataset.subject_ids,
        roi_labels=dataset.roi_labels,
        adjacencies=dataset.adjacencies,
        laplacians=dataset.laplacians,
        times_by_subject=dataset.times_by_subject,
        trajectories=tuple(modified),
        atlas=dataset.atlas,
        roi_alignment=dataset.roi_alignment,
        tau_measurement=dataset.tau_measurement,
        tau_reference=dataset.tau_reference,
        sc_weight_type=dataset.sc_weight_type,
        sc_static_over_time=dataset.sc_static_over_time,
        laplacian_kind=dataset.laplacian_kind,
        data_alignment_ready=dataset.data_alignment_ready,
        formal_evaluation_ready=dataset.formal_evaluation_ready,
    )
    np.testing.assert_allclose(
        forecast_linear_extrapolation(changed_dataset, (0,))["Participant 001"],
        original,
    )


def test_final_visit_metrics_and_bootstrap_are_reproducible(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, visit_counts=(3, 3))
    dataset = load_multi_visit_manifest(manifest).dataset
    forecasts = {
        label: dataset.trajectories[index][-1] + 0.1
        for index, label in enumerate(dataset.subject_ids)
    }

    metrics = evaluate_final_visit_forecasts(dataset, (0, 1), forecasts)

    assert metrics["pooled_rmse"] == pytest.approx(0.1)
    assert metrics["pooled_mae"] == pytest.approx(0.1)
    first = tuple(metrics["residuals_by_subject"].values())
    second = tuple(np.zeros_like(item) for item in first)
    a = paired_participant_bootstrap(first, second, n_resamples=100, seed=42)
    b = paired_participant_bootstrap(first, second, n_resamples=100, seed=42)
    assert a == b
    assert a["rmse"]["difference_first_minus_second"] > 0.0


def test_private_output_permissions_overwrite_and_freeze_protocol(tmp_path: Path) -> None:
    output = prepare_private_multi_visit_output_dir(tmp_path / "private_output")
    assert oct(output.stat().st_mode & 0o777) == "0o700"
    artifact = write_private_multi_visit_json(
        output / "complete.json",
        {"completion_status": "complete", "value": 1},
    )
    assert oct(artifact.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(SchemaError, match="would be overwritten"):
        write_private_multi_visit_json(artifact, {"completion_status": "complete", "value": 2})

    freeze = {
        "schema_version": "tau_sr_multi_visit_analysis_freeze_v1",
        "protocol_hash": compute_protocol_hash(),
    }
    verify_analysis_freeze_protocol(freeze)
    freeze["protocol_hash"] = "changed"
    with pytest.raises(SchemaError, match="protocol hash"):
        verify_analysis_freeze_protocol(freeze)


def test_synthetic_rehearsal_runs_and_resumes_in_private_temp_dir(tmp_path: Path) -> None:
    output = tmp_path / "rehearsal"

    first = run_synthetic_multi_visit_rehearsal(output, bootstrap_resamples=20)
    second = run_synthetic_multi_visit_rehearsal(output, bootstrap_resamples=20)

    assert first == second
    assert first["completion_status"] == "complete"
    assert first["synthetic_only"] is True
    assert first["configuration"]["n_subjects"] == 20
    assert first["fold_completion"] == {"completed": 5, "expected": 5}
    assert set(first["configuration"]["models"]) == {
        "persistence",
        "linear_extrapolation",
        "diffusion",
        "linear",
        "quadratic",
        "neural",
        "symbolic",
    }
