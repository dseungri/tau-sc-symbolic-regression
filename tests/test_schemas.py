from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tau_sr.schemas import (
    ConnectomeBundle,
    SchemaError,
    apply_roi_reorder_to_matrix,
    apply_roi_reorder_to_vector,
    build_roi_reorder_map,
    connectome_bundle_from_mapping,
    invert_roi_reorder_map,
    validate_connectome_bundle,
    validate_tau_table,
)


def test_valid_tau_table_passes() -> None:
    table = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s1", "s1"],
            "visit_id": ["v1", "v1", "v2", "v2"],
            "time_years": [0.0, 0.0, 1.1, 1.1],
            "roi_index": [0, 1, 0, 1],
            "roi_name": ["a", "b", "a", "b"],
            "tau_value": [1.0, 1.2, 1.1, 1.3],
        }
    )

    assert validate_tau_table(table, expected_n_rois=2, require_complete_visits=True) is table


def test_tau_table_requires_canonical_columns() -> None:
    table = pd.DataFrame({"subject_id": ["s1"]})

    with pytest.raises(SchemaError, match="missing required column"):
        validate_tau_table(table, expected_n_rois=2)


def test_tau_table_rejects_duplicate_subject_visit_roi_rows() -> None:
    table = pd.DataFrame(
        {
            "subject_id": ["s1", "s1"],
            "visit_id": ["v1", "v1"],
            "time_years": [0.0, 0.0],
            "roi_index": [0, 0],
            "roi_name": ["a", "a"],
            "tau_value": [1.0, 1.1],
        }
    )

    with pytest.raises(SchemaError, match="duplicate"):
        validate_tau_table(table, expected_n_rois=2)


def test_tau_table_can_require_complete_visits() -> None:
    table = pd.DataFrame(
        {
            "subject_id": ["s1"],
            "visit_id": ["v1"],
            "time_years": [0.0],
            "roi_index": [0],
            "roi_name": ["a"],
            "tau_value": [1.0],
        }
    )

    with pytest.raises(SchemaError, match="incomplete"):
        validate_tau_table(table, expected_n_rois=2, require_complete_visits=True)


def test_tau_table_requires_expected_roi_count_for_complete_visits() -> None:
    table = pd.DataFrame(
        {
            "subject_id": ["s1"],
            "visit_id": ["v1"],
            "time_years": [0.0],
            "roi_index": [0],
            "roi_name": ["a"],
            "tau_value": [1.0],
        }
    )

    with pytest.raises(SchemaError, match="expected_n_rois"):
        validate_tau_table(table, expected_n_rois=None, require_complete_visits=True)


def test_connectome_bundle_from_mapping_and_validation_pass() -> None:
    matrix = np.array([[0.0, 2.0], [2.0, 0.0]])
    bundle = connectome_bundle_from_mapping(
        {
            "subject_id": "s1",
            "roi_labels": np.array(["a", "b"]),
            "count": matrix,
            "fa": matrix * 0.3,
            "length": matrix + np.eye(2),
        }
    )

    checked = validate_connectome_bundle(bundle, expected_n_rois=2)

    assert isinstance(checked, ConnectomeBundle)
    assert checked.subject_id == "s1"
    assert checked.n_rois == 2


def test_connectome_bundle_rejects_non_mapping_input() -> None:
    with pytest.raises(SchemaError, match="mapping"):
        validate_connectome_bundle(["not", "a", "mapping"], expected_n_rois=2)  # type: ignore[arg-type]


def test_connectome_bundle_rejects_non_string_subject_id() -> None:
    matrix = np.zeros((2, 2))

    with pytest.raises(SchemaError, match="subject_id"):
        connectome_bundle_from_mapping(
            {
                "subject_id": 1,
                "roi_labels": ["a", "b"],
                "count": matrix,
                "fa": matrix,
                "length": matrix,
            }
        )


def test_connectome_bundle_rejects_non_string_roi_labels() -> None:
    matrix = np.zeros((2, 2))

    with pytest.raises(SchemaError, match="roi_labels"):
        connectome_bundle_from_mapping(
            {
                "subject_id": "s1",
                "roi_labels": ["a", 1],
                "count": matrix,
                "fa": matrix,
                "length": matrix,
            }
        )


def test_connectome_bundle_rejects_nonnumeric_matrix_values() -> None:
    matrix = np.zeros((2, 2))

    with pytest.raises(SchemaError, match="count matrix"):
        connectome_bundle_from_mapping(
            {
                "subject_id": "s1",
                "roi_labels": ["a", "b"],
                "count": [["bad", "bad"], ["bad", "bad"]],
                "fa": matrix,
                "length": matrix,
            }
        )


def test_connectome_bundle_wraps_attribute_errors_from_matrix_conversion() -> None:
    class BrokenArray:
        def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
            raise AttributeError("broken external array")

    matrix = np.zeros((2, 2))

    with pytest.raises(SchemaError, match="count matrix"):
        connectome_bundle_from_mapping(
            {
                "subject_id": "s1",
                "roi_labels": ["a", "b"],
                "count": BrokenArray(),
                "fa": matrix,
                "length": matrix,
            }
        )


def test_connectome_bundle_rejects_nonfinite_matrix_values() -> None:
    matrix = np.zeros((2, 2))
    count = np.array([[0.0, np.nan], [1.0, 0.0]])

    with pytest.raises(SchemaError, match="nonfinite"):
        validate_connectome_bundle(
            {
                "subject_id": "s1",
                "roi_labels": ["a", "b"],
                "count": count,
                "fa": matrix,
                "length": matrix,
            },
            expected_n_rois=2,
        )


def test_connectome_bundle_copies_and_exposes_readonly_matrices() -> None:
    count = np.array([[0.0, 2.0], [2.0, 0.0]])
    fa = count * 0.3
    length = count + np.eye(2)

    bundle = ConnectomeBundle(
        subject_id="s1",
        roi_labels=("a", "b"),
        count=count,
        fa=fa,
        length=length,
    )

    count[0, 1] = 99.0
    fa[0, 1] = 99.0
    length[0, 1] = 99.0

    assert bundle.count[0, 1] == 2.0
    assert bundle.fa[0, 1] == 0.6
    assert bundle.length[0, 1] == 2.0
    assert not np.shares_memory(bundle.count, count)
    assert not np.shares_memory(bundle.fa, fa)
    assert not np.shares_memory(bundle.length, length)

    for matrix in (bundle.count, bundle.fa, bundle.length, bundle.matrix("count")):
        with pytest.raises(ValueError, match="read-only"):
            matrix[0, 0] = 1.0


def test_connectome_bundle_rejects_shape_mismatch() -> None:
    matrix = np.zeros((2, 2))
    bundle = ConnectomeBundle(
        subject_id="s1",
        roi_labels=("a", "b"),
        count=matrix,
        fa=np.zeros((2, 3)),
        length=matrix,
    )

    with pytest.raises(SchemaError, match="shape"):
        validate_connectome_bundle(bundle, expected_n_rois=2)


def test_connectome_bundle_rejects_duplicate_normalized_labels() -> None:
    matrix = np.zeros((2, 2))
    bundle = ConnectomeBundle(
        subject_id="s1",
        roi_labels=("Left-Hippocampus", "left hippocampus"),
        count=matrix,
        fa=matrix,
        length=matrix,
    )

    with pytest.raises(SchemaError, match="duplicate"):
        validate_connectome_bundle(bundle, expected_n_rois=2)


def test_roi_reorder_map_round_trip_for_vector_and_matrix() -> None:
    source_labels = ["roi_a", "roi_b", "roi_c"]
    target_labels = ["ROI C", "ROI A", "ROI B"]
    reorder_map = build_roi_reorder_map(source_labels, target_labels)

    vector = np.array([10.0, 20.0, 30.0])
    matrix = np.arange(9.0).reshape(3, 3)

    reordered_vector = apply_roi_reorder_to_vector(vector, reorder_map)
    reordered_matrix = apply_roi_reorder_to_matrix(matrix, reorder_map)
    inverse = invert_roi_reorder_map(reorder_map)

    np.testing.assert_array_equal(reordered_vector, np.array([30.0, 10.0, 20.0]))
    np.testing.assert_array_equal(apply_roi_reorder_to_vector(reordered_vector, inverse), vector)
    np.testing.assert_array_equal(apply_roi_reorder_to_matrix(reordered_matrix, inverse), matrix)


def test_roi_reorder_rejects_non_integer_indices() -> None:
    with pytest.raises(SchemaError, match="integer"):
        apply_roi_reorder_to_vector([1.0, 2.0], [0.5, 1.0])

def test_tau_table_rejects_non_dataframe() -> None:
    with pytest.raises(SchemaError, match="DataFrame"):
        validate_tau_table([1, 2, 3], expected_n_rois=2)  # type: ignore[arg-type]
 
 
def test_tau_table_rejects_negative_time_years() -> None:
    table = pd.DataFrame(
        {
            "subject_id": ["s1"],
            "visit_id": ["v1"],
            "time_years": [-1.0],
            "roi_index": [0],
            "roi_name": ["a"],
            "tau_value": [1.0],
        }
    )
 
    with pytest.raises(SchemaError, match="nonnegative"):
        validate_tau_table(table, expected_n_rois=2)
 
 
def test_tau_table_rejects_negative_roi_index() -> None:
    table = pd.DataFrame(
        {
            "subject_id": ["s1"],
            "visit_id": ["v1"],
            "time_years": [0.0],
            "roi_index": [-1],
            "roi_name": ["a"],
            "tau_value": [1.0],
        }
    )
 
    with pytest.raises(SchemaError, match="nonnegative zero-based"):
        validate_tau_table(table, expected_n_rois=2)
 
 
def test_tau_table_rejects_non_integer_roi_index() -> None:
    table = pd.DataFrame(
        {
            "subject_id": ["s1"],
            "visit_id": ["v1"],
            "time_years": [0.0],
            "roi_index": [0.5],
            "roi_name": ["a"],
            "tau_value": [1.0],
        }
    )
 
    with pytest.raises(SchemaError, match="integer"):
        validate_tau_table(table, expected_n_rois=2)
 
 
def test_roi_reorder_rejects_repeated_index() -> None:
    with pytest.raises(SchemaError, match="permutation"):
        apply_roi_reorder_to_vector([1.0, 2.0, 3.0], [0, 0, 2])
 
 
def test_roi_reorder_rejects_out_of_range_index() -> None:
    with pytest.raises(SchemaError, match="permutation"):
        apply_roi_reorder_to_vector([1.0, 2.0, 3.0], [0, 1, 5])
