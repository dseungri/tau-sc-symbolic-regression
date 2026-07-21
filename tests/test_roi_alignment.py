from __future__ import annotations

import pytest

from tau_sr.exceptions import ValidationError
from tau_sr.qc.roi_alignment import (
    build_reorder_map,
    compare_roi_sets,
    round_trip_reorder,
    validate_roi_labels,
)


def test_roi_mapping_detects_duplicates() -> None:
    report = validate_roi_labels(["A", "a", "B"], expected_count=3)
    assert not report.ok


def test_roi_mapping_detects_order_differences() -> None:
    report = compare_roi_sets(["A", "B"], ["B", "A"])
    assert report.ok
    assert any(message.code == "roi_order_difference" for message in report.messages)


def test_roi_reorder_mapping_round_trips() -> None:
    source = ["A", "B", "C"]
    target = ["C", "A", "B"]
    mapping = build_reorder_map(source, target)
    assert mapping == [2, 0, 1]
    assert round_trip_reorder(source, mapping)


def test_roi_duplicate_reorder_rejected() -> None:
    with pytest.raises(ValidationError):
        build_reorder_map(["A", "a"], ["A"])
