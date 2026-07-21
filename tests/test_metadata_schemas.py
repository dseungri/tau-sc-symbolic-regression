from __future__ import annotations

import pandas as pd

from tau_sr.metadata.schemas import validate_manifest


def test_manifest_rejects_duplicate_subject_visit_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "subject_id": "sub-syn001",
                "visit_id": "visit-01",
                "modality": "tau_pet",
                "scan_date": "2026-01-01",
                "time_from_baseline_years": 0.0,
                "tau_file": "tau.csv",
                "count_matrix_file": "count.csv",
                "fa_matrix_file": "fa.csv",
                "length_matrix_file": "length.csv",
                "roi_label_file": "rois.csv",
            },
            {
                "subject_id": "sub-syn001",
                "visit_id": "visit-01",
                "modality": "tau_pet",
                "scan_date": "2026-01-01",
                "time_from_baseline_years": 0.0,
                "tau_file": "tau.csv",
                "count_matrix_file": "count.csv",
                "fa_matrix_file": "fa.csv",
                "length_matrix_file": "length.csv",
                "roi_label_file": "rois.csv",
            },
        ]
    )
    report = validate_manifest(frame)
    assert not report.ok
    assert any(message.code == "duplicate_subject_visit" for message in report.messages)
