from __future__ import annotations

import pandas as pd

from tau_sr.intake.identifier_flags import scan_file
from tau_sr.intake.inventory import inventory_directory


def test_inventory_handles_supported_files(tmp_path) -> None:
    table = tmp_path / "sub-syn001_tau.csv"
    pd.DataFrame({"tau": [1.0]}).to_csv(table, index=False)
    records = inventory_directory(tmp_path)
    assert records[0]["extension"] == ".csv"
    assert records[0]["suspected_modality"] == "tau_table"
    assert records[0]["suspected_subject"] == "sub-syn001"


def test_inventory_does_not_load_full_imaging_data(tmp_path) -> None:
    image = tmp_path / "synthetic.nii.gz"
    image.write_bytes(b"not a real nifti header")
    records = inventory_directory(tmp_path, hash_files=False)
    assert "nibabel" in records[0]["matrix_shape"] or "unavailable" in records[0]["matrix_shape"]


def test_identifier_flagger_reports_field_names_without_values(tmp_path) -> None:
    table = tmp_path / "safe.csv"
    table.write_text(
        "subject_id,mrn,email\nsub-syn001,secret@example.test,value\n", encoding="utf-8"
    )
    flags = scan_file(table)
    assert {flag["field_name"] for flag in flags} == {"mrn", "email"}
    assert all("secret" not in str(flag) for flag in flags)
