from __future__ import annotations

import numpy as np
import pandas as pd

from tau_sr.qc.tau import validate_tau_table
from tau_sr.synthetic.fixtures import synthetic_tau_table


def test_tau_qc_detects_missing_nonnumeric_values() -> None:
    table = synthetic_tau_table(subject_count=1, visits=2)
    table.loc[0, "tau_value"] = np.nan
    report = validate_tau_table(table)
    assert not report.ok
    assert any(message.code == "missing_or_nonnumeric_tau" for message in report.messages)


def test_tau_qc_detects_duplicate_roi_entries() -> None:
    table = synthetic_tau_table(subject_count=1, visits=2)
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    report = validate_tau_table(table)
    assert not report.ok
    assert any(message.code == "duplicate_subject_visit_roi" for message in report.messages)
