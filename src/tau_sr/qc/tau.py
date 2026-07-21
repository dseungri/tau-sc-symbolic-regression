"""Initial table-level tau-PET QC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tau_sr.constants import EXPECTED_ROI_COUNT, UNSET
from tau_sr.qc.common import QCReport


def validate_tau_table(
    table: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    visit_col: str = "visit_id",
    roi_col: str = "roi_name",
    value_col: str = "tau_value",
    time_col: str = "time_from_baseline_years",
    date_col: str = "scan_date",
    expected_roi_count: int = EXPECTED_ROI_COUNT,
    extreme_value_threshold: float | str = UNSET,
    extreme_annualized_change_threshold: float | str = UNSET,
) -> QCReport:
    """Validate tau table structure and numeric sanity without excluding subjects."""

    report = QCReport("tau_table")
    required = [subject_col, visit_col, roi_col, value_col, time_col, date_col]
    missing = [column for column in required if column not in table.columns]
    if missing:
        report.add_error("missing_columns", ", ".join(missing))
        return report

    values = pd.to_numeric(table[value_col], errors="coerce")
    if values.isna().any():
        report.add_error(
            "missing_or_nonnumeric_tau", "tau values contain missing or nonnumeric entries"
        )
    if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
        report.add_error("nonfinite_tau", "tau values contain positive or negative infinity")
    if (values < 0).any():
        report.add_warning("negative_tau", "negative tau values detected")

    if table.duplicated([subject_col, visit_col, roi_col]).any():
        report.add_error("duplicate_subject_visit_roi", "duplicate subject/visit/ROI rows detected")

    roi_counts = table.groupby([subject_col, visit_col])[roi_col].nunique()
    if not roi_counts.empty and (roi_counts != expected_roi_count).any():
        report.add_error(
            "unexpected_roi_count", "one or more subject/visit vectors have wrong ROI count"
        )

    vector_values = table.assign(_tau_value=values).pivot_table(
        index=[subject_col, visit_col],
        columns=roi_col,
        values="_tau_value",
        aggfunc="first",
    )
    for index, row in vector_values.iterrows():
        numeric = row.dropna().to_numpy(dtype=float)
        if len(numeric) == 0:
            continue
        if np.allclose(numeric, numeric[0]):
            report.add_warning(
                "constant_regional_vector", "constant regional vector detected", str(index)
            )
        if np.allclose(numeric, 0.0):
            report.add_warning("all_zero_vector", "all-zero regional vector detected", str(index))

    dates = pd.to_datetime(
        table[[subject_col, visit_col, date_col]].drop_duplicates()[date_col], errors="coerce"
    )
    if dates.isna().any():
        report.add_error("invalid_visit_dates", "scan_date contains invalid dates")
    follow_up = pd.to_numeric(table[time_col], errors="coerce")
    if follow_up.isna().any() or (follow_up < 0).any():
        report.add_error("invalid_follow_up", "follow-up times must be nonnegative")

    visits = table[[subject_col, visit_col, time_col]].drop_duplicates()
    for subject_id, group in visits.groupby(subject_col):
        ordered = group.sort_values(time_col)
        diffs = pd.to_numeric(ordered[time_col], errors="coerce").diff().dropna()
        if (diffs <= 0).any():
            report.add_warning(
                "nonpositive_follow_up_interval", "nonpositive follow-up interval", str(subject_id)
            )

    if extreme_value_threshold != UNSET and (values.abs() > float(extreme_value_threshold)).any():
        report.add_warning("extreme_tau_value", "configured extreme tau threshold exceeded")
    if extreme_annualized_change_threshold != UNSET:
        threshold = float(extreme_annualized_change_threshold)
        for subject_id, group in table.groupby(subject_col):
            pivot = group.pivot_table(
                index=visit_col, columns=roi_col, values=value_col, aggfunc="first"
            )
            times = group.drop_duplicates(visit_col).set_index(visit_col)[time_col]
            pivot = pivot.loc[times.sort_values().index]
            sorted_times = pd.to_numeric(times.loc[pivot.index], errors="coerce").to_numpy(
                dtype=float
            )
            if len(sorted_times) > 1:
                deltas = np.diff(pivot.to_numpy(dtype=float), axis=0)
                time_deltas = np.diff(sorted_times)
                if np.any(time_deltas <= 0):
                    continue
                annualized = np.abs(deltas / time_deltas[:, None])
                if np.nanmax(annualized) > threshold:
                    report.add_warning(
                        "extreme_annualized_change",
                        "configured annualized change threshold exceeded",
                        str(subject_id),
                    )

    report.metrics["row_count"] = int(len(table))
    report.metrics["subject_count"] = int(table[subject_col].nunique())
    return report
