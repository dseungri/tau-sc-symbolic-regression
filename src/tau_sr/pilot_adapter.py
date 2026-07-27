"""Lean privacy-preserving loader for the two-subject pilot data."""

from __future__ import annotations

import datetime as dt
import math
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from tau_sr.graphs import LaplacianKind, build_graph_laplacian
from tau_sr.schemas import SchemaError

PILOT_N_REGIONS = 160
TAU_MEASUREMENT = "SUVR"
TAU_REFERENCE_PENDING = "pending_exact_confirmation"
SC_WEIGHT_TYPE = "normalized_fiber_count"
DEFAULT_LAPLACIAN_KIND: LaplacianKind = "symmetric_normalized"
ROI_LOOKUP_REQUIRED = "An ordered 160-region ROI lookup is required before model training."

_SUBJECT_COLUMN = "PTID"
_SCAN_COLUMN = "SCAN"
_DATE_COLUMN = "EXAMDATE"
_NODE_RE = re.compile(r"^Node ([1-9][0-9]*)$")
_XLSX_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

__all__ = (
    "DEFAULT_LAPLACIAN_KIND",
    "PILOT_N_REGIONS",
    "ROI_LOOKUP_REQUIRED",
    "SC_WEIGHT_TYPE",
    "TAU_MEASUREMENT",
    "TAU_REFERENCE_PENDING",
    "LongitudinalTrajectoryDataset",
    "inspect_pilot_inputs",
    "load_pilot_dataset",
)


@dataclass(frozen=True)
class LongitudinalTrajectoryDataset:
    """Minimal model-input contract for longitudinal reaction-diffusion training."""

    subject_ids: tuple[str, ...]
    roi_labels: tuple[str, ...]
    adjacencies: np.ndarray
    laplacians: np.ndarray
    times_by_subject: tuple[np.ndarray, ...]
    trajectories: tuple[np.ndarray, ...]
    tau_measurement: str = TAU_MEASUREMENT
    tau_reference: str = TAU_REFERENCE_PENDING
    sc_weight_type: str = SC_WEIGHT_TYPE
    laplacian_kind: str = DEFAULT_LAPLACIAN_KIND

    def __post_init__(self) -> None:
        subject_ids = _string_tuple(self.subject_ids, "subject_ids")
        roi_labels = _string_tuple(self.roi_labels, "roi_labels")
        if len(set(subject_ids)) != len(subject_ids):
            raise SchemaError("subject_ids must be unique.")
        if len(set(roi_labels)) != len(roi_labels):
            raise SchemaError("roi_labels must be unique.")
        n_subjects = len(subject_ids)
        n_regions = len(roi_labels)
        adjacencies = _readonly_array3(
            self.adjacencies,
            "adjacencies",
            (n_subjects, n_regions, n_regions),
        )
        laplacians = _readonly_array3(
            self.laplacians,
            "laplacians",
            (n_subjects, n_regions, n_regions),
        )
        times = _validate_times(self.times_by_subject, n_subjects)
        trajectories = _validate_trajectories(self.trajectories, times, n_regions)

        object.__setattr__(self, "subject_ids", subject_ids)
        object.__setattr__(self, "roi_labels", roi_labels)
        object.__setattr__(self, "adjacencies", adjacencies)
        object.__setattr__(self, "laplacians", laplacians)
        object.__setattr__(self, "times_by_subject", times)
        object.__setattr__(self, "trajectories", trajectories)
        object.__setattr__(
            self,
            "tau_measurement",
            _nonempty(self.tau_measurement, "tau_measurement"),
        )
        object.__setattr__(self, "tau_reference", _nonempty(self.tau_reference, "tau_reference"))
        object.__setattr__(self, "sc_weight_type", _nonempty(self.sc_weight_type, "sc_weight_type"))
        object.__setattr__(self, "laplacian_kind", _nonempty(self.laplacian_kind, "laplacian_kind"))

    @property
    def baseline_states(self) -> np.ndarray:
        baseline = np.stack([trajectory[0] for trajectory in self.trajectories])
        baseline.setflags(write=False)
        return baseline

    def __repr__(self) -> str:
        return (
            "LongitudinalTrajectoryDataset("
            f"n_subjects={len(self.subject_ids)}, "
            f"n_regions={len(self.roi_labels)}, "
            f"tau_measurement={self.tau_measurement!r}, "
            f"laplacian_kind={self.laplacian_kind!r})"
        )


@dataclass(frozen=True)
class _TauSubject:
    source_id: str
    neutral_id: str
    times: np.ndarray
    trajectory: np.ndarray


@dataclass(frozen=True)
class _TauData:
    subjects: tuple[_TauSubject, ...]
    missing_count: int
    finite_count: int
    n_regions: int


def load_pilot_dataset(
    tau_workbook_path: str | Path,
    connectome_paths: Sequence[str | Path],
    *,
    roi_labels: Sequence[str] | None,
    tau_reference: str = TAU_REFERENCE_PENDING,
    laplacian_kind: LaplacianKind = DEFAULT_LAPLACIAN_KIND,
) -> LongitudinalTrajectoryDataset:
    """Load pilot files into the generic training contract.

    The ordered 160-label lookup is mandatory; without it, model training stays
    blocked rather than silently guessing atlas alignment.
    """

    labels = _validate_roi_labels(roi_labels)
    tau = _load_tau_workbook(tau_workbook_path, n_regions=PILOT_N_REGIONS)
    connectomes = _load_connectomes(
        connectome_paths,
        tau.subjects,
        laplacian_kind=laplacian_kind,
    )
    return LongitudinalTrajectoryDataset(
        subject_ids=tuple(subject.neutral_id for subject in tau.subjects),
        roi_labels=labels,
        adjacencies=np.stack([adjacency for adjacency, _laplacian, _stats in connectomes]),
        laplacians=np.stack([laplacian for _adjacency, laplacian, _stats in connectomes]),
        times_by_subject=tuple(subject.times for subject in tau.subjects),
        trajectories=tuple(subject.trajectory for subject in tau.subjects),
        tau_measurement=TAU_MEASUREMENT,
        tau_reference=tau_reference,
        sc_weight_type=SC_WEIGHT_TYPE,
        laplacian_kind=laplacian_kind,
    )


def inspect_pilot_inputs(
    tau_workbook_path: str | Path,
    connectome_paths: Sequence[str | Path],
) -> dict[str, object]:
    """Return a privacy-safe structural summary without constructing a dataset."""

    tau = _load_tau_workbook(tau_workbook_path, n_regions=PILOT_N_REGIONS)
    connectomes = _load_connectomes(
        connectome_paths,
        tau.subjects,
        laplacian_kind=DEFAULT_LAPLACIAN_KIND,
    )
    degrees = [stats["minimum_degree"] for _adjacency, _laplacian, stats in connectomes]
    return {
        "subject_count": len(tau.subjects),
        "visit_counts": {
            subject.neutral_id: int(subject.times.shape[0]) for subject in tau.subjects
        },
        "relative_times_years": {
            subject.neutral_id: tuple(float(value) for value in subject.times)
            for subject in tau.subjects
        },
        "region_count": tau.n_regions,
        "missing_tau_values": tau.missing_count,
        "connectome_shapes": tuple((PILOT_N_REGIONS, PILOT_N_REGIONS) for _ in connectomes),
        "symmetrization_valid": all(
            bool(stats["symmetrization_valid"])
            for _adjacency, _laplacian, stats in connectomes
        ),
        "minimum_degree": float(min(degrees)),
        "roi_lookup_required": True,
    }


def _load_tau_workbook(path: str | Path, *, n_regions: int) -> _TauData:
    headers, rows = _read_xlsx_rows(Path(path))
    if not rows:
        raise SchemaError("tau workbook must contain visit rows.")
    for column in (_SUBJECT_COLUMN, _SCAN_COLUMN, _DATE_COLUMN):
        if column not in headers:
            raise SchemaError("tau workbook is missing required columns.")
    node_columns = _node_columns(headers, n_regions)
    seen_visits: set[tuple[str, str]] = set()
    dates_by_subject: dict[str, set[dt.date]] = {}
    visits_by_subject: dict[str, list[tuple[dt.date, np.ndarray]]] = {}
    missing_count = 0
    finite_count = 0

    for row in rows:
        source_id = _cell_text(row.get(_SUBJECT_COLUMN), "subject identifier")
        scan_id = _cell_text(row.get(_SCAN_COLUMN), "scan identifier")
        visit_key = (source_id, scan_id)
        if visit_key in seen_visits:
            raise SchemaError("tau workbook contains duplicate subject/visit rows.")
        seen_visits.add(visit_key)
        exam_date = _parse_date(row.get(_DATE_COLUMN))
        used_dates = dates_by_subject.setdefault(source_id, set())
        if exam_date in used_dates:
            raise SchemaError("tau workbook contains duplicate visit dates.")
        used_dates.add(exam_date)
        values = np.asarray([_float_cell(row.get(column)) for column in node_columns], dtype=float)
        finite_mask = np.isfinite(values)
        finite_count += int(np.count_nonzero(finite_mask))
        missing_count += int(values.size - np.count_nonzero(finite_mask))
        visits_by_subject.setdefault(source_id, []).append((exam_date, values))

    subjects: list[_TauSubject] = []
    for index, source_id in enumerate(sorted(visits_by_subject), start=1):
        visits = sorted(visits_by_subject[source_id], key=lambda item: item[0])
        if len(visits) < 2:
            raise SchemaError("each pilot subject must have at least two visits.")
        baseline_date = visits[0][0]
        times = np.asarray(
            [(visit_date - baseline_date).days / 365.25 for visit_date, _values in visits],
            dtype=float,
        )
        subjects.append(
            _TauSubject(
                source_id=source_id,
                neutral_id=f"Pilot Subject {index}",
                times=times,
                trajectory=np.stack([values for _date, values in visits]),
            )
        )
    return _TauData(
        subjects=tuple(subjects),
        missing_count=missing_count,
        finite_count=finite_count,
        n_regions=n_regions,
    )


def _load_connectomes(
    paths: Sequence[str | Path],
    subjects: Sequence[_TauSubject],
    *,
    laplacian_kind: LaplacianKind,
) -> tuple[tuple[np.ndarray, np.ndarray, dict[str, float | bool]], ...]:
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise SchemaError("connectome_paths must be a sequence.")
    path_objects = tuple(Path(path) for path in paths)
    if len(path_objects) != len(subjects):
        raise SchemaError("Each pilot subject must match exactly one connectome file.")
    used: set[Path] = set()
    output: list[tuple[np.ndarray, np.ndarray, dict[str, float | bool]]] = []
    for subject in subjects:
        matches = [
            path
            for path in path_objects
            if _filename_has_subject_token(path.name, subject.source_id)
        ]
        if len(matches) != 1 or matches[0] in used:
            raise SchemaError("Each pilot subject must match exactly one connectome file.")
        used.add(matches[0])
        output.append(_load_connectome(matches[0], laplacian_kind=laplacian_kind))
    return tuple(output)


def _filename_has_subject_token(filename: str, source_id: str) -> bool:
    token_pattern = rf"(?<![A-Za-z0-9]){re.escape(source_id)}(?![A-Za-z0-9])"
    return re.search(token_pattern, filename) is not None


def _load_connectome(
    path: Path,
    *,
    laplacian_kind: LaplacianKind,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool]]:
    try:
        raw = np.loadtxt(path, delimiter=",")
    except Exception as exc:
        raise SchemaError("connectome matrix must be a numeric CSV.") from exc
    if raw.shape != (PILOT_N_REGIONS, PILOT_N_REGIONS):
        raise SchemaError("connectome matrix has unexpected shape.")
    if raw.dtype.kind in {"b", "O", "S", "U", "c"} or not np.issubdtype(raw.dtype, np.number):
        raise SchemaError("connectome matrix must be real numeric.")
    raw = raw.astype(float, copy=True)
    if not np.isfinite(raw).all():
        raise SchemaError("connectome matrix must contain finite values.")
    if (raw < 0.0).any():
        raise SchemaError("connectome matrix must be nonnegative.")
    if not np.allclose(np.diag(raw), 0.0, rtol=0.0, atol=1e-12):
        raise SchemaError("connectome diagonal must be zero.")

    adjacency = 0.5 * (raw + raw.T)
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    if not np.array_equal(adjacency, adjacency.T):
        raise SchemaError("symmetrized connectome must be symmetric.")
    if not (np.isfinite(degree).all() and np.all(degree > 0.0)):
        raise SchemaError("symmetrized connectome must have positive finite degree.")
    laplacian = build_graph_laplacian(adjacency, kind=laplacian_kind)
    return (
        adjacency.astype(float, copy=True),
        laplacian.astype(float, copy=True),
        {
            "symmetrization_valid": True,
            "minimum_degree": float(np.min(degree)),
        },
    )


def _validate_roi_labels(labels: Sequence[str] | None) -> tuple[str, ...]:
    if labels is None:
        raise SchemaError(ROI_LOOKUP_REQUIRED)
    checked = _string_tuple(labels, "roi_labels")
    if len(checked) != PILOT_N_REGIONS or len(set(checked)) != PILOT_N_REGIONS:
        raise SchemaError(ROI_LOOKUP_REQUIRED)
    return checked


def _node_columns(headers: Sequence[str], n_regions: int) -> tuple[str, ...]:
    found: list[tuple[int, str]] = []
    for header in headers:
        match = _NODE_RE.match(header)
        if match:
            found.append((int(match.group(1)), header))
    if [index for index, _name in found] != list(range(1, n_regions + 1)):
        raise SchemaError("tau workbook must contain consecutive Node 1 through Node 160 columns.")
    return tuple(name for _index, name in found)


def _read_xlsx_rows(path: Path) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    try:
        with zipfile.ZipFile(path) as workbook:
            shared_strings = _shared_strings(workbook)
            sheet_path = _first_sheet_path(workbook)
            root = ET.fromstring(workbook.read(sheet_path))
            raw_rows = [
                _xlsx_row_values(row, shared_strings)
                for row in root.findall(".//main:sheetData/main:row", _XLSX_NS)
            ]
    except SchemaError:
        raise
    except Exception as exc:
        raise SchemaError("tau workbook could not be read.") from exc
    if not raw_rows:
        raise SchemaError("tau workbook must contain a header row.")
    headers = tuple(str(value).strip() for value in raw_rows[0] if value is not None)
    rows: list[dict[str, Any]] = []
    for row in raw_rows[1:]:
        padded = row + [None] * max(0, len(headers) - len(row))
        rows.append({header: padded[index] for index, header in enumerate(headers)})
    return headers, tuple(rows)


def _shared_strings(workbook: zipfile.ZipFile) -> tuple[str, ...]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return ()
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    return tuple(
        "".join(text.text or "" for text in item.findall(".//main:t", _XLSX_NS))
        for item in root.findall("main:si", _XLSX_NS)
    )


def _first_sheet_path(workbook: zipfile.ZipFile) -> str:
    if "xl/worksheets/sheet1.xml" in workbook.namelist():
        return "xl/worksheets/sheet1.xml"
    names = sorted(name for name in workbook.namelist() if name.startswith("xl/worksheets/"))
    if not names:
        raise SchemaError("tau workbook must contain a worksheet.")
    return names[0]


def _xlsx_row_values(row: ET.Element, shared_strings: Sequence[str]) -> list[Any]:
    values_by_index: dict[int, Any] = {}
    for cell in row.findall("main:c", _XLSX_NS):
        values_by_index[_column_index(cell.attrib.get("r", ""))] = _cell_value(cell, shared_strings)
    if not values_by_index:
        return []
    return [values_by_index.get(index) for index in range(max(values_by_index) + 1)]


def _cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> Any:
    if cell.attrib.get("t") == "inlineStr":
        inline = cell.find("main:is", _XLSX_NS)
        if inline is None:
            return None
        return "".join(text.text or "" for text in inline.findall(".//main:t", _XLSX_NS))
    value = cell.find("main:v", _XLSX_NS)
    if value is None:
        return None
    raw = value.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(raw)]
    return raw


def _column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    if not letters:
        raise SchemaError("workbook cell reference is malformed.")
    index = 0
    for character in letters.upper():
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def _parse_date(value: object) -> dt.date:
    if value is None:
        raise SchemaError("exam dates must be parseable.")
    text = str(value).strip()
    if not text:
        raise SchemaError("exam dates must be parseable.")
    if text.isdigit() and len(text) == 8:
        try:
            return dt.datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            pass
    try:
        serial = float(text)
    except ValueError:
        serial = math.nan
    if math.isfinite(serial):
        return (dt.datetime(1899, 12, 30) + dt.timedelta(days=serial)).date()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise SchemaError("exam dates must be parseable.")


def _float_cell(value: object) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError("tau workbook node values must be numeric.") from exc
    if not math.isfinite(number):
        return math.nan
    return number


def _cell_text(value: object, field_name: str) -> str:
    if value is None:
        raise SchemaError(f"tau workbook is missing {field_name}.")
    text = str(value).strip()
    if not text:
        raise SchemaError(f"tau workbook is missing {field_name}.")
    return text


def _validate_times(values: object, n_subjects: int) -> tuple[np.ndarray, ...]:
    if isinstance(values, np.ndarray):
        raw = tuple(values[index] for index in range(values.shape[0]))
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SchemaError("times_by_subject must be a sequence.")
    else:
        raw = tuple(values)
    if len(raw) != n_subjects:
        raise SchemaError("times_by_subject must match subject_ids.")
    output: list[np.ndarray] = []
    for times in raw:
        array = _readonly_array(np.asarray(times), 1, None, "times_by_subject")
        if not math.isclose(float(array[0]), 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise SchemaError("times_by_subject must begin at zero.")
        if array.size > 1 and not np.all(np.diff(array) > 0.0):
            raise SchemaError("times_by_subject must be strictly increasing.")
        output.append(array)
    return tuple(output)


def _validate_trajectories(
    values: object,
    times: tuple[np.ndarray, ...],
    n_regions: int,
) -> tuple[np.ndarray, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SchemaError("trajectories must be a sequence.")
    raw = tuple(values)
    if len(raw) != len(times):
        raise SchemaError("trajectories must match subject_ids.")
    return tuple(
        _readonly_array(np.asarray(trajectory), 2, (times[index].size, n_regions), "trajectories")
        for index, trajectory in enumerate(raw)
    )


def _readonly_array3(value: object, field_name: str, shape: tuple[int, int, int]) -> np.ndarray:
    return _readonly_array(np.asarray(value), 3, shape, field_name)


def _readonly_array(
    array: np.ndarray,
    ndim: int,
    shape: tuple[int, ...] | None,
    field_name: str,
) -> np.ndarray:
    if array.ndim != ndim:
        raise SchemaError(f"{field_name} has unexpected rank.")
    if shape is not None and array.shape != shape:
        raise SchemaError(f"{field_name} has unexpected shape.")
    if array.dtype.kind in {"b", "O", "S", "U", "c"} or not np.issubdtype(array.dtype, np.number):
        raise SchemaError(f"{field_name} must be real numeric.")
    if not np.isfinite(array).all():
        raise SchemaError(f"{field_name} must contain finite values.")
    output = array.astype(float, copy=True)
    output.setflags(write=False)
    return output


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SchemaError(f"{field_name} must be a sequence of strings.")
    output = tuple(_nonempty(value, field_name) for value in values)
    if not output:
        raise SchemaError(f"{field_name} must be nonempty.")
    return output


def _nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{field_name} must contain strings.")
    text = value.strip()
    if not text:
        raise SchemaError(f"{field_name} must contain nonempty strings.")
    return text
