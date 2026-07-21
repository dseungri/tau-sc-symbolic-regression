from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DEFAULT_N_ROIS = 148

TAU_TABLE_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "visit_id",
    "time_years",
    "roi_index",
    "roi_name",
    "tau_value",
)

CONNECTOME_MATRIX_NAMES: tuple[str, ...] = ("count", "fa", "length")


class SchemaError(ValueError):
    """Incorrect input format"""


@dataclass(frozen=True)
class ConnectomeBundle:
    """Holds data so that it can't be accidentally modified"""

    subject_id: str
    roi_labels: tuple[str, ...]
    count: NDArray[np.float64]
    fa: NDArray[np.float64]
    length: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Copy matrices into read-only arrays despite the frozen dataclass wrapper."""

        object.__setattr__(self, "subject_id", _require_string(self.subject_id, "subject_id"))
        object.__setattr__(self, "roi_labels", _coerce_roi_labels(self.roi_labels))
        for name in CONNECTOME_MATRIX_NAMES:
            object.__setattr__(
                self,
                name,
                _readonly_float64_array(getattr(self, name), matrix_name=name),
            )

    @property
    def n_rois(self) -> int:
        """Number of ROI labels represented by this bundle."""

        return len(self.roi_labels)

    def matrix(self, name: str) -> NDArray[np.float64]:
        """Get right matrix by name or raise SchemaError"""

        if name not in CONNECTOME_MATRIX_NAMES:
            raise SchemaError(f"Unknown connectome matrix {name!r}.")
        return getattr(self, name)


def validate_tau_table(
    table: pd.DataFrame,
    *,
    expected_n_rois: int | None = DEFAULT_N_ROIS,
    require_complete_visits: bool = False,
) -> pd.DataFrame:
    """subject_id | visit_id | time_years | roi_index | roi_name | tau_value, throw SchemaError if invalid
    """

    if not isinstance(table, pd.DataFrame):
        raise SchemaError("Tau table must be a pandas DataFrame.")
    _require_columns(table, TAU_TABLE_COLUMNS, table_name="tau table")
    if table.empty:
        raise SchemaError("Tau table is empty.")

    _require_nonmissing(table, ("subject_id", "visit_id", "roi_name"), table_name="tau table")
    time_years = _finite_numeric_series(table["time_years"], "time_years")
    roi_index = _integer_series(table["roi_index"], "roi_index")
    tau_value = _finite_numeric_series(table["tau_value"], "tau_value")

    if (time_years < 0).any():
        raise SchemaError("time_years must be nonnegative.")
    if (roi_index < 0).any():
        raise SchemaError("roi_index must use nonnegative zero-based indices.")
    if not np.isfinite(tau_value).all():
        raise SchemaError("tau_value must be finite.")

    duplicate_mask = table.duplicated(["subject_id", "visit_id", "roi_index"], keep=False)
    if duplicate_mask.any():
        examples = table.loc[duplicate_mask, ["subject_id", "visit_id", "roi_index"]].head(5)
        raise SchemaError(
            "Tau table has duplicate subject_id/visit_id/roi_index rows: "
            f"{examples.to_dict('records')}"
        )

    if require_complete_visits and expected_n_rois is None:
        raise SchemaError("expected_n_rois is required when require_complete_visits is true.")

    if expected_n_rois is not None:
        if expected_n_rois <= 0:
            raise SchemaError("expected_n_rois must be positive when provided.")
        if roi_index.max() >= expected_n_rois:
            raise SchemaError(
                f"roi_index exceeds expected range 0..{expected_n_rois - 1}."
            )
        if require_complete_visits:
            roi_counts = table.groupby(["subject_id", "visit_id"])["roi_index"].nunique()
            incomplete = roi_counts[roi_counts != expected_n_rois]
            if not incomplete.empty:
                raise SchemaError(
                    "Tau table has incomplete subject visits for expected_n_rois="
                    f"{expected_n_rois}: {incomplete.head(5).to_dict()}"
                )

    return table


def connectome_bundle_from_mapping(mapping: Mapping[str, object]) -> ConnectomeBundle:
    """Convert raw dict-like input into standard ConnectomeBundle object"""

    if not isinstance(mapping, Mapping):
        raise SchemaError("Connectome bundle input must be a mapping.")

    required_keys = ("subject_id", "roi_labels", *CONNECTOME_MATRIX_NAMES)
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise SchemaError(f"Connectome bundle missing required key(s): {', '.join(missing)}")

    return ConnectomeBundle(
        subject_id=_require_string(mapping["subject_id"], "subject_id"),
        roi_labels=_coerce_roi_labels(mapping["roi_labels"]),
        count=_readonly_float64_array(mapping["count"], matrix_name="count"),
        fa=_readonly_float64_array(mapping["fa"], matrix_name="fa"),
        length=_readonly_float64_array(mapping["length"], matrix_name="length"),
    )


def validate_connectome_bundle(
    bundle: ConnectomeBundle | Mapping[str, object],
    *,
    expected_n_rois: int | None = DEFAULT_N_ROIS,
) -> ConnectomeBundle:
    """Check matrices and labels"""

    if not isinstance(bundle, ConnectomeBundle):
        bundle = connectome_bundle_from_mapping(bundle)

    subject_id = _require_string(bundle.subject_id, "subject_id")
    roi_labels = _coerce_roi_labels(bundle.roi_labels)

    if not subject_id.strip():
        raise SchemaError("Connectome bundle subject_id is empty.")
    if not roi_labels:
        raise SchemaError("Connectome bundle has no roi_labels.")
    if any(not label.strip() for label in roi_labels):
        raise SchemaError("Connectome bundle contains empty ROI labels.")

    if expected_n_rois is not None:
        if expected_n_rois <= 0:
            raise SchemaError("expected_n_rois must be positive when provided.")
        if len(roi_labels) != expected_n_rois:
            raise SchemaError(
                f"Connectome bundle has {len(roi_labels)} ROI labels; expected {expected_n_rois}."
            )

    duplicate_labels = find_duplicate_normalized_labels(roi_labels)
    if duplicate_labels:
        raise SchemaError(
            "Connectome bundle contains duplicate normalized ROI label(s): "
            f"{duplicate_labels[:5]}"
        )

    matrices = {
        name: _readonly_float64_array(bundle.matrix(name), matrix_name=name)
        for name in CONNECTOME_MATRIX_NAMES
    }
    expected_shape = (len(roi_labels), len(roi_labels))
    for name, matrix in matrices.items():
        if matrix.shape != expected_shape:
            raise SchemaError(
                f"{name} matrix has shape {matrix.shape}; expected {expected_shape}."
            )
        if not np.isfinite(matrix).all():
            raise SchemaError(f"{name} matrix contains nonfinite values.")

    return ConnectomeBundle(
        subject_id=subject_id,
        roi_labels=roi_labels,
        count=matrices["count"],
        fa=matrices["fa"],
        length=matrices["length"],
    )


_LABEL_SEPARATOR_RE = re.compile(r"[\s_\-]+")


def normalize_roi_label(label: str) -> str:
    """Normalize harmless ROI label formatting for alignment checks."""

    return _LABEL_SEPARATOR_RE.sub(" ", str(label).strip().casefold())


def find_duplicate_normalized_labels(labels: Sequence[str]) -> list[str]:
    """Return labels that collide after harmless-format normalization."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for label in labels:
        normalized = normalize_roi_label(label)
        if normalized in seen:
            duplicates.append(str(label))
        seen.add(normalized)
    return duplicates


def build_roi_reorder_map(
    source_labels: Sequence[str],
    target_labels: Sequence[str],
) -> NDArray[np.int64]:
    """Return indices that reorder source arrays into target ROI-label order."""

    if len(source_labels) != len(target_labels):
        raise SchemaError(
            f"Cannot align {len(source_labels)} source labels to "
            f"{len(target_labels)} target labels."
        )

    source_lookup = _label_lookup(source_labels, label_set_name="source")
    _label_lookup(target_labels, label_set_name="target")

    missing = [
        target_label
        for target_label in target_labels
        if normalize_roi_label(target_label) not in source_lookup
    ]
    if missing:
        raise SchemaError(f"Missing target ROI label(s) in source labels: {missing[:5]}")

    return np.asarray(
        [source_lookup[normalize_roi_label(target_label)] for target_label in target_labels],
        dtype=np.int64,
    )


def apply_roi_reorder_to_vector(
    values: Sequence[float],
    reorder_map: Sequence[int],
) -> NDArray[np.float64]:
    """Apply an explicit ROI reorder map to a vector."""

    values_array = np.asarray(values, dtype=float)
    if values_array.ndim != 1:
        raise SchemaError("Vector must be one-dimensional before ROI reordering.")
    reorder_array = _validate_reorder_map(reorder_map, expected_length=values_array.shape[0])
    return values_array[reorder_array].copy()


def apply_roi_reorder_to_matrix(
    matrix: NDArray[np.float64],
    reorder_map: Sequence[int],
) -> NDArray[np.float64]:
    """Apply an explicit ROI reorder map to both axes of a square matrix."""

    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.ndim != 2 or matrix_array.shape[0] != matrix_array.shape[1]:
        raise SchemaError("Matrix must be square before ROI reordering.")
    reorder_array = _validate_reorder_map(reorder_map, expected_length=matrix_array.shape[0])
    return matrix_array[np.ix_(reorder_array, reorder_array)].copy()


def invert_roi_reorder_map(reorder_map: Sequence[int]) -> NDArray[np.int64]:
    """Invert an explicit ROI reorder map."""

    reorder_array = _validate_reorder_map(reorder_map, expected_length=len(reorder_map))
    inverse = np.empty_like(reorder_array)
    inverse[reorder_array] = np.arange(len(reorder_array), dtype=np.int64)
    return inverse


def _require_columns(table: pd.DataFrame, columns: Sequence[str], *, table_name: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise SchemaError(f"{table_name} missing required column(s): {', '.join(missing)}")


def _require_nonmissing(table: pd.DataFrame, columns: Sequence[str], *, table_name: str) -> None:
    missing_value_columns = [column for column in columns if table[column].isna().any()]
    if missing_value_columns:
        raise SchemaError(
            f"{table_name} contains missing values in column(s): "
            f"{', '.join(missing_value_columns)}"
        )


def _finite_numeric_series(series: pd.Series, column_name: str) -> NDArray[np.float64]:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise SchemaError(f"{column_name} must contain finite numeric values.")
    return numeric


def _integer_series(series: pd.Series, column_name: str) -> NDArray[np.int64]:
    numeric = _finite_numeric_series(series, column_name)
    if not np.equal(numeric, np.floor(numeric)).all():
        raise SchemaError(f"{column_name} must contain integer values.")
    return numeric.astype(np.int64)


def _label_lookup(labels: Sequence[str], *, label_set_name: str) -> dict[str, int]:
    empty_labels = [str(label) for label in labels if not str(label).strip()]
    if empty_labels:
        raise SchemaError(f"{label_set_name} labels contain empty ROI label(s).")
    duplicates = find_duplicate_normalized_labels(labels)
    if duplicates:
        raise SchemaError(
            f"{label_set_name} labels contain duplicate normalized ROI label(s): "
            f"{duplicates[:5]}"
        )
    return {normalize_roi_label(label): index for index, label in enumerate(labels)}


def _validate_reorder_map(
    reorder_map: Sequence[int],
    *,
    expected_length: int,
) -> NDArray[np.int64]:
    raw_reorder = np.asarray(reorder_map)
    if raw_reorder.shape != (expected_length,):
        raise SchemaError(
            f"ROI reorder map has shape {raw_reorder.shape}; expected ({expected_length},)."
        )
    if not np.issubdtype(raw_reorder.dtype, np.integer):
        raise SchemaError("ROI reorder map must contain integer indices.")
    reorder_array = raw_reorder.astype(np.int64, copy=False)
    if set(reorder_array.tolist()) != set(range(expected_length)):
        raise SchemaError("ROI reorder map must be a permutation of zero-based ROI indices.")
    return reorder_array


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{field_name} must be a string.")
    return value


def _coerce_roi_labels(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raise SchemaError("roi_labels must be a sequence of strings, not a single string.")
    try:
        labels = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, AttributeError) as exc:
        raise SchemaError("roi_labels must be an iterable sequence of strings.") from exc

    if any(not isinstance(label, str) for label in labels):
        raise SchemaError("roi_labels must contain only strings.")
    return labels


def _readonly_float64_array(values: object, *, matrix_name: str) -> NDArray[np.float64]:
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, AttributeError) as exc:
        raise SchemaError(f"{matrix_name} matrix must contain numeric values.") from exc

    if not np.isfinite(array).all():
        raise SchemaError(f"{matrix_name} matrix contains nonfinite values.")
    array.setflags(write=False)
    return array
