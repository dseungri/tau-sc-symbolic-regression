"""Audit-gated aggregate reporting for the full-cohort tenfold analysis."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tau_sr.cohort_experiment import write_private_json
from tau_sr.cohort_tenfold import (
    SECONDARY_COMPARISONS,
    TENFOLD_MODELS,
    TENFOLD_SUMMARY_NAME,
    validate_private_tenfold_output_directory,
)
from tau_sr.cohort_tenfold_audit import TENFOLD_AUDIT_NAME
from tau_sr.pysr_lifecycle import parse_json_file
from tau_sr.schemas import SchemaError

REPO_ROOT = Path(__file__).resolve().parents[2]
TENFOLD_REPORT_NAME = "cohort_tenfold_report_manifest.json"

__all__ = (
    "TENFOLD_REPORT_NAME",
    "generate_tenfold_report",
)


def generate_tenfold_report(
    output_dir: str | Path,
    *,
    publication_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate privacy-safe aggregate figures and tables after a passed audit."""

    private_dir = validate_private_tenfold_output_directory(output_dir)
    summary = _read_json_object(private_dir / TENFOLD_SUMMARY_NAME, "tenfold summary")
    audit = _read_json_object(private_dir / TENFOLD_AUDIT_NAME, "tenfold audit")
    _validate_reporting_inputs(summary, audit, private_dir)
    destination = _report_directory(private_dir, publication_dir)
    figures = destination / "figures"
    tables = destination / "tables"
    figures.mkdir(parents=True, exist_ok=True, mode=0o700)
    tables.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(figures, 0o700)
    os.chmod(tables, 0o700)

    figure_paths = _write_figures(figures, summary)
    table_paths = _write_tables(tables, summary)
    manifest = {
        "report": "full_cohort_nested_tenfold_aggregate_report",
        "completion_status": "complete",
        "requires_passed_audit": True,
        "audit_pass": True,
        "summary_source": TENFOLD_SUMMARY_NAME,
        "audit_source": TENFOLD_AUDIT_NAME,
        "figures": {
            name: {
                suffix: _relative_to_destination(path, destination)
                for suffix, path in paths.items()
            }
            for name, paths in figure_paths.items()
        },
        "tables": {
            name: _relative_to_destination(path, destination)
            for name, path in table_paths.items()
        },
        "privacy": {
            "participant_labels_included": False,
            "subject_level_points_included": False,
            "private_parent_path_included": False,
        },
    }
    _assert_manifest_privacy(manifest)
    manifest_path = destination / TENFOLD_REPORT_NAME
    write_private_json(manifest_path, manifest)
    os.chmod(manifest_path, 0o600)
    return manifest


def _validate_reporting_inputs(
    summary: Mapping[str, Any],
    audit: Mapping[str, Any],
    private_dir: Path,
) -> None:
    if summary.get("completion_status") != "complete":
        raise SchemaError("tenfold reporting requires a completed summary.")
    if audit.get("audit_pass") is not True:
        raise SchemaError("tenfold reporting requires a passed independent audit.")
    if "model_bootstrap_intervals" not in summary:
        raise SchemaError("tenfold summary is missing model bootstrap intervals.")
    artifacts = _mapping(summary.get("private_prediction_artifacts"), "private artifacts")
    for relative in artifacts.values():
        path = private_dir / _relative_artifact_path(relative)
        if not path.exists():
            raise SchemaError("tenfold reporting requires completed OOF artifacts.")
        if path.stat().st_mode & 0o077:
            raise SchemaError("OOF artifact permissions are not restrictive.")


def _report_directory(private_dir: Path, publication_dir: str | Path | None) -> Path:
    if publication_dir is None:
        destination = private_dir / "aggregate_report"
    else:
        destination = Path(publication_dir).expanduser().resolve()
        if _is_relative_to(destination, REPO_ROOT):
            raise SchemaError("publication report directory must not be inside the repository.")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination, 0o700)
    return destination


def _write_figures(figures: Path, summary: Mapping[str, Any]) -> dict[str, dict[str, Path]]:
    mpl_config = figures / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(mpl_config, 0o700)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    outputs = {
        "model_comparison": _figure_model_comparison(figures, summary, plt),
        "paired_differences": _figure_paired_differences(figures, summary, plt),
        "fold_stability": _figure_fold_stability(figures, summary, plt),
        "parameter_stability": _figure_parameter_stability(figures, summary, plt),
        "symbolic_stability": _figure_symbolic_stability(figures, summary, plt),
    }
    return outputs


def _figure_model_comparison(
    figures: Path,
    summary: Mapping[str, Any],
    plt: Any,
) -> dict[str, Path]:
    metrics = _mapping(summary.get("overall_out_of_fold_metrics"), "overall metrics")
    intervals = _mapping(summary.get("model_bootstrap_intervals"), "model intervals")
    models = [model for model in TENFOLD_MODELS if model in metrics]
    x = np.arange(len(models))
    rmse = np.asarray([float(metrics[model]["rmse"]) for model in models])
    mae = np.asarray([float(metrics[model]["mae"]) for model in models])
    rmse_ci = np.asarray([intervals[model]["rmse"]["ci95"] for model in models], dtype=float)
    mae_ci = np.asarray([intervals[model]["mae"]["ci95"] for model in models], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(
        x - 0.18,
        rmse,
        width=0.36,
        yerr=np.vstack((rmse - rmse_ci[:, 0], rmse_ci[:, 1] - rmse)),
        label="RMSE",
        color="#355C8C",
        capsize=2.5,
    )
    ax.bar(
        x + 0.18,
        mae,
        width=0.36,
        yerr=np.vstack((mae - mae_ci[:, 0], mae_ci[:, 1] - mae)),
        label="MAE",
        color="#8C5A35",
        capsize=2.5,
    )
    ax.set_xticks(x, models, rotation=30, ha="right")
    ax.set_ylabel("Prediction error")
    ax.set_title("Out-of-fold model comparison")
    ax.text(0.01, 0.96, "Lower is better", transform=ax.transAxes, va="top")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save_figure(fig, figures / "figure_1_model_comparison", plt)


def _figure_paired_differences(
    figures: Path,
    summary: Mapping[str, Any],
    plt: Any,
) -> dict[str, Path]:
    comparisons = _mapping(summary.get("paired_bootstrap_comparisons"), "comparisons")
    names = ["quadratic_vs_linear", *[name for name, _first, _second in SECONDARY_COMPARISONS]]
    names = [name for name in names if name in comparisons]
    y = np.arange(len(names))
    diffs = np.asarray(
        [float(comparisons[name]["rmse"]["difference_first_minus_second"]) for name in names]
    )
    ci = np.asarray([comparisons[name]["rmse"]["ci95"] for name in names], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.axvline(0.0, color="0.25", linewidth=0.8)
    colors = ["#1F5A85" if name == "quadratic_vs_linear" else "#666666" for name in names]
    for idx, color in enumerate(colors):
        ax.errorbar(
            diffs[idx],
            y[idx],
            xerr=[[diffs[idx] - ci[idx, 0]], [ci[idx, 1] - diffs[idx]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.5,
        )
    ax.set_yticks(y, [name.replace("_", " ") for name in names])
    ax.set_xlabel("Paired RMSE difference")
    ax.set_title("Participant-level paired bootstrap")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    return _save_figure(fig, figures / "figure_2_paired_rmse_differences", plt)


def _figure_fold_stability(
    figures: Path,
    summary: Mapping[str, Any],
    plt: Any,
) -> dict[str, Path]:
    fold_summary = _mapping(summary.get("fold_metric_summary"), "fold metric summary")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for model in TENFOLD_MODELS:
        if model in fold_summary:
            values = fold_summary[model]["fold_rmse"]
            ax.plot(range(1, len(values) + 1), values, marker="o", linewidth=1.2, label=model)
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("RMSE")
    ax.set_title("Fold-level RMSE stability")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    return _save_figure(fig, figures / "figure_3_fold_rmse_stability", plt)


def _figure_parameter_stability(
    figures: Path,
    summary: Mapping[str, Any],
    plt: Any,
) -> dict[str, Path]:
    folds = _mapping(summary.get("folds"), "folds")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    rho_models = ("diffusion", "linear", "quadratic", "neural")
    rho_values = [
        [
            float(_mapping(_mapping(record, "fold").get("models"), "models")[model]["rho"])
            for record in folds.values()
            if model in _mapping(_mapping(record, "fold").get("models"), "models")
        ]
        for model in rho_models
    ]
    axes[0].boxplot(rho_values, tick_labels=rho_models, showfliers=False)
    axes[0].set_ylabel("rho")
    axes[0].set_title("Diffusion coefficient")
    coefficient_values = [
        [
            float(
                _mapping(_mapping(record, "fold").get("models"), "models")["linear"][
                    "linear_coefficient"
                ]
            )
            for record in folds.values()
        ],
        [
            float(
                _mapping(_mapping(record, "fold").get("models"), "models")["quadratic"][
                    "linear_coefficient"
                ]
            )
            for record in folds.values()
        ],
        [
            float(
                _mapping(_mapping(record, "fold").get("models"), "models")["quadratic"][
                    "quadratic_coefficient"
                ]
            )
            for record in folds.values()
        ],
    ]
    axes[1].boxplot(
        coefficient_values,
        tick_labels=("linear a", "quadratic a", "quadratic b"),
        showfliers=False,
    )
    axes[1].set_title("Reaction coefficients")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
    return _save_figure(fig, figures / "figure_4_parameter_stability", plt)


def _figure_symbolic_stability(
    figures: Path,
    summary: Mapping[str, Any],
    plt: Any,
) -> dict[str, Path]:
    stability = _mapping(
        _mapping(summary.get("fold_stability"), "fold stability").get("symbolic_stability"),
        "symbolic",
    )
    labels = ("linear", "quadratic", "higher order")
    counts = [
        int(stability.get("linear_fold_count", 0)),
        int(stability.get("quadratic_fold_count", 0)),
        int(stability.get("higher_order_fold_count", 0)),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(labels, counts, color="#355C8C")
    ax.set_ylabel("Fold count")
    ax.set_title("Symbolic equation structure")
    ax.spines[["top", "right"]].set_visible(False)
    return _save_figure(fig, figures / "figure_5_symbolic_stability", plt)


def _write_tables(tables: Path, summary: Mapping[str, Any]) -> dict[str, Path]:
    outputs = {
        "model_metrics": _table_model_metrics(tables, summary),
        "paired_comparisons": _table_paired_comparisons(tables, summary),
        "parameter_stability": _table_parameter_stability(tables, summary),
        "symbolic_stability": _table_symbolic_stability(tables, summary),
    }
    return outputs


def _table_model_metrics(tables: Path, summary: Mapping[str, Any]) -> Path:
    metrics = _mapping(summary.get("overall_out_of_fold_metrics"), "overall metrics")
    intervals = _mapping(summary.get("model_bootstrap_intervals"), "model intervals")
    rows = []
    for model in TENFOLD_MODELS:
        row = _mapping(metrics.get(model), f"{model} metrics")
        model_interval = _mapping(intervals.get(model), f"{model} intervals")
        rows.append(
            {
                "model": model,
                "rmse": row["rmse"],
                "rmse_ci95_low": model_interval["rmse"]["ci95"][0],
                "rmse_ci95_high": model_interval["rmse"]["ci95"][1],
                "mae": row["mae"],
                "mae_ci95_low": model_interval["mae"]["ci95"][0],
                "mae_ci95_high": model_interval["mae"]["ci95"][1],
                "mse": row["mse"],
                "bias": row["bias"],
                "residual_variance": row["residual_variance"],
                "residual_standard_deviation": row["residual_standard_deviation"],
                "r_squared_secondary_descriptive": row["r_squared_secondary_descriptive"],
            }
        )
    return _write_csv(tables / "model_metrics.csv", rows)


def _table_paired_comparisons(tables: Path, summary: Mapping[str, Any]) -> Path:
    comparisons = _mapping(summary.get("paired_bootstrap_comparisons"), "comparisons")
    rows = []
    for name, payload_object in comparisons.items():
        payload = _mapping(payload_object, f"{name} comparison")
        rows.append(
            {
                "comparison": name,
                "category": payload["category"],
                "rmse_difference": payload["rmse"]["difference_first_minus_second"],
                "rmse_relative_percent_difference": payload["rmse"][
                    "relative_percent_difference"
                ],
                "rmse_ci95_low": payload["rmse"]["ci95"][0],
                "rmse_ci95_high": payload["rmse"]["ci95"][1],
                "mae_difference": payload["mae"]["difference_first_minus_second"],
                "mae_relative_percent_difference": payload["mae"][
                    "relative_percent_difference"
                ],
                "mae_ci95_low": payload["mae"]["ci95"][0],
                "mae_ci95_high": payload["mae"]["ci95"][1],
            }
        )
    return _write_csv(tables / "paired_comparisons.csv", rows)


def _table_parameter_stability(tables: Path, summary: Mapping[str, Any]) -> Path:
    stability = _mapping(summary.get("fold_stability"), "fold stability")
    rows = []
    for group_name in ("rho_summary", "coefficient_summary"):
        group = _mapping(stability.get(group_name), group_name)
        for name, values in group.items():
            row = _mapping(values, f"{group_name}.{name}")
            rows.append({"parameter": name, **dict(row)})
    return _write_csv(tables / "parameter_stability.csv", rows)


def _table_symbolic_stability(tables: Path, summary: Mapping[str, Any]) -> Path:
    symbolic = _mapping(
        _mapping(summary.get("fold_stability"), "fold stability").get("symbolic_stability"),
        "symbolic stability",
    )
    rows = [
        {"quantity": key, "value": value}
        for key, value in symbolic.items()
        if not isinstance(value, Mapping)
    ]
    return _write_csv(tables / "symbolic_stability.csv", rows)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)
    return path


def _save_figure(fig: Any, stem: Path, plt: Any) -> dict[str, Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    os.chmod(png, 0o600)
    os.chmod(pdf, 0o600)
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def _read_json_object(path: Path, field_name: str) -> dict[str, Any]:
    try:
        payload = parse_json_file(path)
    except Exception as exc:
        raise SchemaError(f"{field_name} could not be read.") from exc
    if not isinstance(payload, dict):
        raise SchemaError(f"{field_name} must contain a JSON object.")
    return payload


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field_name} must be a mapping.")
    return value


def _relative_artifact_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("artifact path must be a relative string.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise SchemaError("artifact path must be relative.")
    return path


def _relative_to_destination(path: Path, destination: Path) -> str:
    return str(path.relative_to(destination))


def _assert_manifest_privacy(manifest: Mapping[str, Any]) -> None:
    text = json.dumps(manifest, sort_keys=True)
    forbidden = (
        "/Users/",
        "Private" + "Research",
        "Down" + "loads",
        "002" + "_S_",
    )
    if any(token in text for token in forbidden):
        raise SchemaError("report manifest contains private-looking content.")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
