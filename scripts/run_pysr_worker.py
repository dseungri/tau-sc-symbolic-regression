"""Dedicated PySR worker used by the identifiability experiment parent process."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tau_sr.pysr_lifecycle import parse_json_file, write_json_atomically  # noqa: E402
from tau_sr.schemas import SchemaError  # noqa: E402

_NEGLIGIBLE_RANGE = 1e-12


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    result_path = Path(args.result)
    try:
        payload = _run_worker(Path(args.request))
        write_json_atomically(result_path, {"status": "ok", "payload": payload})
        return 0
    except Exception as exc:
        error_payload = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            write_json_atomically(result_path, error_payload)
        except Exception as write_exc:  # pragma: no cover - last-ditch diagnostics
            print(
                f"failed to write PySR worker error result: {type(write_exc).__name__}: "
                f"{write_exc}",
                file=sys.stderr,
            )
        return 1


def _run_worker(request_path: Path) -> dict[str, Any]:
    request = _load_request(request_path)

    # Import PySR/Juliacall inside the disposable worker, before importing the
    # torch-using symbolic module. This confines Julia startup and shutdown to
    # this process instead of the long-lived experiment parent.
    importlib.import_module("pysr")
    symbolic = importlib.import_module("tau_sr.symbolic")
    ReactionSamples = symbolic.ReactionSamples
    SymbolicRecoveryMetrics = symbolic.SymbolicRecoveryMetrics
    SymbolicRegressionConfig = symbolic.SymbolicRegressionConfig
    fit_symbolic_reaction = symbolic.fit_symbolic_reaction
    interpret_symbolic_polynomial = symbolic.interpret_symbolic_polynomial
    evaluate_symbolic_regressor = symbolic._evaluate_symbolic_regressor

    samples_payload = request["samples"]
    config_payload = request["config"]
    evaluation_payload = request["evaluation"]
    samples = ReactionSamples(
        input_values=np.asarray(samples_payload["input_values"], dtype=float),
        neural_reaction_values=np.asarray(
            samples_payload["neural_reaction_values"],
            dtype=float,
        ),
        domain_min=float(samples_payload["domain_min"]),
        domain_max=float(samples_payload["domain_max"]),
    )
    config = SymbolicRegressionConfig(
        niterations=int(config_payload["niterations"]),
        populations=int(config_payload["populations"]),
        population_size=int(config_payload["population_size"]),
        maxsize=int(config_payload["maxsize"]),
        seed=int(config_payload["seed"]),
        model_selection=str(config_payload["model_selection"]),
        output_directory=config_payload["output_directory"],
    )
    evaluation_grid = _vector(evaluation_payload["grid"], "evaluation.grid")
    neural_prediction = _vector(
        evaluation_payload["neural_prediction"],
        "evaluation.neural_prediction",
    )
    true_prediction = _vector(
        evaluation_payload["true_prediction"],
        "evaluation.true_prediction",
    )
    if neural_prediction.shape != evaluation_grid.shape:
        raise SchemaError("evaluation.neural_prediction must match evaluation.grid.")
    if true_prediction.shape != evaluation_grid.shape:
        raise SchemaError("evaluation.true_prediction must match evaluation.grid.")

    start = time.perf_counter()
    fit_result = fit_symbolic_reaction(samples, config)
    runtime_seconds = time.perf_counter() - start

    symbolic_prediction = evaluate_symbolic_regressor(
        fit_result.regressor,
        evaluation_grid,
    )
    neural_vs_true = _comparison_metrics(neural_prediction, true_prediction)
    symbolic_vs_neural = _comparison_metrics(symbolic_prediction, neural_prediction)
    symbolic_vs_true = _comparison_metrics(symbolic_prediction, true_prediction)
    true_range = float(np.ptp(true_prediction))
    denominator = true_range if true_range > _NEGLIGIBLE_RANGE else None
    metrics = SymbolicRecoveryMetrics(
        evaluation_domain_min=float(evaluation_grid.min()),
        evaluation_domain_max=float(evaluation_grid.max()),
        n_evaluation_points=int(evaluation_grid.shape[0]),
        neural_vs_true_rmse=neural_vs_true[0],
        neural_vs_true_mae=neural_vs_true[1],
        symbolic_vs_neural_rmse=symbolic_vs_neural[0],
        symbolic_vs_neural_mae=symbolic_vs_neural[1],
        symbolic_vs_true_rmse=symbolic_vs_true[0],
        symbolic_vs_true_mae=symbolic_vs_true[1],
        true_reaction_range=true_range,
        neural_vs_true_normalized_rmse=(
            neural_vs_true[0] / denominator if denominator is not None else None
        ),
        symbolic_vs_neural_normalized_rmse=(
            symbolic_vs_neural[0] / denominator if denominator is not None else None
        ),
        symbolic_vs_true_normalized_rmse=(
            symbolic_vs_true[0] / denominator if denominator is not None else None
        ),
        symbolic_complexity=fit_result.complexity,
    )
    interpretation = interpret_symbolic_polynomial(
        fit_result,
        true_growth_rate=float(evaluation_payload["true_growth_rate"]),
        true_saturation_rate=float(evaluation_payload["true_saturation_rate"]),
    )

    return {
        "fit_domain": [samples.domain_min, samples.domain_max],
        "evaluation_domain": [
            float(evaluation_grid.min()),
            float(evaluation_grid.max()),
        ],
        "equation": fit_result.equation_string,
        "expanded_expression": interpretation.expanded_expression_string,
        "complexity": fit_result.complexity,
        "training_loss": fit_result.training_loss,
        "artifact_directory": fit_result.artifact_directory,
        "runtime_seconds": runtime_seconds,
        "pysr_version": fit_result.pysr_version,
        "julia_version": _julia_version(),
        "metrics": metrics.__dict__,
        "polynomial_interpretation": interpretation.__dict__,
    }


def _load_request(path: Path) -> dict[str, Any]:
    request = parse_json_file(path)
    if not isinstance(request, dict):
        raise SchemaError("worker request must be a JSON object.")
    for key in ("samples", "config", "evaluation"):
        if key not in request:
            raise SchemaError(f"worker request is missing {key!r}.")
        if not isinstance(request[key], dict):
            raise SchemaError(f"worker request field {key!r} must be an object.")
    return request


def _vector(values: object, field_name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise SchemaError(f"{field_name} must be a nonempty one-dimensional array.")
    if not np.isfinite(vector).all():
        raise SchemaError(f"{field_name} must contain only finite values.")
    return vector.copy()


def _comparison_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    difference = left - right
    return float(np.sqrt(np.mean(difference**2))), float(np.mean(np.abs(difference)))


def _julia_version() -> str:
    try:
        from juliacall import Main as JuliaMain

        return str(JuliaMain.VERSION)
    except Exception as exc:  # pragma: no cover - depends on local Julia state
        return f"unavailable: {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())
