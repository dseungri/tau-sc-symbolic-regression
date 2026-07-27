"""Process lifecycle helpers for isolated PySR worker execution."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from tau_sr.schemas import SchemaError

__all__ = (
    "JsonWorkerResult",
    "PySRWorkerError",
    "parse_json_file",
    "run_json_worker",
    "sha256_file",
    "write_json_atomically",
)


@dataclass(frozen=True)
class JsonWorkerResult:
    payload: dict[str, Any]
    process_exited_normally: bool
    cleanup_required: bool
    timed_out: bool
    returncode: int | None
    stdout: str
    stderr: str


class PySRWorkerError(RuntimeError):
    """Raised when an isolated PySR worker fails before a valid result exists."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        cleanup_required: bool = False,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.cleanup_required = cleanup_required
        self.timed_out = timed_out


def write_json_atomically(path: str | Path, payload: object) -> None:
    """Durably write parseable JSON without exposing partial destination files."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        parse_json_file(destination)
    except Exception:
        with suppress(FileNotFoundError):
            temporary_path.unlink()
        raise


def parse_json_file(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_json_worker(
    command: list[str],
    result_path: str | Path,
    *,
    timeout_seconds: float,
    shutdown_grace_seconds: float,
    poll_interval_seconds: float = 0.05,
) -> JsonWorkerResult:
    """Run one owned worker that atomically reports ``{"status": "ok"}`` JSON."""

    if not command:
        raise SchemaError("worker command must be nonempty.")
    timeout = _positive_float(timeout_seconds, "timeout_seconds")
    shutdown_grace = _positive_float(
        shutdown_grace_seconds,
        "shutdown_grace_seconds",
    )
    poll_interval = _positive_float(
        poll_interval_seconds,
        "poll_interval_seconds",
    )
    result = Path(result_path)
    result.parent.mkdir(parents=True, exist_ok=True)
    if result.exists():
        result.unlink()

    stdout_fd, stdout_name = tempfile.mkstemp(
        prefix=".worker_stdout.",
        suffix=".log",
        dir=result.parent,
    )
    stderr_fd, stderr_name = tempfile.mkstemp(
        prefix=".worker_stderr.",
        suffix=".log",
        dir=result.parent,
    )
    stdout_path = Path(stdout_name)
    stderr_path = Path(stderr_name)
    stdout_handle = os.fdopen(stdout_fd, "wb")
    stderr_handle = os.fdopen(stderr_fd, "wb")
    process: subprocess.Popen[bytes] | None = None

    try:
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=(os.name == "posix"),
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()
    except Exception:
        for path in (stdout_path, stderr_path):
            with suppress(FileNotFoundError):
                path.unlink()
        raise

    try:
        deadline = time.monotonic() + timeout
        while True:
            worker_result = _read_worker_result(result)
            if worker_result is not None:
                status, payload = worker_result
                if status == "ok":
                    return _finish_completed_worker(
                        process,
                        payload,
                        shutdown_grace_seconds=shutdown_grace,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                    )
                cleanup_required = _terminate_owned_worker(process, shutdown_grace)
                stdout = _read_text(stdout_path)
                stderr = _read_text(stderr_path)
                error = str(payload.get("error", "worker reported failure"))
                raise PySRWorkerError(
                    error,
                    returncode=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    cleanup_required=cleanup_required,
                )

            returncode = process.poll()
            if returncode is not None:
                # The worker may have written its atomic result exactly as it
                # exited; give the read one more chance before concluding failure.
                worker_result = _read_worker_result(result)
                if worker_result is not None:
                    status, payload = worker_result
                    if status == "ok":
                        return _finish_completed_worker(
                            process,
                            payload,
                            shutdown_grace_seconds=shutdown_grace,
                            stdout_path=stdout_path,
                            stderr_path=stderr_path,
                        )
                stdout = _read_text(stdout_path)
                stderr = _read_text(stderr_path)
                if result.exists():
                    raise PySRWorkerError(
                        "worker result is malformed or incomplete.",
                        returncode=returncode,
                        stdout=stdout,
                        stderr=stderr,
                    )
                raise PySRWorkerError(
                    "worker failed before producing a result.",
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                )

            if time.monotonic() >= deadline:
                cleanup_required = _terminate_owned_worker(process, shutdown_grace)
                stdout = _read_text(stdout_path)
                stderr = _read_text(stderr_path)
                raise PySRWorkerError(
                    "worker timed out before producing a result.",
                    returncode=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    cleanup_required=cleanup_required,
                    timed_out=True,
                )

            time.sleep(poll_interval)
    finally:
        for path in (stdout_path, stderr_path):
            with suppress(FileNotFoundError):
                path.unlink()


def _finish_completed_worker(
    process: subprocess.Popen[bytes],
    payload: dict[str, Any],
    *,
    shutdown_grace_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
) -> JsonWorkerResult:
    cleanup_required = False
    timed_out = False
    try:
        process.wait(timeout=shutdown_grace_seconds)
    except subprocess.TimeoutExpired:
        cleanup_required = _terminate_owned_worker(process, shutdown_grace_seconds)
        timed_out = True

    stdout = _read_text(stdout_path)
    stderr = _read_text(stderr_path)
    if not cleanup_required and process.returncode != 0:
        raise PySRWorkerError(
            "worker exited nonzero after producing a result.",
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return JsonWorkerResult(
        payload=payload,
        process_exited_normally=process.returncode == 0 and not cleanup_required,
        cleanup_required=cleanup_required,
        timed_out=timed_out,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _read_worker_result(path: Path) -> tuple[str, dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        raw = parse_json_file(path)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    status = raw.get("status")
    payload = raw.get("payload")
    if status == "error":
        error_payload = raw if isinstance(raw, dict) else {"error": "worker failed"}
        return "error", error_payload
    if status != "ok" or not isinstance(payload, dict):
        return None
    return "ok", payload


def _terminate_owned_worker(
    process: subprocess.Popen[bytes],
    shutdown_grace_seconds: float,
) -> bool:
    if process.poll() is not None:
        return False

    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    else:  # pragma: no cover - exercised on non-POSIX platforms only
        process.terminate()
    try:
        process.wait(timeout=shutdown_grace_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - exercised on non-POSIX platforms only
            process.kill()
        process.wait(timeout=shutdown_grace_seconds)
    return True


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{field_name} must be a positive number.")
    checked = float(value)
    if not math.isfinite(checked) or not checked > 0.0:
        raise SchemaError(f"{field_name} must be positive.")
    return checked
