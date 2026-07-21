"""Command-line interface for setup and synthetic validation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tau_sr.configuration import find_unset_values, load_config_dir
from tau_sr.graphs.laplacian import (
    build_symmetric_normalized_laplacian,
    build_unnormalized_laplacian,
    transport_direction_test,
    validate_laplacian,
)
from tau_sr.graphs.representations import (
    build_count_network,
    build_fa_weighted_network,
    build_length_adjusted_network,
)
from tau_sr.intake.checksums import (
    checksum_records,
    verify_checksum_manifest,
)
from tau_sr.intake.inventory import inventory_directory, write_inventory
from tau_sr.logging_utils import configure_logging
from tau_sr.metadata.schemas import validate_manifest
from tau_sr.paths import is_incoming_path
from tau_sr.qc.connectome import validate_connectome_matrix
from tau_sr.qc.reports import write_reports_json
from tau_sr.qc.roi_alignment import compare_roi_sets, load_roi_labels, validate_roi_labels
from tau_sr.qc.tau import validate_tau_table
from tau_sr.splitting.grouped import fixed_train_validation_test_split, grouped_kfold, write_split
from tau_sr.synthetic.fixtures import (
    synthetic_connectome_matrices,
    synthetic_roi_labels,
    synthetic_subject_ids,
    synthetic_tau_table,
)

LOGGER = logging.getLogger(__name__)


def _write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_matrix(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    if file_path.suffix == ".npy":
        return np.load(file_path)
    delimiter = "	" if file_path.suffix == ".tsv" else ","
    return np.loadtxt(file_path, delimiter=delimiter)


def _write_matrix(path: str | Path, matrix: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, matrix, delimiter=",")


def cmd_verify_environment(args: argparse.Namespace) -> int:
    configs = load_config_dir(args.config_dir)
    unset = {name: find_unset_values(config) for name, config in configs.items()}
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "config_files": sorted(configs),
        "unset_values": unset,
        "data_root_env": os.getenv("TAU_SR_DATA_ROOT", "UNSET"),
    }
    _write_json(args.output, payload)
    return 0


def cmd_initialize_data_dirs(args: argparse.Namespace) -> int:
    root = Path(args.root)
    LOGGER.warning(
        "Initializing data directories. Incoming data are immutable; "
        "institutional access controls still apply."
    )
    directories = [
        "00_incoming_readonly",
        "01_inventory",
        "02_interim",
        "03_processed",
        "04_splits",
        "05_synthetic",
    ]
    created = []
    for name in directories:
        directory = root / name
        existing_payload = list(directory.iterdir()) if directory.exists() else []
        non_placeholder = [
            item for item in existing_payload if item.name not in {".gitkeep", "README.md"}
        ]
        if non_placeholder:
            raise RuntimeError(f"refusing to touch nonempty data directory: {directory}")
        if not args.dry_run:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ".gitkeep").touch(exist_ok=True)
            if args.apply_posix_permissions and hasattr(os, "chmod"):
                os.chmod(directory, 0o700)
        created.append(directory.as_posix())
    record = {
        "dry_run": args.dry_run,
        "directories": created,
        "posix_permissions_requested": args.apply_posix_permissions,
        "posix_permissions_note": (
            "Filesystem permissions do not replace institutional access controls."
        ),
    }
    if not args.dry_run:
        _write_json(args.output_record, record)
    else:
        _write_json(args.output_record, record)
    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    if is_incoming_path(args.input):
        LOGGER.warning(
            "Inventory is reading incoming immutable data. Source files will not be modified."
        )
    records = inventory_directory(args.input, hash_files=not args.no_hash)
    if not args.dry_run:
        write_inventory(records, args.output)
    return 0


def cmd_verify_checksums(args: argparse.Namespace) -> int:
    mismatches = verify_checksum_manifest(args.input, args.manifest)
    _write_json(args.output, {"ok": not mismatches, "mismatches": mismatches})
    return 1 if mismatches else 0


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.manifest)
    report = validate_manifest(frame, file_root=args.file_root)
    report.write_json(args.output)
    return 0 if report.ok else 1


def cmd_validate_rois(args: argparse.Namespace) -> int:
    tau = load_roi_labels(args.tau_rois)
    connectome = load_roi_labels(args.connectome_rois)
    reports = [
        validate_roi_labels(tau, strict=args.strict),
        validate_roi_labels(connectome, strict=args.strict),
        compare_roi_sets(tau, connectome),
    ]
    write_reports_json(reports, args.output)
    return 0 if all(report.ok for report in reports) else 1


def cmd_validate_tau(args: argparse.Namespace) -> int:
    table = pd.read_csv(args.input, sep="	" if Path(args.input).suffix == ".tsv" else ",")
    report = validate_tau_table(table)
    report.write_json(args.output)
    return 0 if report.ok else 1


def cmd_validate_connectomes(args: argparse.Namespace) -> int:
    matrix = _load_matrix(args.input)
    report = validate_connectome_matrix(matrix)
    report.write_json(args.output)
    return 0 if report.ok else 1


def cmd_build_networks(args: argparse.Namespace) -> int:
    count = _load_matrix(args.count_matrix)
    fa = _load_matrix(args.fa_matrix)
    length = _load_matrix(args.length_matrix)
    outputs = Path(args.output_dir)
    if args.dry_run:
        return 0
    outputs.mkdir(parents=True, exist_ok=True)
    results = {
        "count": build_count_network(count),
        "fa": build_fa_weighted_network(count, fa),
        "length": build_length_adjusted_network(count, length, args.epsilon),
    }
    metadata = {}
    for name, result in results.items():
        _write_matrix(outputs / f"{name}_network.csv", result.matrix)
        metadata[name] = result.metadata
    _write_json(outputs / "network_metadata.json", metadata)
    return 0


def cmd_build_laplacians(args: argparse.Namespace) -> int:
    matrix = _load_matrix(args.input)
    outputs = Path(args.output_dir)
    if args.dry_run:
        return 0
    outputs.mkdir(parents=True, exist_ok=True)
    unnormalized = build_unnormalized_laplacian(matrix)
    symmetric = build_symmetric_normalized_laplacian(matrix)
    _write_matrix(outputs / "laplacian_unnormalized.csv", unnormalized)
    _write_matrix(outputs / "laplacian_symmetric_normalized.csv", symmetric)
    _write_json(
        outputs / "laplacian_validation.json",
        {
            "unnormalized": validate_laplacian(unnormalized, adjacency=matrix).to_dict(),
            "symmetric_normalized": validate_laplacian(symmetric, adjacency=matrix).to_dict(),
            "transport_direction_test": transport_direction_test(),
        },
    )
    return 0


def cmd_create_splits(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.subjects)
    if args.group_column != "subject_id":
        raise RuntimeError("splitting by region or visit is prohibited")
    subjects = frame[args.group_column].astype(str).tolist()
    if args.strategy == "fixed":
        split = fixed_train_validation_test_split(
            subjects,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    else:
        split = grouped_kfold(subjects, n_folds=args.folds, seed=args.seed)
    if not args.dry_run:
        write_split(split, args.output)
    return 0


def cmd_smoke_test(args: argparse.Namespace) -> int:
    temp_dir = Path(tempfile.mkdtemp(prefix="tau_sr_smoke_"))
    try:
        roi_labels = synthetic_roi_labels()
        roi_csv = temp_dir / "roi_labels.csv"
        pd.DataFrame({"roi_name": roi_labels}).to_csv(roi_csv, index=False)
        tau = synthetic_tau_table()
        tau_csv = temp_dir / "tau.csv"
        tau.to_csv(tau_csv, index=False)
        count, fa, length = synthetic_connectome_matrices()
        np.savetxt(temp_dir / "count.csv", count, delimiter=",")
        np.savetxt(temp_dir / "fa.csv", fa, delimiter=",")
        np.savetxt(temp_dir / "length.csv", length, delimiter=",")

        inventory = inventory_directory(temp_dir)
        checksum_manifest = checksum_records(temp_dir)
        roi_report = validate_roi_labels(roi_labels)
        alignment_report = compare_roi_sets(roi_labels, list(reversed(roi_labels)))
        tau_report = validate_tau_table(tau)
        connectome_report = validate_connectome_matrix(count)
        count_network = build_count_network(count)
        fa_network = build_fa_weighted_network(count, fa)
        length_network = build_length_adjusted_network(count, length, 1.0e-8)
        unnormalized = build_unnormalized_laplacian(count_network.matrix)
        symmetric = build_symmetric_normalized_laplacian(count_network.matrix)
        lap_reports = [
            validate_laplacian(unnormalized, adjacency=count_network.matrix),
            validate_laplacian(symmetric, adjacency=count_network.matrix),
        ]
        subjects = synthetic_subject_ids(6)
        split = fixed_train_validation_test_split(
            subjects, test_fraction=0.2, validation_fraction=0.2, seed=13
        )
        payload = {
            "ok": bool(
                roi_report.ok
                and tau_report.ok
                and connectome_report.ok
                and all(report.ok for report in lap_reports)
                and transport_direction_test()
            ),
            "temporary_directory": temp_dir.as_posix() if args.keep_temporary else "deleted",
            "inventory_count": len(inventory),
            "checksum_count": len(checksum_manifest),
            "roi_report": roi_report.to_dict(),
            "alignment_report": alignment_report.to_dict(),
            "tau_report": tau_report.to_dict(),
            "connectome_report": connectome_report.to_dict(),
            "laplacian_reports": [report.to_dict() for report in lap_reports],
            "network_metadata": {
                "count": count_network.metadata,
                "fa": fa_network.metadata,
                "length": length_network.metadata,
            },
            "split_subject_count": int(len(split)),
        }
        _write_json(args.output_report, payload)
        return 0 if payload["ok"] else 1
    finally:
        if not args.keep_temporary:
            shutil.rmtree(temp_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tau-sr")
    parser.add_argument("--log-level", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    env = subparsers.add_parser("verify-environment")
    env.add_argument("--config-dir", required=True)
    env.add_argument("--output", required=True)
    env.set_defaults(func=cmd_verify_environment)

    init = subparsers.add_parser("initialize-data-dirs")
    init.add_argument("--root", required=True)
    init.add_argument("--output-record", required=True)
    init.add_argument("--apply-posix-permissions", action="store_true")
    init.add_argument("--dry-run", action="store_true")
    init.set_defaults(func=cmd_initialize_data_dirs)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--input", required=True)
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--no-hash", action="store_true")
    inventory.add_argument("--dry-run", action="store_true")
    inventory.set_defaults(func=cmd_inventory)

    checksums = subparsers.add_parser("verify-checksums")
    checksums.add_argument("--input", required=True)
    checksums.add_argument("--manifest", required=True)
    checksums.add_argument("--output", required=True)
    checksums.set_defaults(func=cmd_verify_checksums)

    manifest = subparsers.add_parser("validate-manifest")
    manifest.add_argument("--manifest", required=True)
    manifest.add_argument("--file-root", default=None)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(func=cmd_validate_manifest)

    rois = subparsers.add_parser("validate-rois")
    rois.add_argument("--tau-rois", required=True)
    rois.add_argument("--connectome-rois", required=True)
    rois.add_argument("--output", required=True)
    rois.add_argument("--strict", action="store_true")
    rois.set_defaults(func=cmd_validate_rois)

    tau = subparsers.add_parser("validate-tau")
    tau.add_argument("--input", required=True)
    tau.add_argument("--output", required=True)
    tau.set_defaults(func=cmd_validate_tau)

    connectomes = subparsers.add_parser("validate-connectomes")
    connectomes.add_argument("--input", required=True)
    connectomes.add_argument("--output", required=True)
    connectomes.set_defaults(func=cmd_validate_connectomes)

    networks = subparsers.add_parser("build-networks")
    networks.add_argument("--count-matrix", required=True)
    networks.add_argument("--fa-matrix", required=True)
    networks.add_argument("--length-matrix", required=True)
    networks.add_argument("--output-dir", required=True)
    networks.add_argument("--epsilon", type=float, default=1.0e-8)
    networks.add_argument("--dry-run", action="store_true")
    networks.set_defaults(func=cmd_build_networks)

    laplacians = subparsers.add_parser("build-laplacians")
    laplacians.add_argument("--input", required=True)
    laplacians.add_argument("--output-dir", required=True)
    laplacians.add_argument("--dry-run", action="store_true")
    laplacians.set_defaults(func=cmd_build_laplacians)

    splits = subparsers.add_parser("create-splits")
    splits.add_argument("--subjects", required=True)
    splits.add_argument("--output", required=True)
    splits.add_argument("--group-column", default="subject_id")
    splits.add_argument("--strategy", choices=["fixed", "grouped_kfold"], default="fixed")
    splits.add_argument("--test-fraction", type=float, default=0.2)
    splits.add_argument("--validation-fraction", type=float, default=0.2)
    splits.add_argument("--folds", type=int, default=5)
    splits.add_argument("--seed", type=int, default=20260721)
    splits.add_argument("--dry-run", action="store_true")
    splits.set_defaults(func=cmd_create_splits)

    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--output-report", required=True)
    smoke.add_argument("--keep-temporary", action="store_true")
    smoke.set_defaults(func=cmd_smoke_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    configure_logging(args.log_level)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - CLI must return nonzero on failure.
        LOGGER.error("%s", exc)
        return 1
