"""Grouped subject-level splitting."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def _unique_subjects(subject_ids: list[str]) -> list[str]:
    return sorted(dict.fromkeys(str(subject_id) for subject_id in subject_ids))


def validate_group_column(group_column: str) -> None:
    if group_column != "subject_id":
        raise ValueError(
            "splitting by region or visit is prohibited; group_column must be subject_id"
        )


def fixed_train_validation_test_split(
    subject_ids: list[str],
    *,
    test_fraction: float,
    validation_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Create a deterministic subject-level train/validation/test split."""

    subjects = _unique_subjects(subject_ids)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, round(n * test_fraction)) if n >= 3 else 1
    n_validation = max(1, round(n * validation_fraction)) if n >= 3 else 1
    test = set(subjects[:n_test])
    validation = set(subjects[n_test : n_test + n_validation])
    rows = []
    for subject_id in sorted(subjects):
        if subject_id in test:
            partition = "test"
        elif subject_id in validation:
            partition = "validation"
        else:
            partition = "train"
        rows.append({"subject_id": subject_id, "partition": partition, "fold": "", "seed": seed})
    frame = pd.DataFrame(rows)
    validate_no_subject_overlap(frame)
    return frame


def grouped_kfold(subject_ids: list[str], *, n_folds: int, seed: int) -> pd.DataFrame:
    """Assign subjects deterministically to grouped folds."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    subjects = _unique_subjects(subject_ids)
    if n_folds > len(subjects):
        raise ValueError("n_folds cannot exceed subject count")
    rng = random.Random(seed)
    rng.shuffle(subjects)
    rows = [
        {
            "subject_id": subject_id,
            "partition": "cross_validation",
            "fold": index % n_folds,
            "seed": seed,
        }
        for index, subject_id in enumerate(subjects)
    ]
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def validate_no_subject_overlap(split_frame: pd.DataFrame) -> None:
    """Ensure each subject appears in exactly one partition."""

    counts = split_frame.groupby("subject_id")["partition"].nunique()
    if (counts > 1).any():
        raise ValueError("subject overlap detected across partitions")


def write_split(split_frame: pd.DataFrame, output_csv: str | Path) -> None:
    """Write a split file."""

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(destination, index=False)
