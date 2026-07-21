from __future__ import annotations

import pytest

from tau_sr.splitting.grouped import fixed_train_validation_test_split, validate_group_column
from tau_sr.synthetic.fixtures import synthetic_subject_ids


def test_subject_split_contains_no_overlap() -> None:
    split = fixed_train_validation_test_split(
        synthetic_subject_ids(10), test_fraction=0.2, validation_fraction=0.2, seed=1
    )
    assert split.groupby("subject_id")["partition"].nunique().max() == 1


def test_splits_are_deterministic() -> None:
    first = fixed_train_validation_test_split(
        synthetic_subject_ids(10), test_fraction=0.2, validation_fraction=0.2, seed=1
    )
    second = fixed_train_validation_test_split(
        synthetic_subject_ids(10), test_fraction=0.2, validation_fraction=0.2, seed=1
    )
    assert first.equals(second)


def test_refuse_split_by_visit_or_region() -> None:
    with pytest.raises(ValueError):
        validate_group_column("visit_id")
