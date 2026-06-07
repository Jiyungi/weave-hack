"""Unit tests for the Training_Pair / Held_Out_Set contract (Requirement 4).

These are example/edge-case unit tests. The universal property test
(Property 2 / 12) is implemented separately in task 1.4.
"""

from __future__ import annotations

import json

import pytest

from weaveself.contracts import (
    TrainingPair,
    TRAINING_PAIR_FIELDS,
    MissingFieldError,
    read_training_pairs,
    validate_training_pair,
    write_training_pairs,
)


def test_round_trip_preserves_rows(tmp_path, sample_training_pair):
    rows = [
        sample_training_pair,
        {"prompt": "p2", "completion": "c2", "unit_label": "bob"},
    ]
    path = tmp_path / "train.jsonl"
    written = write_training_pairs(path, rows)
    assert written == path
    assert path.exists()

    loaded = read_training_pairs(path)

    assert len(loaded) == len(rows)
    for pair, original in zip(loaded, rows):
        assert isinstance(pair, TrainingPair)
        assert pair.prompt == original["prompt"]
        assert pair.completion == original["completion"]
        assert pair.unit_label == original["unit_label"]


def test_validate_accepts_training_pair_instance(sample_training_pair):
    pair = validate_training_pair(sample_training_pair)
    assert validate_training_pair(pair) is pair


@pytest.mark.parametrize("missing", list(TRAINING_PAIR_FIELDS))
def test_validate_reports_missing_field(sample_training_pair, missing):
    incomplete = {k: v for k, v in sample_training_pair.items() if k != missing}
    with pytest.raises(MissingFieldError) as exc:
        validate_training_pair(incomplete)
    assert exc.value.field_name == missing
    assert missing in str(exc.value)


@pytest.mark.parametrize("missing", list(TRAINING_PAIR_FIELDS))
def test_none_value_treated_as_missing(sample_training_pair, missing):
    sample_training_pair[missing] = None
    with pytest.raises(MissingFieldError) as exc:
        validate_training_pair(sample_training_pair)
    assert exc.value.field_name == missing


def test_read_reports_missing_field_for_bad_row(tmp_path, sample_training_pair):
    path = tmp_path / "train.jsonl"
    good = json.dumps(sample_training_pair)
    bad = json.dumps({"prompt": "p", "completion": "c"})  # missing unit_label
    path.write_text(good + "\n" + bad + "\n", encoding="utf-8")

    with pytest.raises(MissingFieldError) as exc:
        read_training_pairs(path)
    assert exc.value.field_name == "unit_label"


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_training_pairs(tmp_path / "does_not_exist.jsonl")
