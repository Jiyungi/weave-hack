"""Property-based tests for the Data_Pipeline split builder (Track B / Req 11).

Covers:

* task 5.2 — Property 12: training pairs carry valid schema and correct labels
* task 5.3 — Property 13: train and held-out splits do not overlap
* task 5.4 — Property 14: unit inclusion respects the minimum-rows threshold

Each property runs a minimum of 100 generated cases via Hypothesis.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from weaveself.contracts.training_pair import TrainingPair, read_training_pairs
from weaveself.data.pipeline import UnitSource, build_splits

_unit_labels = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8
)
_unit_types = st.sampled_from(["category", "user"])


def _distinct_rows(n: int, prefix: str) -> list[dict]:
    """n distinct prompt/completion rows so de-duplication never changes counts."""
    return [
        {"prompt": f"{prefix}-p{i}", "completion": f"{prefix}-c{i}"} for i in range(n)
    ]


@st.composite
def _unit_sources(draw, *, max_units: int = 5, max_rows: int = 12):
    labels = draw(
        st.lists(_unit_labels, min_size=1, max_size=max_units, unique=True)
    )
    units = []
    for label in labels:
        n = draw(st.integers(min_value=0, max_value=max_rows))
        units.append(
            UnitSource(
                unit_label=label,
                unit_type=draw(_unit_types),
                rows=_distinct_rows(n, label),
            )
        )
    return units


# Feature: weaveself, Property 12: Training pairs carry valid schema and correct unit labels
@settings(max_examples=100)
@given(units=_unit_sources(), min_rows=st.integers(min_value=1, max_value=4))
def test_property_12_valid_schema_and_labels(tmp_path_factory, units, min_rows):
    out_dir = tmp_path_factory.mktemp("splits")
    result = build_splits(units, min_rows=min_rows, out_dir=out_dir)

    source_by_label = {u.unit_label: u for u in units}
    train_rows = read_training_pairs(result.train_path)
    heldout_rows = read_training_pairs(result.heldout_path)

    # Held-out rows live in a file separate from the train rows (Req 4.2, 11.1).
    assert result.train_path != result.heldout_path

    for row in train_rows + heldout_rows:
        # Every emitted row is a valid Training_Pair with string fields (Req 4.1).
        assert isinstance(row, TrainingPair)
        assert isinstance(row.prompt, str)
        assert isinstance(row.completion, str)
        assert isinstance(row.unit_label, str)
        # Its unit_label matches a Unit that was actually included (Req 11.2).
        assert row.unit_label in source_by_label
        assert row.unit_label in result.included_units


# Feature: weaveself, Property 13: Train and held-out splits do not overlap
@settings(max_examples=100)
@given(units=_unit_sources(), min_rows=st.integers(min_value=1, max_value=4))
def test_property_13_no_overlap(tmp_path_factory, units, min_rows):
    out_dir = tmp_path_factory.mktemp("splits")
    result = build_splits(units, min_rows=min_rows, out_dir=out_dir)

    train_rows = read_training_pairs(result.train_path)
    heldout_rows = read_training_pairs(result.heldout_path)

    for label in result.included_units:
        train_keys = {
            (r.prompt, r.completion) for r in train_rows if r.unit_label == label
        }
        heldout_keys = {
            (r.prompt, r.completion) for r in heldout_rows if r.unit_label == label
        }
        # For every Unit, train and held-out rows are disjoint (Req 4.3, 11.3).
        assert train_keys.isdisjoint(heldout_keys)


# Feature: weaveself, Property 14: Unit inclusion respects the minimum-rows threshold
@settings(max_examples=100)
@given(units=_unit_sources(), min_rows=st.integers(min_value=1, max_value=8))
def test_property_14_min_rows_threshold(tmp_path_factory, units, min_rows):
    out_dir = tmp_path_factory.mktemp("splits")
    result = build_splits(units, min_rows=min_rows, out_dir=out_dir)

    expected_included = {u.unit_label for u in units if len(u.rows) >= min_rows}
    expected_excluded = {u.unit_label for u in units if len(u.rows) < min_rows}

    # Exactly the units at/above the threshold are included (Req 11.4).
    assert set(result.included_units) == expected_included
    # Every excluded unit_label is recorded (Req 11.4).
    assert set(result.excluded_units) == expected_excluded
    assert set(result.included_units).isdisjoint(result.excluded_units)
