"""Data_Pipeline split builder (Track B / Requirement 11).

Loads per-Unit source data and emits ``train.jsonl`` and ``heldout.jsonl`` of
:class:`~weaveself.contracts.training_pair.TrainingPair` rows. Guarantees:

* every emitted row carries the ``unit_label`` of the Unit it was derived from
  (Req 11.2 / Property 12);
* a Unit's train rows and held-out rows never overlap (Req 11.3, 4.3 /
  Property 13);
* a Unit is included in the demo set iff its source-row count is ``>= min_rows``,
  and every excluded ``unit_label`` is recorded (Req 11.4 / Property 14).

The held-out split is written to a file separate from the train rows (Req 4.2,
11.1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from weaveself.contracts.training_pair import (
    TrainingPair,
    write_training_pairs,
)


@dataclass(frozen=True)
class UnitSource:
    """Raw source data for a single Unit.

    ``rows`` are raw prompt/completion candidates (each at least a mapping with
    ``prompt`` and ``completion``); the pipeline assigns ``unit_label`` so that
    every emitted Training_Pair is tagged with this Unit (Req 11.2).
    """

    unit_label: str
    unit_type: str
    rows: Sequence[Mapping[str, object]]


@dataclass
class SplitResult:
    """Outcome of :func:`build_splits`."""

    train_path: Path
    heldout_path: Path
    included_units: list[str] = field(default_factory=list)
    excluded_units: list[str] = field(default_factory=list)
    train_counts: dict[str, int] = field(default_factory=dict)
    heldout_counts: dict[str, int] = field(default_factory=dict)

    @property
    def train_total(self) -> int:
        return sum(self.train_counts.values())

    @property
    def heldout_total(self) -> int:
        return sum(self.heldout_counts.values())


def _to_training_pair(row: Mapping[str, object], unit_label: str) -> TrainingPair:
    """Build a Training_Pair from a raw source row, forcing ``unit_label``.

    Any ``unit_label`` present on the source row is overridden so the emitted
    pair always matches the Unit it was derived from (Req 11.2 / Property 12).
    """
    return TrainingPair(
        prompt=str(row["prompt"]),
        completion=str(row["completion"]),
        unit_label=unit_label,
    )


def _dedupe_rows(
    rows: Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Drop rows with a duplicate (prompt, completion) key, preserving order.

    De-duplicating before the split makes the no-overlap guarantee (Property 13)
    hold by content, not just by index: two byte-identical rows can never land
    one in train and one in held-out.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[Mapping[str, object]] = []
    for row in rows:
        key = (str(row["prompt"]), str(row["completion"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _split_indices(n: int, heldout_fraction: float) -> int:
    """Return the held-out size for ``n`` deduped rows.

    Guarantees at least one held-out and at least one train row whenever
    ``n >= 2`` so both files receive rows for every included Unit.
    """
    if n <= 1:
        return 0
    heldout = int(n * heldout_fraction)
    heldout = max(1, heldout)
    heldout = min(heldout, n - 1)
    return heldout


def build_splits(
    source: Iterable[UnitSource],
    min_rows: int,
    out_dir: str | os.PathLike[str],
    *,
    heldout_fraction: float = 0.2,
) -> SplitResult:
    """Build per-Unit train/held-out splits and write them as JSONL.

    Each included Unit contributes rows to ``train.jsonl`` and ``heldout.jsonl``
    (written under ``out_dir``); the two files never share a row for the same
    Unit. A Unit whose source-row count is below ``min_rows`` is excluded and
    its ``unit_label`` is recorded (Req 11.4).

    Args:
        source: the per-Unit source data.
        min_rows: inclusion threshold on the source-row count (Req 11.4).
        out_dir: directory to write ``train.jsonl`` / ``heldout.jsonl`` into.
        heldout_fraction: target fraction of (deduped) rows held out per Unit.

    Returns:
        A :class:`SplitResult` describing the written files and per-Unit counts.
    """
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in the open interval (0, 1)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    heldout_path = out_dir / "heldout.jsonl"

    train_rows: list[TrainingPair] = []
    heldout_rows: list[TrainingPair] = []
    result = SplitResult(train_path=train_path, heldout_path=heldout_path)

    for unit in source:
        source_row_count = len(unit.rows)
        if source_row_count < min_rows:
            result.excluded_units.append(unit.unit_label)
            continue

        unique = _dedupe_rows(unit.rows)
        n = len(unique)
        heldout_size = _split_indices(n, heldout_fraction)

        unit_heldout = unique[:heldout_size]
        unit_train = unique[heldout_size:]

        result.included_units.append(unit.unit_label)
        result.train_counts[unit.unit_label] = len(unit_train)
        result.heldout_counts[unit.unit_label] = len(unit_heldout)

        train_rows.extend(_to_training_pair(r, unit.unit_label) for r in unit_train)
        heldout_rows.extend(
            _to_training_pair(r, unit.unit_label) for r in unit_heldout
        )

    write_training_pairs(train_path, train_rows)
    write_training_pairs(heldout_path, heldout_rows)
    return result
