"""Dataset / Training-Pair schema contract (Track 0 / Requirement 4, owned by Track B).

A Training_Pair is a single JSONL row of shape::

    { "prompt": str, "completion": str, "unit_label": str }

Held_Out_Set rows share exactly this shape (stored in a separate file with no
overlap with the train rows for the same Unit).

This module provides:

* :class:`TrainingPair` — the three-field Pydantic model (Req 4.1).
* :func:`validate_training_pair` — raises :class:`MissingFieldError` naming any
  missing required field (Req 4.4).
* :func:`read_training_pairs` / :func:`write_training_pairs` — the JSONL
  reader/writer pair; the reader validates every row and reports the missing
  field for any bad row (Req 4.4).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from weaveself.contracts.errors import MissingFieldError

# The three required Training_Pair fields, in contract order (Requirement 4.1).
TRAINING_PAIR_FIELDS: tuple[str, ...] = (
    "prompt",
    "completion",
    "unit_label",
)


class TrainingPair(BaseModel):
    """The three-field Training_Pair / Held_Out_Set row model. All required."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    completion: str
    unit_label: str


def validate_training_pair(row: Mapping[str, object] | TrainingPair) -> TrainingPair:
    """Validate a Training_Pair row and return a typed :class:`TrainingPair`.

    If any required field is missing, raises :class:`MissingFieldError` whose
    message names the missing field (Requirement 4.4). Other validation issues
    (wrong type) surface as Pydantic ``ValidationError``.
    """
    if isinstance(row, TrainingPair):
        return row

    if not isinstance(row, Mapping):
        raise TypeError(
            f"training pair must be a mapping or TrainingPair, got {type(row).__name__}"
        )

    # Explicit missing-field detection so we can raise a typed, field-named
    # error. A field that is absent OR present but null counts as missing.
    for field in TRAINING_PAIR_FIELDS:
        if field not in row or row[field] is None:
            raise MissingFieldError(field, context="Training_Pair row")

    return TrainingPair(**dict(row))


def write_training_pairs(
    path: str | os.PathLike[str],
    rows: Iterable[Mapping[str, object] | TrainingPair],
) -> Path:
    """Write Training_Pairs to ``path`` as JSONL and return the path.

    Each row is validated before writing so the emitted file only ever contains
    well-formed Training_Pairs (Req 4.1).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    validated = [validate_training_pair(row) for row in rows]
    with path.open("w", encoding="utf-8") as fh:
        for pair in validated:
            fh.write(json.dumps(pair.model_dump()) + "\n")
    return path


def read_training_pairs(path: str | os.PathLike[str]) -> list[TrainingPair]:
    """Read and validate a JSONL file of Training_Pairs.

    Every row is validated; a row missing ``prompt``, ``completion``, or
    ``unit_label`` raises :class:`MissingFieldError` naming that field
    (Req 4.4). Blank lines are skipped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"training-pair file not found: {path}")

    pairs: list[TrainingPair] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            pairs.append(validate_training_pair(json.loads(line)))
    return pairs
