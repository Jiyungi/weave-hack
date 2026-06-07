"""Eval artifact schema contract (Track 0 / Requirement 5, owned by Track B).

The Weave_Eval emits an ``eval_results.json`` artifact consumed by the Track C
Dashboard. This module locks its shape as Pydantic v2 models so the dashboard can
render proof visuals against a stable file before the real eval runs.

The artifact shape (Requirement 5)::

    {
      "perplexity": { "base": 0.0, "adapter": 0.0, "context_memory": 0.0 },
      "confusion_matrix": { "labels": ["..."], "matrix": [[0.0]] },
      "size_bytes": { "nktmirror": 0, "lora": 0 },
      "examples": [
        { "prompt": "...", "base": "...", "adapter": "...", "reference": "..." }
      ]
    }

This module provides:

* :class:`Perplexity` — ``base`` / ``adapter`` / ``context_memory`` numbers (Req 5.2).
* :class:`ConfusionMatrix` — ``labels`` + square ``matrix``; a model_validator
  enforces the square-matrix invariant (Req 5.3) and raises
  :class:`InvalidConfusionMatrixError` on a dimension mismatch.
* :class:`SizeBytes` — integer ``nktmirror`` / ``lora`` sizes (Req 5.4).
* :class:`EvalExample` — ``prompt`` / ``base`` / ``adapter`` / ``reference`` strings (Req 5.5).
* :class:`EvalResults` — the four-field top-level artifact model (Req 5.1).
* :func:`validate_eval_results` — raises :class:`MissingFieldError` naming any
  missing required top-level field (consistent with the other contracts).
* :func:`write_eval_results` / :func:`read_eval_results` — the JSON writer/reader
  pair, mirroring ``adapter_file.py`` read/write style.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from weaveself.contracts.errors import MissingFieldError

# The four required top-level Eval_Results fields, in contract order (Req 5.1).
EVAL_RESULTS_FIELDS: tuple[str, ...] = (
    "perplexity",
    "confusion_matrix",
    "size_bytes",
    "examples",
)


class InvalidConfusionMatrixError(ValueError):
    """Raised when a Confusion_Matrix violates the square-matrix invariant.

    The matrix dimensions MUST equal the count of ``labels``: the number of rows
    must equal ``len(labels)`` and every row's length must equal ``len(labels)``
    (Requirement 5.3). The message describes the mismatch.
    """


class Perplexity(BaseModel):
    """Held-out perplexity under the base, the adapter, and the context-memory
    baseline (Requirement 5.2)."""

    model_config = ConfigDict(extra="forbid")

    base: float
    adapter: float
    context_memory: float


class ConfusionMatrix(BaseModel):
    """Cross-unit identification matrix (Requirement 5.3).

    ``labels`` is the ordered list of Units; ``matrix`` is a square numeric grid
    whose dimensions equal the count of labels (rows are the true Unit, columns
    are the predicted Unit). The square-matrix invariant is enforced at
    construction by :meth:`_check_square`.
    """

    model_config = ConfigDict(extra="forbid")

    labels: list[str]
    matrix: list[list[float]]

    def __init__(self, **data: Any) -> None:
        # Pydantic wraps a ValueError raised inside a validator into a
        # ValidationError. Unwrap our square-matrix invariant so callers see a
        # clear, typed InvalidConfusionMatrixError at construction (Req 5.3).
        try:
            super().__init__(**data)
        except ValidationError as exc:
            for err in exc.errors():
                original = err.get("ctx", {}).get("error")
                if isinstance(original, InvalidConfusionMatrixError):
                    raise original from None
            raise

    @model_validator(mode="after")
    def _check_square(self) -> "ConfusionMatrix":
        n = len(self.labels)
        row_count = len(self.matrix)
        if row_count != n:
            raise InvalidConfusionMatrixError(
                f"confusion matrix must have {n} rows to match {n} label(s), "
                f"got {row_count} row(s)"
            )
        for i, row in enumerate(self.matrix):
            if len(row) != n:
                raise InvalidConfusionMatrixError(
                    f"confusion matrix row {i} must have {n} column(s) to match "
                    f"{n} label(s), got {len(row)} column(s)"
                )
        return self


class SizeBytes(BaseModel):
    """Adapter size comparison: NKT-Mirror versus LoRA, in bytes (Requirement 5.4)."""

    model_config = ConfigDict(extra="forbid")

    nktmirror: int
    lora: int


class EvalExample(BaseModel):
    """A single base-versus-adapter example with its reference text (Requirement 5.5)."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    base: str
    adapter: str
    reference: str


class EvalResults(BaseModel):
    """The ``eval_results.json`` artifact: the four-field top-level model (Req 5.1)."""

    model_config = ConfigDict(extra="forbid")

    perplexity: Perplexity
    confusion_matrix: ConfusionMatrix
    size_bytes: SizeBytes
    examples: list[EvalExample]


def validate_eval_results(data: Mapping[str, object] | EvalResults) -> EvalResults:
    """Validate an Eval_Results object and return a typed :class:`EvalResults`.

    If any required top-level field is missing, raises :class:`MissingFieldError`
    whose message names the missing field (consistent with the other contracts).
    A non-square ``confusion_matrix.matrix`` raises
    :class:`InvalidConfusionMatrixError` (Req 5.3); other validation issues
    (wrong type) surface as Pydantic ``ValidationError``.
    """
    if isinstance(data, EvalResults):
        return data

    if not isinstance(data, Mapping):
        raise TypeError(
            f"eval results must be a mapping or EvalResults, got {type(data).__name__}"
        )

    # Explicit missing-field detection so we can raise a typed, field-named
    # error. A field that is absent OR present but null counts as missing.
    for field in EVAL_RESULTS_FIELDS:
        if field not in data or data[field] is None:
            raise MissingFieldError(field, context="Eval_Results artifact")

    return EvalResults(**dict(data))


def write_eval_results(
    path: str | os.PathLike[str],
    results: Mapping[str, object] | EvalResults,
) -> Path:
    """Write an Eval_Results artifact to ``path`` as JSON and return the path.

    The results are validated before writing so the emitted file only ever
    contains a well-formed, schema-conformant artifact (Req 5.1).
    """
    validated = validate_eval_results(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated.model_dump(), indent=2), encoding="utf-8")
    return path


def read_eval_results(path: str | os.PathLike[str]) -> EvalResults:
    """Read and validate an ``eval_results.json`` artifact.

    A file missing a required top-level field raises :class:`MissingFieldError`
    naming that field; a non-square confusion matrix raises
    :class:`InvalidConfusionMatrixError` (Req 5.3).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"eval results file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    return validate_eval_results(raw)
