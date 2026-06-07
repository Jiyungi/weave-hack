"""Inference API schema contract (Track 0 / Requirement 2, owned by Track A).

These are the frozen Pydantic v2 request/response shapes for the four
Inference_API endpoints. They are schema definitions ONLY — the FastAPI app and
endpoints are built separately (task 3.7). Locking the shapes here lets Track B
(eval scoring / train) and Track C (chat) mock against stable contracts before
Track A serving is ready.

Endpoints and shapes (Requirement 2):

* ``POST /generate`` — :class:`GenerateRequest` -> :class:`GenerateResponse` (Req 2.1)
* ``POST /score``    — :class:`ScoreRequest`    -> :class:`ScoreResponse`    (Req 2.2)
* ``GET  /adapters`` — :class:`AdaptersResponse` (a ``list[str]`` of adapter ids, Req 2.3)
* ``POST /train``    — :class:`TrainRequest`    -> :class:`TrainResponse`    (Req 2.4)

On both :class:`GenerateRequest` and :class:`ScoreRequest`, ``adapter_id`` is
nullable with a default of ``None``; a null ``adapter_id`` routes to the pure
Base_Model with no adapter applied (Requirement 2.5). Every model uses
``extra="forbid"`` so unknown fields are rejected, and omitting a required field
or supplying a wrong-typed field raises a Pydantic ``ValidationError`` naming the
offending field (supports the Requirement 8.4 contract).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, RootModel

# ---------------------------------------------------------------------------
# /generate (Requirement 2.1)
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request body for ``POST /generate`` (Req 2.1).

    ``adapter_id`` is nullable and defaults to ``None``; null routes to the pure
    Base_Model with no gate tensors applied (Req 2.5).
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    adapter_id: str | None = None
    max_new_tokens: int


class GenerateResponse(BaseModel):
    """Response body for ``POST /generate`` (Req 2.1)."""

    model_config = ConfigDict(extra="forbid")

    text: str
    tokens: int
    latency_ms: int


# ---------------------------------------------------------------------------
# /score (Requirement 2.2)
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    """Request body for ``POST /score`` (Req 2.2).

    ``adapter_id`` is nullable and defaults to ``None``; null routes to the pure
    Base_Model with no gate tensors applied (Req 2.5).
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    target: str
    adapter_id: str | None = None


class ScoreResponse(BaseModel):
    """Response body for ``POST /score`` (Req 2.2)."""

    model_config = ConfigDict(extra="forbid")

    perplexity: float
    nll: float


# ---------------------------------------------------------------------------
# /adapters (Requirement 2.3)
# ---------------------------------------------------------------------------


class AdaptersResponse(RootModel[list[str]]):
    """Response body for ``GET /adapters`` — a list of loadable ``adapter_id``
    strings (Req 2.3)."""


# ---------------------------------------------------------------------------
# /train (Requirement 2.4)
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    """Request body for ``POST /train`` (Req 2.4).

    Triggers ``train_adapter(dataset_path, unit_label, unit_type)``. ``unit_type``
    is constrained to the Unit literal set, matching the Adapter_File contract.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_path: str
    unit_label: str
    unit_type: Literal["category", "user"]


class TrainResponse(BaseModel):
    """Response body for ``POST /train`` — the resulting ``adapter_path`` (Req 2.4)."""

    model_config = ConfigDict(extra="forbid")

    adapter_path: str
