"""FastAPI Inference_API service (Track A / Requirements 8, 9.3).

This module wires the four Requirement 2 endpoints to the resident
:class:`~weaveself.serving.engine.ServingEngine` and the
:func:`~weaveself.training.train_adapter` function:

* ``POST /generate`` -> :class:`GenerateResponse` — generate text under an
  optional adapter, populating ``text``, ``tokens`` and a measured
  ``latency_ms`` (Req 8.1). A null ``adapter_id`` routes to the pure
  Base_Model (Req 2.5).
* ``POST /score``    -> :class:`ScoreResponse`    — teacher-forced ``nll`` and a
  non-negative ``perplexity`` over the supplied ``target`` (Req 8.2).
* ``GET  /adapters`` -> ``list[str]``              — the currently loadable
  ``adapter_id`` values (Req 8.3).
* ``POST /train``    -> :class:`TrainResponse`     — invoke ``train_adapter``
  with the request fields; the returned ``adapter_path`` is identical to what
  the direct ``train_adapter(...)`` call would return for the same inputs
  (Req 9.3).

The app holds ONE resident :class:`ServingEngine` (the Base_Model is loaded
exactly once, Req 7.1). The default engine uses the dependency-free
:class:`~weaveself.serving.backend.StubBackend`, so the app is fully testable
without ``torch``/GPU.

Error handling:

* A malformed body (missing or wrong-typed field) is rejected by Pydantic and
  FastAPI returns HTTP 422 whose body names the offending field (Req 8.4).
* An unknown ``adapter_id`` raises :class:`AdapterNotLoadable`; the API returns
  HTTP 404 whose ``detail`` names the offending ``adapter_id`` (Req 7.5).
* :class:`DatasetNotReadable` -> HTTP 404 naming the path;
  :class:`InsufficientTrainingData` -> HTTP 400 (Req 9.4, 9.5).
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from weaveself.contracts import (
    GenerateRequest,
    GenerateResponse,
    ScoreRequest,
    ScoreResponse,
    TrainRequest,
    TrainResponse,
)
from weaveself.serving.engine import ServingEngine
from weaveself.serving.errors import AdapterNotLoadable
from weaveself.training import train_adapter
from weaveself.training.errors import DatasetNotReadable, InsufficientTrainingData


def create_app(
    engine: ServingEngine | None = None,
    adapters_dir: str | None = None,
    base_model_id: str = "stub-base",
) -> FastAPI:
    """Build the Inference_API FastAPI app around one resident engine.

    Args:
        engine: An already-constructed :class:`ServingEngine`. When ``None`` a
            new engine is created with the dependency-free
            :class:`StubBackend` so the app runs without ``torch``/GPU. The
            Base_Model is loaded exactly once when the engine is constructed
            (Req 7.1).
        adapters_dir: Directory of ``adapter_<id>`` pairs, used only when
            ``engine`` is ``None`` (passed through to the new engine).
        base_model_id: Base_Model id for the default engine.

    Returns:
        A configured :class:`FastAPI` application.
    """
    if engine is None:
        engine = ServingEngine(base_model_id, adapters_dir=adapters_dir)

    app = FastAPI(title="WeaveSelf Inference_API", version="0.1.0")
    app.state.engine = engine

    # -- error handlers -----------------------------------------------------

    @app.exception_handler(AdapterNotLoadable)
    async def _adapter_not_loadable_handler(
        _request: Request, exc: AdapterNotLoadable
    ) -> JSONResponse:
        # Surface Req 7.5: the response body NAMES the missing adapter_id.
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(DatasetNotReadable)
    async def _dataset_not_readable_handler(
        _request: Request, exc: DatasetNotReadable
    ) -> JSONResponse:
        # Req 9.4: name the unreadable dataset path.
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InsufficientTrainingData)
    async def _insufficient_training_data_handler(
        _request: Request, exc: InsufficientTrainingData
    ) -> JSONResponse:
        # Req 9.5: zero-row dataset is a client error.
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # -- endpoints ----------------------------------------------------------

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        # Measure real wall-clock latency around the engine call (Req 8.1).
        start = time.perf_counter()
        result = engine.generate(req.prompt, req.adapter_id, req.max_new_tokens)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return GenerateResponse(
            text=result.text,
            tokens=result.tokens,
            latency_ms=max(0, latency_ms),
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        result = engine.score(req.prompt, req.target, req.adapter_id)
        return ScoreResponse(perplexity=result.perplexity, nll=result.nll)

    @app.get("/adapters", response_model=list[str])
    def adapters() -> list[str]:
        return engine.list_adapters()

    @app.post("/train", response_model=TrainResponse)
    def train(req: TrainRequest) -> TrainResponse:
        # Equivalence to the direct call (Req 9.3): pass the request fields
        # straight through to train_adapter; the returned adapter_path is what
        # the direct call would return for the same inputs.
        adapter_path = train_adapter(
            req.dataset_path, req.unit_label, req.unit_type
        )
        return TrainResponse(adapter_path=adapter_path)

    return app
