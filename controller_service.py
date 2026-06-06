"""Entrypoint for the Track A controller engine.

The implementation lives in the `engine` package (config / model / controllers /
evals / schemas / api). This module just re-exports the FastAPI app so the run
command and tooling stay stable:

    uvicorn controller_service:app --host 0.0.0.0 --port 8000
"""
from engine.api import app

__all__ = ["app"]
