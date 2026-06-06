"""Entrypoint for the Track B control plane.

The implementation lives in the `control_plane` package (config / track_a /
runtime / audit / store / schemas / api). This module re-exports the FastAPI app
so the run command stays stable:

    uvicorn control_plane_service:app --host 0.0.0.0 --port 8100
"""
from control_plane.api import app

__all__ = ["app"]
