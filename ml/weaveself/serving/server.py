"""Real Inference_API server entrypoint (Track A).

Run the FastAPI Inference_API against the *real* HuggingFace Base_Model:

    cd ml
    python -m weaveself.serving.server

Configuration is read from the repo-root ``.env`` (loaded via ``python-dotenv``)
with environment variables taking precedence:

* ``BASE_MODEL_ID``        — instruct Base_Model id (default Qwen2.5-1.5B-Instruct).
* ``WEAVESELF_BACKEND``    — ``hf`` selects the real :class:`HFBackend`; anything
  else (or unset) selects the dependency-free :class:`StubBackend`.
* ``TORCH_DEVICE``         — ``cuda``/``cuda:0``/``cpu``/``mps``; defaults to
  ``cuda`` when a CUDA device is available, else ``cpu``.
* ``MODEL_DTYPE``          — ``float32``/``float16``/``bfloat16`` (HFBackend only).
* ``ADAPTERS_DIR``         — directory of ``adapter_<id>`` pairs to serve.
* ``INFERENCE_API_HOST`` / ``INFERENCE_API_PORT`` — uvicorn bind address.

The single resident :class:`ServingEngine` loads the Base_Model exactly once
(Req 7.1) and is shared by every request; CORS is enabled for the local browser
UI origins so a dashboard/chat front-end can call the API directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from weaveself.serving.api import create_app
from weaveself.serving.backend import HFBackend, ModelBackend
from weaveself.serving.engine import ServingEngine

# Local browser UI origins allowed to call the Inference_API from a browser.
UI_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


def _repo_root() -> Path:
    """Repo root is two levels above the ``ml/`` package (``ml/weaveself/..``)."""
    # server.py -> serving -> weaveself -> ml -> repo root
    return Path(__file__).resolve().parents[3]


def _load_env() -> None:
    """Load the repo-root ``.env`` into the process environment if present.

    Existing environment variables win over ``.env`` values so an explicit
    ``TORCH_DEVICE=cuda python -m ...`` overrides the file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv not installed: rely on real env only.
        return
    env_path = _repo_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _default_device() -> str:
    """Pick ``cuda`` when available, else ``cpu`` (import-safe if torch absent)."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_path(value: str) -> str:
    """Resolve a possibly repo-root-relative path (``./data/adapters``)."""
    p = Path(value)
    if not p.is_absolute():
        p = _repo_root() / p
    return str(p)


def build_backend() -> ModelBackend:
    """Construct the real serving backend (production is HFBackend only).

    ``WEAVESELF_BACKEND`` must be ``hf``; the server never silently serves a
    stub model. Set ``WEAVESELF_BACKEND=hf`` (default in ``.env``).
    """
    backend_kind = os.environ.get("WEAVESELF_BACKEND", "hf").strip().lower()
    if backend_kind != "hf":
        raise RuntimeError(
            f"WEAVESELF_BACKEND={backend_kind!r} is not allowed in the server; "
            "production serving requires the real model (WEAVESELF_BACKEND=hf)."
        )
    device = os.environ.get("TORCH_DEVICE", "").strip() or _default_device()
    dtype = os.environ.get("MODEL_DTYPE", "").strip() or None
    return HFBackend(device=device, torch_dtype=dtype)


def build_engine() -> ServingEngine:
    """Build the single resident :class:`ServingEngine` from the environment."""
    base_model_id = os.environ.get(
        "BASE_MODEL_ID", HFBackend.DEFAULT_BASE_MODEL
    ).strip()
    adapters_dir = _resolve_path(
        os.environ.get("ADAPTERS_DIR", "./data/adapters").strip()
    )
    backend = build_backend()
    return ServingEngine(base_model_id, backend=backend, adapters_dir=adapters_dir)


def build_server_app(engine: ServingEngine | None = None) -> FastAPI:
    """Build the CORS-enabled FastAPI app around a resident engine."""
    if engine is None:
        engine = build_engine()
    app = create_app(engine=engine)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(UI_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def run_server() -> None:
    """Console entrypoint: load config, build the engine, serve with uvicorn."""
    _load_env()
    import uvicorn

    host = os.environ.get("INFERENCE_API_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("INFERENCE_API_PORT", "8000").strip() or "8000")

    backend_kind = os.environ.get("WEAVESELF_BACKEND", "stub").strip().lower()
    base_model_id = os.environ.get(
        "BASE_MODEL_ID", HFBackend.DEFAULT_BASE_MODEL
    ).strip()
    device = (
        os.environ.get("TORCH_DEVICE", "").strip() or _default_device()
        if backend_kind == "hf"
        else "n/a (stub)"
    )
    print(
        f"[weaveself] starting Inference_API: backend={backend_kind} "
        f"base_model={base_model_id} device={device} bind={host}:{port}",
        flush=True,
    )
    # Build the engine now (loads the Base_Model exactly once) so a slow model
    # load happens before uvicorn reports the server as ready.
    app = build_server_app()
    print("[weaveself] Base_Model loaded; serving.", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")


# ``_load_env`` is invoked here so importing the module under uvicorn's factory
# mode (``weaveself.serving.server:build_server_app``) also honors ``.env``.
_load_env()

# Module-level ASGI app for ``uvicorn weaveself.serving.server:app`` usage.
# Constructed lazily only when imported as the ASGI target, not on every import,
# to avoid loading a multi-GB model during unit-test collection.
if os.environ.get("WEAVESELF_SERVE_APP") == "1":  # pragma: no cover
    app = build_server_app()


if __name__ == "__main__":  # pragma: no cover
    run_server()
