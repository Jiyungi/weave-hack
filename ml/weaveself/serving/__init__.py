"""Track A — Serving Engine and Inference API.

Exposes the resident :class:`ServingEngine` (Req 6.1, 7.1), the
:class:`ModelBackend` abstraction with its real (:class:`HFBackend`) and test
(:class:`StubBackend`) implementations, the gate/result value types, and the
:class:`AdapterNotLoadable` serving error (Req 7.5).
"""

from weaveself.serving.backend import (
    GateTensors,
    Generation,
    HFBackend,
    ModelBackend,
    ScoreResult,
    StubBackend,
)
from weaveself.serving.engine import ServingEngine
from weaveself.serving.errors import AdapterNotLoadable

__all__ = [
    "ServingEngine",
    "ModelBackend",
    "HFBackend",
    "StubBackend",
    "GateTensors",
    "Generation",
    "ScoreResult",
    "AdapterNotLoadable",
]

# The FastAPI Inference_API app factory (Req 8). ``fastapi`` lives in the
# optional ``api`` extra, so importing the serving package never hard-requires
# it: if FastAPI is not installed, ``create_app`` is unavailable but the engine,
# backends and errors above still import cleanly.
try:  # pragma: no cover - import guard depends on optional extra
    from weaveself.serving.api import create_app

    __all__.append("create_app")
except ImportError:  # pragma: no cover - fastapi not installed
    pass
