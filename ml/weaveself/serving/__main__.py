"""``python -m weaveself.serving`` entrypoint.

Delegates to :func:`weaveself.serving.server.run_server` so the real
Inference_API can be launched with either:

    python -m weaveself.serving
    python -m weaveself.serving.server

Both load the repo-root ``.env`` and serve the single resident
:class:`~weaveself.serving.engine.ServingEngine` over uvicorn. The heavy server
imports live inside :func:`run_server`, so importing this module is cheap and
never pulls in ``torch``/``uvicorn`` at collection time.
"""

from __future__ import annotations


def main() -> None:
    from weaveself.serving.server import run_server

    run_server()


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    main()
