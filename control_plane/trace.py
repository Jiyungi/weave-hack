"""Optional Weave (W&B) tracing for the control plane.

This is a *zero-risk* integration: decorating a function with ``@op`` is a no-op
until tracing is turned on by a successful ``init()``. So the control plane runs
exactly as before with no Weave installed and no W&B login.

When Weave *is* available and configured, ``init()`` lights up a full trace tree
on the W&B dashboard:

    open_session -> compose (Track A) ............ capability composed
                 -> policy filter ................. authorized vs denied
    act          -> execute (Track A) ............. raw generation
                 -> guard (runtime) .............. allowed vs blocked
    revoke       -> compose [+1,-1] (Track A) ..... lossless subtraction

That trace tree *is* the visualization of how grant/revoke/compose behave at the
model level, with inputs, outputs, and latency captured automatically.

Enable it by installing ``weave`` and logging in (``WANDB_API_KEY`` or
``wandb login``). Disable explicitly with ``WEAVE_DISABLE=1``. Override the
project name with ``WEAVE_PROJECT`` (default: ``OpenMirror``).
"""
from __future__ import annotations

import contextlib
import functools
import os

_weave = None
_ENABLED = False


def init() -> bool:
    """Initialize Weave if installed and configured. Returns True if tracing is on.

    Any failure (no weave, no W&B login, offline) is swallowed and tracing stays
    off, so this is always safe to call at startup.
    """
    global _weave, _ENABLED
    if os.environ.get("WEAVE_DISABLE"):
        _ENABLED = False
        return False
    project = os.environ.get("WEAVE_PROJECT", "OpenMirror")
    try:
        import weave  # optional dependency

        weave.init(project)
        _weave = weave
        _ENABLED = True
    except Exception:
        _weave = None
        _ENABLED = False
    return _ENABLED


def enabled() -> bool:
    return _ENABLED


def current_call():
    """The Weave call currently executing, or None if tracing is off / no call."""
    if not _ENABLED or _weave is None:
        return None
    try:
        return _weave.get_current_call()
    except Exception:
        return None


def trace_headers() -> dict:
    """Headers that propagate the current trace to a downstream HTTP service.

    Inject these into outgoing requests; the receiving service's
    ``WeaveContextMiddleware`` re-parents its ops under this call, so a single
    trace tree spans process boundaries (Track D -> Track B -> Track A).
    """
    call = current_call()
    if call is None:
        return {}
    try:
        return {"x-weave-trace-id": str(call.trace_id), "x-weave-parent-id": str(call.id)}
    except Exception:
        return {}


@contextlib.contextmanager
def remote_parent(trace_id: str | None, parent_id: str | None):
    """Server-side: nest subsequent ops under a remote parent call.

    Reconstructs a stand-in parent ``Call`` from the propagated ids and pushes it
    onto Weave's call stack. A no-op when tracing is off or ids are missing, and
    swallows any failure so request handling is never affected.
    """
    if not _ENABLED or _weave is None or not trace_id or not parent_id:
        yield
        return
    try:
        from weave.trace.context import call_context as _cc
        from weave.trace.context import weave_client_context as _wcc
        from weave.trace.weave_client import Call as _Call
        try:
            project_id = _wcc.get_weave_client()._project_id()
        except Exception:
            project_id = ""
        parent = _Call(_op_name="remote.parent", trace_id=trace_id,
                       project_id=project_id, parent_id=None, inputs={}, id=parent_id)
        with _cc.set_call_stack([parent]):
            yield
    except Exception:
        yield


class WeaveContextMiddleware:
    """Pure-ASGI middleware that re-parents this service's ops under a remote
    caller's trace, propagated via ``x-weave-*`` headers.

    Pure ASGI (not Starlette ``BaseHTTPMiddleware``) so the contextvar set here
    propagates into the sync route handler's threadpool call — BaseHTTPMiddleware
    runs the endpoint in a separate task and would drop it.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not _ENABLED:
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        tid = headers.get("x-weave-trace-id")
        pid = headers.get("x-weave-parent-id")
        if not (tid and pid):
            await self.app(scope, receive, send)
            return
        with remote_parent(tid, pid):
            await self.app(scope, receive, send)


def attributes(values: dict):
    """Context manager attaching searchable metadata to the enclosing trace.

    Tags every op call inside the ``with`` block with ``values`` (e.g. the
    principal, session, or task) so traces are filterable in the Weave UI. A
    no-op when tracing is off, so it's always safe to wrap a code path.
    """
    if not _ENABLED or _weave is None:
        return contextlib.nullcontext()
    try:
        return _weave.attributes(values)
    except Exception:
        return contextlib.nullcontext()


def op(fn=None, *, name=None):
    """Mark a function as a traced op.

    Safe no-op until ``init()`` succeeds. The underlying ``weave.op`` wrapper is
    created lazily on first call (so decoration order vs. ``init()`` never
    matters) and then cached.
    """

    def decorate(f):
        cache: dict = {}

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not _ENABLED or _weave is None:
                return f(*args, **kwargs)
            traced = cache.get("op")
            if traced is None:
                try:
                    traced = _weave.op(f, name=name) if name else _weave.op(f)
                except TypeError:
                    traced = _weave.op(f)
                cache["op"] = traced
            return traced(*args, **kwargs)

        return wrapper

    return decorate(fn) if fn is not None else decorate
