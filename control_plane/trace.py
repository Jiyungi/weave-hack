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
