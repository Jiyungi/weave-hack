"""Track A — NTK-Mirror controller engine.

Submodules:
  config       env-driven settings
  model        single frozen base model (lazy, shared)
  controllers  train / compose / inspect / pair / generate
  evals        evaluate / diagnose / forgetting / jailbreak
  schemas      pydantic request models
  api          FastAPI app (import as: from engine.api import app)

Kept light so non-web callers (e.g. the smoke test) can import controllers/
evals without pulling in FastAPI.
"""

import os as _os
import sys as _sys

# ntkmirror's upstream pip packaging is broken (builds an empty "UNKNOWN"
# package), so it must be used from a clone. If a clone exists at
# ~/ntkmirror_src, put its src/ on the path so `import ntkmirror` works without
# requiring PYTHONPATH to be set for the uvicorn service.
_ntk_src = _os.path.expanduser("~/ntkmirror_src/src")
if _os.path.isdir(_ntk_src) and _ntk_src not in _sys.path:
    _sys.path.insert(0, _ntk_src)
