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
