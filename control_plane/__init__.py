"""Track B - control plane over the Track A controller engine.

Adds capability *governance* on top of the proven compose/subtract math:
  - a skill registry (named skill -> Track A controller id)
  - an authorization policy (principal -> allowed skills)
  - sessions that compose ONLY the authorized skills into a per-session controller
  - a runtime tool-call guard (defense in depth: blocks any unauthorized call
    even if the model emits it)
  - an append-only audit trail (Redis stream if available, else in-memory + file)
  - revocation that recomposes the session via subtract

Talks to Track A over HTTP so the two tracks stay decoupled. Run:
  uvicorn control_plane_service:app --host 0.0.0.0 --port 8100
"""
