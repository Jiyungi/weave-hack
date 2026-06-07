"""FastAPI surface for Track B. Thin: validate -> call store -> return.

Run:  uvicorn control_plane_service:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import config, store, trace
from .audit import audit
from .state import state
from .store import CPError
from .track_a import TrackAError
from .schemas import (ActReq, PersonalizeReq, PolicyReq, RegisterReq,
                      RegisterSkillReq, RevokeReq, SessionReq, TrainSkillReq)

app = FastAPI(title="OpenMirror Control Plane", version="0.1")

_LANDING = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>OpenMirror Control Plane</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #0b0f17; color: #e6edf6;
           display: flex; align-items: center; justify-content: center; min-height: 100vh;
           margin: 0; }
    .box { max-width: 520px; padding: 2rem; border: 1px solid #243044; border-radius: 12px;
           background: #131a26; }
    h1 { font-size: 1.25rem; margin: 0 0 0.5rem; }
    p { color: #8a99b0; font-size: 0.9rem; line-height: 1.5; }
    a { color: #5b9dff; }
    code { background: #0f1521; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <div class="box">
    <h1>OpenMirror Control Plane</h1>
    <p>Track B API is running. The control surface (Track C) is the
       <strong>Next.js + CopilotKit</strong> app on port <code>3000</code>.</p>
    <p><a href="http://localhost:3000">Open the CopilotKit UI →</a></p>
    <p style="margin-top:1.5rem;font-size:0.8rem">
      API docs: <a href="/docs">/docs</a> · health: <a href="/health">/health</a>
    </p>
  </div>
</body>
</html>"""


@app.on_event("startup")
def _start_tracing() -> None:
    """Turn on Weave tracing if it's installed and configured (no-op otherwise)."""
    trace.init()


@app.get("/", response_class=HTMLResponse)
def landing():
    """Pointer to the CopilotKit UI (Track C) on port 3000."""
    return _LANDING


@app.exception_handler(CPError)
async def _cp_error(_req: Request, exc: CPError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(TrackAError)
async def _track_a_error(_req: Request, exc: TrackAError):
    return JSONResponse(status_code=502, content={"detail": f"track A: {exc}"})


@app.get("/health")
def health():
    snap = store.snapshot()
    return {
        "track_a_url": config.TRACK_A_URL,
        "state_backend": state.backend,
        "audit_backend": audit.backend,
        "weave_tracing": trace.enabled(),
        "n_skills": len(snap["skills"]),
        "n_policies": len(snap["policies"]),
        "n_sessions": len(snap["sessions"]),
    }


@app.get("/state")
def get_state():
    return store.snapshot()


@app.post("/skills")
def train_skill(req: TrainSkillReq):
    if not req.examples:
        raise CPError("no examples provided")
    return store.train_skill(req.skill, req.examples)


@app.post("/skills/register")
def register_skill(req: RegisterSkillReq):
    return store.register_skill(req.skill, req.controller_id)


@app.post("/register")
def register_tool(req: RegisterReq):
    """Committee one-shot: mint controller -> register skill -> extend policies.

    Lets an external agent / MCP server bring its own tool to OpenMirror in a
    single call. See ``RegisterReq`` for the payload shape.
    """
    return store.register_tool(req.skill, req.examples,
                               description=req.description, grants=req.grants)


@app.post("/policy")
def set_policy(req: PolicyReq):
    return store.set_policy(req.principal, req.allowed_skills)


@app.post("/personalize")
def personalize(req: PersonalizeReq):
    if not req.examples:
        raise store.CPError("no examples provided")
    return store.personalize(req.user_id, req.examples)


@app.post("/session")
def open_session(req: SessionReq):
    return store.open_session(req.principal, req.skills,
                              compose_skills=req.compose_skills, user_id=req.user_id)


@app.post("/act")
def act(req: ActReq):
    return store.act(req.session_id, req.prompt, req.max_new_tokens)


@app.post("/revoke")
def revoke(req: RevokeReq):
    return store.revoke(req.session_id, req.skill)


@app.get("/audit")
def get_audit(n: int = 50):
    return {"events": audit.tail(n)}
