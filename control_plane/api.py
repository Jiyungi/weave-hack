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
from .schemas import (ActGateReq, ActReq, ApprovalReq, CapabilityRequestReq, MemoryConsolidateReq,
                      MemoryLogReq, PersonalizeReq, PolicyRevokeReq,
                      PolicyReq, RegisterReq, RegisterSkillReq, RevokeReq,
                      SessionReq, SettingsReq, TrainSkillReq)

app = FastAPI(title="OpenMirror Control Plane", version="0.1")
# Re-parent ops under a caller's trace (Track D -> here) for one unified tree.
app.add_middleware(trace.WeaveContextMiddleware)

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


@app.post("/policy/revoke")
def revoke_policy(req: PolicyRevokeReq):
    """Remove one skill from a principal's policy (blocks future sessions)."""
    return store.revoke_policy(req.principal, req.skill)


@app.post("/personalize")
def personalize(req: PersonalizeReq):
    if not req.examples:
        raise store.CPError("no examples provided")
    return store.personalize(req.user_id, req.examples)


@app.post("/session")
def open_session(req: SessionReq):
    return store.open_session(req.principal, req.skills,
                              compose_skills=req.compose_skills,
                              user_id=req.user_id, reuse=req.reuse,
                              session_key=req.session_key)


@app.post("/act")
def act(req: ActReq):
    return store.act(req.session_id, req.prompt, req.max_new_tokens)


@app.post("/act/gate")
def act_gate(req: ActGateReq):
    return store.act_gate(req.session_id, req.skill, req.prompt, req.max_new_tokens)


@app.post("/revoke")
def revoke(req: RevokeReq):
    return store.revoke(req.session_id, req.skill)


@app.post("/capability/request")
def request_capability(req: CapabilityRequestReq):
    """A self-improving agent asks for a skill. Auto-granted if safe; pending if
    sensitive. Returns the request with its (possibly already-decided) status."""
    return store.request_capability(
        req.principal, req.skill, reason=req.reason, session_id=req.session_id,
        sensitive=req.sensitive, examples=req.examples, description=req.description,
    )


@app.post("/capability/approve")
def approve_capability(req: ApprovalReq):
    return store.approve_capability(req.request_id, decided_by=req.decided_by)


@app.post("/capability/deny")
def deny_capability(req: ApprovalReq):
    return store.deny_capability(req.request_id, decided_by=req.decided_by)


@app.post("/settings")
def update_settings(req: SettingsReq):
    return store.set_auto_approve(req.auto_approve_enabled)


@app.get("/capability/request/{request_id}")
def get_capability_request(request_id: str):
    return store.get_capability_request(request_id)


@app.get("/audit")
def get_audit(n: int = 50):
    return {"events": audit.tail(n)}


@app.post("/memory/log")
def memory_log(req: MemoryLogReq):
    return store.log_interaction(req.user_id, req.user, req.assistant)


@app.post("/memory/consolidate")
def memory_consolidate(req: MemoryConsolidateReq):
    return store.consolidate_user(req.user_id)


@app.get("/memory/pending")
def memory_pending():
    return store.snapshot()["memory"]
