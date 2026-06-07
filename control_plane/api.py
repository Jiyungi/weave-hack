"""FastAPI surface for Track B. Thin: validate -> call store -> return.

Run:  uvicorn control_plane_service:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import config, store, trace
from .audit import audit
from .store import CPError
from .track_a import TrackAError
from .schemas import (ActReq, PersonalizeReq, PolicyReq, RegisterSkillReq,
                      RevokeReq, SessionReq, TrainSkillReq)

app = FastAPI(title="OpenMirror Control Plane", version="0.1")

_DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"


@app.on_event("startup")
def _start_tracing() -> None:
    """Turn on Weave tracing if it's installed and configured (no-op otherwise)."""
    trace.init()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Track C — single-file governance dashboard (served same-origin)."""
    return _DASHBOARD.read_text(encoding="utf-8")


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
        "audit_backend": audit.backend,
        "weave_tracing": trace.enabled(),
        "n_skills": len(snap["skills"]),
        "n_policies": len(snap["policies"]),
        "n_sessions": len(snap["sessions"]),
    }


@app.get("/state")
def state():
    return store.snapshot()


@app.post("/skills")
def train_skill(req: TrainSkillReq):
    if not req.examples:
        raise CPError("no examples provided")
    return store.train_skill(req.skill, req.examples)


@app.post("/skills/register")
def register_skill(req: RegisterSkillReq):
    return store.register_skill(req.skill, req.controller_id)


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
