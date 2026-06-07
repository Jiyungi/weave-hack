"""Track D entrypoint: real agents governed by the OpenMirror control plane.

Run on the box (in its own shell, after control plane + Track A are up):

    source ~/venv/bin/activate
    # Brain (optional; needed for live runs). Default = local vLLM on 8001.
    # vllm serve Qwen/Qwen2.5-14B-Instruct --port 8001 \
    #     --max-model-len 8192 --gpu-memory-utilization 0.45
    uvicorn agent_service:app --host 0.0.0.0 --port 8200

Endpoints:
    GET  /health      brain + control plane reachability
    GET  /agents      worker roster + their policies (governance view)
    POST /run         orchestrator: decompose -> delegate -> aggregate
    POST /agent_run   run a single governed agent loop (debug)
    POST /revoke      revoke a skill from a session mid-task (passthrough)
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agents import cp, loop, orchestrator, tools
from agents.brain import get_brain, BrainError
from agents.cp import ControlPlaneError
from control_plane import trace


app = FastAPI(title="OpenMirror Agent Orchestrator", version="0.1")

# The dashboard served by the control plane (port 8100) calls this service on
# port 8200. Allow it (and a few other dev origins) without preflight pain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _start_tracing() -> None:
    # Same opt-in tracing as the control plane: a no-op without weave/login,
    # so the trace tree of every /run is visible in the same Weave project.
    trace.init()


@app.exception_handler(ControlPlaneError)
async def _cp_error(_req: Request, exc: ControlPlaneError):
    return JSONResponse(status_code=502, content={"detail": f"control plane: {exc}"})


@app.exception_handler(BrainError)
async def _brain_error(_req: Request, exc: BrainError):
    return JSONResponse(status_code=502, content={"detail": f"brain: {exc}"})


# --- schemas ------------------------------------------------------------------


class RunReq(BaseModel):
    task: str
    max_delegations: int = 4
    worker_max_steps: int = 4
    worker_max_new_tokens: int = 32
    ensure_seeded: bool = True


class AgentRunReq(BaseModel):
    principal: str
    skills: list[str]
    task: str
    compose_skills: list[str] | None = None
    user_id: str | None = None
    max_steps: int = 6
    max_new_tokens: int = 32


class RevokeReq(BaseModel):
    session_id: str
    skill: str


class RegisterToolReq(BaseModel):
    """Register one of the locally-known tools (agents/tools.py) with the
    control plane in one call. The tool's name + training_examples come from
    the local registry, so callers only need the name + optional grants.
    """
    tool_name: str
    grants: dict[str, list[str]] | None = None


# --- endpoints ----------------------------------------------------------------


@app.get("/health")
def health():
    brain = get_brain()
    # Probe the control plane, but never crash /health if it's down --
    # the agent service is useful for inspecting the brain config alone.
    cp_state: dict | str
    try:
        cp_state = cp.health()
    except ControlPlaneError as e:
        cp_state = f"unreachable: {e}"
    return {
        "service": "openmirror-agents",
        "brain": brain.describe(),
        "control_plane": cp_state,
        "control_plane_url": cp.CP_URL,
        "weave_tracing": trace.enabled(),
        "tools_available": [t["name"] for t in tools.schemas()],
    }


@app.get("/agents")
def agents():
    """Worker roster + the policies the control plane has for each."""
    workers = orchestrator.default_workers()
    try:
        snap = cp.state()
    except ControlPlaneError as e:
        return {"workers": [w.__dict__ for w in workers],
                "control_plane_error": str(e)}
    policies = snap.get("policies", {})
    return {
        "workers": [
            {
                "name": w.name,
                "description": w.description,
                "requested_skills": w.requested_skills,
                "policy": policies.get(w.name, []),
            }
            for w in workers
        ],
        "available_skills": sorted(snap.get("skills", {}).keys()),
    }


@app.post("/run")
def run(req: RunReq):
    result = orchestrator.run(
        req.task,
        max_delegations=req.max_delegations,
        worker_max_steps=req.worker_max_steps,
        worker_max_new_tokens=req.worker_max_new_tokens,
        ensure_seeded=req.ensure_seeded,
    )
    return result.to_dict()


@app.post("/agent_run")
def agent_run(req: AgentRunReq):
    result = loop.run(
        principal=req.principal,
        skills=req.skills,
        task=req.task,
        compose_skills=req.compose_skills,
        user_id=req.user_id,
        max_steps=req.max_steps,
        max_new_tokens=req.max_new_tokens,
    )
    return result.to_dict()


@app.post("/revoke")
def revoke(req: RevokeReq):
    """Pass-through to the control plane so a dashboard can revoke mid-run."""
    return cp.revoke(req.session_id, req.skill)


@app.post("/register_tool")
def register_tool(req: RegisterToolReq):
    """Register a known local tool with the control plane (mints a controller)."""
    try:
        tool = tools.get(req.tool_name)
    except tools.ToolError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    return cp.register_tool(
        skill=tool.name,
        examples=tool.training_examples(),
        description=tool.description,
        grants=req.grants,
    )


@app.get("/tools")
def list_tools():
    """Brain-facing tool list (same shape the agent loop uses)."""
    return {"tools": tools.schemas()}
