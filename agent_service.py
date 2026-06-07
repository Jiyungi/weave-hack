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

from agents import adapters, cp, loop, orchestrator, teacher, tools
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
    # Re-attach any externally-registered tools (MCP / HTTP) so their executors
    # exist again after a restart. The minted controllers persist in the control
    # plane independently.
    adapters.reload_into_registry()


@app.exception_handler(ControlPlaneError)
async def _cp_error(_req: Request, exc: ControlPlaneError):
    return JSONResponse(status_code=502, content={"detail": f"control plane: {exc}"})


@app.exception_handler(BrainError)
async def _brain_error(_req: Request, exc: BrainError):
    return JSONResponse(status_code=502, content={"detail": f"brain: {exc}"})


# --- schemas ------------------------------------------------------------------


class ChatTurn(BaseModel):
    role: str
    content: str


class RunReq(BaseModel):
    task: str
    max_delegations: int = 6
    worker_max_steps: int = 6
    worker_max_new_tokens: int = 64
    ensure_seeded: bool = True
    user_id: str | None = None
    chat_id: str | None = None
    history: list[ChatTurn] = []
    force_worker: str | None = None


class AgentRunReq(BaseModel):
    principal: str
    skills: list[str]
    task: str
    compose_skills: list[str] | None = None
    user_id: str | None = None
    max_steps: int = 6
    max_new_tokens: int = 64


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


class McpListReq(BaseModel):
    """Discover the tools advertised by an MCP server (Streamable-HTTP)."""
    server_url: str
    headers: dict[str, str] | None = None


class RegisterExternalReq(BaseModel):
    """Register an external tool (MCP server tool or arbitrary HTTP endpoint) as
    a governed OpenMirror skill: builds the executor, mints a controller (~36s),
    and grants it. ``kind`` selects the adapter.
    """
    kind: str  # "mcp" | "http"
    name: str
    description: str = ""
    grants: dict[str, list[str]] | None = None
    sample_args: list[str] | None = None
    input_schema: dict | None = None
    headers: dict[str, str] | None = None
    # kind == "mcp"
    server_url: str | None = None
    remote_name: str | None = None
    arg_key: str | None = None
    transport: str | None = None  # "http" (Streamable) or "sse" (legacy)
    # kind == "http"
    method: str | None = None
    url_template: str | None = None
    body_template: str | None = None
    encode_arg: bool = False


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
        user_id=req.user_id,
        chat_id=req.chat_id,
        history=[t.model_dump() for t in req.history],
        force_worker=req.force_worker,
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
    examples, source = teacher.mint_examples(tool)
    result = cp.register_tool(
        skill=tool.name,
        examples=examples,
        description=tool.description,
        grants=req.grants,
    )
    return {
        "registered": tool.name,
        "examples_source": source,
        "n_examples": len(examples),
        "control_plane": result,
    }


@app.post("/mcp/list")
def mcp_list(req: McpListReq):
    """List the tools an MCP server advertises (no registration yet)."""
    try:
        resolved, transport, raw = adapters.discover(req.server_url, req.headers)
    except adapters.McpAuthError as e:
        return JSONResponse(status_code=401, content={"detail": str(e)})
    except adapters.AdapterError as e:
        return JSONResponse(status_code=502, content={"detail": str(e)})
    except Exception as e:  # noqa: BLE001 — surface the real cause, never a bare 500
        return JSONResponse(
            status_code=502,
            content={"detail": f"MCP discover failed: {type(e).__name__}: {e}"},
        )
    out = []
    for t in raw:
        schema = t.get("inputSchema") or t.get("input_schema")
        out.append({
            "name": t.get("name"),
            "description": t.get("description", ""),
            "primary_arg": adapters.primary_arg(schema),
            "input_schema": schema,
        })
    return {"server_url": resolved, "transport": transport, "tools": out}


@app.post("/register_external")
def register_external(req: RegisterExternalReq):
    """Register an MCP-server tool or an arbitrary HTTP endpoint as a governed
    skill: live-register the executor, then mint + grant a controller for it."""
    cfg: dict = {"kind": req.kind, "name": req.name, "description": req.description}
    if req.sample_args:
        cfg["sample_args"] = req.sample_args
    if req.input_schema:
        cfg["input_schema"] = req.input_schema
    if req.headers:
        cfg["headers"] = req.headers
    if req.kind == "mcp":
        if not req.server_url:
            return JSONResponse(status_code=400, content={"detail": "mcp requires server_url"})
        cfg["server_url"] = req.server_url
        cfg["remote_name"] = req.remote_name or req.name
        cfg["arg_key"] = req.arg_key or "input"
        cfg["transport"] = req.transport or "http"
    elif req.kind == "http":
        if not req.url_template:
            return JSONResponse(status_code=400, content={"detail": "http requires url_template"})
        cfg["url_template"] = req.url_template
        cfg["method"] = req.method or "GET"
        if req.body_template:
            cfg["body_template"] = req.body_template
        cfg["encode_arg"] = req.encode_arg
    else:
        return JSONResponse(status_code=400, content={"detail": f"unknown kind {req.kind!r}"})

    try:
        tool = adapters.register_config(cfg)
    except adapters.AdapterError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            status_code=400,
            content={"detail": f"register failed: {type(e).__name__}: {e}"},
        )

    examples, source = teacher.mint_examples(
        tool,
        context=(req.description or "").strip() or None,
        arg_key=cfg.get("arg_key") or "input",
        schema=cfg.get("input_schema"),
        extra_args=cfg.get("sample_args"),
    )
    result = cp.register_tool(
        skill=tool.name,
        examples=examples,
        description=tool.description,
        grants=req.grants,
    )
    return {
        "registered": tool.name,
        "kind": req.kind,
        "examples_source": source,
        "n_examples": len(examples),
        "control_plane": result,
    }


@app.get("/tools")
def list_tools():
    """Brain-facing tool list (same shape the agent loop uses)."""
    return {"tools": tools.schemas()}
