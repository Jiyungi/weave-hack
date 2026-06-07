"""Thin HTTP client for the OpenMirror control plane (Track B).

Mirrors ``control_plane/track_a.py`` -- stdlib only, no extra deps for the
agents layer. Read CP_URL from env so the orchestrator can run in a separate
process than the control plane (the multi-process / shared-governance setup
the Redis state layer was built for).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from control_plane.trace import op, trace_headers


CP_URL = os.environ.get("CP_URL", "http://localhost:8100")


class ControlPlaneError(RuntimeError):
    """The control plane returned an error or was unreachable."""


def _request(method: str, path: str, body: dict | None = None, timeout: float = 1800) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    headers.update(trace_headers())  # propagate Weave trace across the HTTP hop
    req = urllib.request.Request(CP_URL + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ControlPlaneError(f"{method} {path} -> {e.code}: {e.read().decode()}") from e
    except urllib.error.URLError as e:
        raise ControlPlaneError(f"{method} {path} unreachable at {CP_URL} ({e})") from e


@op(name="cp_client.open_session")
def open_session(principal: str, skills: list[str],
                 compose_skills: list[str] | None = None,
                 user_id: str | None = None) -> dict:
    body: dict = {"principal": principal, "skills": skills}
    if compose_skills is not None:
        body["compose_skills"] = compose_skills
    if user_id is not None:
        body["user_id"] = user_id
    return _request("POST", "/session", body)


@op(name="cp_client.act")
def act(session_id: str, prompt: str, max_new_tokens: int = 16) -> dict:
    return _request("POST", "/act", {
        "session_id": session_id,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
    })


@op(name="cp_client.act_gate")
def act_gate(session_id: str, skill: str, prompt: str, max_new_tokens: int = 16) -> dict:
    return _request("POST", "/act/gate", {
        "session_id": session_id,
        "skill": skill,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
    })


@op(name="cp_client.revoke")
def revoke(session_id: str, skill: str) -> dict:
    return _request("POST", "/revoke", {"session_id": session_id, "skill": skill})


@op(name="cp_client.set_policy")
def set_policy(principal: str, allowed_skills: list[str]) -> dict:
    return _request("POST", "/policy",
                    {"principal": principal, "allowed_skills": allowed_skills})


@op(name="cp_client.register_tool")
def register_tool(skill: str, examples: list[dict], description: str = "",
                  grants: dict[str, list[str]] | None = None) -> dict:
    body: dict = {"skill": skill, "examples": examples, "description": description}
    if grants is not None:
        body["grants"] = grants
    return _request("POST", "/register", body)


@op(name="cp_client.request_capability")
def request_capability(principal: str, skill: str, *, reason: str = "",
                       session_id: str | None = None, sensitive: bool = False,
                       examples: list[dict] | None = None,
                       description: str = "") -> dict:
    body: dict = {"principal": principal, "skill": skill, "reason": reason,
                  "sensitive": sensitive, "description": description}
    if session_id is not None:
        body["session_id"] = session_id
    if examples is not None:
        body["examples"] = examples
    return _request("POST", "/capability/request", body)


@op(name="cp_client.approve_capability")
def approve_capability(request_id: str, decided_by: str = "human") -> dict:
    return _request("POST", "/capability/approve",
                    {"request_id": request_id, "decided_by": decided_by})


@op(name="cp_client.deny_capability")
def deny_capability(request_id: str, decided_by: str = "human") -> dict:
    return _request("POST", "/capability/deny",
                    {"request_id": request_id, "decided_by": decided_by})


def get_capability_request(request_id: str) -> dict:
    return _request("GET", f"/capability/request/{request_id}")


def state() -> dict:
    return _request("GET", "/state")


def health() -> dict:
    return _request("GET", "/health")


@op(name="cp_client.memory_log")
def log_interaction(user_id: str, user: str, assistant: str) -> dict:
    return _request("POST", "/memory/log",
                    {"user_id": user_id, "user": user, "assistant": assistant})


@op(name="cp_client.memory_consolidate")
def consolidate_user(user_id: str) -> dict:
    return _request("POST", "/memory/consolidate", {"user_id": user_id}, timeout=1800)
