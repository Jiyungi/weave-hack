"""Governance state + operations: skills, policies, sessions, act, revoke.

State is in-memory (single-process demo). All mutations are audited. Every
compose/subtract is delegated to Track A so the control plane never re-implements
the gate math — it just decides *which* controllers to combine for *whom*.
"""
from __future__ import annotations

import uuid

from . import track_a
from .audit import audit
from .runtime import authorize_calls, extract_tool_calls


class CPError(RuntimeError):
    """A governance precondition failed (unknown skill/session, not authorized)."""


# skill name -> Track A controller id
_skills: dict[str, str] = {}
# principal -> set of allowed skill names
_policy: dict[str, set[str]] = {}
# session id -> {principal, authorized: set[str], controller_id: str}
_sessions: dict[str, dict] = {}


def register_skill(skill: str, controller_id: str) -> dict:
    _skills[skill] = controller_id
    audit.record("register_skill", skill=skill, controller_id=controller_id)
    return {"skill": skill, "controller_id": controller_id}


def train_skill(skill: str, examples: list[dict]) -> dict:
    """Train a controller on Track A and register it under `skill` in one step."""
    res = track_a.train(skill, examples)
    register_skill(skill, res["controller_id"])
    return {"skill": skill, **res}


def set_policy(principal: str, allowed_skills: list[str]) -> dict:
    unknown = [s for s in allowed_skills if s not in _skills]
    if unknown:
        raise CPError(f"unknown skills (register/train first): {unknown}")
    _policy[principal] = set(allowed_skills)
    audit.record("set_policy", principal=principal, allowed=sorted(_policy[principal]))
    return {"principal": principal, "allowed_skills": sorted(_policy[principal])}


def _compose_for(skills: list[str], controller_id: str) -> str:
    """Compose the given skills' controllers into one session controller via Track A."""
    ids = [_skills[s] for s in skills]
    res = track_a.compose(ids, [1.0] * len(ids), new_id=controller_id)
    return res["controller_id"]


def open_session(principal: str, requested_skills: list[str]) -> dict:
    allowed = _policy.get(principal, set())
    authorized = [s for s in requested_skills if s in allowed]
    denied = [s for s in requested_skills if s not in allowed]
    if not authorized:
        raise CPError(f"principal {principal!r} authorized for none of "
                      f"{requested_skills} (allowed: {sorted(allowed)})")
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    controller_id = _compose_for(authorized, f"{sid}-ctrl")
    _sessions[sid] = {"principal": principal, "authorized": set(authorized),
                      "controller_id": controller_id}
    audit.record("open_session", session_id=sid, principal=principal,
                 authorized=authorized, denied=denied, controller_id=controller_id)
    return {"session_id": sid, "principal": principal, "authorized": authorized,
            "denied": denied, "controller_id": controller_id}


def get_session(session_id: str) -> dict:
    s = _sessions.get(session_id)
    if s is None:
        raise CPError(f"unknown session {session_id}")
    return s


def act(session_id: str, prompt: str, max_new_tokens: int) -> dict:
    s = get_session(session_id)
    out = track_a.execute(s["controller_id"], prompt, max_new_tokens)
    completion = out["completion"]
    calls = extract_tool_calls(completion)
    allowed, blocked = authorize_calls(calls, s["authorized"])
    audit.record("act", session_id=session_id, principal=s["principal"],
                 prompt=prompt, completion=completion, tool_calls=calls,
                 allowed=allowed, blocked=blocked)
    return {
        "session_id": session_id,
        "principal": s["principal"],
        "completion": completion,
        "tool_calls": calls,
        "allowed_calls": allowed,
        "blocked_calls": blocked,
        "authorized": sorted(s["authorized"]),
    }


def revoke(session_id: str, skill: str) -> dict:
    s = get_session(session_id)
    if skill not in s["authorized"]:
        raise CPError(f"skill {skill!r} is not active in session {session_id}")
    # Model-level revoke: subtract the skill's controller from the session
    # controller (Track A handles the headroom so this is lossless).
    new_id = f"{session_id}-ctrl-rev-{uuid.uuid4().hex[:4]}"
    res = track_a.compose([s["controller_id"], _skills[skill]], [1.0, -1.0], new_id=new_id)
    s["controller_id"] = res["controller_id"]
    s["authorized"].discard(skill)  # runtime-level revoke too
    audit.record("revoke", session_id=session_id, skill=skill,
                 controller_id=s["controller_id"], authorized=sorted(s["authorized"]))
    return {"session_id": session_id, "revoked": skill,
            "authorized": sorted(s["authorized"]), "controller_id": s["controller_id"]}


def snapshot() -> dict:
    return {
        "skills": dict(_skills),
        "policies": {p: sorted(v) for p, v in _policy.items()},
        "sessions": {sid: {"principal": v["principal"],
                           "authorized": sorted(v["authorized"]),
                           "controller_id": v["controller_id"]}
                     for sid, v in _sessions.items()},
    }
