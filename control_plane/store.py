"""Governance state + operations: skills, policies, sessions, act, revoke.

State lives behind the ``state`` abstraction (in-memory by default, Redis when
REDIS_URL is set) so it can be durable and shared across processes. All mutations
are audited. Every compose/subtract is delegated to Track A so the control plane
never re-implements the gate math — it just decides *which* controllers to
combine for *whom*.
"""
from __future__ import annotations

import os
import threading
import time
import uuid

from . import track_a
from .audit import audit
from .runtime import authorize_calls, extract_tool_calls
from .state import state
from .trace import op


class CPError(RuntimeError):
    """A governance precondition failed (unknown skill/session, not authorized)."""


@op(name="cp.register_skill")
def register_skill(skill: str, controller_id: str, *,
                   mint_args: list[str] | None = None) -> dict:
    state.set_skill(skill, controller_id)
    if mint_args:
        state.set_skill_args(skill, mint_args)
    audit.record("register_skill", skill=skill, controller_id=controller_id,
                 n_probe_args=len(mint_args or []))
    return {"skill": skill, "controller_id": controller_id,
            "probe_args": mint_args or []}


@op(name="cp.train_skill")
def train_skill(skill: str, examples: list[dict]) -> dict:
    """Train a controller on Track A and register it under `skill` in one step."""
    from agents.teacher import args_from_examples

    res = track_a.train(skill, examples)
    mint_args = args_from_examples(skill, examples)
    register_skill(skill, res["controller_id"], mint_args=mint_args)
    return {"skill": skill, **res, "probe_args": mint_args}


@op(name="cp.register_tool")
def register_tool(skill: str, examples: list[dict], description: str = "",
                  grants: dict[str, list[str]] | None = None) -> dict:
    """Committee one-shot: mint controller -> register skill -> extend policies.

    Idempotent on the skill name: re-registering replaces the controller and
    leaves grants untouched (unless explicitly extended again here).
    """
    if not examples:
        raise CPError("no training examples provided")
    minted = train_skill(skill, examples)
    extended: dict[str, list[str]] = {}
    if grants:
        for principal, extra in grants.items():
            current = state.get_policy(principal)
            wanted = sorted(current | set(extra) | {skill})
            # All wanted skills must exist (skill is now registered, extras
            # might not be). set_policy enforces this; surface a clean error
            # by raising CPError ourselves with the offending names.
            unknown = [s for s in wanted if not state.has_skill(s)]
            if unknown:
                raise CPError(
                    f"register_tool: principal {principal!r} grants reference "
                    f"unknown skills {unknown}; register them first"
                )
            state.set_policy(principal, set(wanted))
            audit.record("set_policy", principal=principal, allowed=wanted)
            extended[principal] = wanted
    audit.record("register_tool", skill=skill, description=description,
                 controller_id=minted["controller_id"], grants=list(extended))
    return {
        "skill": skill,
        "description": description,
        "controller_id": minted["controller_id"],
        "loss_first": minted.get("loss_first"),
        "loss_last": minted.get("loss_last"),
        "train_seconds": minted.get("train_seconds"),
        "artifact_bytes": minted.get("artifact_bytes"),
        "policies": extended,
    }


@op(name="cp.set_policy")
def set_policy(principal: str, allowed_skills: list[str]) -> dict:
    unknown = [s for s in allowed_skills if not state.has_skill(s)]
    if unknown:
        raise CPError(f"unknown skills (register/train first): {unknown}")
    allowed = set(allowed_skills)
    state.set_policy(principal, allowed)
    audit.record("set_policy", principal=principal, allowed=sorted(allowed))
    return {"principal": principal, "allowed_skills": sorted(allowed)}


@op(name="cp.revoke_policy")
def revoke_policy(principal: str, skill: str) -> dict:
    """Remove a skill from a principal's policy so new sessions cannot grant it."""
    current = state.get_policy(principal)
    if skill not in current:
        raise CPError(f"skill {skill!r} not in policy for {principal!r} "
                      f"(allowed: {sorted(current)})")
    allowed = current - {skill}
    state.set_policy(principal, allowed)
    audit.record("revoke_policy", principal=principal, skill=skill,
                 allowed=sorted(allowed))
    for sid in state.all_principal_session_ids(principal):
        if state.get_session(sid) is not None:
            try:
                _sync_session_with_policy(sid, principal, sorted(allowed))
            except CPError:
                pass
    return {"principal": principal, "revoked": skill,
            "allowed_skills": sorted(allowed)}


@op(name="cp.personalize")
def personalize(user_id: str, examples: list[dict]) -> dict:
    """Mint/refresh a user's personalization (style) adapter via Track A.

    The memory track calls this on its update cadence with styled (prompt,
    completion) examples (HOW the user likes things, not facts). The resulting
    controller is composed into that user's sessions alongside their tools.
    """
    res = track_a.train(f"user_style-{user_id}", examples)
    state.set_personalization(user_id, res["controller_id"])
    audit.record("personalize", user_id=user_id,
                 controller_id=res["controller_id"], n_examples=len(examples))
    return {"user_id": user_id, **res}


def _compose_ids(ids: list[str], controller_id: str) -> str:
    """Compose the given Track A controller ids into one controller (weights all 1)."""
    res = track_a.compose(ids, [1.0] * len(ids), new_id=controller_id)
    return res["controller_id"]


def _session_scope(session_key: str | None, user_id: str | None) -> str | None:
    return session_key or user_id


@op(name="cp.open_session")
def open_session(principal: str, requested_skills: list[str],
                 compose_skills: list[str] | None = None,
                 user_id: str | None = None,
                 *, reuse: bool = True, session_key: str | None = None) -> dict:
    """Open or reuse a sticky governed session for ``principal``."""
    scope = _session_scope(session_key, user_id)
    if reuse:
        sid = state.get_principal_session(principal, scope)
        if sid and state.get_session(sid) is not None:
            s = state.get_session(sid)
            if s.get("user_id") == user_id:
                _sync_session_with_policy(sid, principal, requested_skills)
                s = get_session(sid)
                if not s["authorized"] and not s.get("bootstrap"):
                    raise CPError(f"principal {principal!r} has no authorized skills "
                                  f"in reused session {sid} (policy: "
                                  f"{sorted(state.get_policy(principal))})")
                denied = [sk for sk in requested_skills if sk not in s["authorized"]]
                audit.record("reuse_session", session_id=sid, principal=principal,
                             authorized=sorted(s["authorized"]),
                             denied=denied, capability=sorted(s.get("capability", set())),
                             user_id=user_id, controller_id=s["controller_id"])
                return _session_response(sid, s, denied=denied)
    return _create_session(principal, requested_skills,
                           compose_skills=compose_skills, user_id=user_id,
                           session_key=session_key)


def _session_response(session_id: str, s: dict, *, denied: list[str] | None = None) -> dict:
    denied = denied or []
    return {
        "session_id": session_id,
        "principal": s["principal"],
        "authorized": sorted(s["authorized"]),
        "denied": denied,
        "capability": sorted(s.get("capability", set())),
        "user_id": s.get("user_id"),
        "personalized": s.get("personalized", False),
        "controller_id": s["controller_id"],
        "reused": True,
    }


def _sync_session_with_policy(session_id: str, principal: str,
                              requested_skills: list[str]) -> None:
    """Align a live session's grants with current policy (+ prior revokes)."""
    policy = state.get_policy(principal)
    target = {sk for sk in requested_skills if sk in policy}
    s = get_session(session_id)
    current = set(s.get("authorized", set()))
    for skill in sorted(current - target):
        try:
            revoke(session_id, skill)
        except CPError:
            pass
    for skill in sorted(target - current):
        if state.has_skill(skill):
            _grant_into_session(skill, session_id)


def _create_bootstrap_session(principal: str, requested_skills: list[str],
                              user_id: str | None, scope: str | None) -> dict:
    """Session with no tool grants yet — worker can REQUEST skills into it."""
    denied = list(requested_skills)
    style_id = state.get_personalization(user_id) if user_id else None
    personalized = style_id is not None
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    controller_id = _compose_ids([style_id], f"{sid}-ctrl") if style_id else None
    state.set_session(sid, {
        "principal": principal,
        "authorized": set(),
        "capability": set(),
        "controller_id": controller_id,
        "user_id": user_id,
        "personalized": personalized,
        "bootstrap": True,
    })
    audit.record("open_session", session_id=sid, principal=principal,
                 authorized=[], denied=denied, capability=[],
                 user_id=user_id, personalized=personalized,
                 controller_id=controller_id, bootstrap=True)
    state.set_principal_session(principal, sid, scope)
    return {
        "session_id": sid, "principal": principal, "authorized": [],
        "denied": denied, "capability": [],
        "user_id": user_id, "personalized": personalized,
        "controller_id": controller_id, "reused": False, "bootstrap": True,
    }


def _create_session(principal: str, requested_skills: list[str],
                    compose_skills: list[str] | None = None,
                    user_id: str | None = None,
                    session_key: str | None = None) -> dict:
    scope = _session_scope(session_key, user_id)
    allowed = state.get_policy(principal)
    authorized = [s for s in requested_skills if s in allowed]
    denied = [s for s in requested_skills if s not in allowed]
    if not authorized:
        return _create_bootstrap_session(principal, requested_skills, user_id, scope)
    # `capability` = what the session controller is actually composed from (the
    # model-level reach). Normally this equals the authorized set, so the model
    # simply cannot do anything it isn't allowed to. compose_skills lets a caller
    # provision a broader controller (shared/over-capable, or a reduce-only skill)
    # to demonstrate that the runtime guard still blocks the excess.
    capability = list(compose_skills) if compose_skills is not None else authorized
    unknown = [s for s in capability if not state.has_skill(s)]
    if unknown:
        raise CPError(f"unknown skills in compose_skills: {unknown}")
    if not capability:
        raise CPError("session has no capability skills to compose")
    # The session controller composes the user's personalization adapter (style,
    # if any) with the capability controllers (tools). Style is not a tool and
    # emits no tool calls, so it rides along the same compose() but never enters
    # the runtime-authorized set.
    style_id = state.get_personalization(user_id) if user_id else None
    personalized = style_id is not None
    compose_ids = ([style_id] if style_id else []) + [state.get_skill(s) for s in capability]
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    controller_id = _compose_ids(compose_ids, f"{sid}-ctrl")
    state.set_session(sid, {"principal": principal, "authorized": set(authorized),
                            "capability": set(capability), "controller_id": controller_id,
                            "user_id": user_id, "personalized": personalized,
                            "bootstrap": False})
    audit.record("open_session", session_id=sid, principal=principal,
                 authorized=authorized, denied=denied, capability=sorted(capability),
                 user_id=user_id, personalized=personalized, controller_id=controller_id)
    state.set_principal_session(principal, sid, scope)
    out = {"session_id": sid, "principal": principal, "authorized": authorized,
           "denied": denied, "capability": sorted(capability),
           "user_id": user_id, "personalized": personalized,
           "controller_id": controller_id, "reused": False}
    return out


def get_session(session_id: str) -> dict:
    s = state.get_session(session_id)
    if s is None:
        raise CPError(f"unknown session {session_id}")
    return s


@op(name="cp.act")
def act(session_id: str, prompt: str, max_new_tokens: int) -> dict:
    s = get_session(session_id)
    out = track_a.execute(s["controller_id"], prompt, max_new_tokens)
    completion = out["completion"]
    calls = extract_tool_calls(completion)
    allowed, blocked = authorize_calls(calls, s["authorized"])
    permitted = not blocked
    audit.record("act", session_id=session_id, principal=s["principal"],
                 prompt=prompt, completion=completion, tool_calls=calls,
                 allowed=allowed, blocked=blocked, permitted=permitted)
    return {
        "session_id": session_id,
        "principal": s["principal"],
        "completion": completion,
        "tool_calls": calls,
        "allowed_calls": allowed,
        "blocked_calls": blocked,
        "permitted": permitted,  # False => runtime guard caught an unauthorized call
        "authorized": sorted(s["authorized"]),
    }


@op(name="cp.act_gate")
def act_gate(session_id: str, skill: str, prompt: str, max_new_tokens: int) -> dict:
    """Run ``/act`` against a skill's *solo* controller, not the session compose.

    Composed session controllers (weather + calendar + …) often fail to emit a
    specific tool call even when each solo controller would. Gate-mode tools probe
    here so the 7B is evaluated on the skill it was minted for, while the
    runtime guard still checks ``skill`` is in the session's authorized set.
    """
    s = get_session(session_id)
    if skill not in s["authorized"]:
        raise CPError(f"skill {skill!r} not authorized in session {session_id}")
    if not state.has_skill(skill):
        raise CPError(f"skill {skill!r} not registered")
    ctrl = state.get_skill(skill)
    out = track_a.execute(ctrl, prompt, max_new_tokens)
    completion = out["completion"]
    calls = extract_tool_calls(completion)
    allowed, blocked = authorize_calls(calls, s["authorized"])
    permitted = not blocked
    audit.record("act_gate", session_id=session_id, principal=s["principal"],
                 skill=skill, controller_id=ctrl, prompt=prompt,
                 completion=completion, tool_calls=calls,
                 allowed=allowed, blocked=blocked, permitted=permitted)
    return {
        "session_id": session_id,
        "principal": s["principal"],
        "skill": skill,
        "completion": completion,
        "tool_calls": calls,
        "allowed_calls": allowed,
        "blocked_calls": blocked,
        "permitted": permitted,
        "authorized": sorted(s["authorized"]),
    }


@op(name="cp.revoke")
def revoke(session_id: str, skill: str) -> dict:
    s = get_session(session_id)
    if skill not in s["authorized"]:
        raise CPError(f"skill {skill!r} is not active in session {session_id}")
    # Model-level revoke: subtract the skill's controller from the session
    # controller (Track A handles the headroom so this is lossless).
    new_id = f"{session_id}-ctrl-rev-{uuid.uuid4().hex[:4]}"
    res = track_a.compose([s["controller_id"], state.get_skill(skill)], [1.0, -1.0], new_id=new_id)
    s["controller_id"] = res["controller_id"]
    s["authorized"].discard(skill)              # runtime-level revoke too
    s.get("capability", set()).discard(skill)   # capability shrinks with it
    state.set_session(session_id, s)            # persist the mutation (Redis write-back)
    audit.record("revoke", session_id=session_id, skill=skill,
                 controller_id=s["controller_id"], authorized=sorted(s["authorized"]))
    return {"session_id": session_id, "revoked": skill,
            "authorized": sorted(s["authorized"]), "controller_id": s["controller_id"]}


# ---------------------------------------------------------------------------
# Capability requests: hybrid human-in-the-loop approval
# ---------------------------------------------------------------------------
#
# A self-improving agent never grants itself a skill. It *requests* one; an
# authority decides. In hybrid mode the authority is a rule + a human:
#   - "safe" skills (read-only, no key) are auto-approved instantly, so the
#     common case stays fast;
#   - "sensitive" skills (need an API key / spend money / have side effects)
#     park as pending until a human approves in the UI.
# Track D flags `sensitive` from the tool's requires_key; these env lists let an
# operator force a skill either way regardless of that hint (defense in depth).


def _force_require() -> set[str]:
    return {s.strip() for s in os.environ.get("OPENMIRROR_REQUIRE_APPROVAL", "").split(",") if s.strip()}


def _force_auto() -> set[str]:
    return {s.strip() for s in os.environ.get("OPENMIRROR_AUTOAPPROVE", "").split(",") if s.strip()}


def _needs_human(skill: str, sensitive: bool) -> bool:
    if skill in _force_require():
        return True
    if skill in _force_auto():
        return False
    return bool(sensitive)


# Serializes decide() so a request can't be approved twice concurrently -- e.g.
# the auto-approve (inside request_capability) racing a human click during the
# ~36s mint window, which would otherwise mint the controller twice. Per-process
# (fine: a single control plane owns the decision path).
_decide_lock = threading.Lock()


def _public_request(rec: dict) -> dict:
    """Request view safe to return/snapshot (drops the bulky examples blob)."""
    out = {k: v for k, v in rec.items() if k != "examples"}
    out["has_examples"] = bool(rec.get("examples"))
    return out


def _grant_into_session(skill: str, session_id: str | None) -> str | None:
    """Compose a granted skill into a live session controller (+1.0) so the
    worker can use it immediately -- the additive mirror of revoke's subtract."""
    if not session_id:
        return None
    s = state.get_session(session_id)
    if s is None or skill in s.get("authorized", set()):
        return None
    skill_ctrl = state.get_skill(skill)
    new_id = f"{session_id}-ctrl-add-{uuid.uuid4().hex[:4]}"
    base = s.get("controller_id")
    if base:
        res = track_a.compose([base, skill_ctrl], [1.0, 1.0], new_id=new_id)
    else:
        user_id = s.get("user_id")
        style_id = state.get_personalization(user_id) if user_id else None
        ids = ([style_id] if style_id else []) + [skill_ctrl]
        res = track_a.compose(ids, [1.0] * len(ids), new_id=new_id)
    s["controller_id"] = res["controller_id"]
    s["authorized"].add(skill)
    s["bootstrap"] = False
    cap = s.get("capability")
    if isinstance(cap, set):
        cap.add(skill)
    state.set_session(session_id, s)
    return res["controller_id"]


@op(name="cp.request_capability")
def request_capability(principal: str, skill: str, *, reason: str = "",
                       session_id: str | None = None, sensitive: bool = False,
                       examples: list[dict] | None = None,
                       description: str = "") -> dict:
    """An agent asks for a skill. Auto-approved if safe, else parked as pending.

    ``examples`` (optional) lets the caller supply teacher-synthesized training
    data so approval can MINT a not-yet-registered skill before granting it.
    """
    rid = f"req-{uuid.uuid4().hex[:8]}"
    rec = {
        "request_id": rid, "principal": principal, "skill": skill,
        "reason": reason, "session_id": session_id, "sensitive": bool(sensitive),
        "description": description, "examples": examples or [],
        "status": "pending", "decided_by": None, "controller_id": None,
        "created": time.time(),
    }
    state.set_request(rid, rec)
    audit.record("request_capability", request_id=rid, principal=principal,
                 skill=skill, sensitive=bool(sensitive), reason=reason,
                 session_id=session_id, mint=bool(examples) and not state.has_skill(skill),
                 n_examples=len(examples or []))
    if not _needs_human(skill, sensitive):
        return _decide(rid, approve=True, decided_by="auto")
    return _public_request(rec)


@op(name="cp.approve_capability")
def approve_capability(request_id: str, *, decided_by: str = "human") -> dict:
    return _decide(request_id, approve=True, decided_by=decided_by)


@op(name="cp.deny_capability")
def deny_capability(request_id: str, *, decided_by: str = "human") -> dict:
    return _decide(request_id, approve=False, decided_by=decided_by)


def _decide(request_id: str, *, approve: bool, decided_by: str) -> dict:
    with _decide_lock:
        return _decide_locked(request_id, approve=approve, decided_by=decided_by)


def _decide_locked(request_id: str, *, approve: bool, decided_by: str) -> dict:
    rec = state.get_request(request_id)
    if rec is None:
        raise CPError(f"unknown capability request {request_id!r}")
    if rec["status"] != "pending":
        return _public_request(rec)  # idempotent: already decided
    skill, principal = rec["skill"], rec["principal"]
    if not approve:
        rec.update(status="denied", decided_by=decided_by)
        state.set_request(request_id, rec)
        audit.record("deny_capability", request_id=request_id,
                     principal=principal, skill=skill, decided_by=decided_by)
        return _public_request(rec)
    # Approved. Case 2: mint a not-yet-registered skill from supplied examples.
    if not state.has_skill(skill):
        if rec.get("examples"):
            train_skill(skill, rec["examples"])
        else:
            raise CPError(f"cannot approve {skill!r}: not registered and no "
                          "training examples were supplied to mint it")
    # Grant policy only when the skill is newly allowed for this principal.
    current = state.get_policy(principal)
    if skill not in current:
        state.set_policy(principal, current | {skill})
        audit.record("set_policy", principal=principal, allowed=sorted(current | {skill}))
    composed = _grant_into_session(skill, rec.get("session_id"))
    rec.update(status="approved", decided_by=decided_by, controller_id=composed)
    state.set_request(request_id, rec)
    audit.record("approve_capability", request_id=request_id, principal=principal,
                 skill=skill, decided_by=decided_by, controller_id=composed)
    return _public_request(rec)


def get_capability_request(request_id: str) -> dict:
    rec = state.get_request(request_id)
    if rec is None:
        raise CPError(f"unknown capability request {request_id!r}")
    return _public_request(rec)


# ---------------------------------------------------------------------------
# Memory track — chat logs → consolidation → user_style adapter
# ---------------------------------------------------------------------------

@op(name="cp.memory_log")
def log_interaction(user_id: str, user: str, assistant: str) -> dict:
    """Append one chat turn for later consolidation (deleted after mint)."""
    user_id = user_id.strip()
    if not user_id:
        raise CPError("user_id required")
    if not user.strip() or not assistant.strip():
        raise CPError("user and assistant messages required")
    n = state.append_interaction(user_id, {"user": user.strip(), "assistant": assistant.strip()})
    audit.record("memory_log", user_id=user_id, count=n)
    return {"user_id": user_id, "logged": True, "pending_interactions": n}


def get_interactions(user_id: str) -> list[dict]:
    return state.get_interactions(user_id)


def clear_interactions(user_id: str) -> int:
    n = state.clear_interactions(user_id)
    audit.record("memory_clear", user_id=user_id, deleted=n)
    return n


@op(name="cp.memory_consolidate")
def consolidate_user(user_id: str) -> dict:
    """Run memory consolidation (curate → personalize → delete logs)."""
    from memory.consolidate import consolidate_user as _run
    return _run(user_id)


def snapshot() -> dict:
    return {
        "skills": state.all_skills(),
        "skill_probe_args": state.all_skill_args(),
        "policies": {p: sorted(v) for p, v in state.all_policies().items()},
        "personalization": state.all_personalization(),
        "sessions": {sid: {"principal": v["principal"],
                           "authorized": sorted(v["authorized"]),
                           "capability": sorted(v.get("capability", set())),
                           "user_id": v.get("user_id"),
                           "personalized": v.get("personalized", False),
                           "controller_id": v["controller_id"]}
                     for sid, v in state.all_sessions().items()},
        "requests": sorted((_public_request(r) for r in state.all_requests().values()),
                           key=lambda r: r.get("created", 0), reverse=True),
        "memory": {
            "pending": {uid: len(state.get_interactions(uid))
                        for uid in state.interaction_users()},
        },
    }
