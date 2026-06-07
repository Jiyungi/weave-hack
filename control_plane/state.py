"""Durable governance state: skills, policies, personalization, sessions.

State was originally module-level dicts (single-process only). This abstraction
keeps that exact behavior as the default, but transparently uses Redis when
``REDIS_URL`` is set and reachable -- which makes the state **durable** (survives
a control-plane restart) and **shared** (multiple processes / agents see the same
policies and sessions). That shared view is the prerequisite for a multi-agent
orchestrator where several agents act under one governance authority.

Backends, mirroring ``audit.py``:
  1. Redis (hashes) if REDIS_URL is set and reachable  [sponsor integration]
  2. in-memory dicts (always available, used as fallback)

Redis layout (all decode_responses=True):
  cp:skills           hash  skill        -> controller_id
  cp:policy           hash  principal    -> json list of allowed skills
  cp:personalization  hash  user_id      -> controller_id
  cp:sessions         hash  session_id   -> json blob (sets stored as lists)

Sessions carry set-valued fields (``authorized``, ``capability``); they're
serialized to sorted lists for Redis and rehydrated to sets on read, so callers
in ``store.py`` always work with sets regardless of backend.
"""
from __future__ import annotations

import json

from . import config

_SKILLS = "cp:skills"
_POLICY = "cp:policy"
_PERS = "cp:personalization"
_SESS = "cp:sessions"
_REQ = "cp:requests"

_SESSION_SET_FIELDS = ("authorized", "capability")


def _ser_session(session: dict) -> str:
    out = dict(session)
    for f in _SESSION_SET_FIELDS:
        if isinstance(out.get(f), set):
            out[f] = sorted(out[f])
    return json.dumps(out)


def _deser_session(raw: str) -> dict:
    out = json.loads(raw)
    for f in _SESSION_SET_FIELDS:
        if f in out and not isinstance(out[f], set):
            out[f] = set(out[f])
    return out


class State:
    def __init__(self) -> None:
        self._redis = None
        self.backend = "memory"
        # In-memory fallbacks (also the live store when Redis is absent).
        self._skills: dict[str, str] = {}
        self._policy: dict[str, set[str]] = {}
        self._personalization: dict[str, str] = {}
        self._sessions: dict[str, dict] = {}
        self._requests: dict[str, dict] = {}
        if config.REDIS_URL:
            try:
                import redis  # optional dependency

                client = redis.from_url(config.REDIS_URL, decode_responses=True)
                client.ping()
                self._redis = client
                self.backend = "redis"
            except Exception:
                self._redis = None

    # ----- skills -------------------------------------------------------------
    def set_skill(self, skill: str, controller_id: str) -> None:
        if self._redis is not None:
            self._redis.hset(_SKILLS, skill, controller_id)
        else:
            self._skills[skill] = controller_id

    def get_skill(self, skill: str) -> str | None:
        if self._redis is not None:
            return self._redis.hget(_SKILLS, skill)
        return self._skills.get(skill)

    def has_skill(self, skill: str) -> bool:
        if self._redis is not None:
            return bool(self._redis.hexists(_SKILLS, skill))
        return skill in self._skills

    def all_skills(self) -> dict[str, str]:
        if self._redis is not None:
            return dict(self._redis.hgetall(_SKILLS))
        return dict(self._skills)

    # ----- policies -----------------------------------------------------------
    def set_policy(self, principal: str, allowed: set[str]) -> None:
        if self._redis is not None:
            self._redis.hset(_POLICY, principal, json.dumps(sorted(allowed)))
        else:
            self._policy[principal] = set(allowed)

    def get_policy(self, principal: str) -> set[str]:
        if self._redis is not None:
            raw = self._redis.hget(_POLICY, principal)
            return set(json.loads(raw)) if raw else set()
        return set(self._policy.get(principal, set()))

    def all_policies(self) -> dict[str, set[str]]:
        if self._redis is not None:
            return {p: set(json.loads(v)) for p, v in self._redis.hgetall(_POLICY).items()}
        return {p: set(v) for p, v in self._policy.items()}

    # ----- personalization ----------------------------------------------------
    def set_personalization(self, user_id: str, controller_id: str) -> None:
        if self._redis is not None:
            self._redis.hset(_PERS, user_id, controller_id)
        else:
            self._personalization[user_id] = controller_id

    def get_personalization(self, user_id: str) -> str | None:
        if self._redis is not None:
            return self._redis.hget(_PERS, user_id)
        return self._personalization.get(user_id)

    def all_personalization(self) -> dict[str, str]:
        if self._redis is not None:
            return dict(self._redis.hgetall(_PERS))
        return dict(self._personalization)

    # ----- sessions -----------------------------------------------------------
    def set_session(self, session_id: str, session: dict) -> None:
        if self._redis is not None:
            self._redis.hset(_SESS, session_id, _ser_session(session))
        else:
            self._sessions[session_id] = session

    def get_session(self, session_id: str) -> dict | None:
        if self._redis is not None:
            raw = self._redis.hget(_SESS, session_id)
            return _deser_session(raw) if raw else None
        return self._sessions.get(session_id)

    def all_sessions(self) -> dict[str, dict]:
        if self._redis is not None:
            return {sid: _deser_session(v) for sid, v in self._redis.hgetall(_SESS).items()}
        return dict(self._sessions)

    # ----- capability requests (human-in-the-loop approval) -------------------
    def set_request(self, request_id: str, req: dict) -> None:
        if self._redis is not None:
            self._redis.hset(_REQ, request_id, json.dumps(req))
        else:
            self._requests[request_id] = req

    def get_request(self, request_id: str) -> dict | None:
        if self._redis is not None:
            raw = self._redis.hget(_REQ, request_id)
            return json.loads(raw) if raw else None
        return self._requests.get(request_id)

    def all_requests(self) -> dict[str, dict]:
        if self._redis is not None:
            return {rid: json.loads(v) for rid, v in self._redis.hgetall(_REQ).items()}
        return dict(self._requests)


state = State()
