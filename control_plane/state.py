"""Durable governance state: skills, policies, personalization, sessions.

All state lives in Redis (sponsor integration). ``REDIS_URL`` must be set and
reachable at startup — see ``redis_client.get_redis()``.

Redis layout (decode_responses=True):
  cp:skills           hash  skill        -> controller_id
  cp:skill_args       hash  skill        -> json list of args from last mint
  cp:policy           hash  principal    -> json list of allowed skills
  cp:personalization  hash  user_id      -> controller_id
  cp:sessions         hash  session_id   -> json blob (sets stored as lists)
  cp:principal_sessions hash  principal:scope -> session_id (sticky reuse; scope = session_key or user_id)
  cp:requests         hash  request_id   -> json blob
  cp:interactions     hash  user_id      -> json list of chat turns (memory)
"""
from __future__ import annotations

import json

from .redis_client import get_redis

_SKILLS = "cp:skills"
_SKILL_ARGS = "cp:skill_args"
_POLICY = "cp:policy"
_PERS = "cp:personalization"
_SESS = "cp:sessions"
_PRINC_SESS = "cp:principal_sessions"
_REQ = "cp:requests"
_INTERACTIONS = "cp:interactions"

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
        self._redis = get_redis()
        self.backend = "redis"

    def set_skill(self, skill: str, controller_id: str) -> None:
        self._redis.hset(_SKILLS, skill, controller_id)

    def set_skill_args(self, skill: str, args: list[str]) -> None:
        self._redis.hset(_SKILL_ARGS, skill, json.dumps(args))

    def get_skill_args(self, skill: str) -> list[str]:
        raw = self._redis.hget(_SKILL_ARGS, skill)
        return json.loads(raw) if raw else []

    def all_skill_args(self) -> dict[str, list[str]]:
        return {k: json.loads(v) for k, v in self._redis.hgetall(_SKILL_ARGS).items()}

    def get_skill(self, skill: str) -> str | None:
        return self._redis.hget(_SKILLS, skill)

    def has_skill(self, skill: str) -> bool:
        return bool(self._redis.hexists(_SKILLS, skill))

    def all_skills(self) -> dict[str, str]:
        return dict(self._redis.hgetall(_SKILLS))

    def set_policy(self, principal: str, allowed: set[str]) -> None:
        self._redis.hset(_POLICY, principal, json.dumps(sorted(allowed)))

    def get_policy(self, principal: str) -> set[str]:
        raw = self._redis.hget(_POLICY, principal)
        return set(json.loads(raw)) if raw else set()

    def all_policies(self) -> dict[str, set[str]]:
        return {p: set(json.loads(v)) for p, v in self._redis.hgetall(_POLICY).items()}

    def set_personalization(self, user_id: str, controller_id: str) -> None:
        self._redis.hset(_PERS, user_id, controller_id)

    def get_personalization(self, user_id: str) -> str | None:
        return self._redis.hget(_PERS, user_id)

    def all_personalization(self) -> dict[str, str]:
        return dict(self._redis.hgetall(_PERS))

    def set_session(self, session_id: str, session: dict) -> None:
        self._redis.hset(_SESS, session_id, _ser_session(session))

    def get_session(self, session_id: str) -> dict | None:
        raw = self._redis.hget(_SESS, session_id)
        return _deser_session(raw) if raw else None

    def all_sessions(self) -> dict[str, dict]:
        return {sid: _deser_session(v) for sid, v in self._redis.hgetall(_SESS).items()}

    def _principal_session_key(self, principal: str, scope: str | None) -> str:
        return f"{principal}:{scope or ''}"

    def get_principal_session(self, principal: str, scope: str | None = None) -> str | None:
        return self._redis.hget(_PRINC_SESS, self._principal_session_key(principal, scope))

    def set_principal_session(self, principal: str, session_id: str,
                              scope: str | None = None) -> None:
        self._redis.hset(_PRINC_SESS, self._principal_session_key(principal, scope),
                         session_id)

    def clear_principal_session(self, principal: str, scope: str | None = None) -> None:
        self._redis.hdel(_PRINC_SESS, self._principal_session_key(principal, scope))

    def all_principal_session_ids(self, principal: str) -> list[str]:
        """All sticky session ids for ``principal`` (every chat/user scope)."""
        seen: set[str] = set()
        out: list[str] = []
        for key, sid in self._redis.hgetall(_PRINC_SESS).items():
            if key.startswith(f"{principal}:"):
                if sid not in seen:
                    seen.add(sid)
                    out.append(sid)
        return out

    def set_request(self, request_id: str, req: dict) -> None:
        self._redis.hset(_REQ, request_id, json.dumps(req))

    def get_request(self, request_id: str) -> dict | None:
        raw = self._redis.hget(_REQ, request_id)
        return json.loads(raw) if raw else None

    def all_requests(self) -> dict[str, dict]:
        return {rid: json.loads(v) for rid, v in self._redis.hgetall(_REQ).items()}

    def append_interaction(self, user_id: str, interaction: dict) -> int:
        raw = self._redis.hget(_INTERACTIONS, user_id)
        items = json.loads(raw) if raw else []
        items.append(interaction)
        self._redis.hset(_INTERACTIONS, user_id, json.dumps(items))
        return len(items)

    def get_interactions(self, user_id: str) -> list[dict]:
        raw = self._redis.hget(_INTERACTIONS, user_id)
        return json.loads(raw) if raw else []

    def clear_interactions(self, user_id: str) -> int:
        n = len(self.get_interactions(user_id))
        self._redis.hdel(_INTERACTIONS, user_id)
        return n

    def interaction_users(self) -> list[str]:
        return sorted(self._redis.hkeys(_INTERACTIONS))


state = State()
