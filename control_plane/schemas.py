"""Pydantic request models for the control-plane HTTP layer."""
from __future__ import annotations

from pydantic import BaseModel, Field

from . import config


class TrainSkillReq(BaseModel):
    skill: str
    examples: list[dict] = Field(..., description="[{prompt, completion}, ...]")


class RegisterSkillReq(BaseModel):
    skill: str
    controller_id: str


class PolicyReq(BaseModel):
    principal: str
    allowed_skills: list[str]


class PersonalizeReq(BaseModel):
    user_id: str
    examples: list[dict] = Field(..., description="[{prompt, completion}, ...] styled to the user")


class SessionReq(BaseModel):
    principal: str
    skills: list[str]
    # Optional: compose this user's personalization (style) adapter into the
    # session alongside the authorized tools. No-op if the user has none yet.
    user_id: str | None = None
    # Advanced/demo: the skills actually composed into the session controller
    # (the model-level capability). Defaults to the authorized set. When this is
    # broader than policy -- e.g. a shared/over-capable controller, or a
    # REDUCE-only skill that subtraction cannot fully erase -- the runtime guard
    # is what still blocks the unauthorized calls (defense in depth).
    compose_skills: list[str] | None = None


class ActReq(BaseModel):
    session_id: str
    prompt: str
    max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS


class RevokeReq(BaseModel):
    session_id: str
    skill: str
