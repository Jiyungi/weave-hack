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


class SessionReq(BaseModel):
    principal: str
    skills: list[str]


class ActReq(BaseModel):
    session_id: str
    prompt: str
    max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS


class RevokeReq(BaseModel):
    session_id: str
    skill: str
