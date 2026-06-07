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


class CapabilityRequestReq(BaseModel):
    """A (self-improving) agent requests a skill it lacks. Hybrid approval:
    auto-granted if safe, parked as pending if ``sensitive``. ``examples`` lets
    approval MINT a not-yet-registered skill before granting it.
    """
    principal: str
    skill: str
    reason: str = ""
    session_id: str | None = None
    sensitive: bool = False
    description: str = ""
    examples: list[dict] | None = None


class ApprovalReq(BaseModel):
    request_id: str
    decided_by: str = "human"


class RegisterReq(BaseModel):
    """One-shot 'committee' registration: mint -> register -> grant.

    Provided so an external agent or MCP server can bring its own tool to
    OpenMirror in one call. The control plane mints the controller on Track A
    (~36 s, lossless gate fit), registers the skill, then optionally extends
    each named principal's policy to include this skill. Idempotent on the
    skill name (re-registration replaces the controller).
    """
    skill: str
    description: str = ""
    examples: list[dict] = Field(..., description="[{prompt, completion}, ...]")
    grants: dict[str, list[str]] | None = Field(
        default=None,
        description="Optional principal -> additional skills to add to their "
                    "policy. The newly-registered skill is appended to each "
                    "listed principal even if they had no policy yet.",
    )
