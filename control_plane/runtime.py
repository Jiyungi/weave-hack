"""Runtime tool-call guard.

This is the *second* governance layer. Track A's compose/subtract removes a skill
at the model level (the session controller can't emit it). The runtime guard is a
hard, deterministic boundary on top: it parses whatever the model emitted and
blocks any tool call the principal is not authorized for — so even a model-level
leak (or a future, less-clean controller) cannot result in an unauthorized action.
Defense in depth: model-level governance + runtime authorization.
"""
from __future__ import annotations

import re

from .trace import op

# Matches a tool invocation like  weather("Berlin")  ->  "weather".
_TOOL_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")


@op(name="guard.extract_tool_calls")
def extract_tool_calls(text: str) -> list[str]:
    """Ordered, de-duplicated tool names found in a generation."""
    return list(dict.fromkeys(_TOOL_RE.findall(text)))


@op(name="guard.authorize_calls")
def authorize_calls(tool_calls: list[str], authorized_skills: set[str]) -> tuple[list[str], list[str]]:
    """Split detected calls into (allowed, blocked) against the authorized set.

    Skill name == tool function name by convention (skill "weather" emits
    weather(...)). Anything not in the authorized set is blocked.
    """
    allowed = [t for t in tool_calls if t in authorized_skills]
    blocked = [t for t in tool_calls if t not in authorized_skills]
    return allowed, blocked
