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


def _skip_string(text: str, i: int) -> int:
    """Advance past a single- or double-quoted string starting at *i* (on the quote)."""
    quote = text[i]
    i += 1
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n:
            i += 2
            continue
        if text[i] == quote:
            return i + 1
        i += 1
    return n


@op(name="guard.extract_tool_calls")
def extract_tool_calls(text: str) -> list[str]:
    """Ordered, de-duplicated tool names found in a generation.

    Ignores ``name(`` inside quoted string arguments so e.g.
    ``python("print(2)")`` registers only ``python``, not ``print``.
    """
    calls: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in '"\'':
            i = _skip_string(text, i)
            continue
        m = _TOOL_RE.match(text, i)
        if m:
            calls.append(m.group(1))
            i = m.end()  # past '('
            depth = 1
            while i < n and depth:
                if text[i] in '"\'':
                    i = _skip_string(text, i)
                    continue
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            continue
        i += 1
    return list(dict.fromkeys(calls))


@op(name="guard.authorize_calls")
def authorize_calls(tool_calls: list[str], authorized_skills: set[str]) -> tuple[list[str], list[str]]:
    """Split detected calls into (allowed, blocked) against the authorized set.

    Skill name == tool function name by convention (skill "weather" emits
    weather(...)). Anything not in the authorized set is blocked.
    """
    allowed = [t for t in tool_calls if t in authorized_skills]
    blocked = [t for t in tool_calls if t not in authorized_skills]
    return allowed, blocked
