"""Brain-synthesized training data for minting controllers.

A controller is only as good as the examples it's trained on. The static
sample-arg lists in ``adapters.py`` (``_QUERY_ARGS`` etc.) are arxiv/search-
shaped, so for an *arbitrary* MCP / HTTP tool they teach a domain-mismatched
mapping -- e.g. a weather tool trained on "large language models" as its arg.
This module instead asks the brain to produce realistic, diverse argument
values *from the tool's own name + description + JSON schema*, so registration
generalizes to any tool rather than just the ones we hard-coded for.

The same generator is the seed of on-demand skill acquisition: when a worker
lacks a capability, synthesize examples here, mint a controller on Track A,
compose it into the live session, and retry.

Everything here is best-effort: callers wrap it and fall back to the schema /
heuristic args so registration never hard-fails when the brain is offline.
"""
from __future__ import annotations

import json
import re

from control_plane.trace import op

from .brain import Brain, get_brain


_ARG_SYS = (
    "You generate training data for a tool-using language model. Given a tool's "
    "name, description, and the single parameter it takes, you output realistic, "
    "diverse example VALUES for that parameter -- the kind a real user request "
    "would actually supply. Output ONLY a JSON array of strings, nothing else."
)

_ARG_USER = """Tool name: {name}
Tool description: {description}
Parameter name: {arg_key}
Parameter spec (JSON Schema fragment, may be empty): {arg_spec}

Produce {n} realistic, diverse values for the parameter "{arg_key}".
- Each value must be a plausible real argument for THIS specific tool, not a placeholder like "example" or "test".
- Vary the length and specifics; prefer natural multi-word values where appropriate so the model learns to quote them.
- Respect the parameter's type / format / enum if the spec gives one.
- Output ONLY a JSON array of exactly {n} strings."""


def _parse_str_array(text: str) -> list[str]:
    """Pull a list of strings from a brain response, tolerant of fences/prose."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
    # Fall back to non-empty lines, stripping bullets/numbering/quotes.
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip().lstrip("-*").strip()
        s = re.sub(r"^\d+[.)]\s*", "", s).strip().strip('"').strip("'").strip()
        if s and not s.startswith("```"):
            out.append(s)
    return out


@op(name="teacher.synthesize_args")
def synthesize_args(
    name: str,
    description: str,
    arg_key: str,
    schema: dict | None = None,
    *,
    n: int = 8,
    brain: Brain | None = None,
) -> list[str]:
    """Ask the brain for realistic example values for one tool parameter.

    Raises ``BrainError`` (from ``brain.chat``) if the brain is unreachable so
    the caller can fall back to schema/heuristic args. Returns a de-duplicated,
    order-preserving list capped at ``n``.
    """
    brain = brain or get_brain()
    schema = schema or {}
    props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
    arg_spec = props.get(arg_key, {}) if isinstance(props, dict) else {}
    msgs = [
        {"role": "system", "content": _ARG_SYS},
        {"role": "user", "content": _ARG_USER.format(
            name=name,
            description=description or "(none)",
            arg_key=arg_key,
            arg_spec=(json.dumps(arg_spec)[:600] or "{}"),
            n=n,
        )},
    ]
    raw = brain.chat(msgs, temperature=0.7, max_tokens=400)
    seen: set[str] = set()
    uniq: list[str] = []
    for a in _parse_str_array(raw):
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq[: max(n, 1)]
