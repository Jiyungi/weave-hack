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
{context_block}
Produce {n} realistic, diverse values for the parameter "{arg_key}".
- Each value must be a plausible real argument for THIS specific tool, not a placeholder like "example" or "test".
- Vary the length and specifics; prefer natural multi-word values where appropriate so the model learns to quote them.
- Respect the parameter's type / format / enum if the spec gives one.
- Output ONLY a JSON array of exactly {n} strings."""

_CONTEXT_BLOCK = """Agent context (why this skill is needed — use to pick realistic args):
{context}
"""


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
    context: str | None = None,
    n: int = 8,
    brain: Brain | None = None,
) -> list[str]:
    """Ask the brain for realistic example values for one tool parameter.

    Raises ``BrainError`` (from ``brain.chat``) if the brain is unreachable so
    the caller can fall back to schema/heuristic args. Returns a de-duplicated,
    order-preserving list capped at ``n``.

    ``context`` (optional) carries organic task/reason text from the agent loop
    so minted controllers see args shaped like the live request — not hardcoded
    domain rules.
    """
    brain = brain or get_brain()
    schema = schema or {}
    props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
    arg_spec = props.get(arg_key, {}) if isinstance(props, dict) else {}
    ctx = (context or "").strip()
    context_block = _CONTEXT_BLOCK.format(context=ctx) if ctx else ""
    msgs = [
        {"role": "system", "content": _ARG_SYS},
        {"role": "user", "content": _ARG_USER.format(
            name=name,
            description=description or "(none)",
            arg_key=arg_key,
            arg_spec=(json.dumps(arg_spec)[:600] or "{}"),
            context_block=context_block,
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


def args_from_examples(skill: str, examples: list[dict]) -> list[str]:
    """Extract de-duplicated tool args from mint (prompt, completion) pairs."""
    from . import tools

    seen: set[str] = set()
    out: list[str] = []
    for ex in examples:
        comp = str(ex.get("completion", ""))
        arg = tools.extract_arg(comp, skill)
        if arg and arg not in seen:
            seen.add(arg)
            out.append(arg)
    return out


def pick_probe_arg(mint_args: list[str], fallback: list[str], *, max_len: int = 160) -> str:
    """Gate probe: first mint arg that fits, else shortest mint, else static floor."""
    for pool in (mint_args, fallback):
        if not pool:
            continue
        for a in pool:
            if a and len(a) <= max_len:
                return a
        return min(pool, key=len)
    return "..."


def _merge_args(*groups: list[str]) -> list[str]:
    """De-duplicated union preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for a in group:
            s = str(a).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


@op(name="teacher.mint_examples")
def mint_examples(
    tool,
    *,
    context: str | None = None,
    arg_key: str = "arg",
    schema: dict | None = None,
    extra_args: list[str] | None = None,
    n: int = 8,
    brain: Brain | None = None,
) -> tuple[list[dict], str]:
    """Build (prompt, completion) pairs for minting a controller on Track A.

    Returns ``(examples, source)`` where ``source`` is ``"teacher"`` when the
    brain contributed args, else ``"static"``. Always folds in the tool's static
    ``sample_args`` as a floor so mint never depends solely on brain output.
    """
    from . import tools as tools_mod  # local import avoids cycle at module load

    if not isinstance(tool, tools_mod.Tool):
        raise TypeError(f"expected tools.Tool, got {type(tool)!r}")

    synthesized: list[str] = []
    source = "static"
    try:
        synthesized = synthesize_args(
            tool.name, tool.description, arg_key, schema,
            context=context, n=n, brain=brain,
        )
        if synthesized:
            source = "teacher"
    except Exception:  # noqa: BLE001 — callers must always get mintable examples
        pass

    args = _merge_args(synthesized, extra_args or [], tool.sample_args)
    if not args:
        args = list(tool.sample_args)
    examples = tool.examples_for_args(args)
    return examples, source
