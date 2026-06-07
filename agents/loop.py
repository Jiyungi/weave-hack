"""Governed ReAct loop for a single agent.

The brain (any OpenAI-compatible LLM) reasons and proposes tool calls. Each
proposed call is *governed*: the loop hands it to the control plane's
``/act`` endpoint, which runs it through the per-session composed controllers
on Qwen2.5-7B and the runtime guard. The loop only executes the tools the
control plane says are allowed; everything else becomes an observation telling
the brain it was blocked or the model couldn't emit the call.

Why this split:

- Brain decides *what to try* (ungoverned, swappable).
- Control plane decides *what is expressible/allowed* (governed at the weight
  level by composed NTK controllers + runtime guard).
- This module decides *what happens* when an allowed call runs (real tool
  execution from ``agents/tools.py``).

That separation is the whole point of OpenMirror: revoking a tool mid-run
literally subtracts its controller, so the governed model can no longer emit
the call -- the brain's intent becomes inert.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from control_plane.trace import op

from . import cp, tools
from .brain import Brain, get_brain


SYSTEM_TEMPLATE = """You are an agent named '{principal}'.

Your task: {task}

You have these tools (the OpenMirror control plane may block calls outside
your authorized capability set):

{tools_doc}

Protocol -- respond with EXACTLY ONE of the following, each on its own line:

  THOUGHT: <one short sentence of reasoning>     (optional; may precede an action)
  ACTION: tool_name("argument")                  (request one tool call)
  FINAL: <concise answer to the task>            (end the loop)

Rules:
- Exactly one ACTION per turn (or one FINAL).
- Use the exact tool call format: tool_name("argument") with double quotes.
- If a previous ACTION was BLOCKED or DROPPED, choose a different tool or finish
  with FINAL using what you already know.
- Keep going until you can answer. Do not loop -- stop with FINAL.
"""


_ACTION_RE = re.compile(r"^ACTION:\s*([A-Za-z_]\w*)\s*\(\s*['\"]([^'\"]*)['\"]\s*\)\s*$",
                        re.MULTILINE)
_FINAL_RE = re.compile(r"^FINAL:\s*(.+?)(?:\n|$)", re.MULTILINE | re.DOTALL)
_THOUGHT_RE = re.compile(r"^THOUGHT:\s*(.+?)$", re.MULTILINE)


@dataclass
class Step:
    """One iteration of the loop, captured for tracing/audit."""
    thought: str = ""
    proposed_tool: Optional[str] = None
    proposed_arg: str = ""
    governed_completion: str = ""
    allowed: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)  # one per allowed call
    final: Optional[str] = None
    note: str = ""                                          # parse / unknown-tool notes

    def to_dict(self) -> dict:
        return {
            "thought": self.thought,
            "proposed_tool": self.proposed_tool,
            "proposed_arg": self.proposed_arg,
            "governed_completion": self.governed_completion,
            "allowed": self.allowed,
            "blocked": self.blocked,
            "observations": self.observations,
            "final": self.final,
            "note": self.note,
        }


@dataclass
class RunResult:
    principal: str
    task: str
    session_id: str
    authorized: list[str]
    denied: list[str]
    steps: list[Step]
    final_answer: Optional[str]
    stopped_reason: str

    def to_dict(self) -> dict:
        return {
            "principal": self.principal,
            "task": self.task,
            "session_id": self.session_id,
            "authorized": self.authorized,
            "denied": self.denied,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "stopped_reason": self.stopped_reason,
        }


def _tools_doc(only: list[str] | None = None) -> str:
    items = tools.schemas()
    if only is not None:
        wanted = set(only)
        items = [s for s in items if s["name"] in wanted]
    lines = []
    for s in items:
        flag = " [requires key]" if s["requires_key"] else ""
        lines.append(f"- {s['name']}: {s['description']}{flag}\n    example: {s['example_call']}")
    return "\n".join(lines) if lines else "(no tools available)"


def _parse_brain(text: str) -> tuple[str, Optional[tuple[str, str]], Optional[str]]:
    """Return (thought, (tool, arg) | None, final | None) from a brain response."""
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""
    final_m = _FINAL_RE.search(text)
    if final_m:
        return thought, None, final_m.group(1).strip()
    action_m = _ACTION_RE.search(text)
    if action_m:
        return thought, (action_m.group(1), action_m.group(2)), None
    return thought, None, None


def _format_observation(step: Step) -> str:
    """Render a step into the OBSERVATION message fed back to the brain."""
    if step.note and not step.allowed and not step.blocked:
        return f"OBSERVATION: {step.note}"
    parts: list[str] = []
    if step.allowed:
        for tname, obs in zip(step.allowed, step.observations):
            parts.append(f"ALLOWED {tname}(...): {obs}")
    if step.blocked:
        for tname in step.blocked:
            parts.append(f"BLOCKED {tname}(...): runtime guard denied -- not in your authorized set")
    if not parts:
        parts.append(
            f"DROPPED {step.proposed_tool}(...): governed model did not emit this call "
            f"(capability may have been revoked or never granted)"
        )
    return "OBSERVATION:\n  " + "\n  ".join(parts)


@op(name="agent.execute_allowed")
def _execute_allowed(allowed: list[str], completion: str) -> list[str]:
    """Run each allowed tool call by extracting its arg from the governed completion."""
    out: list[str] = []
    for name in allowed:
        arg = tools.extract_arg(completion, name) or ""
        out.append(tools.execute(name, arg))
    return out


@op(name="agent.step")
def _step(brain: Brain, messages: list[dict], session_id: str,
          max_new_tokens: int) -> Step:
    """One brain turn: reason, propose, govern, execute."""
    step = Step()
    raw = brain.chat(messages)
    thought, action, final = _parse_brain(raw)
    step.thought = thought
    if final is not None:
        step.final = final
        return step
    if action is None:
        step.note = (
            "could not parse a valid ACTION or FINAL line; please reply with exactly "
            "one of: 'ACTION: tool_name(\"arg\")' or 'FINAL: <answer>'."
        )
        return step
    tool_name, arg = action
    step.proposed_tool, step.proposed_arg = tool_name, arg
    if tool_name not in tools.registry():
        step.note = f"unknown tool {tool_name!r}; available: {sorted(tools.registry())}"
        return step
    tool = tools.get(tool_name)
    prompt = tool.prompt_template.format(arg=arg)
    governed = cp.act(session_id, prompt, max_new_tokens=max_new_tokens)
    step.governed_completion = governed.get("completion", "")
    step.allowed = list(governed.get("allowed_calls", []))
    step.blocked = list(governed.get("blocked_calls", []))
    step.observations = _execute_allowed(step.allowed, step.governed_completion)
    return step


@op(name="agent.run")
def run(principal: str, skills: list[str], task: str, *,
        compose_skills: list[str] | None = None,
        user_id: str | None = None,
        max_steps: int = 6,
        max_new_tokens: int = 32,
        brain: Brain | None = None,
        tools_filter: list[str] | None = None) -> RunResult:
    """Run a governed agent loop end to end.

    ``principal`` and ``skills`` are sent straight to the control plane to open
    a session; the resulting authorized set drives both the governed emission
    and the runtime guard. ``compose_skills`` lets you provision a broader
    model-level capability than policy (defense-in-depth demo).

    ``tools_filter``, if given, narrows what the brain is *told about*. The
    governance set still comes from the session's authorized list -- this only
    keeps the brain's prompt focused.
    """
    sess = cp.open_session(principal, skills, compose_skills=compose_skills, user_id=user_id)
    session_id = sess["session_id"]
    authorized = list(sess.get("authorized", []))
    denied = list(sess.get("denied", []))

    brain = brain or get_brain()
    visible_tools = tools_filter if tools_filter is not None else authorized
    sys_msg = SYSTEM_TEMPLATE.format(
        principal=principal,
        task=task,
        tools_doc=_tools_doc(visible_tools),
    )
    messages: list[dict] = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": task},
    ]

    steps: list[Step] = []
    stopped_reason = "max_steps"
    for _ in range(max_steps):
        step = _step(brain, messages, session_id, max_new_tokens)
        steps.append(step)
        if step.final is not None:
            stopped_reason = "final"
            break
        messages.append({"role": "assistant", "content":
                         (f"THOUGHT: {step.thought}\n" if step.thought else "")
                         + (f"ACTION: {step.proposed_tool}(\"{step.proposed_arg}\")"
                            if step.proposed_tool else "")})
        messages.append({"role": "user", "content": _format_observation(step)})

    final_answer = steps[-1].final if steps and steps[-1].final else None
    return RunResult(
        principal=principal,
        task=task,
        session_id=session_id,
        authorized=authorized,
        denied=denied,
        steps=steps,
        final_answer=final_answer,
        stopped_reason=stopped_reason,
    )
