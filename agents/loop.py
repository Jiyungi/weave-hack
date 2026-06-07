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

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from control_plane.trace import attributes, op

from . import cp, grounding, teacher, tools
from .brain import Brain, get_brain
from .workers import worker_scope_guidance


# How long a worker waits for a *human* to decide a sensitive request before it
# gives up and proceeds without the skill. Safe skills are auto-approved instantly
# so this only bites the human-in-the-loop path.
APPROVAL_TIMEOUT_S = float(os.environ.get("OPENMIRROR_APPROVAL_TIMEOUT_S", "120"))
APPROVAL_POLL_S = float(os.environ.get("OPENMIRROR_APPROVAL_POLL_S", "2"))
_OBS_MAX = int(os.environ.get("OPENMIRROR_OBS_MAX_CHARS", "600"))
_STYLE_MAX = int(os.environ.get("OPENMIRROR_STYLE_MAX_NEW_TOKENS", "128"))


def _clip_obs(text: str) -> str:
    return grounding.clip_observation(text, _OBS_MAX)


def _act_prompt(task: str) -> str:
    return f"User: {task}\nAssistant:"


def _completion_is_prose(completion: str) -> bool:
    from control_plane.runtime import extract_tool_calls

    text = (completion or "").strip()
    if not text:
        return False
    return not extract_tool_calls(text)


def _style_tokens(max_new_tokens: int) -> int:
    return max(max_new_tokens, _STYLE_MAX)


@op(name="agent.styled_completion")
def styled_completion(session_id: str, task: str, *, max_new_tokens: int,
                      fallback: str = "") -> str:
    """Run ``/act`` on a personalized session controller for natural-language output."""
    try:
        out = cp.act(session_id, _act_prompt(task),
                     max_new_tokens=_style_tokens(max_new_tokens))
        comp = (out.get("completion") or "").strip()
        if _completion_is_prose(comp):
            return comp.lstrip()
    except cp.ControlPlaneError:
        pass
    return fallback


def _user_has_style(user_id: str) -> bool:
    try:
        return bool(cp.state().get("personalization", {}).get(user_id))
    except Exception:  # noqa: BLE001
        return False


@op(name="agent.styled_completion_for_user")
def styled_completion_for_user(user_id: str, principal: str, skills: list[str],
                               task: str, *, max_new_tokens: int,
                               fallback: str = "",
                               session_key: str | None = None) -> str:
    """Open (or reuse) a session with ``user_style-{user_id}`` composed in."""
    if not user_id or not _user_has_style(user_id):
        return fallback
    key = session_key or f"style-{user_id}"
    try:
        sess = cp.open_session(principal, skills, user_id=user_id,
                               session_key=key, reuse=True)
        if not sess.get("personalized"):
            return fallback
        return styled_completion(sess["session_id"], task,
                                 max_new_tokens=max_new_tokens, fallback=fallback)
    except cp.ControlPlaneError:
        return fallback


def _styled_passes_grounding(styled: str, *, ground_task: str, evidence: str,
                             had_tool_steps: bool, original: str) -> bool:
    has_useful = bool(evidence.strip())
    issue = grounding.final_grounding_issue(
        styled,
        evidence,
        require_evidence=had_tool_steps and not has_useful,
    )
    if issue:
        return False
    if not grounding.final_completeness_issue(
        ground_task, styled, evidence, had_delegations=had_tool_steps,
    ):
        orig = grounding.extract_claim_tokens(original)
        ev = grounding.extract_claim_tokens(evidence)
        required = orig & ev
        if required and not required <= grounding.extract_claim_tokens(styled):
            return False
        return True
    return False


@op(name="agent.personalize_final")
def personalize_final(*, session_id: str | None, user_id: str | None,
                      principal: str, skills: list[str], task: str,
                      answer: str, evidence: str = "",
                      ground_task: str | None = None,
                      had_tool_steps: bool = False,
                      max_new_tokens: int = 64,
                      session_key: str | None = None) -> str:
    """Apply ``user_style`` via composed 7B ``/act`` when grounding still holds."""
    ground_task = ground_task or task
    styled = ""
    if session_id:
        styled = styled_completion(session_id, task, max_new_tokens=max_new_tokens,
                                   fallback="")
    elif user_id:
        styled = styled_completion_for_user(
            user_id, principal, skills, task,
            max_new_tokens=max_new_tokens, fallback="",
            session_key=session_key,
        )
    if not styled or styled == answer:
        return answer
    if _styled_passes_grounding(styled, ground_task=ground_task, evidence=evidence,
                                had_tool_steps=had_tool_steps, original=answer):
        return styled
    return answer


SYSTEM_TEMPLATE = """You are an agent named '{principal}'.

Your task: {task}

{scope_guidance}
You have these tools (the OpenMirror control plane may block calls outside
your authorized capability set):

{tools_doc}
{request_block}{revoked_block}
Protocol -- respond with EXACTLY ONE of the following, each on its own line:

  THOUGHT: <one short sentence of reasoning>     (optional; may precede an action)
  ACTION: tool_name("argument")                  (request one tool call)
  FINAL: <concise answer to the task>            (end the loop){request_protocol}

Rules:
- Exactly one ACTION per turn (or one FINAL).
- Use the exact tool call format: tool_name("argument") with double quotes --
  EXCEPT tools shown with a ```python``` block, which use that block form instead.
- If a previous ACTION was BLOCKED or DROPPED, choose a different tool or finish
  with FINAL using what you already know.{request_rule}
- Do not invent numbers, dates, or facts — FINAL must only cite values that
  appeared in prior OBSERVATION lines from allowed tool calls.
- Keep going until you can answer. Do not loop -- stop with FINAL.
"""

_REQUEST_BLOCK = """
You can also ACQUIRE a capability you don't have yet. These skills exist but you
are NOT currently authorized for them:

{requestable_doc}
"""
_REVOKED_OVERRIDE_BLOCK = """
These skills were revoked for this chat. If one fits your task, REQUEST it (not
ACTION) to ask the operator to re-grant for this session:

{revoked_doc}
"""
_REQUEST_PROTOCOL = '\n  REQUEST: skill_name | why you need it          (ask to be granted a skill)'
_REQUEST_RULE = (
    "\n- Prefer your authorized tools. Only REQUEST when none of them can do the task."
    "\n- If a session-revoked skill matches the task (e.g. web_search for facts),"
    " REQUEST that skill instead of minting an unrelated tool like python."
)


_ACTION_RE = re.compile(r"^ACTION:\s*([A-Za-z_]\w*)\s*\(\s*['\"]([^'\"]*)['\"]\s*\)\s*$",
                        re.MULTILINE)
# FINAL is terminal, so capture everything after it (greedy, multi-line) -- a
# lazy match here would truncate code blocks / multi-line answers to one line.
_FINAL_RE = re.compile(r"^FINAL:\s*(.+)\Z", re.MULTILINE | re.DOTALL)
_THOUGHT_RE = re.compile(r"^THOUGHT:\s*(.+?)$", re.MULTILINE)
_REQUEST_RE = re.compile(
    r"REQUEST:\s*([A-Za-z_]\w*)\s*(?:\|\s*(.*))?",
    re.MULTILINE | re.IGNORECASE,
)
# Block-mode action: a bare `ACTION: <tool>` line whose argument is a multi-line
# fenced code block on the following lines (for tools like `python`).
_ACTION_BARE_RE = re.compile(r"^ACTION:\s*([A-Za-z_]\w*)\s*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)


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
    requested_skill: Optional[str] = None                  # self-improvement: skill asked for
    request_reason: str = ""
    request_status: str = ""                               # approved | denied | pending | timeout
    request_decided_by: str = ""                           # auto | human

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
            "requested_skill": self.requested_skill,
            "request_reason": self.request_reason,
            "request_status": self.request_status,
            "request_decided_by": self.request_decided_by,
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
        if s.get("sensitive"):
            flag = " [sensitive]"
        elif s["requires_key"]:
            flag = " [requires key]"
        else:
            flag = ""
        if s.get("arg_mode") == "block":
            usage = (f"    to use, reply EXACTLY in this form:\n    ACTION: {s['name']}\n"
                     f"    ```python\n    <complete self-contained script>\n    ```\n"
                     f"    (include all imports/defs each time; then FINAL with the result)")
        else:
            usage = f"    example: {s['example_call']}"
        lines.append(f"- {s['name']}: {s['description']}{flag}\n{usage}")
    return "\n".join(lines) if lines else "(no tools available)"


def _usage_hint(skill: str) -> str:
    """One-line reminder of how to invoke a skill (block tools use a fence)."""
    try:
        if tools.get(skill).arg_mode == "block":
            return f'ACTION: {skill} then a ```python ... ``` block'
    except tools.ToolError:
        pass
    return f'ACTION: {skill}("...")'


def _action_echo(tool_name: str, arg: str) -> str:
    """Render an assistant action for the conversation history (block-aware)."""
    try:
        block = tools.get(tool_name).arg_mode == "block"
    except tools.ToolError:
        block = False
    if block:
        return f"ACTION: {tool_name}\n```python\n{arg}\n```"
    return f'ACTION: {tool_name}("{arg}")'


def _parse_block_action(text: str) -> tuple[str, str] | None:
    """Parse ACTION: tool + fenced block (lines need not be adjacent)."""
    bare_m = _ACTION_BARE_RE.search(text)
    if not bare_m:
        bare_m = re.search(r"ACTION:\s*([A-Za-z_]\w*)\s*(?:\n|$)", text)
    fence_m = _CODE_FENCE_RE.search(text)
    if bare_m and fence_m:
        return bare_m.group(1), fence_m.group(1).rstrip("\n")
    return None


def _parse_error_hint(*, requestable: list[str], block_tools: frozenset[str]) -> str:
    parts = ['ACTION: tool_name("arg")', "FINAL: <answer>"]
    if requestable:
        parts.append("REQUEST: skill_name | why you need it")
    for name in sorted(block_tools):
        parts.append(f"ACTION: {name}\\n```python\\n<complete script>\\n```")
    return (
        "could not parse a valid response; reply with exactly one of: "
        + "; ".join(parts)
        + "."
    )


def _parse_brain(
    text: str,
    *,
    block_tools: frozenset[str] = frozenset(),
) -> tuple[str, Optional[tuple[str, str]], Optional[str], Optional[tuple[str, str]]]:
    """Return (thought, (tool, arg) | None, final | None, (skill, reason) | None)."""
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""
    final_m = _FINAL_RE.search(text)
    if final_m:
        return thought, None, final_m.group(1).strip(), None
    req_m = _REQUEST_RE.search(text)
    if req_m:
        reason = (req_m.group(2) or "").strip()
        if not reason:
            # Brain sometimes puts the reason on the next line after REQUEST: skill
            tail = text[req_m.end():].strip().splitlines()
            if tail and not tail[0].startswith(("ACTION:", "FINAL:", "THOUGHT:", "REQUEST:", "```")):
                reason = tail[0].strip()
        return thought, None, None, (req_m.group(1), reason)
    block = _parse_block_action(text)
    if block:
        return thought, block, None, None
    if block_tools:
        fence_m = _CODE_FENCE_RE.search(text)
        if fence_m and len(block_tools) == 1:
            name = next(iter(block_tools))
            return thought, (name, fence_m.group(1).rstrip("\n")), None, None
    action_m = _ACTION_RE.search(text)
    if action_m:
        return thought, (action_m.group(1), action_m.group(2)), None, None
    return thought, None, None, None


def _prior_evidence(steps: list[Step]) -> str:
    parts: list[str] = []
    for step in steps:
        parts.extend(step.observations)
    return "\n".join(parts)


def _step_has_useful_obs(step: Step) -> bool:
    return any(grounding.observation_is_useful(obs) for obs in step.observations)


def _grounding_task(task: str, root_task: str | None) -> str:
    return root_task or task


def _factual_stalled(steps: list[Step], ground_task: str) -> bool:
    if not grounding.task_expects_concrete_answer(ground_task):
        return False
    if grounding.extract_claim_tokens(_prior_evidence(steps)):
        return False
    tools = [s.proposed_tool for s in steps if s.proposed_tool]
    return len(tools) >= 2 and len(set(tools)) == 1


def _format_observation(step: Step) -> str:
    """Render a step into the OBSERVATION message fed back to the brain."""
    if step.note and not step.allowed and not step.blocked:
        return f"OBSERVATION: {step.note}"
    parts: list[str] = []
    if step.allowed:
        for tname, obs in zip(step.allowed, step.observations):
            parts.append(f"ALLOWED {tname}(...): {_clip_obs(obs)}")
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


def _resolve_probe_arg(tool_name: str, mint_args: list[str], task: str,
                       brain: Brain | None) -> str | None:
    arg = teacher.pick_probe_arg(mint_args)
    if arg:
        return arg
    if not task:
        return None
    try:
        tool = tools.get(tool_name)
        syn = teacher.synthesize_args(
            tool.name, tool.description, "arg", context=task, n=3, brain=brain,
        )
        return teacher.pick_probe_arg(syn)
    except Exception:  # noqa: BLE001
        return None


def _govern_and_execute(step: Step, session_id: str, tool_name: str, arg: str,
                        max_new_tokens: int, *, task: str = "",
                        brain: Brain | None = None) -> None:
    """Emit via the skill's solo controller (act_gate), never session compose."""
    tool = tools.get(tool_name)
    mint_args: list[str] = []
    try:
        mint_args = cp.state().get("skill_probe_args", {}).get(tool_name, [])
    except Exception:  # noqa: BLE001
        pass
    if tool.arg_mode in ("gate", "block"):
        probe_arg = _resolve_probe_arg(tool_name, mint_args, task, brain)
        if not probe_arg:
            step.note = (f"no gate probe arg for {tool_name!r} "
                         f"(re-register or REQUEST-mint with context)")
            step.blocked = [tool_name]
            return
        prompt = tool.prompt_template.format(arg=probe_arg)
        exec_arg = arg
    else:
        prompt = tool.prompt_template.format(arg=arg)
        exec_arg = arg
    try:
        governed = cp.act_gate(session_id, tool_name, prompt,
                               max_new_tokens=max_new_tokens)
    except cp.ControlPlaneError as e:
        step.note = str(e)
        step.blocked = [tool_name]
        return
    step.governed_completion = governed.get("completion", "")
    step.allowed = list(governed.get("allowed_calls", []))
    step.blocked = list(governed.get("blocked_calls", []))
    if tool_name not in step.allowed and tool.needle:
        if tool.needle in (step.governed_completion or ""):
            step.allowed = [tool_name]
    if tool_name in step.allowed:
        step.observations = [tools.execute(tool_name, exec_arg)]


@op(name="agent.step")
def _step(brain: Brain, messages: list[dict], session_id: str,
          max_new_tokens: int, *,
          block_tools: frozenset[str] = frozenset(),
          requestable: list[str] | None = None,
          task: str = "",
          authorized: set[str] | None = None) -> Step:
    """One brain turn: reason, propose, govern, execute."""
    step = Step()
    raw = brain.chat(messages)
    thought, action, final, request = _parse_brain(raw, block_tools=block_tools)
    step.thought = thought
    if final is not None:
        step.final = final
        return step
    if request is not None:
        skill, reason = request
        if authorized and skill in authorized:
            step.note = f"already authorized for {skill!r}; use ACTION, not REQUEST"
            step.requested_skill = skill
            step.request_reason = reason
            step.request_status = "denied"
            return step
        step.requested_skill, step.request_reason = skill, reason
        return step
    if action is None:
        step.note = _parse_error_hint(
            requestable=requestable or [],
            block_tools=block_tools,
        )
        return step
    tool_name, arg = action
    step.proposed_tool, step.proposed_arg = tool_name, arg
    if tool_name not in tools.registry():
        step.note = f"unknown tool {tool_name!r}; available: {sorted(tools.registry())}"
        return step
    tool = tools.get(tool_name)
    _govern_and_execute(step, session_id, tool_name, arg, max_new_tokens,
                        task=task, brain=brain)
    return step


def _requestable_doc(requestable: list[str]) -> str:
    lines = []
    for name in requestable:
        try:
            t = tools.get(name)
            flag = " [sensitive: needs approval]" if (t.sensitive or t.requires_key) else ""
            lines.append(f"- {name}: {t.description}{flag}")
        except tools.ToolError:
            lines.append(f"- {name}")
    return "\n".join(lines) if lines else "(none)"


def _build_system(principal: str, task: str, visible_tools: list[str] | None,
                  requestable: list[str], revoked_overridable: list[str],
                  allow_requests: bool) -> str:
    can_request = allow_requests and (bool(requestable) or bool(revoked_overridable))
    scope = worker_scope_guidance(principal)
    scope_block = f"{scope}\n\n" if scope else ""
    return SYSTEM_TEMPLATE.format(
        principal=principal,
        task=task,
        scope_guidance=scope_block,
        tools_doc=_tools_doc(visible_tools),
        request_block=_REQUEST_BLOCK.format(requestable_doc=_requestable_doc(requestable))
        if allow_requests and requestable else "",
        revoked_block=_REVOKED_OVERRIDE_BLOCK.format(
            revoked_doc=_requestable_doc(revoked_overridable),
        ) if allow_requests and revoked_overridable else "",
        request_protocol=_REQUEST_PROTOCOL if can_request else "",
        request_rule=_REQUEST_RULE if can_request else "",
    )


def _mint_context(reason: str, task: str) -> str:
    """Organic context for teacher minting (REQUEST reason + worker task)."""
    parts = [p.strip() for p in (reason, task) if p and p.strip()]
    return "\n".join(parts)


def _provision_meta(skill: str, catalog: set[str], *,
                    context: str = "",
                    brain: Brain | None = None) -> tuple[bool, Optional[list[dict]], str, str]:
    """Resolve (sensitive, mint_examples, description, examples_source).

    A skill already in the catalog needs no examples (its controller exists) --
    approval just grants it. A known local tool not yet minted gets
    teacher-synthesized training data (with static sample_args as a floor).
    """
    try:
        t = tools.get(skill)
    except tools.ToolError:
        return False, None, "", "static"
    if skill in catalog:
        return bool(t.sensitive or t.requires_key), None, t.description, "static"
    examples, source = teacher.mint_examples(t, context=context or None, brain=brain)
    return bool(t.sensitive or t.requires_key), examples, t.description, source


def _await_decision(request_id: str) -> tuple[str, str]:
    """Poll a pending request until decided or timed out. Returns (status, by)."""
    deadline = time.time() + APPROVAL_TIMEOUT_S
    while time.time() < deadline:
        rec = cp.get_capability_request(request_id)
        if rec.get("status") != "pending":
            return rec.get("status", "denied"), rec.get("decided_by", "")
        time.sleep(APPROVAL_POLL_S)
    return "timeout", ""


@op(name="agent.acquire_skill")
def _acquire_skill(principal: str, skill: str, reason: str, session_id: str,
                   catalog: set[str], *, task: str = "",
                   brain: Brain | None = None) -> Step:
    """Run one self-improvement step: request a skill, wait for the hybrid
    decision, and (on approval) it is already granted + composed into the live
    session by the control plane. Returns a Step recording the outcome."""
    step = Step(requested_skill=skill, request_reason=reason)
    if skill not in tools.registry():
        step.request_status = "denied"
        step.note = f"cannot acquire {skill!r}: no such tool exists to grant"
        return step
    if not tools.key_configured(skill):
        step.request_status = "denied"
        step.note = f"{skill} requires an API key that is not configured"
        return step
    sensitive, examples, description, source = _provision_meta(
        skill, catalog, context=_mint_context(reason, task), brain=brain,
    )
    resp = cp.request_capability(principal, skill, reason=reason, session_id=session_id,
                                 sensitive=sensitive, examples=examples,
                                 description=description)
    status = resp.get("status", "pending")
    decided_by = resp.get("decided_by", "") or ""
    if status == "pending":
        status, decided_by = _await_decision(resp["request_id"])
    step.request_status = status
    step.request_decided_by = decided_by
    if examples and source == "teacher":
        step.note = f"mint examples from teacher ({len(examples)} pairs)"
    elif examples:
        step.note = f"mint examples from static registry ({len(examples)} pairs)"
    return step


@op(name="agent.run")
def run(principal: str, skills: list[str], task: str, *,
        compose_skills: list[str] | None = None,
        user_id: str | None = None,
        session_key: str | None = None,
        root_task: str | None = None,
        max_steps: int = 6,
        max_new_tokens: int = 64,
        brain: Brain | None = None,
        tools_filter: list[str] | None = None,
        allow_requests: bool = True) -> RunResult:
    """Run a governed agent loop end to end.

    ``principal`` and ``skills`` are sent straight to the control plane to open
    a session; the resulting authorized set drives both the governed emission
    and the runtime guard. ``compose_skills`` lets you provision a broader
    model-level capability than policy (defense-in-depth demo).

    ``tools_filter``, if given, narrows what the brain is *told about*. The
    governance set still comes from the session's authorized list -- this only
    keeps the brain's prompt focused.
    """
    sess = cp.open_session(principal, skills, compose_skills=compose_skills,
                           user_id=user_id, session_key=session_key)
    session_id = sess["session_id"]
    personalized = bool(sess.get("personalized"))
    authorized = list(sess.get("authorized", []))
    denied = list(sess.get("denied", []))
    session_revoked = set(sess.get("session_revoked", []))

    brain = brain or get_brain()
    ground_task = _grounding_task(task, root_task)

    # Skills the worker could ACQUIRE
    # plus locally-known tools not yet minted (approval mints them). This is what
    # turns "I can't" into "let me ask for the capability".
    catalog: set[str] = set()
    requestable: list[str] = []
    if allow_requests:
        try:
            catalog = set(cp.state().get("skills", {}).keys())
        except Exception:  # noqa: BLE001 — self-improvement is best-effort
            catalog = set()
        seen = set(authorized) | session_revoked
        for name in sorted(catalog | set(tools.registry())):
            if name in seen:
                continue
            if not tools.key_configured(name):
                continue
            requestable.append(name)

    revoked_overridable: list[str] = []
    if allow_requests:
        for name in sorted(session_revoked):
            if name in authorized:
                continue
            if name not in catalog and name not in tools.registry():
                continue
            if not tools.key_configured(name):
                continue
            revoked_overridable.append(name)

    visible_tools = tools_filter if tools_filter is not None else authorized

    def _block_tools() -> frozenset[str]:
        names: set[str] = set()
        for name in visible_tools:
            try:
                if tools.get(name).arg_mode == "block":
                    names.add(name)
            except tools.ToolError:
                pass
        return frozenset(names)

    def _system() -> dict:
        return {"role": "system",
                "content": _build_system(principal, task, visible_tools,
                                         requestable, revoked_overridable,
                                         allow_requests)}

    messages: list[dict] = [_system(), {"role": "user", "content": task}]

    steps: list[Step] = []
    stopped_reason = "max_steps"
    # Tag every traced op in this loop with the governance context so the Weave
    # trace tree is filterable by principal / session / authorized capability.
    with attributes({"principal": principal, "session_id": session_id,
                     "authorized": authorized, "denied": denied}):
        for _ in range(max_steps):
            step = _step(
                brain, messages, session_id, max_new_tokens,
                block_tools=_block_tools(),
                requestable=requestable,
                task=task,
                authorized=set(authorized),
            )
            if step.final is not None:
                evidence = _prior_evidence(steps)
                has_useful = any(_step_has_useful_obs(s) for s in steps)
                issue = grounding.final_grounding_issue(
                    step.final,
                    evidence,
                    require_evidence=not has_useful,
                )
                if not issue:
                    issue = grounding.final_completeness_issue(
                        ground_task, step.final, evidence, had_delegations=bool(steps),
                    )
                if issue:
                    steps.append(step)
                    messages.append({"role": "assistant",
                                     "content": f"FINAL: {step.final}"})
                    messages.append({"role": "user",
                                     "content": f"OBSERVATION: {issue} "
                                     "Run another ACTION or revise FINAL."})
                    continue
                if personalized and step.final:
                    step.final = personalize_final(
                        session_id=session_id,
                        user_id=user_id,
                        principal=principal,
                        skills=skills,
                        task=task,
                        answer=step.final,
                        evidence=evidence,
                        ground_task=ground_task,
                        had_tool_steps=bool(steps),
                        max_new_tokens=max_new_tokens,
                        session_key=session_key,
                    )
                steps.append(step)
                stopped_reason = "final"
                break

            # Self-improvement: the brain asked for a capability it lacks.
            if step.requested_skill is not None and allow_requests:
                skill = step.requested_skill
                if step.request_status == "denied" and step.note:
                    steps.append(step)
                    messages.append({"role": "assistant",
                                     "content": f"REQUEST: {skill} | {step.request_reason}"})
                    messages.append({"role": "user", "content":
                                     f"OBSERVATION: {step.note}"})
                    continue
                outcome = _acquire_skill(principal, skill, step.request_reason,
                                         session_id, catalog, task=task, brain=brain)
                step.request_status = outcome.request_status
                step.request_decided_by = outcome.request_decided_by
                step.note = outcome.note
                steps.append(step)
                messages.append({"role": "assistant",
                                 "content": f"REQUEST: {skill} | {step.request_reason}"})
                if step.request_status == "approved":
                    # The control plane granted + composed the skill into this
                    # live session; reflect it locally so the brain can use it.
                    if skill not in authorized:
                        authorized.append(skill)
                    session_revoked.discard(skill)
                    if skill in revoked_overridable:
                        revoked_overridable.remove(skill)
                    if tools_filter is None and skill not in visible_tools:
                        visible_tools.append(skill)
                    if skill in requestable:
                        requestable.remove(skill)
                    catalog.add(skill)
                    messages[0] = _system()
                    by = step.request_decided_by or "authority"
                    messages.append({"role": "user", "content":
                                     f"OBSERVATION: GRANTED {skill} (approved by {by}). "
                                     f"You may now use it -- {_usage_hint(skill)}."})
                else:
                    detail = step.note or f"request was {step.request_status}"
                    messages.append({"role": "user", "content":
                                     f"OBSERVATION: REQUEST {skill} not granted "
                                     f"({detail}). Use another tool or finish with FINAL."})
                continue

            steps.append(step)
            messages.append({"role": "assistant", "content":
                             (f"THOUGHT: {step.thought}\n" if step.thought else "")
                             + (_action_echo(step.proposed_tool, step.proposed_arg)
                                if step.proposed_tool else "")})
            messages.append({"role": "user", "content": _format_observation(step)})
            if _factual_stalled(steps, ground_task):
                messages.append({
                    "role": "user",
                    "content": (
                        "OBSERVATION: Repeated tool attempts did not produce a numeric "
                        "answer. FINAL now with an explicit could-not-verify statement "
                        "(do not refer users to external websites)."
                    ),
                })

    final_answer = steps[-1].final if steps and steps[-1].final else None
    if user_id and final_answer:
        try:
            cp.log_interaction(user_id, task, final_answer)
        except Exception:  # noqa: BLE001 — memory logging is best-effort
            pass
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
