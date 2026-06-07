"""Track D: multi-agent orchestrator.

A planner decomposes a user task and delegates each sub-task to one of the
governed worker agents. The orchestrator itself has no tools and runs no
governed session -- its only privilege is *deciding who to ask*. Workers act
under their own OpenMirror policies, so the same sub-task delegated to two
different workers will succeed or be blocked based on each worker's authorized
capabilities.

Default worker roster (see ``agents.workers``):

    research-agent -> web / docs lookup
    ops-agent      -> code / compute / workspace
    support-agent  -> weather only (narrow; calendar blocked for contrast)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from control_plane.trace import attributes, op

from . import cp, grounding, loop
from .brain import Brain, get_brain
from .workers import (
    OPS_AGENT,
    RESEARCH_AGENT,
    WorkerSpec,
    default_policy_for,
    default_workers,
    merge_policy,
    orchestrator_routing_hints,
)

_OBS_MAX = int(os.environ.get("OPENMIRROR_OBS_MAX_CHARS", "600"))

Worker = WorkerSpec


def _clip(text: str, limit: int | None = None) -> str:
    limit = limit if limit is not None else _OBS_MAX
    return grounding.clip_observation(text, limit)


ORCH_SYSTEM = """You are the orchestrator. You coordinate a team of governed
worker agents to complete a task. You do NOT call tools directly. You decide
who to ask and what to ask them.

Workers and their authorized tools (the OpenMirror control plane enforces this
at the weight level; if you delegate a sub-task to a worker that lacks the
needed tool, the run will be BLOCKED -- choose wisely):

{worker_doc}

Routing hints:
{routing_hints}

Task: {task}

Protocol -- respond with EXACTLY ONE of:

  THOUGHT: <short reasoning>
  DELEGATE: <worker_name> | <one-sentence sub-task for the worker>
  FINAL: <answer composed from prior results>

Rules:
- Issue one DELEGATE per turn. Wait for the result before the next step.
- You are the planner: choose ONE worker and ONE tool strategy per delegation.
  Workers do not pick alternate routes — if a delegation fails, YOU pick the next worker.
- If a delegation comes back BLOCKED, you MUST DELEGATE the same sub-task to a
  different worker before FINAL. Do not give up after one BLOCKED.
- If a delegation reports quote source failed (stock_price/crypto_price), DELEGATE
  research-agent with a web_search sub-task — do not send http_fetch or python.
- Do not FINAL until every sub-task either succeeded or you exhausted workers.
- Stop with FINAL once you have enough information. Do not loop endlessly.
- FINAL must cite only values that appear in delegation observations. Do not
  invent numbers, dates, or facts, and never use placeholder text like {{output}}.
"""


_DELEGATE_RE = re.compile(
    r"^DELEGATE:\s*([A-Za-z_][\w\-]*)\s*\|\s*(.+?)\s*$", re.MULTILINE
)
_FINAL_RE = re.compile(r"^FINAL:\s*(.+)\Z", re.MULTILINE | re.DOTALL)
_THOUGHT_RE = re.compile(r"^THOUGHT:\s*(.+?)$", re.MULTILINE)


def _worker_doc(workers: list[WorkerSpec], snapshot: dict) -> str:
    policies = snapshot.get("policies", {})
    lines = []
    for w in workers:
        allowed = policies.get(w.name, [])
        lines.append(
            f"- {w.name}: {w.description} | authorized tools: {allowed or '(none yet)'}"
        )
    return "\n".join(lines) if lines else "(no workers)"


@dataclass
class Delegation:
    worker: str
    subtask: str
    thought: str = ""
    result: Optional[loop.RunResult] = None
    note: str = ""

    def had_blocked(self) -> bool:
        if self.result is None:
            return False
        return any(s.blocked for s in self.result.steps)

    def summarize(self) -> str:
        if self.note and self.result is None:
            return f"DELEGATION ERROR ({self.worker}): {self.note}"
        r = self.result
        allowed = sorted({a for s in r.steps for a in s.allowed})
        blocked = sorted({b for s in r.steps for b in s.blocked})
        obs_lines = []
        for i, s in enumerate(r.steps, 1):
            if s.allowed:
                for tname, obs in zip(s.allowed, s.observations):
                    obs_lines.append(f"  step{i} {tname}: {_clip(obs)}")
            if s.blocked:
                for tname in s.blocked:
                    obs_lines.append(f"  step{i} BLOCKED {tname}")
        summary = [
            f"DELEGATION ({self.worker}) -> {self.subtask!r}",
            f"  authorized: {r.authorized}  denied: {r.denied}",
            f"  tools used (allowed): {allowed}  blocked: {blocked}",
            f"  final: {_clip(r.final_answer or '(no FINAL emitted)', 800)}",
        ]
        if blocked and not allowed:
            summary.append(
                "  GOVERNANCE: this worker was BLOCKED. Retry the same sub-task "
                "with a different worker (research-agent, ops-agent, or support-agent)."
            )
        if obs_lines:
            summary.append("  observations:")
            summary.extend(obs_lines)
        elif allowed:
            summary.append("  observations: (empty — do not FINAL with invented facts)")
        elif not blocked:
            summary.append(
                "  observations: (no usable tool output — try another worker or tool)"
            )
        hint = _delegation_routing_hint(self)
        if hint:
            summary.append(hint)
        return "\n".join(summary)


def _delegation_routing_hint(d: Delegation) -> str | None:
    """Tell the orchestrator what to try next — workers do not chain random tools."""
    if d.result is None:
        return None
    obs_text = "\n".join(
        o for s in d.result.steps for o in s.observations
    ).lower()
    if d.worker == OPS_AGENT and (
        "no price found" in obs_text
        or "quote source unavailable" in obs_text
        or "stock_price failed" in obs_text
    ):
        return (
            "  ORCHESTRATOR: structured quote failed on ops-agent — next DELEGATE "
            "research-agent with web_search (not http_fetch or python)."
        )
    if d.worker == RESEARCH_AGENT and "no results for" in obs_text:
        return (
            "  ORCHESTRATOR: web_search empty — try a shorter query or ops-agent "
            "stock_price if a ticker is known."
        )
    tools_used = {t for s in d.result.steps for t in s.allowed}
    if len(tools_used) >= 3 and not _delegation_has_evidence(d):
        return (
            "  ORCHESTRATOR: worker tried many tools without useful output — "
            "DELEGATE to a different worker with a single focused sub-task."
        )
    return None


@dataclass
class OrchestratorResult:
    task: str
    delegations: list[Delegation]
    final_answer: Optional[str]
    stopped_reason: str

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "delegations": [
                {
                    "worker": d.worker,
                    "subtask": d.subtask,
                    "thought": d.thought,
                    "note": d.note,
                    "result": d.result.to_dict() if d.result else None,
                }
                for d in self.delegations
            ],
            "final_answer": self.final_answer,
            "stopped_reason": self.stopped_reason,
        }


def _pending_blocked(delegations: list[Delegation]) -> bool:
    if not delegations:
        return False
    return delegations[-1].had_blocked()


def _delegation_has_evidence(d: Delegation) -> bool:
    if d.result is None:
        return False
    for step in d.result.steps:
        if _step_has_useful_obs(step):
            return True
    return False


def _step_has_useful_obs(step: loop.Step) -> bool:
    return any(grounding.observation_is_useful(obs) for obs in step.observations)


def _collect_observations(delegations: list[Delegation]) -> str:
    parts: list[str] = []
    for d in delegations:
        if d.result is None:
            continue
        for step in d.result.steps:
            parts.extend(step.observations)
    return "\n".join(parts)


def _final_grounding_issue(task: str, final: str,
                           delegations: list[Delegation]) -> str | None:
    """Return a rejection message when FINAL is not supported by observations."""
    evidence = _collect_observations(delegations)
    has_evidence = any(_delegation_has_evidence(d) for d in delegations)
    issue = grounding.final_grounding_issue(
        final,
        evidence,
        require_evidence=not has_evidence,
    )
    if issue:
        return issue
    return grounding.final_completeness_issue(
        task,
        final,
        evidence,
        had_delegations=bool(delegations),
    )


@op(name="orch.ensure_workers_seeded")
def ensure_workers_seeded(workers: list[WorkerSpec] | None = None, *,
                          reset_policies: bool = False) -> dict:
    """Ensure each orchestrator worker has an appropriate policy."""
    workers = workers or default_workers()
    snap = cp.state()
    available = set(snap.get("skills", {}).keys())
    policies = snap.get("policies", {})
    roster = []
    for w in workers:
        current = set(policies.get(w.name, []))
        if reset_policies:
            grants_set = default_policy_for(w.name, available)
        elif w.name not in policies:
            grants_set = default_policy_for(w.name, available)
        else:
            grants_set = merge_policy(w.name, current, available)
        grants = sorted(grants_set)
        if not grants:
            roster.append({"worker": w.name, "policy": [], "minted": False,
                           "note": "no skills minted yet"})
            continue
        if grants != sorted(current):
            cp.set_policy(w.name, grants)
            roster.append({"worker": w.name, "policy": grants, "minted": True})
        else:
            roster.append({"worker": w.name, "policy": grants, "minted": False})
    return {"workers": roster}


def _parse_orchestrator(text: str) -> tuple[str, Optional[tuple[str, str]], Optional[str]]:
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""
    final_m = _FINAL_RE.search(text)
    if final_m:
        return thought, None, final_m.group(1).strip()
    delegate_m = _DELEGATE_RE.search(text)
    if delegate_m:
        return thought, (delegate_m.group(1).strip(), delegate_m.group(2).strip()), None
    return thought, None, None


def _synthesize_when_exhausted(task: str, delegations: list[Delegation]) -> str:
    """Honest orchestrator answer when delegations are used up without a grounded FINAL."""
    observation_bits: list[str] = []
    for d in delegations:
        if d.result is None:
            continue
        for step in d.result.steps:
            for tname, obs in zip(step.allowed, step.observations):
                if not grounding.observation_is_useful(obs):
                    continue
                clipped = grounding.clip_observation(obs, 220)
                observation_bits.append(f"{tname}: {clipped}")
    if observation_bits:
        joined = "; ".join(observation_bits)
        if len(observation_bits) == 1:
            return f"Based on tool results: {joined}"
        return f"Based on tool results: {joined}"

    notes: list[str] = []
    for d in delegations:
        if d.result and d.result.final_answer:
            fa = d.result.final_answer.strip()
            evidence = "\n".join(
                o for s in d.result.steps for o in s.observations
            )
            if not grounding.final_grounding_issue(fa, evidence):
                return fa
            notes.append(fa)
    if notes:
        return notes[-1]
    return f"Could not verify an answer for: {task}"


@op(name="orch.delegate")
def _delegate(worker: WorkerSpec, subtask: str, *, max_steps: int,
              max_new_tokens: int, brain: Brain,
              user_id: str | None = None,
              session_key: str | None = None,
              root_task: str | None = None) -> Delegation:
    d = Delegation(worker=worker.name, subtask=subtask)
    try:
        d.result = loop.run(
            principal=worker.name,
            skills=worker.requested_skills,
            task=subtask,
            root_task=root_task,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            brain=brain,
            user_id=user_id,
            session_key=session_key,
        )
    except Exception as e:  # noqa: BLE001
        d.note = f"{type(e).__name__}: {e}"
    return d


@op(name="orch.run")
def run(task: str, *,
        workers: list[WorkerSpec] | None = None,
        max_delegations: int = 6,
        worker_max_steps: int = 6,
        worker_max_new_tokens: int = 64,
        brain: Brain | None = None,
        ensure_seeded: bool = True,
        user_id: str | None = None,
        chat_id: str | None = None,
        history: list[dict] | None = None,
        force_worker: str | None = None) -> OrchestratorResult:
    """Run the orchestrator end-to-end on a task."""
    workers = workers or default_workers()
    force_worker = force_worker or os.environ.get("OPENMIRROR_FORCE_WORKER", "").strip() or None
    if ensure_seeded:
        ensure_workers_seeded(workers)

    brain = brain or get_brain()
    snap = cp.state()
    available = sorted(snap.get("skills", {}).keys())

    sys_msg = ORCH_SYSTEM.format(
        worker_doc=_worker_doc(workers, snap),
        routing_hints=orchestrator_routing_hints(workers),
        task=task,
    )
    messages: list[dict] = [{"role": "system", "content": sys_msg}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": task})

    by_name = {w.name: w for w in workers}
    delegations: list[Delegation] = []
    stopped_reason = "max_delegations"
    final_answer: Optional[str] = None

    with attributes({"task": task, "workers": sorted(by_name),
                     "available_skills": available}):
        for _ in range(max_delegations):
            if force_worker and not delegations:
                worker_name, subtask = force_worker, task
                thought = f"forced worker {force_worker!r}"
                raw = f"DELEGATE: {worker_name} | {subtask}"
            else:
                raw = brain.chat(messages)
                thought, action, final = _parse_orchestrator(raw)
                if final is not None:
                    if _pending_blocked(delegations):
                        messages.append({"role": "assistant", "content": raw.strip()})
                        messages.append({
                            "role": "user",
                            "content": (
                                "You may not FINAL yet: the last delegation was BLOCKED. "
                                "DELEGATE the same sub-task to a different worker first."
                            ),
                        })
                        continue
                    grounding = _final_grounding_issue(task, final, delegations)
                    if grounding:
                        messages.append({"role": "assistant", "content": raw.strip()})
                        messages.append({"role": "user", "content": grounding})
                        continue
                    final_answer = final
                    stopped_reason = "final"
                    break
                if action is None:
                    d = Delegation(worker="(planner)", subtask="",
                                   note="planner returned no DELEGATE or FINAL")
                    delegations.append(d)
                    stopped_reason = "parse_error"
                    break
                worker_name, subtask = action

            if worker_name not in by_name:
                d = Delegation(worker=worker_name, subtask=subtask, thought=thought,
                               note=f"unknown worker {worker_name!r}; "
                                    f"available: {sorted(by_name)}")
                delegations.append(d)
                messages.append({"role": "assistant", "content": raw.strip()})
                messages.append({"role": "user", "content": d.summarize()})
                continue

            d = _delegate(by_name[worker_name], subtask,
                          max_steps=worker_max_steps,
                          max_new_tokens=worker_max_new_tokens,
                          brain=brain, user_id=user_id,
                          session_key=chat_id, root_task=task)
            d.thought = thought
            delegations.append(d)
            messages.append({"role": "assistant", "content": raw.strip()})
            messages.append({"role": "user", "content": d.summarize()})

    if final_answer is None and stopped_reason == "max_delegations":
        final_answer = _synthesize_when_exhausted(task, delegations)
        stopped_reason = "exhausted"

    if user_id and final_answer:
        try:
            cp.log_interaction(user_id, task, final_answer)
        except Exception:  # noqa: BLE001
            pass

    return OrchestratorResult(
        task=task,
        delegations=delegations,
        final_answer=final_answer,
        stopped_reason=stopped_reason,
    )
