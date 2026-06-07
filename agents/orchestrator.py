"""Track D: multi-agent orchestrator.

A planner ("orchestrator" principal) decomposes a user task and delegates each
sub-task to one of the governed worker agents. The orchestrator itself has no
tools and runs no governed session -- its only privilege is *deciding who to
ask*. Workers act under their own OpenMirror policies, so the same sub-task
delegated to two different workers will succeed or be blocked based on each
worker's authorized capabilities.

Default worker roster (set up by ``ensure_workers_seeded`` so the demo starts
from a known state):

    exec-assistant -> all minted tool skills        (broad)
    support-bot    -> [weather] only                (restricted)

Why three agents (orchestrator + two workers): two workers with *different*
grants are the minimum to demonstrate differential governance; a third agent
that *delegates* makes "shared governance authority" concrete. Beyond three,
the demo gets noisy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from control_plane.trace import op

from . import cp, loop, tools
from .brain import Brain, get_brain


ORCH_SYSTEM = """You are the orchestrator. You coordinate a team of governed
worker agents to complete a task. You do NOT call tools directly. You decide
who to ask and what to ask them.

Workers and their authorized tools (the OpenMirror control plane enforces this
at the weight level; if you delegate a sub-task to a worker that lacks the
needed tool, the run will be BLOCKED -- choose wisely):

{worker_doc}

Task: {task}

Protocol -- respond with EXACTLY ONE of:

  THOUGHT: <short reasoning>
  DELEGATE: <worker_name> | <one-sentence sub-task for the worker>
  FINAL: <answer composed from prior results>

Rules:
- Issue one DELEGATE per turn. Wait for the result before the next step.
- If a delegation comes back BLOCKED, retry with a different worker or finish.
- Stop with FINAL once you have enough information. Do not loop.
"""


_DELEGATE_RE = re.compile(
    r"^DELEGATE:\s*([A-Za-z_][\w\-]*)\s*\|\s*(.+?)\s*$", re.MULTILINE
)
_FINAL_RE = re.compile(r"^FINAL:\s*(.+?)(?:\n|$)", re.MULTILINE | re.DOTALL)
_THOUGHT_RE = re.compile(r"^THOUGHT:\s*(.+?)$", re.MULTILINE)


@dataclass
class Worker:
    name: str               # the OpenMirror principal
    description: str        # human-readable note for the orchestrator
    # Skills the worker REQUESTS when opening a session. The control plane
    # filters this against the principal's policy; only authorized ones become
    # the live capability set.
    requested_skills: list[str]


def _worker_doc(workers: list[Worker], snapshot: dict) -> str:
    policies = snapshot.get("policies", {})
    lines = []
    for w in workers:
        allowed = policies.get(w.name, [])
        lines.append(f"- {w.name}: {w.description} | authorized tools: {allowed or '(none yet)'}")
    return "\n".join(lines) if lines else "(no workers)"


@dataclass
class Delegation:
    worker: str
    subtask: str
    thought: str = ""
    result: Optional[loop.RunResult] = None
    note: str = ""

    def summarize(self) -> str:
        """Brain-facing one-paragraph summary of a delegation outcome."""
        if self.note and self.result is None:
            return f"DELEGATION ERROR ({self.worker}): {self.note}"
        r = self.result
        allowed = sorted({a for s in r.steps for a in s.allowed})
        blocked = sorted({b for s in r.steps for b in s.blocked})
        obs_lines = []
        for i, s in enumerate(r.steps, 1):
            if s.allowed:
                for tname, obs in zip(s.allowed, s.observations):
                    obs_lines.append(f"  step{i} {tname}: {obs}")
            if s.blocked:
                for tname in s.blocked:
                    obs_lines.append(f"  step{i} BLOCKED {tname}")
        summary = [
            f"DELEGATION ({self.worker}) -> {self.subtask!r}",
            f"  authorized: {r.authorized}  denied: {r.denied}",
            f"  tools used (allowed): {allowed}  blocked: {blocked}",
            f"  final: {r.final_answer or '(no FINAL emitted)'}",
        ]
        if obs_lines:
            summary.append("  observations:")
            summary.extend(obs_lines)
        return "\n".join(summary)


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


# --- worker seeding -------------------------------------------------------


def default_workers() -> list[Worker]:
    """Two-worker roster matching the existing seed: one broad, one restricted."""
    return [
        Worker(
            name="exec-assistant",
            description="broadly-capable assistant",
            requested_skills=["weather", "calendar", "web_search"],
        ),
        Worker(
            name="support-bot",
            description="customer support, weather-only by policy",
            requested_skills=["weather", "calendar", "web_search"],
        ),
    ]


@op(name="orch.ensure_workers_seeded")
def ensure_workers_seeded(workers: list[Worker] | None = None) -> dict:
    """Make sure each worker has a policy in the control plane.

    Idempotent: if a policy already exists for a principal it's left as-is.
    Skills that aren't yet registered are skipped silently -- the registry
    endpoint (or the dashboard seed) is the right place to mint them.
    """
    workers = workers or default_workers()
    snap = cp.state()
    available_skills = set(snap.get("skills", {}).keys())
    policies = snap.get("policies", {})
    roster = []
    for w in workers:
        if w.name in policies:
            roster.append({"worker": w.name, "policy": policies[w.name], "minted": False})
            continue
        # Default grants: support-bot -> [weather] only; everyone else -> all known
        grants = ["weather"] if w.name == "support-bot" else sorted(available_skills)
        grants = [g for g in grants if g in available_skills]
        if not grants:
            roster.append({"worker": w.name, "policy": [], "minted": False,
                           "note": "no skills minted yet"})
            continue
        cp.set_policy(w.name, grants)
        roster.append({"worker": w.name, "policy": grants, "minted": True})
    return {"workers": roster}


# --- orchestrator loop ----------------------------------------------------


def _parse_orchestrator(text: str) -> tuple[str, Optional[tuple[str, str]], Optional[str]]:
    """Return (thought, (worker, subtask) | None, final | None)."""
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""
    final_m = _FINAL_RE.search(text)
    if final_m:
        return thought, None, final_m.group(1).strip()
    delegate_m = _DELEGATE_RE.search(text)
    if delegate_m:
        return thought, (delegate_m.group(1).strip(), delegate_m.group(2).strip()), None
    return thought, None, None


@op(name="orch.delegate")
def _delegate(worker: Worker, subtask: str, *, max_steps: int,
              max_new_tokens: int, brain: Brain) -> Delegation:
    """Run one worker on one sub-task; tolerate worker errors."""
    d = Delegation(worker=worker.name, subtask=subtask)
    try:
        d.result = loop.run(
            principal=worker.name,
            skills=worker.requested_skills,
            task=subtask,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            brain=brain,
        )
    except Exception as e:  # noqa: BLE001
        d.note = f"{type(e).__name__}: {e}"
    return d


@op(name="orch.run")
def run(task: str, *,
        workers: list[Worker] | None = None,
        max_delegations: int = 4,
        worker_max_steps: int = 4,
        worker_max_new_tokens: int = 32,
        brain: Brain | None = None,
        ensure_seeded: bool = True) -> OrchestratorResult:
    """Run the orchestrator end-to-end on a task."""
    workers = workers or default_workers()
    if ensure_seeded:
        ensure_workers_seeded(workers)

    brain = brain or get_brain()
    snap = cp.state()
    # Workers request EVERY registered skill; the control plane authorizes only
    # those in each worker's policy. This makes a newly registered + granted
    # tool usable immediately, without editing the static worker roster. The
    # restricted worker (support-bot) still gets only its policy's skills.
    available = sorted(snap.get("skills", {}).keys())
    if available:
        for w in workers:
            w.requested_skills = available
    sys_msg = ORCH_SYSTEM.format(worker_doc=_worker_doc(workers, snap), task=task)
    messages: list[dict] = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": task},
    ]

    by_name = {w.name: w for w in workers}
    delegations: list[Delegation] = []
    stopped_reason = "max_delegations"
    final_answer: Optional[str] = None

    for _ in range(max_delegations):
        raw = brain.chat(messages)
        thought, action, final = _parse_orchestrator(raw)
        if final is not None:
            final_answer = final
            stopped_reason = "final"
            break
        if action is None:
            d = Delegation(worker="(planner)", subtask="",
                           note="planner returned no DELEGATE or FINAL; expected "
                                "'DELEGATE: <worker> | <sub-task>' or 'FINAL: ...'")
            delegations.append(d)
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
                      brain=brain)
        d.thought = thought
        delegations.append(d)
        messages.append({"role": "assistant", "content": raw.strip()})
        messages.append({"role": "user", "content": d.summarize()})

    return OrchestratorResult(
        task=task,
        delegations=delegations,
        final_answer=final_answer,
        stopped_reason=stopped_reason,
    )
