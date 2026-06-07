"""Role-based worker roster and skill grant defaults for multi-agent orchestration.

Each worker is a control-plane *principal* with its own policy. The orchestrator
routes sub-tasks to the worker whose grants fit the job; differential governance
shows up when the wrong worker is chosen (BLOCKED) and the planner retries another.
"""
from __future__ import annotations

from dataclasses import dataclass

RESEARCH_AGENT = "research-agent"
OPS_AGENT = "ops-agent"
SUPPORT_AGENT = "support-agent"

# Legacy principals (CP smoke tests + migration only — not in orchestrator roster).
EXEC_ASSISTANT = "exec-assistant"
SUPPORT_BOT = "support-bot"

ORCHESTRATOR_WORKERS = (RESEARCH_AGENT, OPS_AGENT, SUPPORT_AGENT)

# Skill name sets used to seed policies and route mint grants.
_RESEARCH_SKILLS = frozenset({
    "web_search", "http_fetch", "wikipedia", "news", "doc_search", "doc_index",
    "pdf_read", "dictionary", "synonyms", "geocode", "country_info", "translate",
    "brightdata_scrape", "ip_info",
})

_OPS_SKILLS = frozenset({
    "python", "calculator", "datetime_now", "shell", "read_file", "write_file",
    "list_dir", "apply_patch", "csv_query", "sql_query", "hash_text", "base64_tool",
    "uuid_gen", "password_gen", "json_format", "regex_test", "roman",
    "number_base", "morse", "slugify", "epoch_convert", "unit_convert",
    "currency", "stock_price", "crypto_price",
})

_SUPPORT_SKILLS = frozenset({"weather", "calendar", "forecast", "timezone"})

# Narrow support policy for differential demo (calendar blocked on support-agent).
_SUPPORT_POLICY_SKILLS = frozenset({"weather"})


def skill_owners(skill: str) -> list[str]:
    """Principals that should receive a newly minted skill by default."""
    if skill in _RESEARCH_SKILLS:
        return [RESEARCH_AGENT]
    if skill in _OPS_SKILLS:
        return [OPS_AGENT]
    if skill in _SUPPORT_SKILLS:
        return [SUPPORT_AGENT]
    if skill == "http_fetch":
        return [RESEARCH_AGENT, OPS_AGENT]
    # Heuristic for tools_extra names not in the static sets.
    name = skill.lower()
    if any(k in name for k in ("search", "wiki", "news", "doc", "pdf", "fetch", "dict")):
        return [RESEARCH_AGENT]
    if any(k in name for k in ("python", "calc", "shell", "file", "sql", "csv", "hash")):
        return [OPS_AGENT]
    if any(k in name for k in ("weather", "calendar", "forecast")):
        return [SUPPORT_AGENT]
    return [RESEARCH_AGENT, OPS_AGENT]


def grants_for_skill(skill: str) -> dict[str, list[str]]:
    """``grants`` payload for ``POST /register`` — one entry per owning principal."""
    return {p: [skill] for p in skill_owners(skill)}


def default_policy_for(worker: str, available: set[str]) -> set[str]:
    """Initial policy skills for a worker (intersected with registered skills)."""
    if worker == RESEARCH_AGENT:
        base = _RESEARCH_SKILLS | {"http_fetch"}
    elif worker == OPS_AGENT:
        base = _OPS_SKILLS | {"http_fetch"}
    elif worker == SUPPORT_AGENT:
        base = _SUPPORT_POLICY_SKILLS
    else:
        return set()
    return {s for s in base if s in available}


@dataclass
class WorkerSpec:
    name: str
    description: str
    requested_skills: list[str]


def default_workers() -> list[WorkerSpec]:
    return [
        WorkerSpec(
            name=RESEARCH_AGENT,
            description="facts and web lookup (web_search, http_fetch, wikipedia, docs)",
            requested_skills=sorted(_RESEARCH_SKILLS),
        ),
        WorkerSpec(
            name=OPS_AGENT,
            description="code, compute, and workspace ops (python, calculator, files)",
            requested_skills=sorted(_OPS_SKILLS | {"http_fetch"}),
        ),
        WorkerSpec(
            name=SUPPORT_AGENT,
            description="customer support — weather only by policy (calendar denied)",
            requested_skills=sorted(_SUPPORT_SKILLS),
        ),
    ]


def merge_policy(worker: str, current: set[str], available: set[str]) -> set[str]:
    """Add role-default skills that are registered but missing from ``current``."""
    wanted = default_policy_for(worker, available)
    return current | wanted
