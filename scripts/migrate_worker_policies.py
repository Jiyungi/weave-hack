#!/usr/bin/env python3
"""Non-destructive migration: map legacy principals to role-based workers.

Reads existing policies from the control plane and *merges* skills into the
new role workers without deleting legacy principals.

    python scripts/migrate_worker_policies.py
    python scripts/migrate_worker_policies.py --dry-run

Requires CP_URL (default http://127.0.0.1:8100).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

from agents.workers import (
    EXEC_ASSISTANT,
    OPS_AGENT,
    RESEARCH_AGENT,
    SUPPORT_AGENT,
    SUPPORT_BOT,
    default_policy_for,
    merge_policy,
    skill_owners,
)

CP_URL = os.environ.get("CP_URL", "http://127.0.0.1:8100").rstrip("/")

_RESEARCH_HINTS = frozenset({"web_search", "http_fetch", "wikipedia", "news", "doc_search"})
_OPS_HINTS = frozenset({"python", "calculator", "shell", "read_file", "write_file"})
_SUPPORT_HINTS = frozenset({"weather", "calendar", "forecast"})


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{CP_URL}{path}", timeout=30) as r:
        return json.loads(r.read().decode())


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{CP_URL}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _split_skill(skill: str) -> str:
    """Return target role worker for a skill name."""
    return skill_owners(skill)[0]


def migrate(*, dry_run: bool = False) -> dict:
    snap = _get("/state")
    available = set(snap.get("skills", {}).keys())
    policies = snap.get("policies", {})

    targets: dict[str, set[str]] = {
        RESEARCH_AGENT: set(policies.get(RESEARCH_AGENT, [])),
        OPS_AGENT: set(policies.get(OPS_AGENT, [])),
        SUPPORT_AGENT: set(policies.get(SUPPORT_AGENT, [])),
    }

    for legacy in (EXEC_ASSISTANT, SUPPORT_BOT):
        for skill in policies.get(legacy, []):
            if skill in _RESEARCH_HINTS or skill in _OPS_HINTS:
                if skill in _RESEARCH_HINTS:
                    targets[RESEARCH_AGENT].add(skill)
                if skill in _OPS_HINTS:
                    targets[OPS_AGENT].add(skill)
            elif skill in _SUPPORT_HINTS:
                targets[SUPPORT_AGENT].add(skill)
            else:
                targets[_split_skill(skill)].add(skill)

    for worker in targets:
        targets[worker] = merge_policy(worker, targets[worker], available)
        if worker not in policies:
            targets[worker] |= default_policy_for(worker, available)

    out = {}
    for worker, skills in targets.items():
        grants = sorted(skills & available)
        out[worker] = grants
        if dry_run:
            print(f"would set {worker} -> {grants}")
        else:
            _post("/policy", {"principal": worker, "allowed_skills": grants})
            print(f"set {worker} -> {grants}")

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        migrate(dry_run=args.dry_run)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
