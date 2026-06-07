#!/usr/bin/env python3
"""Verify multi-agent orchestration: differential BLOCKED -> retry across workers.

Uses a stub brain (no vLLM). For live stack + real workers, run with --live
(requires CP :8100, Track A :8000, agent :8200, brain :8001).

    python verify_orchestrator.py          # offline stub (default)
    python verify_orchestrator.py --live   # POST /run on localhost:8200
"""
from __future__ import annotations

import argparse
import json
import sys
import unittest
import urllib.request

from agents import orchestrator
from agents.test_orchestrator import OrchestratorTests
from agents.workers import OPS_AGENT, SUPPORT_AGENT


def offline_check() -> int:
    suite = unittest.TestSuite()
    suite.addTest(OrchestratorTests("test_blocked_then_retry"))
    suite.addTest(OrchestratorTests("test_rejects_final_while_blocked_pending"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def live_check(ag_url: str) -> int:
    task = (
        "First ask support-agent for calendar events on 2026-05-05, then ops-agent "
        "if blocked. Weather in Berlin if needed."
    )
    body = json.dumps({"task": task, "ensure_seeded": True}).encode()
    req = urllib.request.Request(
        f"{ag_url.rstrip('/')}/run",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        print(f"live run failed: {e}", file=sys.stderr)
        return 1

    delegations = data.get("delegations", [])
    workers = {d.get("worker") for d in delegations}
    blocked = any(
        any(step.get("blocked") for step in (d.get("result") or {}).get("steps", []))
        for d in delegations
    )
    print(json.dumps(data, indent=2)[:4000])
    print()
    print(f"workers used: {sorted(workers)}")
    print(f"any BLOCKED step: {blocked}")
    print(f"stopped: {data.get('stopped_reason')}")

    ok = len(workers) >= 2 and blocked and data.get("final_answer")
    print("\nPASS" if ok else "\nFAIL (need >=2 workers, a BLOCKED step, and FINAL)")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify orchestrator multi-agent path")
    parser.add_argument("--live", action="store_true", help="Run against agent_service :8200")
    parser.add_argument("--ag-url", default="http://127.0.0.1:8200")
    args = parser.parse_args()
    if args.live:
        return live_check(args.ag_url)
    return offline_check()


if __name__ == "__main__":
    raise SystemExit(main())
