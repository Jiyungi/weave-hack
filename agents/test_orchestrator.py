"""Tests for multi-agent orchestrator (stub brain, mocked worker loop)."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from agents import loop, orchestrator
from agents.brain import Brain
from agents.workers import OPS_AGENT, RESEARCH_AGENT, SUPPORT_AGENT, default_workers


class _StubBrain(Brain):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._i = 0

    def chat(self, messages: list[dict]) -> str:
        if self._i >= len(self._responses):
            return "FINAL: done"
        out = self._responses[self._i]
        self._i += 1
        return out

    def describe(self) -> dict:
        return {"kind": "stub"}


def _run_result(principal: str, *, blocked: list[str] | None = None,
                allowed: list[str] | None = None) -> loop.RunResult:
    from agents.loop import RunResult, Step

    blocked = blocked or []
    allowed = allowed or []
    steps = [Step(
        proposed_tool=blocked[0] if blocked else (allowed[0] if allowed else None),
        blocked=blocked,
        allowed=allowed,
        observations=["ok"] if allowed else [],
    )]
    return RunResult(
        principal=principal,
        task="sub",
        session_id="sess-test",
        authorized=allowed,
        denied=blocked,
        steps=steps,
        final_answer="worker done" if allowed else None,
        stopped_reason="final" if allowed else "blocked",
    )


class OrchestratorTests(unittest.TestCase):
    def test_parse_delegate_and_final(self) -> None:
        t, action, final = orchestrator._parse_orchestrator(
            "THOUGHT: route\nDELEGATE: ops-agent | run python"
        )
        self.assertEqual(t, "route")
        self.assertEqual(action, ("ops-agent", "run python"))
        self.assertIsNone(final)

        _, action2, final2 = orchestrator._parse_orchestrator("FINAL: answer here")
        self.assertIsNone(action2)
        self.assertEqual(final2, "answer here")

    @patch("agents.orchestrator.cp")
    @patch("agents.orchestrator.loop.run")
    @patch("agents.orchestrator.ensure_workers_seeded")
    def test_blocked_then_retry(self, _seed, mock_run, mock_cp) -> None:
        mock_cp.state.return_value = {
            "skills": {"weather": "w", "calendar": "c", "python": "p"},
            "policies": {
                SUPPORT_AGENT: ["weather"],
                OPS_AGENT: ["calendar", "python"],
                RESEARCH_AGENT: ["web_search"],
            },
        }
        mock_run.side_effect = [
            _run_result(SUPPORT_AGENT, blocked=["calendar"]),
            _run_result(OPS_AGENT, allowed=["calendar"]),
        ]
        brain = _StubBrain([
            f"DELEGATE: {SUPPORT_AGENT} | check calendar on 2026-05-05",
            f"DELEGATE: {OPS_AGENT} | check calendar on 2026-05-05",
            "FINAL: events on the 5th",
        ])
        result = orchestrator.run(
            "calendar task",
            workers=default_workers(),
            brain=brain,
            ensure_seeded=False,
        )
        self.assertEqual(result.stopped_reason, "final")
        self.assertEqual(len(result.delegations), 2)
        self.assertTrue(result.delegations[0].had_blocked())
        workers_used = {d.worker for d in result.delegations}
        self.assertEqual(workers_used, {SUPPORT_AGENT, OPS_AGENT})

    @patch("agents.orchestrator.cp")
    @patch("agents.orchestrator.loop.run")
    @patch("agents.orchestrator.ensure_workers_seeded")
    def test_rejects_final_while_blocked_pending(self, _seed, mock_run, mock_cp) -> None:
        mock_cp.state.return_value = {
            "skills": {"calendar": "c"},
            "policies": {SUPPORT_AGENT: ["weather"]},
        }
        mock_run.return_value = _run_result(SUPPORT_AGENT, blocked=["calendar"])
        brain = _StubBrain([
            f"DELEGATE: {SUPPORT_AGENT} | calendar query",
            "FINAL: too early",
            f"DELEGATE: {OPS_AGENT} | calendar query",
            "FINAL: ok now",
        ])
        with patch("agents.orchestrator._delegate") as mock_del:
            mock_del.side_effect = [
                orchestrator.Delegation(
                    worker=SUPPORT_AGENT, subtask="calendar query",
                    result=_run_result(SUPPORT_AGENT, blocked=["calendar"]),
                ),
                orchestrator.Delegation(
                    worker=OPS_AGENT, subtask="calendar query",
                    result=_run_result(OPS_AGENT, allowed=["calendar"]),
                ),
            ]
            result = orchestrator.run(
                "task",
                workers=default_workers(),
                brain=brain,
                ensure_seeded=False,
            )
        self.assertEqual(result.final_answer, "ok now")


if __name__ == "__main__":
    unittest.main()
