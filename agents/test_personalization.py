"""Tests for weight-memory personalization on FINAL answers."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from agents import loop


class PersonalizationTests(unittest.TestCase):
    @patch("agents.loop.cp")
    def test_styled_completion_uses_act(self, mock_cp) -> None:
        mock_cp.act.return_value = {"completion": " TL;DR: photosynthesis basics."}
        out = loop.styled_completion("sess-1", "explain photosynthesis",
                                       max_new_tokens=64, fallback="plain")
        self.assertEqual(out, "TL;DR: photosynthesis basics.")
        mock_cp.act.assert_called_once()
        prompt = mock_cp.act.call_args[0][1]
        self.assertIn("explain photosynthesis", prompt)

    @patch("agents.loop.cp")
    def test_styled_completion_rejects_tool_emit(self, mock_cp) -> None:
        mock_cp.act.return_value = {"completion": ' weather("Paris")'}
        out = loop.styled_completion("sess-1", "weather?", max_new_tokens=64,
                                     fallback="plain")
        self.assertEqual(out, "plain")

    @patch("agents.loop.cp")
    def test_personalize_final_keeps_answer_when_style_fails_grounding(self, mock_cp) -> None:
        mock_cp.act.return_value = {"completion": " TL;DR: no numbers here."}
        answer = "NVDA is $205.10 today."
        evidence = "stock_price(NVDA) = 205.10"
        out = loop.personalize_final(
            session_id="sess-1",
            user_id=None,
            principal="ops-agent",
            skills=["stock_price"],
            task="NVDA price?",
            answer=answer,
            evidence=evidence,
            ground_task="NVDA price?",
            had_tool_steps=True,
            max_new_tokens=64,
        )
        self.assertEqual(out, answer)

    @patch("agents.loop.cp")
    def test_personalize_final_applies_style_when_grounded(self, mock_cp) -> None:
        mock_cp.act.return_value = {
            "completion": " TL;DR: Biden, Trump, Obama, Bush, Clinton.",
        }
        answer = "Joe Biden, Donald Trump, Barack Obama, George W. Bush, and Bill Clinton."
        out = loop.personalize_final(
            session_id="sess-1",
            user_id=None,
            principal="research-agent",
            skills=["web_search"],
            task="last five US presidents",
            answer=answer,
            evidence="",
            ground_task="last five US presidents",
            had_tool_steps=False,
            max_new_tokens=64,
        )
        self.assertTrue(out.startswith("TL;DR:"))

    @patch("agents.loop.cp")
    def test_styled_completion_for_user_opens_session(self, mock_cp) -> None:
        mock_cp.state.return_value = {"personalization": {"carl": "user_style-carl-abc"}}
        mock_cp.open_session.return_value = {
            "session_id": "sess-carl",
            "personalized": True,
        }
        mock_cp.act.return_value = {"completion": " TL;DR: hello."}
        out = loop.styled_completion_for_user(
            "carl", "exec-assistant", ["web_search"],
            "say hi", max_new_tokens=64, fallback="hi",
        )
        self.assertEqual(out, "TL;DR: hello.")
        mock_cp.open_session.assert_called_once()
        kwargs = mock_cp.open_session.call_args.kwargs
        self.assertEqual(kwargs.get("user_id"), "carl")


if __name__ == "__main__":
    unittest.main()
