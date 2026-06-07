"""Tests for shared grounding and observation normalization."""
from __future__ import annotations

import unittest

from agents import grounding


class GroundingTests(unittest.TestCase):
    def test_sanitize_rejects_binary(self) -> None:
        out = grounding.sanitize_observation("\x00\xff" * 50)
        self.assertIn("omitted", out)

    def test_html_to_text_strips_tags(self) -> None:
        html = "<html><body><p>NVDA closed at <b>214.86</b></p></body></html>"
        text = grounding.html_to_text(html)
        self.assertIn("214.86", text)
        self.assertNotIn("<p>", text)

    def test_normalize_tool_output_flattens_html(self) -> None:
        out = grounding.normalize_tool_output("<div>hello <span>world</span></div>")
        self.assertEqual(out, "hello world")

    def test_final_rejects_ungrounded_decimal(self) -> None:
        issue = grounding.final_grounding_issue(
            "The answer is 99.99.",
            "weather: 72F in Boston",
        )
        self.assertIsNotNone(issue)
        self.assertIn("99.99", issue or "")

    def test_final_accepts_matching_decimal(self) -> None:
        issue = grounding.final_grounding_issue(
            "NVDA closed at 214.86 yesterday.",
            "NVDA: 214.86 (date 2026-06-04, previous trading day)",
        )
        self.assertIsNone(issue)

    def test_final_accepts_explicit_uncertainty(self) -> None:
        issue = grounding.final_grounding_issue(
            "I could not verify the price.",
            "",
            require_evidence=True,
        )
        self.assertIsNone(issue)

    def test_observation_is_useful_filters_errors(self) -> None:
        self.assertFalse(grounding.observation_is_useful("[web_search error] timeout"))
        self.assertFalse(grounding.observation_is_useful("no results for 'foo'"))
        self.assertTrue(grounding.observation_is_useful("Boston: 72F, partly cloudy"))


if __name__ == "__main__":
    unittest.main()
