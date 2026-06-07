"""Tests for HTTP helpers (gzip decompression, text heuristics)."""
from __future__ import annotations

import gzip
import unittest
from unittest.mock import MagicMock, patch

from agents import tools


class HttpGetTests(unittest.TestCase):
    def test_looks_like_text_rejects_binary_garbage(self) -> None:
        garbage = "\x00\x01\x02" * 50 + "ok"
        self.assertFalse(tools.looks_like_text(garbage))

    def test_looks_like_text_accepts_plain_html(self) -> None:
        self.assertTrue(tools.looks_like_text("<html><body>NVDA 214.86</body></html>"))

    def test_decode_http_body_decompresses_gzip(self) -> None:
        raw = gzip.compress(b"hello world")
        self.assertEqual(tools._decode_http_body(raw, ""), "hello world")

    def test_decode_http_body_gzip_magic_without_header(self) -> None:
        raw = gzip.compress(b"price data")
        self.assertEqual(tools._decode_http_body(raw, "identity"), "price data")

    @patch("agents.tools.urllib.request.urlopen")
    def test_http_get_decompresses_gzip_response(self, mock_urlopen: MagicMock) -> None:
        payload = gzip.compress(b"<html>NVDA</html>")
        resp = MagicMock()
        resp.read.return_value = payload
        resp.headers = {"Content-Encoding": "gzip"}
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        body = tools._http_get("https://example.com/test")
        self.assertIn("NVDA", body)
        self.assertTrue(tools.looks_like_text(body))

    def test_http_fetch_rejects_binary_body(self) -> None:
        with patch("agents.tools._http_get", return_value="\x00\xff" * 100):
            with self.assertRaises(tools.ToolError) as ctx:
                tools._http_fetch("https://example.com/page")
        self.assertIn("binary", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
