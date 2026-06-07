"""Pluggable reasoning brain.

OpenAI-compatible client: works against any OpenAI-spec endpoint, so the brain
is swappable without changing the agent loop. Defaults to a *local* vLLM server
(no API key, all data stays on the box) and falls back to real OpenAI by simply
flipping the env. That swappability is itself part of the pitch: the brain is
ungoverned and replaceable; the governance is what's local and novel (the
composed NTK controllers on Qwen2.5-7B).

Defaults:
    OPENMIRROR_BRAIN_BASE_URL   http://localhost:8001/v1   (local vLLM)
    OPENMIRROR_BRAIN_MODEL      Qwen/Qwen2.5-14B-Instruct
    OPENMIRROR_BRAIN_API_KEY    sk-no-key                  (vLLM ignores this)

Run vLLM on the box (in its own shell, venv activated):
    pip install vllm openai
    vllm serve Qwen/Qwen2.5-14B-Instruct --port 8001 \
        --max-model-len 8192 --gpu-memory-utilization 0.45

The 0.45 utilization is to share the 80GB A100 with the governed 7B
(controller_service on port 8000) without OOM.

To point at real OpenAI instead:
    export OPENMIRROR_BRAIN_BASE_URL=https://api.openai.com/v1
    export OPENMIRROR_BRAIN_API_KEY=sk-...
    export OPENMIRROR_BRAIN_MODEL=gpt-4o-mini
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from control_plane.trace import op


@dataclass
class BrainConfig:
    base_url: str
    model: str
    api_key: str
    temperature: float
    max_tokens: int

    @classmethod
    def from_env(cls) -> "BrainConfig":
        return cls(
            base_url=os.environ.get("OPENMIRROR_BRAIN_BASE_URL", "http://localhost:8001/v1"),
            model=os.environ.get("OPENMIRROR_BRAIN_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
            api_key=os.environ.get("OPENMIRROR_BRAIN_API_KEY", "sk-no-key"),
            temperature=float(os.environ.get("OPENMIRROR_BRAIN_TEMPERATURE", "0.2")),
            max_tokens=int(os.environ.get("OPENMIRROR_BRAIN_MAX_TOKENS", "512")),
        )


class BrainError(RuntimeError):
    """The brain endpoint was unreachable or returned a non-OK response."""


class Brain:
    """Thin wrapper around the OpenAI Python SDK. Sync, easy to trace."""

    def __init__(self, cfg: BrainConfig | None = None) -> None:
        self.cfg = cfg or BrainConfig.from_env()
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise BrainError(
                    "openai package not installed. Run: pip install openai"
                ) from e
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        return self._client

    @op(name="brain.chat")
    def chat(self, messages: list[dict], **overrides) -> str:
        client = self._client_lazy()
        try:
            resp = client.chat.completions.create(
                model=overrides.get("model", self.cfg.model),
                messages=messages,
                temperature=overrides.get("temperature", self.cfg.temperature),
                max_tokens=overrides.get("max_tokens", self.cfg.max_tokens),
            )
        except Exception as e:
            raise BrainError(f"brain call failed ({self.cfg.base_url}): {e}") from e
        try:
            return resp.choices[0].message.content or ""
        except (AttributeError, IndexError) as e:
            raise BrainError(f"unexpected brain response shape: {resp!r}") from e

    def describe(self) -> dict:
        """Diagnostics for /health-style endpoints."""
        return {
            "base_url": self.cfg.base_url,
            "model": self.cfg.model,
            "local": self.cfg.base_url.startswith("http://localhost"),
        }


_singleton: Brain | None = None


def get_brain() -> Brain:
    """Process-wide brain singleton built from environment."""
    global _singleton
    if _singleton is None:
        _singleton = Brain()
    return _singleton
