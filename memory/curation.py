"""Turn raw chat interactions into NTK training pairs for style adapters."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_pair(user: str, assistant: str) -> dict | None:
    user, assistant = _as_text(user), _as_text(assistant)
    if not user or not assistant:
        return None
    return {
        "prompt": f"User: {user}\nAssistant:",
        "completion": f" {assistant}" if not assistant.startswith(" ") else assistant,
    }


def curate_interactions(interactions: list[dict]) -> tuple[list[dict], int]:
    """Convert logged interactions to styled (prompt, completion) pairs."""
    pairs: list[dict] = []
    discarded = 0
    for raw in interactions:
        if "prompt" in raw and "completion" in raw:
            p, c = _as_text(raw["prompt"]), _as_text(raw["completion"])
            if p and c:
                pairs.append({"prompt": p, "completion": c})
            else:
                discarded += 1
            continue
        pair = _format_pair(
            raw.get("user") or raw.get("message"),
            raw.get("assistant") or raw.get("reply"),
        )
        if pair:
            pairs.append(pair)
        else:
            discarded += 1
    return pairs, discarded


def curate_with_openai(interactions: list[dict], user_id: str) -> tuple[list[dict], int]:
    """Optional OpenAI pass; falls back to heuristic curation."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or not interactions:
        return curate_interactions(interactions)
    try:
        payload = {
            "model": os.environ.get("CURATION_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": (
                    "Extract training pairs for a style personalization adapter. "
                    "Output JSON array of {prompt, completion}. Preserve HOW the "
                    "user likes answers (tone/format), not new facts. JSON only."
                )},
                {"role": "user", "content": json.dumps({"user_id": user_id, "interactions": interactions})},
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        text = body["choices"][0]["message"]["content"]
        m = re.search(r"\[[\s\S]*\]", text)
        if not m:
            return curate_interactions(interactions)
        parsed = json.loads(m.group(0))
        pairs = [p for p in parsed if isinstance(p, dict) and p.get("prompt") and p.get("completion")]
        if pairs:
            return pairs, max(0, len(interactions) - len(pairs))
    except Exception:
        pass
    return curate_interactions(interactions)
