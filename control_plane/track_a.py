"""Thin HTTP client for Track A. Stdlib only, so Track B has no extra deps."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config
from .trace import op


class TrackAError(RuntimeError):
    """Track A returned an error or was unreachable."""


def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(config.TRACK_A_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise TrackAError(f"{path} -> {e.code}: {e.read().decode()}") from e
    except urllib.error.URLError as e:
        raise TrackAError(f"{path} unreachable at {config.TRACK_A_URL} ({e})") from e


@op(name="track_a.train")
def train(task_id: str, examples: list[dict]) -> dict:
    return _post("/train", {"task_id": task_id, "examples": examples})


@op(name="track_a.compose")
def compose(controller_ids: list[str], weights: list[float],
            new_id: str | None = None) -> dict:
    return _post("/compose", {"controller_ids": controller_ids,
                              "weights": weights, "new_id": new_id})


@op(name="track_a.execute")
def execute(controller_id: str | None, prompt: str, max_new_tokens: int) -> dict:
    return _post("/execute", {"controller_id": controller_id,
                              "prompt": prompt, "max_new_tokens": max_new_tokens})
